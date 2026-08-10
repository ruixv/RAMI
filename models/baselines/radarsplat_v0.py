"""RadarSplat v0 — first minimal radar-aware 3DGS.

Architecture: identical to B1 (gsplat DefaultStrategy + SH degree 3 + SH
warmup) plus one new loss term anchored to the LIRA anchor cloud:

    L_radar = mean_{voxel v in LIRA} max(0, min_dist(v, Gaussian means) - τ)

i.e. for every LIRA-occupied voxel center v, penalize if the nearest
Gaussian mean is further than τ meters away. This says "wherever the radar
sees a surface, there must be at least one Gaussian near it" — a coverage
constraint that does NOT depend on photons.

Total loss:
    L = (1 - λ_ssim) · L1 + λ_ssim · (1 - SSIM) + λ_radar · L_radar

The radar loss is independent of α, so it provides clean geometric signal
under low light where the photometric L1+SSIM signal collapses.

This is the *simplest possible* radar-side contribution. Phase 3 will add
the full L_cmc 3D-occupancy consistency, intrinsic decomposition, and
albedo-on-surface invariance. v0 establishes whether radar geometry helps
at all under low light.

Hyperparameters (Phase 2 P1 defaults):
    λ_radar = 0.05
    τ (radar coverage tolerance) = 0.10 m  (= scene_scale / 18 on B408)
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import torch

from ._gsplat_common import (
    SH_DEGREE_MAX,
    ColmapData,
    GaussianModel,
    K_from_camera,
    colmap_image_to_w2c,
    init_gaussians_from_points3D,
    load_checkpoint,
    load_image_to_tensor,
    parse_colmap_export,
    render_view,
    save_checkpoint,
)
from .base import Baseline, BaselineMeta, TrainResult, register
from .vanilla_3dgs import _build_optimizers, _scene_scale_from_cameras, _ssim


def _load_anchor_positions(export_dir: str) -> np.ndarray:
    """Re-read the points3D.txt we wrote from LIRA. Returns (N, 3)."""
    pts: list[list[float]] = []
    path = os.path.join(export_dir, "sparse", "0", "points3D.txt")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            pts.append([float(p) for p in parts[1:4]])
    return np.asarray(pts, dtype=np.float32)


def _radar_coverage_loss(
    gauss_means: torch.Tensor,
    anchor_positions: torch.Tensor,
    tau_m: float,
    sample_k: int | None = None,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """For each anchor in `anchor_positions` (N_a, 3), find nearest Gaussian
    mean and penalize the gap above `tau_m`. Optionally sub-sample the
    anchors to `sample_k` per iter for cost control.

    All distances in scene meters; the loss is in [0, ∞)."""
    if anchor_positions.shape[0] == 0:
        return torch.tensor(0.0, device=gauss_means.device)
    if sample_k is not None and sample_k < anchor_positions.shape[0]:
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(anchor_positions.shape[0], size=sample_k, replace=False)
        anchors = anchor_positions[idx]
    else:
        anchors = anchor_positions
    # (N_a, N_g) pairwise distances. With ~400 anchors and ~400k Gaussians
    # → 1.6e8 entries = 640 MB in fp32. We can do this directly on V100
    # but use cdist in chunks if it blows up.
    dists = torch.cdist(anchors, gauss_means)
    min_d = dists.min(dim=1).values
    return (min_d - tau_m).clamp(min=0).mean()


def _train_loop(
    data: ColmapData,
    export_dir: str,
    output_dir: str,
    config: dict,
    device: str,
) -> TrainResult:
    import gsplat

    n_iters = int(config.get("n_iters", 30000))
    lambda_ssim = float(config.get("lambda_ssim", 0.2))
    lambda_radar = float(config.get("lambda_radar", 0.05))
    tau_m = float(config.get("radar_tau_m", 0.10))
    radar_sample_k = config.get("radar_sample_k", None)
    if radar_sample_k is not None:
        radar_sample_k = int(radar_sample_k)
    seed = int(config.get("seed", 1))
    sh_warmup_every = int(config.get("sh_warmup_every", 1000))
    save_every = int(config.get("save_every", n_iters))

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    train_records = [im for im in data.images.values()
                     if im.name not in set(data.test_image_names)]
    if not train_records:
        raise RuntimeError(f"no training images in {export_dir}")
    cam = next(iter(data.cameras.values()))
    K = torch.tensor(K_from_camera(cam), dtype=torch.float32, device=device)
    width, height = cam.width, cam.height
    images_dir = os.path.join(export_dir, "images")
    train_w2cs = [
        torch.tensor(colmap_image_to_w2c(im), dtype=torch.float32, device=device)
        for im in train_records
    ]
    train_imgs = [
        load_image_to_tensor(os.path.join(images_dir, im.name), device=device)
        for im in train_records
    ]

    anchor_np = _load_anchor_positions(export_dir)
    anchor_t = torch.tensor(anchor_np, dtype=torch.float32, device=device) if anchor_np.shape[0] else None
    print(f"[RadarSplat-v0] {len(anchor_np)} radar anchors, τ={tau_m:.3f} m, λ_radar={lambda_radar}")

    scene_scale = _scene_scale_from_cameras(data)
    print(f"[RadarSplat-v0] scene_scale={scene_scale:.3f} m")

    gm = init_gaussians_from_points3D(data.points3D, device=device)
    print(f"[RadarSplat-v0] init {gm.num_gaussians} Gaussians")
    optimizers = _build_optimizers(gm, scene_scale)
    params = gm.params_dict()

    strategy = gsplat.DefaultStrategy(
        prune_opa=0.005,
        grow_grad2d=2e-4,
        grow_scale3d=0.01,
        prune_scale3d=0.1,
        refine_scale2d_stop_iter=0,
        refine_start_iter=int(config.get("densify_from_iter", 500)),
        refine_stop_iter=int(config.get("densify_until_iter", 15000)),
        reset_every=int(config.get("opacity_reset_interval", 3000)),
        refine_every=int(config.get("densify_interval", 100)),
        absgrad=bool(config.get("absgrad", False)),
        verbose=False,
    )
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    os.makedirs(output_dir, exist_ok=True)
    log_f = open(os.path.join(output_dir, "train.log"), "w")
    t0 = time.time()
    last_log = 0.0
    loss = torch.tensor(0.0, device=device)
    l_radar_log = 0.0

    for step in range(1, n_iters + 1):
        sh_degree = min(SH_DEGREE_MAX, (step - 1) // sh_warmup_every)

        i = int(rng.integers(0, len(train_records)))
        scales = torch.exp(gm.scales)
        quats = torch.nn.functional.normalize(gm.quats, dim=-1)
        opacities = torch.sigmoid(gm.opacities)
        colors = gm.colors_for_render(sh_degree)
        renders, alphas, info = gsplat.rasterization(
            means=gm.means, quats=quats, scales=scales, opacities=opacities,
            colors=colors,
            viewmats=train_w2cs[i].unsqueeze(0), Ks=K.unsqueeze(0),
            width=width, height=height,
            sh_degree=sh_degree, packed=False,
            absgrad=strategy.absgrad,
        )
        rendered = renders[0].clamp(0.0, 1.0)
        gt = train_imgs[i].permute(1, 2, 0)

        strategy.step_pre_backward(params, optimizers, strategy_state, step, info)

        l1 = (rendered - gt).abs().mean()
        ssim_val = _ssim(rendered, gt)
        l_photo = (1.0 - lambda_ssim) * l1 + lambda_ssim * (1.0 - ssim_val)

        if anchor_t is not None and lambda_radar > 0:
            finite_mask = torch.isfinite(gm.means).all(dim=1)
            if finite_mask.any():
                l_radar = _radar_coverage_loss(
                    gm.means[finite_mask], anchor_t, tau_m,
                    sample_k=radar_sample_k, rng=rng,
                )
                if torch.isfinite(l_radar):
                    loss = l_photo + lambda_radar * l_radar
                    l_radar_log = float(l_radar.detach().cpu())
                else:
                    loss = l_photo
                    l_radar_log = float("nan")
            else:
                loss = l_photo
                l_radar_log = float("nan")
        else:
            loss = l_photo
            l_radar_log = 0.0

        loss.backward()
        strategy.step_post_backward(params, optimizers, strategy_state, step, info)
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)

        now = time.time()
        if now - last_log > 5.0 or step == 1 or step == n_iters:
            msg = (f"[RadarSplat-v0] iter {step}/{n_iters}  loss={float(loss.detach().cpu()):.4f}  "
                   f"L_radar={l_radar_log:.4f}  |G|={gm.num_gaussians}  sh={sh_degree}  "
                   f"elapsed={now - t0:.0f}s")
            print(msg); log_f.write(msg + "\n"); log_f.flush()
            last_log = now

        if (step % save_every == 0) or step == n_iters:
            save_checkpoint(gm, os.path.join(output_dir, "checkpoints", f"gaussians_iter{step}.pt"))

    log_f.close()
    elapsed = time.time() - t0
    return TrainResult(
        checkpoint_dir=os.path.join(output_dir, "checkpoints"),
        n_iters=n_iters, seconds_elapsed=elapsed,
        extra={"final_loss": float(loss.detach().cpu()),
               "final_n_gaussians": gm.num_gaussians,
               "scene_scale": scene_scale,
               "lambda_radar": lambda_radar, "tau_m": tau_m,
               "n_anchors": len(anchor_np)},
    )


@register
class RadarSplatV0(Baseline):
    meta = BaselineMeta(
        key="radarsplat_v0",
        paper_id="radarsplat_v0",
        venue="ours, Phase 2",
        tier=0,
        requires_gsplat=True,
        upstream_repo="",
        notes="Minimal radar-aware 3DGS: B1 + L_radar coverage loss anchored to LIRA voxels.",
    )

    def train(self, export_dir: str, output_dir: str, config: Optional[dict] = None) -> TrainResult:
        config = config or {}
        if not torch.cuda.is_available():
            raise RuntimeError("RadarSplat-v0 requires CUDA")
        data = parse_colmap_export(export_dir)
        return _train_loop(data, export_dir, output_dir, config, device="cuda")

    def render(self, checkpoint_dir: str, w2c_matrices: np.ndarray, K: np.ndarray,
               image_w: int, image_h: int) -> np.ndarray:
        ckpts = sorted(
            f for f in os.listdir(checkpoint_dir)
            if f.endswith(".pt") and "gaussians" in f
        )
        gm = load_checkpoint(os.path.join(checkpoint_dir, ckpts[-1]), device="cuda")
        K_t = torch.tensor(K, dtype=torch.float32, device="cuda")
        outs = []
        with torch.no_grad():
            for w2c in w2c_matrices:
                w2c_t = torch.tensor(w2c, dtype=torch.float32, device="cuda")
                img, _ = render_view(gm, w2c_t, K_t, image_w, image_h, sh_degree=SH_DEGREE_MAX)
                outs.append(img.cpu().numpy())
        return np.stack(outs, axis=0).astype(np.float32)
