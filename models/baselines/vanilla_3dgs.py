"""B1 — Vanilla 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023).

Implemented on top of gsplat's `DefaultStrategy` + SH degree 3, following
the official 3DGS training recipe:

  - Adam optimizers per parameter group with the canonical learning rates
    (lr_means scaled by scene extent, lr_sh0 = 0.0025, lr_shN = lr_sh0/20,
    lr_opacities = 0.05, lr_scales = 0.005, lr_quats = 0.001)
  - L1 + SSIM photometric loss (λ_ssim = 0.2)
  - SH-degree warmup: start at degree 0, +1 every `sh_warmup_every` iters
    up to degree 3
  - gsplat.DefaultStrategy handles densification (clone/split/prune/opacity
    reset). 30000 default iters with refine in [500, 15000] and opacity
    reset every 3000 iters.

This produces 3DGS results comparable to the official 3DGS / nerfstudio
implementations, which is the bar for B1 in the Phase 2 comparison.
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


def _ssim(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """SSIM on (H, W, 3) tensors in [0, 1]."""
    pred = pred.permute(2, 0, 1).unsqueeze(0)
    gt = gt.permute(2, 0, 1).unsqueeze(0)
    ws = 11
    sigma = 1.5
    coords = torch.arange(ws, dtype=pred.dtype, device=pred.device) - ws // 2
    g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1d = (g1d / g1d.sum()).view(1, 1, -1)
    g2d = (g1d.transpose(-1, -2) @ g1d).expand(3, 1, ws, ws)
    pad = ws // 2
    pad_fn = lambda x: torch.nn.functional.pad(x, (pad,) * 4, mode="reflect")
    mu_p = torch.nn.functional.conv2d(pad_fn(pred), g2d, groups=3)
    mu_g = torch.nn.functional.conv2d(pad_fn(gt), g2d, groups=3)
    mu_p2, mu_g2, mu_pg = mu_p * mu_p, mu_g * mu_g, mu_p * mu_g
    sig_p = torch.nn.functional.conv2d(pad_fn(pred * pred), g2d, groups=3) - mu_p2
    sig_g = torch.nn.functional.conv2d(pad_fn(gt * gt), g2d, groups=3) - mu_g2
    sig_pg = torch.nn.functional.conv2d(pad_fn(pred * gt), g2d, groups=3) - mu_pg
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_pg + c1) * (2 * sig_pg + c2)
    den = (mu_p2 + mu_g2 + c1) * (sig_p + sig_g + c2)
    return (num / den).mean()


def _scene_scale_from_cameras(data: ColmapData) -> float:
    """Approximate scene extent for scaling learning rates and the strategy."""
    centers = []
    for im in data.images.values():
        w2c = colmap_image_to_w2c(im)
        c2w = np.linalg.inv(w2c)
        centers.append(c2w[:3, 3])
    centers = np.asarray(centers)
    if centers.shape[0] < 2:
        return 1.0
    cam_center = centers.mean(axis=0)
    diag = np.linalg.norm(centers - cam_center, axis=-1).max()
    return float(max(diag, 1e-3))


def _build_optimizers(gm: GaussianModel, scene_scale: float) -> dict[str, torch.optim.Optimizer]:
    """gsplat strategies expect one Adam per parameter; lrs follow the official 3DGS recipe."""
    lr_means = 1.6e-4 * scene_scale
    return {
        "means": torch.optim.Adam([gm.means], lr=lr_means, eps=1e-15),
        "scales": torch.optim.Adam([gm.scales], lr=5e-3, eps=1e-15),
        "quats": torch.optim.Adam([gm.quats], lr=1e-3, eps=1e-15),
        "opacities": torch.optim.Adam([gm.opacities], lr=5e-2, eps=1e-15),
        "sh0": torch.optim.Adam([gm.sh0], lr=2.5e-3, eps=1e-15),
        "shN": torch.optim.Adam([gm.shN], lr=2.5e-3 / 20.0, eps=1e-15),
    }


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
    seed = int(config.get("seed", 1))
    sh_warmup_every = int(config.get("sh_warmup_every", 1000))   # +1 SH degree every N iters
    save_every = int(config.get("save_every", n_iters))

    torch.manual_seed(seed)
    np.random.seed(seed)

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

    scene_scale = _scene_scale_from_cameras(data)
    print(f"[B1] scene_scale={scene_scale:.3f} m")

    gm = init_gaussians_from_points3D(data.points3D, device=device)
    print(f"[B1] init {gm.num_gaussians} Gaussians from {data.points3D.shape[0]} points3D")

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
        absgrad=bool(config.get("absgrad", True)),
        verbose=bool(config.get("strategy_verbose", False)),
    )
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train.log")
    log_f = open(log_path, "w")
    rng = np.random.default_rng(seed)
    t0 = time.time()
    last_log = 0.0

    for step in range(1, n_iters + 1):
        # SH degree warmup: 0 → 1 → 2 → 3 every sh_warmup_every iters
        sh_degree = min(SH_DEGREE_MAX, (step - 1) // sh_warmup_every)

        i = int(rng.integers(0, len(train_records)))
        scales = torch.exp(gm.scales)
        quats = torch.nn.functional.normalize(gm.quats, dim=-1)
        opacities = torch.sigmoid(gm.opacities)
        colors = gm.colors_for_render(sh_degree)

        renders, alphas, info = gsplat.rasterization(
            means=gm.means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=train_w2cs[i].unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=width, height=height,
            sh_degree=sh_degree,
            packed=False,
            absgrad=strategy.absgrad,
        )
        rendered = renders[0].clamp(0.0, 1.0)            # (H, W, 3)
        gt = train_imgs[i].permute(1, 2, 0)              # (H, W, 3)

        # gsplat.DefaultStrategy reads `info["means2d"].absgrad` (or `.grad`)
        # after backward to decide clone/split/prune; the tensor is non-leaf
        # so its grad is dropped by default. step_pre_backward sets the
        # retain flag, but we run it before backward as the strategy spec
        # requires.
        strategy.step_pre_backward(params, optimizers, strategy_state, step, info)

        l1 = (rendered - gt).abs().mean()
        ssim_val = _ssim(rendered, gt)
        loss = (1.0 - lambda_ssim) * l1 + lambda_ssim * (1.0 - ssim_val)

        loss.backward()
        strategy.step_post_backward(params, optimizers, strategy_state, step, info)

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)

        now = time.time()
        if now - last_log > 5.0 or step == 1 or step == n_iters:
            msg = (f"[B1] iter {step}/{n_iters}  loss={float(loss.detach().cpu()):.4f}  "
                   f"|G|={gm.num_gaussians}  sh={sh_degree}  elapsed={now - t0:.0f}s")
            print(msg); log_f.write(msg + "\n"); log_f.flush()
            last_log = now

        if (step % save_every == 0) or step == n_iters:
            ckpt_path = os.path.join(output_dir, "checkpoints", f"gaussians_iter{step}.pt")
            save_checkpoint(gm, ckpt_path)

    log_f.close()
    elapsed = time.time() - t0
    final_path = os.path.join(output_dir, "checkpoints")
    return TrainResult(
        checkpoint_dir=final_path,
        n_iters=n_iters,
        seconds_elapsed=elapsed,
        extra={"final_loss": float(loss.detach().cpu()),
               "final_n_gaussians": gm.num_gaussians,
               "scene_scale": scene_scale},
    )


@register
class VanillaThreeDGS(Baseline):
    meta = BaselineMeta(
        key="vanilla_3dgs",
        paper_id="kerbl2023_3dgs",
        venue="SIGGRAPH 2023",
        tier=1,
        requires_gsplat=True,
        upstream_repo="https://github.com/graphdeco-inria/gaussian-splatting",
        notes="gsplat-backed implementation with DefaultStrategy + SH degree 3.",
    )

    def train(self, export_dir: str, output_dir: str, config: Optional[dict] = None) -> TrainResult:
        config = config or {}
        if not torch.cuda.is_available():
            raise RuntimeError("B1 requires CUDA")
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
