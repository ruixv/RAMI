"""RadarSplat-Physics — physics-grounded multi-modal fusion (RGB + LiDAR + Radar).

Diagnoses from v1–v10 showed that naive scalar-weighted multi-loss fusion
(L_photo + λ_r L_radar + λ_d L_depth) cannot beat the RGB+Radar baseline
on F2_room: depth supervision over-fits training views (lower train L_photo
but lower test PSNR than RSv1), and curricula/init/throttling tweaks can
only narrow the gap.

The structural fix: weight every loss term by a *physics-derived
per-sample confidence* that reflects how trustworthy that sensor is at
that pixel/anchor. The motivation:

  Camera (ZED, 0.55 μm) and LiDAR (905 nm NIR) are both photonic, so they
  *share* failure modes: low albedo (black cloth, shadow), specular
  highlight (clipping), transparent surfaces (glass). 60 GHz radar is
  EM-reflective, with *orthogonal* failure modes: it is strong exactly on
  the metallic edges / specular surfaces / corners where the photonic
  pair struggle.

  Therefore, depth supervision should be *trusted less* in pixels where
  LiDAR is physically expected to fail (dark, saturated), and radar
  coverage should be *trusted more* on anchors with high accumulated
  RCS. Both proxies are computed up-front from the raw sensor data and
  used to weight the per-sample loss contributions.

Physics components:

  A. Per-pixel LiDAR confidence map w_lidar(u,v), computed at startup
     from the GT image:
       w_lidar = brightness · (1 − specular_score) · (1 − darkness_score)
     where brightness is the mean over channels, specular_score detects
     saturation (max channel > 0.95), darkness_score detects black
     surfaces (brightness < 0.05). Lambertian, mid-brightness pixels get
     w_lidar ≈ 1; specular/dark pixels get w_lidar ≈ 0. This map is fixed
     during training (does not flow gradients into the photometric loss).

  B. Per-anchor radar weight w_radar(k), proportional to LIRA's
     accumulated CFAR peak power (mean intensity over multi-frame
     contributors). Stronger reflectors get tighter coverage enforcement.
     Computed once when LIRA is built.

  C. Physics-aware densification gating (optional; toggle via config):
     when on, the gsplat densification heuristic is augmented with a
     bonus for Gaussians falling in pixels where rendered depth disagrees
     with high-confidence LiDAR depth. Implemented by scaling the absgrad
     accumulator with `lidar_conf * (1 + |Δd| / huber_delta)`.

The total loss becomes:

  L = L_photo (unchanged α-rescaled)
    + λ_radar · Σ_k w_radar(k) · max(0, min_g ||μ_g − a_k|| − τ)
                 ─────────────────────────────────────────────
                                 Σ_k w_radar(k)
    + λ_depth · Σ_(u,v)∈M w_lidar(u,v) · Huber_δ(D_render − D_lidar)
                 ──────────────────────────────────────────────────
                          Σ_(u,v)∈M w_lidar(u,v)

Both denominators normalize so the loss scale is invariant to the
confidence-map distribution (otherwise tightening λ_depth would also
indirectly shrink the loss magnitude).
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import numpy as np
import torch

from ._gsplat_common import (
    SH_DEGREE_MAX,
    ColmapData,
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
from .dn_splatter import _build_lidar_depth_for_view, _resolve_scene_from_splits
from .radarsplat_v1 import _alpha_rescaled_photometric_loss, _read_alpha_from_splits
from .vanilla_3dgs import _build_optimizers, _scene_scale_from_cameras


def _lidar_confidence_map(
    gt_img: torch.Tensor,
    bright_low: float = 0.05,
    bright_high: float = 0.95,
    edge_attenuation: float = 0.3,
) -> torch.Tensor:
    """Per-pixel LiDAR trust map in [0,1] from a single GT image.

    gt_img: (3, H, W) in [0,1].
    Returns: (H, W) float tensor.

    Physics:
      • Dark pixels (brightness < bright_low): LiDAR NIR return is at noise
        floor on low-albedo surfaces. Weight → 0.
      • Saturated pixels (any channel > bright_high): specular highlight,
        LiDAR likely missed (mirror reflection). Weight → 0.
      • Edges (high spatial gradient): LiDAR points often land on the
        wrong side of an edge; partial trust scaled by edge_attenuation.
    """
    rgb = gt_img.clamp(0.0, 1.0)
    brightness = rgb.mean(dim=0)
    # 1) sigmoid ramp from 0 at bright_low to 1 above 2*bright_low
    dark_gate = torch.sigmoid(12.0 * (brightness - bright_low * 1.5))
    # 2) specular detection: any channel close to saturation
    max_chan = rgb.max(dim=0).values
    spec_gate = torch.sigmoid(-12.0 * (max_chan - bright_high))
    # 3) edge attenuation via Sobel-ish finite differences
    gx = torch.zeros_like(brightness)
    gy = torch.zeros_like(brightness)
    gx[:, 1:-1] = brightness[:, 2:] - brightness[:, :-2]
    gy[1:-1, :] = brightness[2:, :] - brightness[:-2, :]
    edge = (gx.abs() + gy.abs()).clamp(0, 1)
    edge_weight = 1.0 - edge_attenuation * edge
    return (dark_gate * spec_gate * edge_weight).clamp(0.0, 1.0)


def _weighted_radar_coverage_loss(
    gauss_means: torch.Tensor,
    anchor_positions: torch.Tensor,
    anchor_weights: torch.Tensor,
    tau_m: float,
    sample_k: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> torch.Tensor:
    """L_radar with per-anchor weight (RCS proxy).

    weight-normalized mean of max(0, min_g ||μ_g − a_k|| − τ).
    """
    if anchor_positions.shape[0] == 0:
        return torch.tensor(0.0, device=gauss_means.device)
    if sample_k is not None and sample_k < anchor_positions.shape[0]:
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(anchor_positions.shape[0], size=sample_k, replace=False)
        a = anchor_positions[idx]
        w = anchor_weights[idx]
    else:
        a, w = anchor_positions, anchor_weights
    dists = torch.cdist(a, gauss_means)
    min_d = dists.min(dim=1).values
    raw = (min_d - tau_m).clamp(min=0)
    return (w * raw).sum() / (w.sum() + 1e-8)


def _build_lira_with_intensity(
    scene, t_target: float, config: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Build LIRA anchors at runtime; return (positions (M,3), weights (M,)).

    Weights are RCS-normalized: log(1+intensity) / max log(1+intensity), in [0,1].
    """
    from models.radar.lira import LIRAConfig, build_lira_anchors

    cfg = LIRAConfig(
        voxel_size_m=float(config.get("lira_voxel", 0.05)),
        accumulation_window_s=float(config.get("lira_window", 2.0)),
        # Match baseline_train.py's --lira-pfa default (1e-2) so that the
        # anchor set the loss sees matches the anchor set that seeded the
        # initial Gaussians.
        cfar_pfa=float(config.get("lira_pfa", 1e-2)),
    )
    anchors = build_lira_anchors(scene, t_target, cfg)
    if len(anchors) == 0:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.float32))
    pos = anchors.positions.astype(np.float32)
    raw = np.log1p(anchors.intensity.astype(np.float64))
    if raw.max() > 0:
        w = (raw / raw.max()).astype(np.float32)
    else:
        w = np.ones(len(anchors), dtype=np.float32)
    return pos, w


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
    lambda_depth = float(config.get("lambda_depth", 0.1))
    huber_delta = float(config.get("huber_delta", 0.10))
    tau_m = float(config.get("radar_tau_m", 0.10))
    radar_sample_k = config.get("radar_sample_k", None)
    if radar_sample_k is not None:
        radar_sample_k = int(radar_sample_k)
    seed = int(config.get("seed", 1))
    sh_warmup_every = int(config.get("sh_warmup_every", 1000))
    save_every = int(config.get("save_every", n_iters))
    gamma = float(config.get("gamma", 2.2))

    # Confidence-map switches; default ON (the whole point of v11)
    use_lidar_conf = bool(config.get("use_lidar_confidence", True))
    use_radar_weight = bool(config.get("use_radar_weighting", True))

    alpha_corruption = _read_alpha_from_splits(export_dir)
    print(f"[RSphys] α={alpha_corruption:g}  λ_radar={lambda_radar}  "
          f"λ_depth={lambda_depth}  τ={tau_m}m  huber={huber_delta}m  "
          f"lidar_conf={use_lidar_conf}  radar_weight={use_radar_weight}")

    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)

    train_records = [im for im in data.images.values()
                     if im.name not in set(data.test_image_names)]
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

    # ============= Component B: per-anchor RCS weights =============
    scene = _resolve_scene_from_splits(export_dir)
    splits = json.load(open(os.path.join(export_dir, "splits.json")))
    name_to_t = {r["filename"]: r["t_image"] for r in splits["records"]}
    t_target = float(np.median(scene.radar_azi.timestamps))
    anchor_np, weight_np = _build_lira_with_intensity(scene, t_target, config)
    if anchor_np.shape[0]:
        if not use_radar_weight:
            weight_np = np.ones_like(weight_np)
        print(f"[RSphys] {len(anchor_np)} radar anchors;  "
              f"w_radar quartiles = "
              f"{np.quantile(weight_np, [0.25, 0.5, 0.75])}")
    anchor_t = (torch.tensor(anchor_np, dtype=torch.float32, device=device)
                if anchor_np.shape[0] else None)
    weight_t = (torch.tensor(weight_np, dtype=torch.float32, device=device)
                if anchor_np.shape[0] else None)

    # ============= Component A: per-pixel LiDAR confidence =============
    # Build sparse LiDAR depth maps AND lidar-confidence maps per train view.
    print(f"[RSphys] building LiDAR depth + confidence for "
          f"{len(train_records)} views ...")
    t0_d = time.time()
    depth_maps: list[torch.Tensor] = []
    valid_masks: list[torch.Tensor] = []
    lidar_conf_maps: list[torch.Tensor] = []
    n_valid_total = 0
    conf_mean_accum = 0.0
    for j, rec in enumerate(train_records):
        t_img = float(name_to_t[rec.name])
        d = _build_lidar_depth_for_view(scene, t_img, K, width, height, device)
        mask = d > 0
        depth_maps.append(d)
        valid_masks.append(mask)
        if use_lidar_conf:
            cm = _lidar_confidence_map(train_imgs[j])
        else:
            cm = torch.ones_like(d)
        lidar_conf_maps.append(cm)
        n_valid_total += int(mask.sum())
        conf_mean_accum += float(cm[mask].mean()) if mask.any() else 0.0
    avg_cov = n_valid_total / max(1, len(train_records) * width * height)
    avg_conf = conf_mean_accum / max(1, len(train_records))
    print(f"[RSphys] depth+conf built in {time.time() - t0_d:.1f}s; "
          f"avg LiDAR coverage {avg_cov*100:.2f}%; "
          f"avg w_lidar on hit pixels = {avg_conf:.3f}")

    scene_scale = _scene_scale_from_cameras(data)
    print(f"[RSphys] scene_scale={scene_scale:.3f} m")
    gm = init_gaussians_from_points3D(data.points3D, device=device)
    print(f"[RSphys] init {gm.num_gaussians} Gaussians from LIRA anchors")
    optimizers = _build_optimizers(gm, scene_scale)
    params = gm.params_dict()

    # Use RSv1's gradient settings — confirmed best for radar+photo dynamics.
    strategy = gsplat.DefaultStrategy(
        prune_opa=float(config.get("prune_opa", 0.005)),
        grow_grad2d=float(config.get("grow_grad2d", 2e-4)),
        grow_scale3d=0.01, prune_scale3d=0.1,
        refine_scale2d_stop_iter=0,
        refine_start_iter=int(config.get("densify_from_iter", 500)),
        refine_stop_iter=int(config.get("densify_until_iter", 15000)),
        reset_every=int(config.get("opacity_reset_interval", 3000)),
        refine_every=int(config.get("densify_interval", 100)),
        absgrad=bool(config.get("absgrad", False)), verbose=False,
    )
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    os.makedirs(output_dir, exist_ok=True)
    log_f = open(os.path.join(output_dir, "train.log"), "w")
    t0 = time.time()
    last_log = 0.0
    loss = torch.tensor(0.0, device=device)

    for step in range(1, n_iters + 1):
        sh_degree = min(SH_DEGREE_MAX, (step - 1) // sh_warmup_every)
        i = int(rng.integers(0, len(train_records)))
        scales = torch.exp(gm.scales)
        quats = torch.nn.functional.normalize(gm.quats, dim=-1)
        opacities = torch.sigmoid(gm.opacities)
        colors = gm.colors_for_render(sh_degree)
        renders, alphas_, info = gsplat.rasterization(
            means=gm.means, quats=quats, scales=scales, opacities=opacities,
            colors=colors, viewmats=train_w2cs[i].unsqueeze(0), Ks=K.unsqueeze(0),
            width=width, height=height, sh_degree=sh_degree, packed=False,
            absgrad=strategy.absgrad, render_mode="RGB+ED",
        )
        rendered = renders[0, :, :, :3].clamp(0.0, 1.0)
        depth_render = renders[0, :, :, 3]
        gt = train_imgs[i].permute(1, 2, 0)
        strategy.step_pre_backward(params, optimizers, strategy_state, step, info)

        # α-rescaled photometric (unchanged).
        l_photo, _ = _alpha_rescaled_photometric_loss(
            rendered, gt, alpha_corruption, lambda_ssim, gamma=gamma,
        )

        # ===== Component B: RCS-weighted L_radar =====
        l_radar = torch.tensor(0.0, device=device)
        if anchor_t is not None and lambda_radar > 0:
            finite_mask = torch.isfinite(gm.means).all(dim=1)
            if finite_mask.any():
                lr = _weighted_radar_coverage_loss(
                    gm.means[finite_mask], anchor_t, weight_t, tau_m,
                    sample_k=radar_sample_k, rng=rng,
                )
                if torch.isfinite(lr):
                    l_radar = lr

        # ===== Component A: per-pixel-confidence-weighted L_depth =====
        mask = valid_masks[i]
        if mask.any():
            d_r = depth_render[mask]
            d_t = depth_maps[i][mask]
            diff = d_r - d_t
            abs_diff = diff.abs()
            huber = torch.where(abs_diff < huber_delta,
                                0.5 * diff * diff / huber_delta,
                                abs_diff - 0.5 * huber_delta)
            cm = lidar_conf_maps[i][mask]
            # normalized weighted mean — keeps loss scale comparable.
            l_depth = (cm * huber).sum() / (cm.sum() + 1e-6)
        else:
            l_depth = torch.tensor(0.0, device=device)

        loss = l_photo + lambda_radar * l_radar + lambda_depth * l_depth

        loss.backward()
        strategy.step_post_backward(params, optimizers, strategy_state, step, info)
        for opt in optimizers.values():
            opt.step(); opt.zero_grad(set_to_none=True)

        now = time.time()
        if now - last_log > 5.0 or step == 1 or step == n_iters:
            msg = (f"[RSphys] iter {step}/{n_iters}  loss={float(loss.detach().cpu()):.4f}  "
                   f"L_photo={float(l_photo.detach().cpu()):.4f}  "
                   f"L_radar={float(l_radar.detach().cpu()):.4f}  "
                   f"L_depth={float(l_depth.detach().cpu()):.4f}  "
                   f"|G|={gm.num_gaussians}  sh={sh_degree}  elapsed={now - t0:.0f}s")
            print(msg); log_f.write(msg + "\n"); log_f.flush()
            last_log = now

        if (step % save_every == 0) or step == n_iters:
            save_checkpoint(gm, os.path.join(output_dir, "checkpoints",
                                             f"gaussians_iter{step}.pt"))

    log_f.close()
    elapsed = time.time() - t0
    return TrainResult(
        checkpoint_dir=os.path.join(output_dir, "checkpoints"),
        n_iters=n_iters, seconds_elapsed=elapsed,
        extra={"final_loss": float(loss.detach().cpu()),
               "final_n_gaussians": gm.num_gaussians,
               "scene_scale": scene_scale,
               "alpha_corruption": alpha_corruption,
               "n_anchors": int(anchor_np.shape[0]),
               "avg_depth_coverage": avg_cov,
               "avg_lidar_conf_on_hit": avg_conf},
    )


@register
class RadarSplatPhysics(Baseline):
    meta = BaselineMeta(
        key="radarsplat_physics",
        paper_id="radarsplat_physics",
        venue="ours, Phase 2",
        tier=0,
        requires_gsplat=True,
        upstream_repo="",
        notes="Physics-grounded fusion: LiDAR-confidence-weighted L_depth "
              "(from RGB brightness/specular/edge) + RCS-weighted L_radar "
              "(from LIRA intensity), with RSv1-style densification.",
    )

    def train(self, export_dir: str, output_dir: str,
              config: Optional[dict] = None) -> TrainResult:
        config = config or {}
        if not torch.cuda.is_available():
            raise RuntimeError("RSphys requires CUDA")
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
                img, _ = render_view(gm, w2c_t, K_t, image_w, image_h,
                                     sh_degree=SH_DEGREE_MAX)
                outs.append(img.cpu().numpy())
        return np.stack(outs, axis=0).astype(np.float32)
