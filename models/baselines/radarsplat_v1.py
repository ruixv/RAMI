"""RadarSplat v1 — radar-aware 3DGS + α-rescaled photometric loss.

v0 added a radar coverage loss but did nothing under low light because the
photometric L1+SSIM was still pulling the model toward "render black"
(matching the dark GT). v1 fixes the photometric loss to be α-aware so the
model is encouraged to render the underlying CLEAN scene.

Forward model assumption (matches data/lowlight/noise_model.py):
    GT_dark ≈ α · I_clean + noise

So if `render` is the model's prediction of the CLEAN scene, the matching
quantity is `α · render`, and the photometric loss becomes

    L_photo = L1(α · render_linear, GT_dark_linear)
                                                                + λ_ssim·(1 − SSIM(α · render, GT_dark))

evaluated in approximately-linear color space (gamma 2.2 de-gamma applied
to both sides). At α=1.0 this reduces to the standard 3DGS loss.

Total loss:
    L = L_photo  +  λ_radar · L_radar     (L_radar same as v0)

This is the simplest version of the noise-aware photometric loss we plan
for Phase 3 (full Poisson-Gaussian NLL). v1 establishes whether the
darkness-aware correction alone is enough to unlock the radar contribution.
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
from .radarsplat_v0 import _load_anchor_positions, _radar_coverage_loss
from .vanilla_3dgs import _build_optimizers, _scene_scale_from_cameras, _ssim


def _read_alpha_from_splits(export_dir: str) -> float:
    p = os.path.join(export_dir, "splits.json")
    if os.path.isfile(p):
        with open(p) as f:
            return float(json.load(f).get("alpha", 1.0))
    return 1.0


def _alpha_rescaled_photometric_loss(
    render_srgb: torch.Tensor,
    gt_srgb: torch.Tensor,
    alpha: float,
    lambda_ssim: float,
    gamma: float = 2.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (loss, alpha_scaled_render_srgb_for_SSIM).
    All inputs are (H, W, 3) in sRGB [0, 1]. Compares α·render to GT in
    approximate linear space (gamma 2.2)."""
    render_linear = render_srgb.clamp(0.0, 1.0) ** gamma
    gt_linear = gt_srgb.clamp(0.0, 1.0) ** gamma
    scaled_render_linear = (alpha * render_linear).clamp(0.0, 1.0)
    # L1 in linear space.
    l1 = (scaled_render_linear - gt_linear).abs().mean()
    # Convert α·render back to sRGB for SSIM, which expects sRGB-like input.
    scaled_render_srgb = scaled_render_linear ** (1.0 / gamma)
    ssim_val = _ssim(scaled_render_srgb, gt_srgb)
    loss = (1.0 - lambda_ssim) * l1 + lambda_ssim * (1.0 - ssim_val)
    return loss, scaled_render_srgb


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
    gamma = float(config.get("gamma", 2.2))
    # Learned depth-prior supervision (2026-06-08): confidence-weighted depth
    # loss vs a precomputed per-view dense prior (from RadarDepthNet). For the
    # decisive radar-prior-vs-rgb-prior 3DGS ablation. lambda=0 => off.
    depth_prior_dir = config.get("depth_prior_dir", None)
    lambda_depthprior = float(config.get("lambda_depthprior", 0.0))
    # early-only supervision control: prior loss active only while
    # step <= depthprior_until_iter (default: always on).
    depthprior_until = int(config.get("depthprior_until_iter", 10**9))
    # RAIN-GS progressive Gaussian low-pass filter control (off by default):
    # low_pass = clip(H*W / N / (9*pi), 0.3, c2f_max_lowpass), recomputed
    # every c2f_every_step iters while step < densify_until_iter, then frozen
    # (ported from the official RAIN-GS train.py; feeds gsplat's eps2d).
    c2f = bool(config.get("c2f", False))
    c2f_every = int(config.get("c2f_every_step", 1000))
    c2f_max = float(config.get("c2f_max_lowpass", 300.0))
    # Review-response supervision variants (all default-off):
    #   depthprior_schedule: "const" | "linear15k" (lambda -> 0 linearly by
    #     15k) | "exp3k" (half-life 3k iters)
    #   depthprior_edgeaware: down-weight pixels with large prior-depth
    #     gradient, w = exp(-|grad d| / s), s = per-view median |grad d|
    depthprior_schedule = str(config.get("depthprior_schedule", "const"))
    depthprior_edgeaware = bool(config.get("depthprior_edgeaware", False))

    alpha_corruption = _read_alpha_from_splits(export_dir)
    print(f"[RadarSplat-v1] α (from splits.json) = {alpha_corruption:g}")

    torch.manual_seed(seed); np.random.seed(seed)
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

    # Per-view learned depth priors (depth + confidence), aligned to train_records.
    prior_depth = [None] * len(train_records)
    prior_conf = [None] * len(train_records)
    if depth_prior_dir and lambda_depthprior > 0:
        nload = 0
        for j, im in enumerate(train_records):
            p = os.path.join(depth_prior_dir, im.name + ".npz")
            if os.path.exists(p):
                z = np.load(p)
                prior_depth[j] = torch.tensor(z["depth"].astype(np.float32), device=device)
                cf_np = z["conf"].astype(np.float32)
                if depthprior_edgeaware:
                    gy, gx = np.gradient(z["depth"].astype(np.float32))
                    gmag = np.abs(gy) + np.abs(gx)
                    s = max(float(np.median(gmag)), 1e-6)
                    cf_np = cf_np * np.exp(-gmag / s).astype(np.float32)
                prior_conf[j] = torch.tensor(cf_np, device=device)
                nload += 1
        print(f"[RadarSplat-v1] loaded {nload}/{len(train_records)} depth priors "
              f"(λ_depthprior={lambda_depthprior}, schedule={depthprior_schedule}, "
              f"edgeaware={depthprior_edgeaware}) from {depth_prior_dir}")

    anchor_np = _load_anchor_positions(export_dir)
    anchor_t = torch.tensor(anchor_np, dtype=torch.float32, device=device) if anchor_np.shape[0] else None
    print(f"[RadarSplat-v1] {len(anchor_np)} radar anchors, τ={tau_m:.3f} m, λ_radar={lambda_radar}")

    scene_scale = _scene_scale_from_cameras(data)
    print(f"[RadarSplat-v1] scene_scale={scene_scale:.3f} m")
    gm = init_gaussians_from_points3D(data.points3D, device=device)
    print(f"[RadarSplat-v1] init {gm.num_gaussians} Gaussians")
    optimizers = _build_optimizers(gm, scene_scale)
    params = gm.params_dict()
    # Review-response B2.8 (off by default): Luminance-GS-style per-view tone
    # adaptation. A learnable per-view, per-channel affine curve is applied to
    # the RENDERED image before the photometric loss, so the 3DGS model learns
    # a canonical scene while the curves absorb per-view exposure/tone
    # variation (simplified from Luminance-GS's per-view curve adjustment;
    # labeled "-style" in the paper per the mechanism-reproduction convention).
    perview_tone = bool(config.get("perview_tone", False))
    tone_s = tone_b = tone_opt = None
    if perview_tone:
        tone_s = torch.zeros(len(train_records), 3, device=device, requires_grad=True)
        tone_b = torch.zeros(len(train_records), 3, device=device, requires_grad=True)
        tone_opt = torch.optim.Adam([tone_s, tone_b], lr=1e-3)
        print(f"[RadarSplat-v1] per-view tone adaptation ON "
              f"({len(train_records)} views x 3ch affine)")

    # Review-response B3.4 (off by default): geometry-guided densification.
    # The prior enters NO loss; clone/split proposals are biased toward
    # prior-occupied voxels by scaling the accumulated 2D gradients
    # (stop-gradient guidance) before the standard threshold test.
    guidance_cloud = config.get("densify_guidance_cloud", None)

    class _GuidedStrategy(gsplat.DefaultStrategy):
        _g_keys = None

        def set_guidance(self, cloud_np, voxel, boost, device):
            v = np.floor(cloud_np / voxel).astype(np.int64)
            key = v[:, 0] * 73856093 ^ v[:, 1] * 19349663 ^ v[:, 2] * 83492791
            self._g_keys = torch.tensor(np.unique(key), device=device)
            self._g_vox = float(voxel); self._g_boost = float(boost)

        def _grow_gs(self, params, optimizers, state, step):
            if self._g_keys is not None:
                with torch.no_grad():
                    v = torch.floor(params["means"].detach() / self._g_vox).to(torch.int64)
                    key = v[:, 0] * 73856093 ^ v[:, 1] * 19349663 ^ v[:, 2] * 83492791
                    occ = torch.isin(key, self._g_keys)
                    state["grad2d"] = state["grad2d"] * torch.where(
                        occ, 1.0 + self._g_boost, 1.0)
            return super()._grow_gs(params, optimizers, state, step)

    _strategy_cls = _GuidedStrategy if guidance_cloud else gsplat.DefaultStrategy
    strategy = _strategy_cls(
        prune_opa=0.005, grow_grad2d=2e-4, grow_scale3d=0.01, prune_scale3d=0.1,
        refine_scale2d_stop_iter=0,
        refine_start_iter=int(config.get("densify_from_iter", 500)),
        refine_stop_iter=int(config.get("densify_until_iter", 15000)),
        reset_every=int(config.get("opacity_reset_interval", 3000)),
        refine_every=int(config.get("densify_interval", 100)),
        absgrad=bool(config.get("absgrad", False)),
        verbose=False,
    )
    if guidance_cloud:
        strategy.set_guidance(
            np.load(guidance_cloud).astype(np.float64),
            float(config.get("densify_guidance_voxel", 0.2)),
            float(config.get("densify_guidance_boost", 1.0)), device)
        print(f"[RadarSplat-v1] densify guidance from {guidance_cloud} "
              f"(voxel={config.get('densify_guidance_voxel', 0.2)}, "
              f"boost={config.get('densify_guidance_boost', 1.0)})")
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    os.makedirs(output_dir, exist_ok=True)
    log_f = open(os.path.join(output_dir, "train.log"), "w")
    t0 = time.time()
    last_log = 0.0
    loss = torch.tensor(0.0, device=device)
    l_radar_log = 0.0

    low_pass = 0.3
    densify_until = int(config.get("densify_until_iter", 15000))
    for step in range(1, n_iters + 1):
        sh_degree = min(SH_DEGREE_MAX, (step - 1) // sh_warmup_every)
        i = int(rng.integers(0, len(train_records)))
        if c2f and (step == 1 or (step % c2f_every == 0 and step < densify_until)):
            low_pass = float(min(max(width * height / gm.num_gaussians / (9 * np.pi), 0.3),
                                 c2f_max))
            if step == 1 or step % 1000 == 0:
                print(f"[c2f] step {step}: low_pass={low_pass:.2f} (N={gm.num_gaussians})")
        scales = torch.exp(gm.scales)
        quats = torch.nn.functional.normalize(gm.quats, dim=-1)
        opacities = torch.sigmoid(gm.opacities)
        colors = gm.colors_for_render(sh_degree)
        _rmode = "RGB+ED" if lambda_depthprior > 0 else "RGB"
        renders, alphas_, info = gsplat.rasterization(
            means=gm.means, quats=quats, scales=scales, opacities=opacities,
            colors=colors, viewmats=train_w2cs[i].unsqueeze(0), Ks=K.unsqueeze(0),
            width=width, height=height, sh_degree=sh_degree, packed=False,
            absgrad=strategy.absgrad, render_mode=_rmode,
            eps2d=low_pass,
        )
        rendered = renders[0][:, :, :3].clamp(0.0, 1.0)
        depth_r = renders[0][:, :, 3] if lambda_depthprior > 0 else None
        gt = train_imgs[i].permute(1, 2, 0)

        strategy.step_pre_backward(params, optimizers, strategy_state, step, info)

        if perview_tone:
            # clamp floor 1e-4, not 0: the alpha-rescaled loss applies
            # x^(1/gamma) whose gradient is infinite at exactly 0, and a
            # negative tone bias saturates whole regions to 0 -> 0*inf = NaN
            rendered = ((1.0 + tone_s[i]) * rendered + tone_b[i]).clamp(1e-4, 1.0)
        l_photo, _ = _alpha_rescaled_photometric_loss(
            rendered, gt, alpha_corruption, lambda_ssim, gamma=gamma,
        )

        if anchor_t is not None and lambda_radar > 0:
            # Guard against numerical blowups: if any Gaussian mean drifted
            # to NaN/Inf, the cdist call returns NaN and contaminates the
            # whole loss. Filter to finite means; if none, skip L_radar.
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

        # Confidence-weighted depth-prior supervision (full view, incl. blind).
        if (lambda_depthprior > 0 and prior_depth[i] is not None
                and step <= depthprior_until):
            lam_dp = lambda_depthprior
            if depthprior_schedule == "linear15k":
                lam_dp = lambda_depthprior * max(0.0, 1.0 - step / 15000.0)
            elif depthprior_schedule == "exp3k":
                lam_dp = lambda_depthprior * (0.5 ** (step / 3000.0))
            if lam_dp > 0:
                pdp = prior_depth[i]; cf = prior_conf[i]
                diff = depth_r - pdp
                hub = torch.where(diff.abs() < 0.2, 0.5 * diff * diff / 0.2, diff.abs() - 0.1)
                l_dp = (cf * hub).sum() / (cf.sum() + 1e-6)
                if torch.isfinite(l_dp):
                    loss = loss + lam_dp * l_dp

        loss.backward()
        strategy.step_post_backward(params, optimizers, strategy_state, step, info)
        for opt in optimizers.values():
            opt.step(); opt.zero_grad(set_to_none=True)
        if tone_opt is not None:
            tone_opt.step(); tone_opt.zero_grad(set_to_none=True)
            # project out the global component: per-view curves must carry
            # only view-RESIDUAL exposure, the shared exposure stays in the
            # model (otherwise raw test-view renders drift arbitrarily)
            with torch.no_grad():
                tone_s -= tone_s.mean(0, keepdim=True)
                tone_b -= tone_b.mean(0, keepdim=True)

        now = time.time()
        if now - last_log > 5.0 or step == 1 or step == n_iters:
            msg = (f"[RSv1] iter {step}/{n_iters}  loss={float(loss.detach().cpu()):.4f}  "
                   f"L_photo={float(l_photo.detach().cpu()):.4f}  "
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
               "scene_scale": scene_scale, "lambda_radar": lambda_radar,
               "tau_m": tau_m, "alpha_corruption": alpha_corruption,
               "n_anchors": len(anchor_np)},
    )


@register
class RadarSplatV1(Baseline):
    meta = BaselineMeta(
        key="radarsplat_v1",
        paper_id="radarsplat_v1",
        venue="ours, Phase 2",
        tier=0,
        requires_gsplat=True,
        upstream_repo="",
        notes="RadarSplat v0 + α-rescaled photometric loss (renders CLEAN scene from dark observations).",
    )

    def train(self, export_dir: str, output_dir: str, config: Optional[dict] = None) -> TrainResult:
        config = config or {}
        if not torch.cuda.is_available():
            raise RuntimeError("RadarSplat-v1 requires CUDA")
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
