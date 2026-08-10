"""B6 — DN-Splatter (Turkulainen et al., WACV 2025).

DN-Splatter adds **dense depth + normal supervision from a depth sensor**
to vanilla 3DGS. In the original paper the depth comes from a stereo /
ToF / LiDAR sensor projected per-frame. Here we use the RadarEyes Velodyne
LiDAR: for each training ZED frame we project the nearest-in-time LiDAR
sweep onto the image plane to get a *sparse* depth map (~ 1-2% pixel
coverage at 720p).

This is the **"is radar redundant when LiDAR is on the rig?" baseline**.
DN-Splatter has access to the full LiDAR; RadarSplat has access only to
radar. If DN-Splatter matches or beats RadarSplat under low light, the
radar contribution is questionable.

Implementation:
- Per-view sparse LiDAR depth maps are precomputed at training start
  (LiDAR-frame → world → camera → image plane). Cached in memory for
  the training run. Use only "in-front + in-frustum + 0.3 ≤ depth ≤ 20 m"
  pixels.
- Loss:
    L = L_photo (L1+SSIM)
        + λ_depth · masked_L1(depth_rendered, depth_lidar)
        + λ_normal · (1 − cos(normal_rendered, normal_from_depth))
- Normal supervision uses the gsplat rasterizer's RGB+ED depth output to
  build a depth-derived normal map (cross-product of finite differences),
  then encourages rendered surface normals from the same depth to be
  consistent. Note: this is the *self-supervised normal regularizer*
  from the paper, not LiDAR-sensed normals (LiDAR doesn't give normals
  directly without a meshing step).

Hypothesis: DN-Splatter helps at all α levels (LiDAR is photon-
independent) but does not specifically address the α-rescale failure
mode. Should give ~ flat performance across α, possibly higher absolute
than B1 but without our v1's monotone-widening edge.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Optional

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.radareyes import Scene
from data.radareyes.sync import interpolate_pose, nearest_index
from data.radareyes.transforms import SENSOR_OFFSETS, sensor_to_world
from tools.fov_mask import project_to_camera
from tools.zed_calibration import ZED_BODY_TO_OPTICAL_R

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


def _resolve_scene_from_splits(export_dir: str) -> Scene:
    """splits.json has the scene name; the scene directory may be in the
    export_dir's parent (set by training/baseline_train.py) or under the
    default dataset root."""
    splits = json.load(open(os.path.join(export_dir, "splits.json")))
    name = splits["scene"]
    cand = os.path.join(os.environ.get("RAMI_DATA_ROOT", "data"), name)
    if os.path.isdir(cand):
        return Scene(cand)
    raise FileNotFoundError(f"cannot resolve scene {name!r} for DN-Splatter")


def _build_lidar_depth_for_view(
    scene: Scene,
    t_image: float,
    K: torch.Tensor,
    width: int, height: int,
    device: str,
    near: float = 0.3,
    far: float = 20.0,
    window_s: float = 0.2,
) -> torch.Tensor:
    """Project all LiDAR points within ±window_s of t_image to the camera
    view at t_image. Returns a (H, W) tensor of depths in metres; 0 where
    no LiDAR point projects.
    """
    depth = torch.zeros(height, width, device=device, dtype=torch.float32)
    if scene.lidar is None:
        return depth

    times = scene.lidar.timestamps
    z_pos, z_quat = interpolate_pose(scene.zed.pose_stream, t_image)
    fx = float(K[0, 0]); fy = float(K[1, 1]); cx = float(K[0, 2]); cy = float(K[1, 2])

    i0 = nearest_index(times, t_image - window_s)
    i1 = nearest_index(times, t_image + window_s)
    i0 = max(0, int(i0)); i1 = min(len(scene.lidar) - 1, int(i1))
    if i1 < i0:
        return depth

    # Pre-allocate small buffer; depth uses min-fill so we keep nearest hit.
    for j in range(i0, i1 + 1):
        pcd = scene.lidar.load_pcd(j)
        if pcd.shape[0] == 0:
            continue
        t_lid = float(times[j])
        body_pos, body_quat = interpolate_pose(scene.zed.pose_stream, t_lid)
        pcd_world = sensor_to_world(pcd[:, :3], body_pos, body_quat, SENSOR_OFFSETS["lidar"])
        pixels, depths, valid = project_to_camera(
            pcd_world, z_pos, z_quat,
            fx=fx, fy=fy, cx=cx, cy=cy,
            image_w=width, image_h=height,
            near=near, far=far,
            body_to_optical_R=ZED_BODY_TO_OPTICAL_R,
        )
        if not valid.any():
            continue
        u = pixels[valid, 0].astype(np.int64).clip(0, width - 1)
        v = pixels[valid, 1].astype(np.int64).clip(0, height - 1)
        d = depths[valid].astype(np.float32)

        # Z-buffer style min-fill so closer points win over farther ones at
        # the same pixel; multi-hit pixels keep the smallest depth.
        d_t = torch.from_numpy(d).to(device)
        u_t = torch.from_numpy(u).to(device)
        v_t = torch.from_numpy(v).to(device)
        flat_idx = v_t * width + u_t
        depth_flat = depth.view(-1)
        # current depths at those flat indices
        cur = depth_flat[flat_idx]
        # if pixel empty (0) or this depth is closer → overwrite
        new = torch.where((cur == 0) | (d_t < cur), d_t, cur)
        depth_flat.index_copy_(0, flat_idx, new)
    return depth


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
    # 0.5 over-drove DefaultStrategy's gradient-based densification to ~10M
    # Gaussians → OOM. The depth loss is in metres so the unscaled gradient
    # magnitude dwarfs the photometric one; 0.1 keeps both contributions
    # comparable.
    lambda_depth = float(config.get("lambda_depth", 0.1))
    # Distinct key from B4 2DGS's `lambda_normal` so a shared sweep config
    # can disable 2DGS's normal-consistency loss (which depends on the 2DGS
    # rasterizer populating surf_normals) without also disabling DN-Splatter's
    # depth-smoothness regularizer (a finite-difference term on the rendered
    # depth that has nothing to do with the 2DGS path).
    lambda_normal = float(config.get("lambda_normal_dn", 0.05))
    huber_delta = float(config.get("huber_delta", 0.10))
    seed = int(config.get("seed", 1))
    sh_warmup_every = int(config.get("sh_warmup_every", 1000))
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

    # ---- Build per-view sparse LiDAR depth maps. ----
    print(f"[B6] building LiDAR-derived depth maps for {len(train_records)} views ...")
    t0_d = time.time()
    scene = _resolve_scene_from_splits(export_dir)
    splits = json.load(open(os.path.join(export_dir, "splits.json")))
    # Map image name → t_image from splits.json records.
    name_to_t = {r["filename"]: r["t_image"] for r in splits["records"]}
    depth_maps: list[torch.Tensor] = []
    valid_masks: list[torch.Tensor] = []
    n_valid_total = 0
    for k, rec in enumerate(train_records):
        t_img = float(name_to_t[rec.name])
        d = _build_lidar_depth_for_view(scene, t_img, K, width, height, device)
        mask = d > 0
        depth_maps.append(d)
        valid_masks.append(mask)
        n_valid_total += int(mask.sum())
    avg_cov = n_valid_total / (len(train_records) * width * height)
    print(f"[B6] depth maps built in {time.time() - t0_d:.1f}s; "
          f"avg coverage {avg_cov*100:.2f}% of pixels")

    scene_scale = _scene_scale_from_cameras(data)
    print(f"[B6] scene_scale={scene_scale:.3f} m  λ_depth={lambda_depth}  λ_normal={lambda_normal}")

    gm = init_gaussians_from_points3D(data.points3D, device=device)
    print(f"[B6] init {gm.num_gaussians} Gaussians")

    optimizers = _build_optimizers(gm, scene_scale)
    params = gm.params_dict()

    # Throttle densification vs B1: depth supervision sharpens the 2D
    # gradient at LiDAR-hit pixels, which makes DefaultStrategy clone/split
    # ~50× more aggressively than for photometric-only B1. Higher
    # grow_grad2d, earlier stop, and more aggressive pruning together keep
    # |G| under ~1M (we OOM'd at 9.8M with the B1 defaults).
    strategy = gsplat.DefaultStrategy(
        prune_opa=0.01, grow_grad2d=5e-4, grow_scale3d=0.01, prune_scale3d=0.1,
        refine_scale2d_stop_iter=0,
        refine_start_iter=int(config.get("densify_from_iter", 500)),
        refine_stop_iter=int(config.get("densify_until_iter", 5000)),
        reset_every=int(config.get("opacity_reset_interval", 3000)),
        refine_every=int(config.get("densify_interval", 200)),
        absgrad=True, verbose=False,
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
        sh_degree = min(SH_DEGREE_MAX, (step - 1) // sh_warmup_every)
        i = int(rng.integers(0, len(train_records)))

        scales = torch.exp(gm.scales)
        quats = torch.nn.functional.normalize(gm.quats, dim=-1)
        opacities = torch.sigmoid(gm.opacities)
        colors = gm.colors_for_render(sh_degree)
        renders, alphas, info = gsplat.rasterization(
            means=gm.means, quats=quats, scales=scales, opacities=opacities,
            colors=colors, viewmats=train_w2cs[i].unsqueeze(0), Ks=K.unsqueeze(0),
            width=width, height=height, sh_degree=sh_degree, packed=False,
            absgrad=True, render_mode="RGB+ED",
        )
        # renders is (1, H, W, 4): RGB + expected-depth
        rendered = renders[0, :, :, :3].clamp(0.0, 1.0)
        depth_render = renders[0, :, :, 3]
        gt = train_imgs[i].permute(1, 2, 0)
        strategy.step_pre_backward(params, optimizers, strategy_state, step, info)

        l1 = (rendered - gt).abs().mean()
        ssim_val = _ssim(rendered, gt)
        l_photo = (1.0 - lambda_ssim) * l1 + lambda_ssim * (1.0 - ssim_val)

        # Sparse LiDAR depth supervision (Huber loss on valid pixels only).
        mask = valid_masks[i]
        if mask.any():
            d_r = depth_render[mask]
            d_t = depth_maps[i][mask]
            diff = d_r - d_t
            abs_diff = diff.abs()
            huber = torch.where(abs_diff < huber_delta,
                                0.5 * diff * diff / huber_delta,
                                abs_diff - 0.5 * huber_delta)
            l_depth = huber.mean()
        else:
            l_depth = torch.tensor(0.0, device=device)

        loss = l_photo + lambda_depth * l_depth

        # Self-supervised normal regularizer (DN-Splatter Eq.7 simplified):
        # encourage smoothness of the depth-derived normal map. We compute
        # finite-difference normals from `depth_render` and penalize gradient
        # magnitude. This is a cheap proxy for the full surface-normal
        # consistency term and stays photon-independent.
        if lambda_normal > 0:
            d = depth_render
            dx = d[:, 1:] - d[:, :-1]
            dy = d[1:, :] - d[:-1, :]
            l_normal = dx.abs().mean() + dy.abs().mean()
            loss = loss + lambda_normal * l_normal

        loss.backward()
        strategy.step_post_backward(params, optimizers, strategy_state, step, info)
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)

        now = time.time()
        if now - last_log > 5.0 or step == 1 or step == n_iters:
            msg = (f"[B6] iter {step}/{n_iters}  loss={float(loss.detach().cpu()):.4f}  "
                   f"l_d={float(l_depth.detach().cpu()):.3f}  "
                   f"|G|={gm.num_gaussians}  sh={sh_degree}  elapsed={now - t0:.0f}s")
            print(msg); log_f.write(msg + "\n"); log_f.flush()
            last_log = now

        if (step % save_every == 0) or step == n_iters:
            ckpt_path = os.path.join(output_dir, "checkpoints", f"gaussians_iter{step}.pt")
            save_checkpoint(gm, ckpt_path)

    log_f.close()
    elapsed = time.time() - t0
    return TrainResult(
        checkpoint_dir=os.path.join(output_dir, "checkpoints"),
        n_iters=n_iters,
        seconds_elapsed=elapsed,
        extra={"final_loss": float(loss.detach().cpu()),
               "final_n_gaussians": gm.num_gaussians,
               "scene_scale": scene_scale,
               "avg_depth_coverage": avg_cov},
    )


@register
class DNSplatter(Baseline):
    meta = BaselineMeta(
        key="dn_splatter",
        paper_id="turkulainen2025_dn_splatter",
        venue="WACV 2025",
        tier=1,
        requires_gsplat=True,
        upstream_repo="https://github.com/maturk/dn-splatter",
        notes=("LiDAR-supervised. Critical: if this matches RadarSplat under low "
               "light, radar may be redundant when LiDAR is on the rig."),
    )

    def train(self, export_dir: str, output_dir: str, config: Optional[dict] = None) -> TrainResult:
        config = config or {}
        if not torch.cuda.is_available():
            raise RuntimeError("B6 requires CUDA")
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
