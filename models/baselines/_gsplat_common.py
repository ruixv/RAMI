"""Shared utilities for gsplat-based baselines (B1-B5).

Implements:
    parse_colmap_export(export_dir) → COLMAPData
    GaussianModel  — named-param dataclass (means, scales, quats, opacities,
                     sh0, shN). Supports SH degree 0-3 like the official 3DGS.
    init_gaussians_from_points3D → GaussianModel
    save_checkpoint / load_checkpoint
    render_view  — single-image gsplat call with optional SH

Each baseline (B1 vanilla, B2 mip, B3 scaffold, B4 2dgs, B5 mcmc) wraps this
with its own densification strategy (gsplat.DefaultStrategy / MCMCStrategy)
and loss tweaks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from PIL import Image


# ============================ COLMAP parser ============================ #


@dataclass
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass
class ColmapImage:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


@dataclass
class ColmapData:
    cameras: dict[int, ColmapCamera]
    images: dict[int, ColmapImage]
    points3D: np.ndarray
    test_image_names: list[str] = field(default_factory=list)


def parse_colmap_export(export_dir: str) -> ColmapData:
    sparse = os.path.join(export_dir, "sparse", "0")

    cameras: dict[int, ColmapCamera] = {}
    with open(os.path.join(sparse, "cameras.txt")) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cid = int(parts[0])
            model = parts[1]
            w, h = int(parts[2]), int(parts[3])
            params = np.array([float(p) for p in parts[4:]], dtype=np.float64)
            cameras[cid] = ColmapCamera(cid, model, w, h, params)

    # COLMAP's images.txt stores 2 lines per image: a header + a 2D-feature
    # line that may be empty. After we strip blank lines and comments only
    # the header survives, so each surviving line is one full image record
    # and we step by 1 (NOT 2 — that was a previous bug that silently dropped
    # half the views).
    images: dict[int, ColmapImage] = {}
    with open(os.path.join(sparse, "images.txt")) as f:
        lines = [l for l in f.read().splitlines() if l and not l.startswith("#")]
    for line in lines:
        parts = line.split()
        if len(parts) < 10:
            # 2D-feature line that survived (rare) — skip.
            continue
        image_id = int(parts[0])
        qvec = np.array([float(x) for x in parts[1:5]], dtype=np.float64)
        tvec = np.array([float(x) for x in parts[5:8]], dtype=np.float64)
        cid = int(parts[8])
        name = parts[9]
        images[image_id] = ColmapImage(image_id, qvec, tvec, cid, name)

    points_list: list[tuple[float, ...]] = []
    with open(os.path.join(sparse, "points3D.txt")) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            x, y, z = (float(p) for p in parts[1:4])
            r, g, b = (int(p) for p in parts[4:7])
            points_list.append((x, y, z, r, g, b))
    points3D = np.array(points_list, dtype=np.float64) if points_list else np.zeros((0, 6))

    test_image_names: list[str] = []
    test_txt = os.path.join(export_dir, "test_image_names.txt")
    if os.path.isfile(test_txt):
        with open(test_txt) as f:
            test_image_names = [l.strip() for l in f if l.strip()]

    return ColmapData(cameras=cameras, images=images, points3D=points3D,
                      test_image_names=test_image_names)


# ============================ Pose math =============================== #


def quat_to_R_torch(quat: torch.Tensor) -> torch.Tensor:
    quat = quat.to(dtype=torch.float64) if quat.dtype != torch.float64 else quat
    qw, qx, qy, qz = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    n = qw * qw + qx * qx + qy * qy + qz * qz
    s = torch.where(n > 0, 2.0 / n, torch.zeros_like(n))
    xx, yy, zz = qx * qx * s, qy * qy * s, qz * qz * s
    xy, xz, yz = qx * qy * s, qx * qz * s, qy * qz * s
    wx, wy, wz = qw * qx * s, qw * qy * s, qw * qz * s
    R = torch.empty(quat.shape[:-1] + (3, 3), dtype=torch.float64, device=quat.device)
    R[..., 0, 0] = 1.0 - (yy + zz); R[..., 0, 1] = xy - wz;        R[..., 0, 2] = xz + wy
    R[..., 1, 0] = xy + wz;         R[..., 1, 1] = 1.0 - (xx + zz); R[..., 1, 2] = yz - wx
    R[..., 2, 0] = xz - wy;         R[..., 2, 1] = yz + wx;        R[..., 2, 2] = 1.0 - (xx + yy)
    return R


def colmap_image_to_w2c(img: ColmapImage) -> np.ndarray:
    q = torch.tensor(img.qvec, dtype=torch.float64)
    R = quat_to_R_torch(q).numpy()
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = img.tvec
    return M


def K_from_camera(cam: ColmapCamera) -> np.ndarray:
    if cam.model != "PINHOLE":
        raise NotImplementedError(f"only PINHOLE supported, got {cam.model}")
    fx, fy, cx, cy = cam.params
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


# ============================ Gaussian model ============================ #
# SH coefficient counts: degree d has (d+1)^2 coefficients per channel.
# degree 0: 1, degree 1: 4, degree 2: 9, degree 3: 16.
# We split into sh0 (the DC term, 1 coef) and shN (the rest = 15 coefs at deg 3).

SH_DEGREE_MAX = 3
SH_COEFS_TOTAL = (SH_DEGREE_MAX + 1) ** 2     # 16
SH_COEFS_REST = SH_COEFS_TOTAL - 1            # 15
# Convert RGB in [0, 1] to SH degree-0 coefficient. SH basis Y_0^0 = 1 / (2 sqrt(pi)),
# so f(dir) = sh0 * (1/(2 sqrt(pi))). To make f = RGB, sh0 = 2 sqrt(pi) * (RGB - 0.5)
# (gsplat's sigmoid-on-DC convention: sh0 = (RGB - 0.5) / SH_C0 where SH_C0 = 0.28209479).
SH_C0 = 0.28209479177387814


class GaussianModel:
    """Wraps a `dict[str, torch.nn.Parameter]` so that mutations by a gsplat
    strategy (which replaces dict entries on densify/prune) are picked up by
    the trainer transparently. Always access via the properties; never cache
    `.means` etc. outside of a single iteration."""

    PARAM_NAMES = ("means", "scales", "quats", "opacities", "sh0", "shN")

    def __init__(self, params: dict[str, torch.nn.Parameter]):
        missing = set(self.PARAM_NAMES) - set(params)
        if missing:
            raise ValueError(f"GaussianModel missing required params: {missing}")
        self._params = params   # mutated in place by strategy callbacks

    @property
    def means(self) -> torch.nn.Parameter: return self._params["means"]
    @property
    def scales(self) -> torch.nn.Parameter: return self._params["scales"]
    @property
    def quats(self) -> torch.nn.Parameter: return self._params["quats"]
    @property
    def opacities(self) -> torch.nn.Parameter: return self._params["opacities"]
    @property
    def sh0(self) -> torch.nn.Parameter: return self._params["sh0"]
    @property
    def shN(self) -> torch.nn.Parameter: return self._params["shN"]

    @property
    def num_gaussians(self) -> int:
        return self.means.shape[0]

    @property
    def device(self) -> torch.device:
        return self.means.device

    def params_dict(self) -> dict[str, torch.nn.Parameter]:
        """Return the backing dict for use by gsplat strategies."""
        return self._params

    def colors_for_render(self, sh_degree: int) -> torch.Tensor:
        active = (sh_degree + 1) ** 2
        out = torch.cat([self.sh0, self.shN], dim=1)
        return out[:, :active, :].contiguous()


def _knn_scale_init(means_np: np.ndarray, k: int = 3) -> np.ndarray:
    """For each point return the mean distance to its k nearest neighbors.
    Used to seed Gaussian scales so they cover the local point density
    (3DGS official initialization)."""
    from scipy.spatial import cKDTree
    n = means_np.shape[0]
    if n <= k + 1:
        return np.full((n,), 0.1, dtype=np.float64)
    tree = cKDTree(means_np)
    # k+1 because the nearest neighbor includes the point itself at distance 0.
    d, _ = tree.query(means_np, k=k + 1)
    return d[:, 1:].mean(axis=-1).clip(min=1e-4)


def init_gaussians_from_points3D(
    points3D: np.ndarray,
    initial_opacity: float = 0.1,
    device: str = "cuda",
    fallback_n: int = 1000,
    scene_bbox: Optional[tuple[float, float, float, float, float, float]] = None,
    scale_multiplier: float = 1.0,
    fallback_scale_m: float = 0.05,
) -> GaussianModel:
    """Initialize from a (N, ≥6) points3D = (x, y, z, r, g, b). RGB in [0,255]."""
    if points3D.shape[0] == 0:
        if scene_bbox is None:
            scene_bbox = (-2.0, -2.0, -2.0, 2.0, 2.0, 2.0)
        rng = np.random.default_rng(0)
        means_np = rng.uniform(low=scene_bbox[:3], high=scene_bbox[3:], size=(fallback_n, 3))
        colors_np = np.full((fallback_n, 3), 0.6)
        per_pt_scale = np.full((fallback_n,), fallback_scale_m, dtype=np.float64)
    else:
        means_np = points3D[:, :3]
        colors_np = (points3D[:, 3:6] / 255.0).clip(0, 1)
        # k-NN distance gives each Gaussian a scale that matches the local
        # point density. Critical for sparse init clouds (e.g. LIRA's
        # ~400 anchors over a 10 m³ room → per-point ~0.3 m vs the previous
        # 0.05 m flat default that left Gaussians sub-pixel.).
        per_pt_scale = _knn_scale_init(means_np, k=3) * scale_multiplier

    n = means_np.shape[0]

    def _P(x: np.ndarray) -> torch.nn.Parameter:
        return torch.nn.Parameter(torch.tensor(x, dtype=torch.float32, device=device))

    means = _P(means_np)
    scales_np = np.log(per_pt_scale)[:, None].repeat(3, axis=1)  # (N, 3) log space
    scales = _P(scales_np)
    quats_t = torch.zeros(n, 4, dtype=torch.float32, device=device); quats_t[:, 0] = 1.0
    quats = torch.nn.Parameter(quats_t)
    logit_opacity = float(np.log(initial_opacity / (1.0 - initial_opacity)))
    opacities = _P(np.full((n,), logit_opacity))
    sh0_np = ((colors_np - 0.5) / SH_C0)[:, None, :]   # (N, 1, 3)
    sh0 = _P(sh0_np)
    shN = torch.nn.Parameter(torch.zeros(n, SH_COEFS_REST, 3, dtype=torch.float32, device=device))

    return GaussianModel({
        "means": means, "scales": scales, "quats": quats,
        "opacities": opacities, "sh0": sh0, "shN": shN,
    })


def save_checkpoint(gm: GaussianModel, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "means": gm.means.detach().cpu(),
        "scales": gm.scales.detach().cpu(),
        "quats": gm.quats.detach().cpu(),
        "opacities": gm.opacities.detach().cpu(),
        "sh0": gm.sh0.detach().cpu(),
        "shN": gm.shN.detach().cpu(),
    }, path)


def load_checkpoint(path: str, device: str = "cuda") -> GaussianModel:
    state = torch.load(path, map_location=device, weights_only=True)
    return GaussianModel({
        k: torch.nn.Parameter(state[k].to(device))
        for k in GaussianModel.PARAM_NAMES
    })


# ============================ Render via gsplat ========================= #


def render_view(
    gm: GaussianModel,
    w2c: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    sh_degree: int = SH_DEGREE_MAX,
    near_plane: float = 0.01,
    far_plane: float = 200.0,
    packed: bool = False,
    return_info: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Single-view forward render. Returns (image[H,W,3] in [0,1], info)."""
    import gsplat

    scales = torch.exp(gm.scales)
    quats = torch.nn.functional.normalize(gm.quats, dim=-1)
    opacities = torch.sigmoid(gm.opacities)
    colors = gm.colors_for_render(sh_degree)

    out = gsplat.rasterization(
        means=gm.means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=w2c.unsqueeze(0).to(gm.means),
        Ks=K.unsqueeze(0).to(gm.means),
        width=width, height=height,
        near_plane=near_plane, far_plane=far_plane,
        sh_degree=sh_degree,
        packed=packed,
    )
    if isinstance(out, tuple):
        img_batched = out[0]
        info = out[2] if len(out) >= 3 else {}
    else:
        img_batched = out["render"]
        info = out
    return img_batched[0].clamp(0.0, 1.0), info


def save_image_uint8(img: torch.Tensor, path: str) -> None:
    t = img.detach().cpu()
    if t.ndim == 3 and t.shape[0] == 3:
        t = t.permute(1, 2, 0)
    arr = (t.clamp(0, 1).numpy() * 255.0 + 0.5).astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    Image.fromarray(arr).save(path)


def load_image_to_tensor(path: str, device: str = "cuda") -> torch.Tensor:
    with Image.open(path) as im:
        if im.mode != "RGB":
            im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).to(device)
