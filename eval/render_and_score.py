"""Render → score harness for Phase 2 / 3 NVS comparisons.

Computes the three headline metrics on a directory of rendered PNGs against
held-out ground-truth PNGs:

    PSNR — per-pixel intensity fidelity
    SSIM — structural similarity (skimage)
    LPIPS — perceptual (lpips library, AlexNet weights)

Plus utility for paired-bootstrap CIs that will land in the eval pipeline in
Phase 3.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image


@dataclass
class ImagePairMetrics:
    """Per-image scores from compute_image_pair."""

    psnr: float
    ssim: float
    lpips: float
    filename: str


def _load_rgb_uint8(path: str, size: Optional[tuple[int, int]] = None) -> np.ndarray:
    with Image.open(path) as im:
        if im.mode != "RGB":
            im = im.convert("RGB")
        if size is not None and (im.width, im.height) != size:
            im = im.resize(size, Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def compute_psnr(pred_u8: np.ndarray, gt_u8: np.ndarray) -> float:
    if pred_u8.shape != gt_u8.shape:
        raise ValueError(f"shape mismatch: pred {pred_u8.shape} vs gt {gt_u8.shape}")
    pred = pred_u8.astype(np.float64) / 255.0
    gt = gt_u8.astype(np.float64) / 255.0
    mse = float(np.mean((pred - gt) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(-10.0 * np.log10(mse))


def compute_ssim(pred_u8: np.ndarray, gt_u8: np.ndarray) -> float:
    from skimage.metrics import structural_similarity as ssim

    pred = pred_u8.astype(np.float64) / 255.0
    gt = gt_u8.astype(np.float64) / 255.0
    # channel_axis=-1 because pred/gt are (H, W, 3); data_range=1 since we
    # normalized to [0, 1].
    return float(ssim(gt, pred, data_range=1.0, channel_axis=-1))


_LPIPS_NET = None


def compute_lpips(pred_u8: np.ndarray, gt_u8: np.ndarray) -> float:
    """LPIPS with AlexNet weights. The first call lazily loads the network,
    subsequent calls reuse it. Cache lives in module-level state."""
    global _LPIPS_NET
    try:
        import lpips
        import torch
    except ImportError as e:
        raise ImportError(
            "lpips and torch are required for compute_lpips; "
            "see docs/phase2_env.md for env setup"
        ) from e

    if _LPIPS_NET is None:
        net = lpips.LPIPS(net="alex")
        net.eval()
        if torch.cuda.is_available():
            net = net.cuda()
        _LPIPS_NET = net

    device = next(_LPIPS_NET.parameters()).device

    def _to_t(u8: np.ndarray) -> "torch.Tensor":
        t = torch.from_numpy(u8.astype(np.float32) / 255.0)  # (H, W, 3)
        t = t.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0      # (1, 3, H, W) in [-1, 1]
        return t.to(device)

    with torch.no_grad():
        d = _LPIPS_NET(_to_t(pred_u8), _to_t(gt_u8))
    return float(d.item())


def compute_image_pair(
    pred_path: str, gt_path: str, with_lpips: bool = True
) -> ImagePairMetrics:
    gt_u8 = _load_rgb_uint8(gt_path)
    pred_u8 = _load_rgb_uint8(pred_path, size=(gt_u8.shape[1], gt_u8.shape[0]))
    return ImagePairMetrics(
        psnr=compute_psnr(pred_u8, gt_u8),
        ssim=compute_ssim(pred_u8, gt_u8),
        lpips=compute_lpips(pred_u8, gt_u8) if with_lpips else float("nan"),
        filename=os.path.basename(gt_path),
    )


def score_directory(
    pred_dir: str,
    gt_dir: str,
    filenames: Optional[list[str]] = None,
    with_lpips: bool = True,
) -> tuple[list[ImagePairMetrics], dict]:
    """Iterate over `filenames` (or `gt_dir`'s PNGs) and compute the per-image
    metrics. Returns (per_image_list, aggregate_summary)."""
    if filenames is None:
        filenames = sorted(f for f in os.listdir(gt_dir) if f.lower().endswith(".png"))
    rows: list[ImagePairMetrics] = []
    for name in filenames:
        gt_path = os.path.join(gt_dir, name)
        pred_path = os.path.join(pred_dir, name)
        if not (os.path.isfile(gt_path) and os.path.isfile(pred_path)):
            continue
        rows.append(compute_image_pair(pred_path, gt_path, with_lpips=with_lpips))
    if not rows:
        return rows, {"n": 0, "mean_psnr": float("nan"), "mean_ssim": float("nan"), "mean_lpips": float("nan")}
    summary = {
        "n": len(rows),
        "mean_psnr": float(np.mean([r.psnr for r in rows if np.isfinite(r.psnr)])),
        "mean_ssim": float(np.mean([r.ssim for r in rows])),
        "mean_lpips": float(np.mean([r.lpips for r in rows if np.isfinite(r.lpips)])) if with_lpips else float("nan"),
    }
    return rows, summary


def paired_bootstrap_ci(
    deltas: np.ndarray,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 20260523,
) -> tuple[float, float, float]:
    """Paired-bootstrap CI on a mean delta (e.g., RadarSplat-PSNR minus
    baseline-PSNR per test view). Returns (mean, lower, upper) at confidence
    1 - alpha."""
    if deltas.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    n = deltas.size
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[b] = float(np.mean(deltas[idx]))
    mean = float(np.mean(deltas))
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return mean, lo, hi
