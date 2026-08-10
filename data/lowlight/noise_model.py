"""Physically grounded low-light corruption for ZED RGB frames.

Approximation of a Sony IMX390 ZED 2i pipeline as documented in
docs/low_light_protocol.md:

    1. de-gamma: I_sRGB → I_linear  (gamma = 2.2 approximation of sRGB)
    2. scale by darkness factor α:  signal = α · I_linear
    3. add shot noise: shot ~ Normal(0, σ_shot_scale · sqrt(signal))
    4. add read noise: read ~ Normal(0, α · σ_read)
       (read noise scales linearly with sensor gain we'd use to compensate)
    5. clip to [0, 1] and re-gamma to sRGB
    6. quantize to uint8

This is a deliberately simple but physically motivated approximation: shot
noise has signal-dependent variance (Poisson → Gaussian approx since the
signal magnitudes here are large in DN units), and read noise is
signal-independent. Real sensors also have dark current, fixed-pattern noise,
PRNU, and PRNU × signal terms — we leave those for Phase 2 if the
synthetic-vs-real gap shows up.

Per the calibration_status disclosure (hardware unavailable), the default
σ_read / σ_shot_scale are derived from the published Sony IMX390 datasheet
typical values, not measured from this specific unit. Reviewers see the
uncertainty disclosure in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Canonical darkness levels — must match configs/lowlight/darkness_levels.yaml.
DARKNESS_LEVELS = (1.00, 0.30, 0.10, 0.03, 0.01)


@dataclass(frozen=True)
class NoiseParams:
    """Sensor noise hyperparameters in normalized [0, 1] intensity units.

    σ_read = 0.005 corresponds to ~1.3 e- RMS read noise at typical gain on
    the IMX390, mapped through the ZED 2i's 8-bit sRGB ISP at HD720 binning.
    σ_shot_scale = 0.05 calibrates the Poisson-approximation Gaussian: noise
    stddev scales as σ_shot_scale · sqrt(α · I_linear).
    """

    sigma_read: float = 0.005
    sigma_shot_scale: float = 0.05
    gamma: float = 2.2
    bias: float = 0.0      # sensor pedestal in normalized units


DEFAULT_NOISE_PARAMS = NoiseParams()


def _to_linear(image_sRGB_unit: np.ndarray, gamma: float) -> np.ndarray:
    """Approximate inverse sRGB OETF using a single-gamma model."""
    return np.power(np.clip(image_sRGB_unit, 0.0, 1.0), gamma)


def _to_sRGB(image_linear_unit: np.ndarray, gamma: float) -> np.ndarray:
    return np.power(np.clip(image_linear_unit, 0.0, 1.0), 1.0 / gamma)


def corrupt(
    image: np.ndarray,
    alpha: float,
    params: NoiseParams = DEFAULT_NOISE_PARAMS,
    seed: int | None = None,
) -> np.ndarray:
    """Corrupt a normal-light image to a target darkness level.

    Args:
        image: uint8 (H, W, 3) sRGB array. Other dtypes / ranges raise.
        alpha: darkness factor in (0, 1]. 1.0 returns essentially the same
               image (up to noise injection); 0.01 is ≈ 6.6 stops darker.
        params: sensor noise hyperparameters.
        seed: per-call RNG seed for reproducibility.

    Returns:
        uint8 (H, W, 3) sRGB array.
    """
    if image.dtype != np.uint8:
        raise TypeError(f"image must be uint8 sRGB, got dtype {image.dtype}")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"image must be (H, W, 3), got shape {image.shape}")
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")

    rng = np.random.default_rng(seed)

    i_unit = image.astype(np.float64) / 255.0
    i_lin = _to_linear(i_unit, params.gamma)

    signal = alpha * i_lin
    shot = params.sigma_shot_scale * rng.standard_normal(signal.shape) * np.sqrt(np.maximum(signal, 0.0))
    read = alpha * params.sigma_read * rng.standard_normal(signal.shape)
    noisy_lin = np.clip(signal + shot + read + params.bias, 0.0, 1.0)

    noisy_sRGB = _to_sRGB(noisy_lin, params.gamma)
    return np.clip(np.round(noisy_sRGB * 255.0), 0, 255).astype(np.uint8)


def corrupt_batch(
    image: np.ndarray,
    alphas: Sequence[float] = DARKNESS_LEVELS,
    params: NoiseParams = DEFAULT_NOISE_PARAMS,
    seed: int | None = None,
) -> dict[float, np.ndarray]:
    """Produce corrupted copies at multiple darkness levels. Deterministic
    given (seed, alpha)."""
    out: dict[float, np.ndarray] = {}
    for a in alphas:
        # Mix a per-α seed so two adjacent α's don't share noise sample.
        s = None if seed is None else (seed + int(round(a * 10_000)))
        out[float(a)] = corrupt(image, a, params=params, seed=s)
    return out
