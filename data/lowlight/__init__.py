"""Low-light synthesis protocol — Poisson-Gaussian sensor noise model.

Public API:
    corrupt              — corrupt a normal-light image to a target darkness
    NoiseParams          — sensor-noise hyperparameters (Phase 1 IMX390 defaults)
    DEFAULT_NOISE_PARAMS
    DARKNESS_LEVELS      — the 5 canonical α values from configs/lowlight/darkness_levels.yaml
"""

from .noise_model import (
    DARKNESS_LEVELS,
    DEFAULT_NOISE_PARAMS,
    NoiseParams,
    corrupt,
    corrupt_batch,
)

__all__ = [
    "DARKNESS_LEVELS",
    "DEFAULT_NOISE_PARAMS",
    "NoiseParams",
    "corrupt",
    "corrupt_batch",
]
