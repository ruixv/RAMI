"""Baseline registry.

Importing an adapter module registers its baseline (see base.py). This
release ships the models used by the main benchmark: vanilla 3DGS,
RadarSplat-v1 (the training entry for all initialization arms; photometric
loss only at lambda_radar=0), and the DN-Splatter-style depth-supervised
baseline. radarsplat_v0 / radarsplat_physics are imported for their helper
functions used by the preprocessing tools.
"""

from .base import Baseline, BaselineMeta, get_baseline, list_baselines, register

from . import vanilla_3dgs  # noqa: F401
from . import dn_splatter  # noqa: F401
from . import radarsplat_v0  # noqa: F401
from . import radarsplat_v1  # noqa: F401
from . import radarsplat_physics  # noqa: F401

__all__ = [
    "Baseline",
    "BaselineMeta",
    "get_baseline",
    "list_baselines",
    "register",
]
