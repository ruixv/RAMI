"""Radar signal-processing chain for RadarSplat.

Public API:
    RadarConfig           — physical-constant dataclass per radar arm
    AZI_CONFIG            — locked params for 1843_azi (horizontal arm)
    ELE_CONFIG            — locked params for 1843_ele (vertical arm)
    adc_to_rdae_cube      — ADC complex tensor → 4D RDAE magnitude cube
    build_bin_geometry    — per-axis physical bin coordinates
    ca_cfar_2d            — 2D CA-CFAR detector with integral-image speedup
    cube_to_detections    — RDAE cube + CFAR → (range, az, el, doppler) detections
    detections_to_points  — detections → radar-local Cartesian (N, 5)
    LIRAConfig            — Light-Invariant Radar Anchor builder config
    build_lira_anchors    — Scene + target time → world-frame anchor set
"""

from .cfar import ca_cfar_2d, cube_to_detections, detections_to_points
from .lira import LIRAConfig, build_lira_anchors, AnchorSet
from .rdae import (
    AZI_CONFIG,
    ELE_CONFIG,
    RadarConfig,
    adc_to_rdae_cube,
    build_bin_geometry,
)

__all__ = [
    "AZI_CONFIG",
    "ELE_CONFIG",
    "AnchorSet",
    "LIRAConfig",
    "RadarConfig",
    "adc_to_rdae_cube",
    "build_bin_geometry",
    "build_lira_anchors",
    "ca_cfar_2d",
    "cube_to_detections",
    "detections_to_points",
]
