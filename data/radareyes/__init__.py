"""RadarEyes dataset adapter.

Public API:
    Scene             — per-scene multi-modal handle
    PoseStream        — high-frequency 6-DoF pose with SLERP interpolation
    RadarStream       — radar ADC frames with del_frame applied
    LidarStream       — per-frame LiDAR point clouds
    ZedStream         — ZED RGB images + dense pose stream
    MultiModalSample  — synchronized sample across modalities at a target time
"""

from .loader import LidarStream, PoseStream, RadarStream, Scene, ZedStream
from .sync import MultiModalSample, interpolate_pose, nearest_index

__all__ = [
    "LidarStream",
    "MultiModalSample",
    "PoseStream",
    "RadarStream",
    "Scene",
    "ZedStream",
    "interpolate_pose",
    "nearest_index",
]
