"""Time-sync utilities across RadarEyes modalities.

The rig's modalities run at different rates: pose ~100 Hz, ZED images ~10 Hz,
radar ~10 Hz, lidar ~10 Hz. A multi-modal sample at target time t is built by
(a) nearest-timestamp matching for the discrete streams (images, radar frames,
lidar frames) and (b) SLERP interpolation of the dense pose stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .loader import LidarStream, PoseStream, RadarStream, Scene, ZedStream


def nearest_index(times: np.ndarray, t: float) -> int:
    """Return the index in `times` closest to `t`. `times` is assumed sorted
    ascending; behaviour is still correct (but slower) for unsorted input."""
    if times.size == 0:
        raise ValueError("empty time array")
    if np.all(np.diff(times) >= 0):
        pos = int(np.searchsorted(times, t))
        if pos == 0:
            return 0
        if pos == times.size:
            return times.size - 1
        before = pos - 1
        after = pos
        return before if abs(times[before] - t) <= abs(times[after] - t) else after
    return int(np.argmin(np.abs(times - t)))


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Slerp between two quaternions in (qx, qy, qz, qw) order at fraction t."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = (1.0 - t) * q0 + t * q1
        return out / (np.linalg.norm(out) + 1e-12)
    theta_0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_theta_0 = float(np.sin(theta_0))
    theta = theta_0 * t
    s0 = float(np.cos(theta) - dot * np.sin(theta) / sin_theta_0)
    s1 = float(np.sin(theta) / sin_theta_0)
    return s0 * q0 + s1 * q1


def interpolate_pose(
    pose: PoseStream, t: float
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate the dense pose stream at time `t`. Returns (position(3,),
    quat(4,)). Outside the stream's time range, the nearest endpoint is
    returned (no extrapolation)."""
    ts = pose.timestamps
    if ts.size == 0:
        raise ValueError("empty pose stream")
    if t <= ts[0]:
        return pose.positions[0].copy(), pose.quats[0].copy()
    if t >= ts[-1]:
        return pose.positions[-1].copy(), pose.quats[-1].copy()

    right = int(np.searchsorted(ts, t))
    left = right - 1
    span = ts[right] - ts[left]
    frac = 0.0 if span <= 0 else float((t - ts[left]) / span)
    position = pose.positions[left] * (1.0 - frac) + pose.positions[right] * frac
    quat = _slerp(pose.quats[left], pose.quats[right], frac)
    return position, quat


@dataclass
class MultiModalSample:
    """Result of synchronizing all modalities at a target time.

    Indices into the per-stream arrays; not loaded data. Call the corresponding
    Scene.<stream>.load_*(idx) method to materialize a payload.
    """

    target_time: float
    position: np.ndarray         # (3,)
    quat: np.ndarray             # (4,) qx qy qz qw
    zed_idx: Optional[int]
    radar_azi_idx: Optional[int]
    radar_ele_idx: Optional[int]
    lidar_idx: Optional[int]
    # Per-stream timestamps for the picked frames; useful for sync-quality
    # diagnostics (max(|dt|) across modalities ⇒ alignment slack).
    zed_time: Optional[float]
    radar_azi_time: Optional[float]
    radar_ele_time: Optional[float]
    lidar_time: Optional[float]

    def max_dt(self) -> float:
        """Largest |Δt| between any picked frame and the target time. Useful
        for filtering bad samples (e.g. drop if > 100 ms)."""
        dts = []
        for ts in (self.zed_time, self.radar_azi_time, self.radar_ele_time, self.lidar_time):
            if ts is not None:
                dts.append(abs(ts - self.target_time))
        return float(max(dts)) if dts else 0.0


def sync_at(scene: Scene, t: float) -> MultiModalSample:
    """Pick the nearest frame in each available stream and interpolate pose."""
    if scene.zed is None:
        raise RuntimeError(f"scene {scene.name!r} has no ZED pose stream to anchor sync")

    position, quat = interpolate_pose(scene.zed.pose_stream, t)

    zed_idx, zed_time = _pick_zed(scene.zed, t)
    azi_idx, azi_time = _pick_radar(scene.radar_azi, t)
    ele_idx, ele_time = _pick_radar(scene.radar_ele, t)
    lidar_idx, lidar_time = _pick_lidar(scene.lidar, t)

    return MultiModalSample(
        target_time=t,
        position=position,
        quat=quat,
        zed_idx=zed_idx,
        radar_azi_idx=azi_idx,
        radar_ele_idx=ele_idx,
        lidar_idx=lidar_idx,
        zed_time=zed_time,
        radar_azi_time=azi_time,
        radar_ele_time=ele_time,
        lidar_time=lidar_time,
    )


def _pick_zed(stream: ZedStream, t: float) -> tuple[Optional[int], Optional[float]]:
    if stream is None or len(stream) == 0:
        return None, None
    idx = nearest_index(stream.image_timestamps, t)
    return idx, float(stream.image_timestamps[idx])


def _pick_radar(
    stream: Optional[RadarStream], t: float
) -> tuple[Optional[int], Optional[float]]:
    if stream is None or len(stream) == 0:
        return None, None
    idx = nearest_index(stream.timestamps, t)
    return idx, float(stream.timestamps[idx])


def _pick_lidar(
    stream: Optional[LidarStream], t: float
) -> tuple[Optional[int], Optional[float]]:
    if stream is None or len(stream) == 0:
        return None, None
    idx = nearest_index(stream.timestamps, t)
    return idx, float(stream.timestamps[idx])
