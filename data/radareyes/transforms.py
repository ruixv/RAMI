"""Coordinate transforms for RadarEyes.

The rig's body frame is the ZED camera's optical center (this is the convention
the legacy code at IPLab_mmwavePCD/FuseData/* adopted, by always using
camera_path's pose as the canonical pose). The two radars and the lidar live
at small fixed offsets from the body frame.

Per-sensor offset values are taken from the legacy reference; they are
PLACEHOLDERS to be verified by tools/calibration_sanity.py during Phase 0.

Coordinate convention (ZED scenes):
    body / world: right-handed; +X right, +Y up, +Z forward as the ZED reports.
    Quaternion order in pose files: (qx, qy, qz, qw).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Hardcoded extrinsic offsets observed in the legacy reference; verify before
# trusting. See docs/datasets.md "Calibration offsets" and Phase 0 deliverable
# tools/calibration_sanity.py.
SENSOR_OFFSETS = {
    # body → sensor translation, meters (in body frame, before world rotation)
    "radar_azi": np.array([-0.086, -0.01, 0.102], dtype=np.float64),
    "radar_ele": np.array([0.09, -0.04, 0.153], dtype=np.float64),
    # Initial value from rig photo (user-provided): (-0.075, -0.010, -0.331).
    # Refined by `tools/calib_grid_search.py` against F2_room (views
    # 60/90/120/150/179), maximizing edge-coincidence + surface smoothness +
    # centroid match between LiDAR projection and brightened RGB. The dy/dz
    # update (-3 cm Y, -5 cm Z vs. user's value) lands LiDAR scanlines on
    # chair seats and table edges instead of floating above them; dx was
    # degenerate in the metric so we kept the user's measured value. See
    # logs/phase0/alignment_F2room/pair_user_vs_tuned/.
    "lidar": np.array([-0.075, +0.020, -0.381], dtype=np.float64),
}


def quat_to_R(quat: np.ndarray) -> np.ndarray:
    """Convert a (qx, qy, qz, qw) quaternion to a 3x3 rotation matrix.

    Vectorized over leading dimensions: input (..., 4) → output (..., 3, 3).
    """
    quat = np.asarray(quat, dtype=np.float64)
    if quat.shape[-1] != 4:
        raise ValueError(f"quaternion last dim must be 4, got {quat.shape}")
    qx, qy, qz, qw = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    n = qx * qx + qy * qy + qz * qz + qw * qw
    s = np.where(n > 0, 2.0 / n, 0.0)
    xx, yy, zz = qx * qx * s, qy * qy * s, qz * qz * s
    xy, xz, yz = qx * qy * s, qx * qz * s, qy * qz * s
    wx, wy, wz = qw * qx * s, qw * qy * s, qw * qz * s
    R = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1.0 - (yy + zz)
    R[..., 0, 1] = xy - wz
    R[..., 0, 2] = xz + wy
    R[..., 1, 0] = xy + wz
    R[..., 1, 1] = 1.0 - (xx + zz)
    R[..., 1, 2] = yz - wx
    R[..., 2, 0] = xz - wy
    R[..., 2, 1] = yz + wx
    R[..., 2, 2] = 1.0 - (xx + yy)
    return R


def sensor_to_world(
    points_sensor: np.ndarray,
    body_position: np.ndarray,
    body_quat: np.ndarray,
    sensor_offset: np.ndarray,
) -> np.ndarray:
    """Transform points from a sensor's local frame to world frame.

    Args:
        points_sensor: (N, 3) or (N, 4); only the first 3 cols are rotated.
            Extra columns (intensity, doppler) pass through unchanged.
        body_position: (3,) body origin in world.
        body_quat: (4,) body orientation (qx, qy, qz, qw).
        sensor_offset: (3,) sensor origin in body frame.

    Returns:
        Same shape as points_sensor, in world frame.
    """
    points_sensor = np.asarray(points_sensor, dtype=np.float64)
    if points_sensor.ndim != 2 or points_sensor.shape[1] < 3:
        raise ValueError(
            f"points_sensor must be (N, >=3), got {points_sensor.shape}"
        )
    R = quat_to_R(body_quat)
    xyz = points_sensor[:, :3]
    sensor_origin_world = body_position + R @ sensor_offset
    xyz_world = xyz @ R.T + sensor_origin_world  # rotate then translate
    out = points_sensor.copy()
    out[:, :3] = xyz_world
    return out


def world_to_sensor(
    points_world: np.ndarray,
    body_position: np.ndarray,
    body_quat: np.ndarray,
    sensor_offset: np.ndarray,
) -> np.ndarray:
    """Inverse of sensor_to_world. Same input/output convention."""
    points_world = np.asarray(points_world, dtype=np.float64)
    if points_world.ndim != 2 or points_world.shape[1] < 3:
        raise ValueError(
            f"points_world must be (N, >=3), got {points_world.shape}"
        )
    R = quat_to_R(body_quat)
    sensor_origin_world = body_position + R @ sensor_offset
    xyz = points_world[:, :3]
    xyz_sensor = (xyz - sensor_origin_world) @ R  # = R.T @ delta
    out = points_world.copy()
    out[:, :3] = xyz_sensor
    return out


def pose_to_transform(
    body_position: np.ndarray, body_quat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (R, t) world ← body, both numpy arrays."""
    return quat_to_R(body_quat), np.asarray(body_position, dtype=np.float64)
