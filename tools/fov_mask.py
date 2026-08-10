"""Per-pose FOV masks for ZED, mmWave radar, and LiDAR sensors.

The dataset's three modalities cover different angular regions of the world
(see docs/fov_geometry.md). Loss terms must respect those regions: photometric
loss only applies inside the ZED frustum at training-view time; the radar
consistency loss only applies where radar has support. This module supplies
the masks.

All functions take points in WORLD frame and a sensor's WORLD pose
(position + quaternion), and return a boolean mask of shape (N,).

Vectorized over both dimensions; no Python-side iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from data.radareyes.transforms import quat_to_R

# Sensor FOV parameters used as defaults. See docs/fov_geometry.md for the
# physical justification. Phase 1 should refine these against rig calibration.

ZED_HFOV_DEG = 110.0       # ZED 2i wide-angle horizontal
ZED_VFOV_DEG = 70.0        # vertical
ZED_NEAR_M = 0.3
ZED_FAR_M = 20.0
ZED_INTRINSICS_DEFAULT = None  # set in Phase 0 calibration step; until then use FOV cone

RADAR_AZI_HFOV_DEG = 120.0   # ±60° azimuth
RADAR_AZI_VFOV_DEG = 40.0    # ±20° elevation
RADAR_ELE_HFOV_DEG = 40.0    # ±20° azimuth (rotated antenna)
RADAR_ELE_VFOV_DEG = 120.0   # ±60° elevation
RADAR_NEAR_M = 0.1
RADAR_FAR_M = 15.0

LIDAR_HFOV_DEG = 360.0
LIDAR_VFOV_DEG = 30.0        # ±15°, typical Velodyne VLP-16-class
LIDAR_NEAR_M = 0.5
LIDAR_FAR_M = 80.0


@dataclass(frozen=True)
class FOVCone:
    """Symmetric truncated cone in a sensor's local frame.

    Local convention: +Z is the sensor look-axis. (Matches the convention used
    by the legacy IPLab_mmwavePCD code when expressing sensor-local geometry.)
    """

    hfov_deg: float
    vfov_deg: float
    near_m: float
    far_m: float

    def mask(self, points_sensor: np.ndarray) -> np.ndarray:
        """Return (N,) boolean mask: True if the point falls inside the cone."""
        pts = np.asarray(points_sensor, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"points_sensor must be (N, >=3), got {pts.shape}")
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        range_sq = x * x + y * y + z * z
        in_range = (range_sq >= self.near_m**2) & (range_sq <= self.far_m**2)
        # +Z is the look axis. Points behind the sensor are out.
        in_front = z > 0
        # Half-angle thresholds; use tan-form to avoid arctan per point.
        h_half = np.tan(np.deg2rad(self.hfov_deg) * 0.5)
        v_half = np.tan(np.deg2rad(self.vfov_deg) * 0.5)
        in_h = np.abs(x) <= h_half * z
        in_v = np.abs(y) <= v_half * z
        return in_range & in_front & in_h & in_v


def points_world_to_sensor(
    points_world: np.ndarray,
    sensor_position: np.ndarray,
    sensor_quat: np.ndarray,
    body_to_local_R: np.ndarray | None = None,
) -> np.ndarray:
    """Transform a (N, >=3) world-frame array into the sensor's local frame.

    The two-step convention:
        P_body  = R(sensor_quat).T @ (P_world - sensor_position)
        P_local = body_to_local_R   @ P_body          (if provided)

    If `body_to_local_R` is None, the body frame and the sensor's local frame
    are assumed to coincide. For ZED projections, pass
    ZED_BODY_TO_OPTICAL_R from tools/zed_calibration; for radar/lidar FOV
    cones whose look-axis is +Z in the body frame, leave it None.
    """
    pts = np.asarray(points_world, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"points_world must be (N, >=3), got {pts.shape}")
    R = quat_to_R(sensor_quat)
    delta = pts[:, :3] - np.asarray(sensor_position, dtype=np.float64)
    pts_body = delta @ R  # = R.T @ delta with per-row broadcasting
    if body_to_local_R is None:
        return pts_body
    body_to_local_R = np.asarray(body_to_local_R, dtype=np.float64)
    return pts_body @ body_to_local_R.T


def fov_mask(
    points_world: np.ndarray,
    sensor_position: np.ndarray,
    sensor_quat: np.ndarray,
    cone: FOVCone,
    body_to_local_R: np.ndarray | None = None,
) -> np.ndarray:
    pts_sensor = points_world_to_sensor(
        points_world, sensor_position, sensor_quat, body_to_local_R
    )
    return cone.mask(pts_sensor)


# Convenience constructors with the project defaults. -------------------------

ZED_CONE = FOVCone(ZED_HFOV_DEG, ZED_VFOV_DEG, ZED_NEAR_M, ZED_FAR_M)
RADAR_AZI_CONE = FOVCone(RADAR_AZI_HFOV_DEG, RADAR_AZI_VFOV_DEG, RADAR_NEAR_M, RADAR_FAR_M)
RADAR_ELE_CONE = FOVCone(RADAR_ELE_HFOV_DEG, RADAR_ELE_VFOV_DEG, RADAR_NEAR_M, RADAR_FAR_M)
LIDAR_CONE = FOVCone(LIDAR_HFOV_DEG, LIDAR_VFOV_DEG, LIDAR_NEAR_M, LIDAR_FAR_M)


def project_to_camera(
    points_world: np.ndarray,
    cam_position: np.ndarray,
    cam_quat: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    image_w: int,
    image_h: int,
    near: float = ZED_NEAR_M,
    far: float = ZED_FAR_M,
    body_to_optical_R: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pinhole-project world points into a camera frame.

    Arguments:
        points_world: (N, 3) world coordinates.
        cam_position, cam_quat: camera body's pose in world (quat order
            qx, qy, qz, qw).
        fx, fy, cx, cy: pinhole intrinsics in the OPTICAL frame
            (+X right, +Y down, +Z forward).
        image_w, image_h: image dimensions in pixels.
        near, far: depth gating in optical-frame +Z.
        body_to_optical_R: rotation from the body frame to the optical frame.
            For ZED's RIGHT_HANDED_Z_UP body convention pass
            ZED_BODY_TO_OPTICAL_R (from tools.zed_calibration). If None, body
            and optical frames are assumed identical (i.e. body already uses
            +Z look-axis).

    Returns:
        pixels: (N, 2) float64, image-plane (u, v) coordinates. NaN where the
            point is behind the camera or out of frame.
        depths: (N,) float64 depth along optical +Z (NaN if invalid).
        valid: (N,) bool mask: True if the projection lands inside the image
            with depth in [near, far].
    """
    pts_cam = points_world_to_sensor(
        points_world, cam_position, cam_quat, body_to_optical_R
    )
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * (x / z) + cx
        v = fy * (y / z) + cy
    in_front = z > 0
    in_range = (z >= near) & (z <= far)
    in_bounds = (u >= 0) & (u < image_w) & (v >= 0) & (v < image_h)
    valid = in_front & in_range & in_bounds
    pixels = np.stack([u, v], axis=-1)
    pixels[~valid] = np.nan
    depths = np.where(valid, z, np.nan)
    return pixels, depths, valid


def per_pixel_min_depth(
    pixels: np.ndarray,
    depths: np.ndarray,
    valid: np.ndarray,
    image_w: int,
    image_h: int,
) -> np.ndarray:
    """Rasterize point projections into a (H, W) depth image (min over points
    falling in the same pixel). Empty pixels are NaN."""
    out = np.full((image_h, image_w), np.nan, dtype=np.float64)
    if not valid.any():
        return out
    u = pixels[valid, 0].astype(np.int64)
    v = pixels[valid, 1].astype(np.int64)
    d = depths[valid]
    # Naive per-pixel min via np.minimum.reduceat would need sorting; for
    # Phase 0 sanity, the simple loop is fine — Phase 1 cythonizes if needed.
    order = np.argsort(d)  # smaller depths last so they overwrite
    for k in order[::-1]:
        out[v[k], u[k]] = d[k]
    return out
