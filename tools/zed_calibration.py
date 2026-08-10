"""Locked ZED 2i HD720 intrinsics and coordinate conventions for RadarSplat.

Background
----------
The RadarEyes capture rig used a Stereolabs ZED 2i in HD720 mode
(1280 × 720, factory-rectified single-eye left view; see
SelfMadePackage/DataCollection_code_0603/Camera/save_camera_data_ZED.py).
Per-unit SN<digits>.conf calibration files are NOT accessible on the server
and the hardware is no longer reachable, so we cannot read the unit-specific
calibration via pyzed.sl.

We therefore lock canonical ZED 2i HD720 **factory-typical** intrinsics here.
Unit-to-unit variation on Stereolabs cameras is typically ±2% for fx/fy and
±5-15 px for cx/cy — small enough that our Phase 2/3 fusion experiments are
not biased toward a wrong answer, but reviewers will (correctly) treat any
absolute geometry claims with that uncertainty band. We disclose it in the
paper.

Coordinate convention
---------------------
save_camera_data_ZED.py opens the camera with
  init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
which fixes the world frame as right-handed with:
  +X = right (in image), +Y = forward (into scene), +Z = up.
The rig's body frame coincides with this convention (the ZED is the body
origin). The pose stored in Camera_ZED/pose.txt is the body's position +
quaternion in this world frame.

The optical / pinhole projection frame, however, follows the standard CV
convention used by every NVS code path on the planet:
  +X = right, +Y = down, +Z = forward (look-axis).

So projection of a world point P into the rectified ZED image is:
  P_body    = R(pose.quat).T @ (P_world - pose.position)
  P_optical = ZED_BODY_TO_OPTICAL_R @ P_body          # the rotation below
  u = fx * P_optical[0] / P_optical[2] + cx
  v = fy * P_optical[1] / P_optical[2] + cy

The body→optical rotation maps body axes to optical axes as:
  body +X (right)   → optical +X (right)
  body +Y (forward) → optical +Z (forward, look-axis)
  body +Z (up)      → optical -Y (down is +Y in CV, so up is -Y)
"""

from __future__ import annotations

import numpy as np


# ZED 2i HD720 factory-typical intrinsics (per Stereolabs documentation /
# datasheet for the rectified left eye at HD720). Unit-specific values differ
# by O(1%); see docs/calibration_status.md for the disclosure.
ZED_2I_HD720_FX: float = 528.0
ZED_2I_HD720_FY: float = 528.0
ZED_2I_HD720_CX: float = 640.0   # = 1280 / 2
ZED_2I_HD720_CY: float = 360.0   # =  720 / 2
ZED_2I_HD720_IMAGE_W: int = 1280
ZED_2I_HD720_IMAGE_H: int = 720
ZED_2I_BASELINE_M: float = 0.12   # ZED 2i nominal stereo baseline (meters)

# Look-axis convention: in ZED's RIGHT_HANDED_Z_UP body frame, +Y points
# forward (the camera's look direction). Standard CV optical frame has +Z
# forward. The constant rotation below converts body-frame points to optical-
# frame points:  P_optical = ZED_BODY_TO_OPTICAL_R @ P_body
ZED_BODY_TO_OPTICAL_R: np.ndarray = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def zed_hd720_intrinsics() -> tuple[float, float, float, float]:
    """Returns (fx, fy, cx, cy) for ZED 2i HD720 factory-typical defaults."""
    return ZED_2I_HD720_FX, ZED_2I_HD720_FY, ZED_2I_HD720_CX, ZED_2I_HD720_CY


def zed_hd720_image_size() -> tuple[int, int]:
    """Returns (width, height) for HD720 mode."""
    return ZED_2I_HD720_IMAGE_W, ZED_2I_HD720_IMAGE_H
