"""Export a RadarEyes scene as a COLMAP-format directory.

The 3DGS / Scaffold-GS / 2DGS / Mip-Splatting / 3DGS-MCMC code paths all
consume the same on-disk layout:

    <out_dir>/
    ├── images/
    │   ├── frame_000000.png
    │   ├── frame_000001.png
    │   └── ...
    └── sparse/0/
        ├── cameras.txt   # one line per camera intrinsic set (we use one)
        ├── images.txt    # one record per image (2 lines each)
        └── points3D.txt  # sparse pointcloud used for splat initialization

`points3D.txt` is normally produced by COLMAP's SfM pipeline. We replace it
with the LIRA anchor cloud derived in Phase 1, which (a) avoids running
COLMAP — important under low-light where SfM features fail — and (b) is the
"radar-anchored init" we will compare against COLMAP-init in Phase 3
ablations.

The export also handles:
  * Body-to-optical rotation: ZED's RIGHT_HANDED_Z_UP body frame has +Y as
    look-axis; COLMAP / CV pinhole convention is +Z look-axis. We apply
    ZED_BODY_TO_OPTICAL_R from tools/zed_calibration when writing poses.
  * Low-light corruption: pass a non-1.0 `alpha` to corrupt the ZED RGB
    frames in place before writing them out, using the Phase 1 noise model.
  * Train/test split: COLMAP doesn't have a native split file, but most
    3DGS variants accept a `test_indices.txt` or `--eval` flag with
    every-Nth-as-test. We write `splits.json` for our own training loop and
    also a `test_image_names.txt` listing held-out filenames.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from data.lowlight import DEFAULT_NOISE_PARAMS, NoiseParams, corrupt
from data.radareyes import Scene
from data.radareyes.sync import interpolate_pose
from tools.zed_calibration import ZED_BODY_TO_OPTICAL_R, zed_hd720_intrinsics


@dataclass(frozen=True)
class ExportConfig:
    """Defaults for Phase 2 baselines. Override per experiment."""

    alpha: float = 1.0
    seed: int = 20260523
    noise: NoiseParams = DEFAULT_NOISE_PARAMS
    test_every_nth: int = 8       # ~12.5% held-out, ≈ standard 3DGS practice
    # Subsample images uniformly to keep export size manageable for the
    # initial 3DGS smoke tests. None = use every image.
    keep_every_nth: int | None = None
    # If True, replace LIRA anchor cloud with a small random pointcloud
    # (useful when LIRA itself is being ablated against COLMAP-init).
    use_random_anchors: bool = False
    random_anchors_n: int = 1000
    anchor_color: tuple[int, int, int] = (200, 200, 200)


def _quaternion_w2c_from_body_quat(body_quat: np.ndarray) -> tuple[float, float, float, float]:
    """Convert body-in-world quaternion (qx, qy, qz, qw) to the
    world-to-camera (optical) quaternion (qw, qx, qy, qz) that COLMAP wants.

    Sequence:
      R_b2w(quat)          — body-to-world rotation from input
      R_w2b = R_b2w.T      — world-to-body
      R_b2o = ZED_BODY_TO_OPTICAL_R
      R_w2c = R_b2o @ R_w2b
      → quat(R_w2c) in (qw, qx, qy, qz)
    """
    from data.radareyes.transforms import quat_to_R

    R_b2w = quat_to_R(np.asarray(body_quat, dtype=np.float64))
    R_w2b = R_b2w.T
    R_w2c = ZED_BODY_TO_OPTICAL_R @ R_w2b
    # Matrix → quaternion (qw, qx, qy, qz) via Shepperd's method (numerically
    # robust for any 3×3 rotation matrix).
    t = np.trace(R_w2c)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R_w2c[2, 1] - R_w2c[1, 2]) / s
        qy = (R_w2c[0, 2] - R_w2c[2, 0]) / s
        qz = (R_w2c[1, 0] - R_w2c[0, 1]) / s
    elif (R_w2c[0, 0] > R_w2c[1, 1]) and (R_w2c[0, 0] > R_w2c[2, 2]):
        s = np.sqrt(1.0 + R_w2c[0, 0] - R_w2c[1, 1] - R_w2c[2, 2]) * 2.0
        qw = (R_w2c[2, 1] - R_w2c[1, 2]) / s
        qx = 0.25 * s
        qy = (R_w2c[0, 1] + R_w2c[1, 0]) / s
        qz = (R_w2c[0, 2] + R_w2c[2, 0]) / s
    elif R_w2c[1, 1] > R_w2c[2, 2]:
        s = np.sqrt(1.0 + R_w2c[1, 1] - R_w2c[0, 0] - R_w2c[2, 2]) * 2.0
        qw = (R_w2c[0, 2] - R_w2c[2, 0]) / s
        qx = (R_w2c[0, 1] + R_w2c[1, 0]) / s
        qy = 0.25 * s
        qz = (R_w2c[1, 2] + R_w2c[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R_w2c[2, 2] - R_w2c[0, 0] - R_w2c[1, 1]) * 2.0
        qw = (R_w2c[1, 0] - R_w2c[0, 1]) / s
        qx = (R_w2c[0, 2] + R_w2c[2, 0]) / s
        qy = (R_w2c[1, 2] + R_w2c[2, 1]) / s
        qz = 0.25 * s
    return float(qw), float(qx), float(qy), float(qz)


def _translation_w2c_from_body(
    body_position: np.ndarray, body_quat: np.ndarray
) -> tuple[float, float, float]:
    """COLMAP's `T` in `images.txt` is the camera origin expressed in the
    optical frame; equivalently, `T = -R_w2c @ position`."""
    from data.radareyes.transforms import quat_to_R

    R_b2w = quat_to_R(np.asarray(body_quat, dtype=np.float64))
    R_w2b = R_b2w.T
    R_w2c = ZED_BODY_TO_OPTICAL_R @ R_w2b
    t = -R_w2c @ np.asarray(body_position, dtype=np.float64)
    return float(t[0]), float(t[1]), float(t[2])


def export_scene(
    scene: Scene,
    out_dir: str,
    config: Optional[ExportConfig] = None,
    anchor_positions: Optional[np.ndarray] = None,
) -> dict:
    """Write a scene to COLMAP-format directory.

    Args:
        scene: a loaded Scene.
        out_dir: target directory; created if missing. Existing contents are
            NOT removed — pass a fresh path or clean it yourself.
        config: ExportConfig; see dataclass for defaults.
        anchor_positions: (N, 3) float array of LIRA anchor world positions
            to use as points3D. Pass None to fall back to random anchors per
            `config.use_random_anchors`.

    Returns:
        A summary dict (image counts, splits, etc.) for logging.
    """
    if config is None:
        config = ExportConfig()
    if scene.zed is None or len(scene.zed) == 0:
        raise RuntimeError(f"scene {scene.name!r} has no ZED images to export")

    images_dir = os.path.join(out_dir, "images")
    sparse_dir = os.path.join(out_dir, "sparse", "0")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    # 1. Pick which image indices to export.
    n = len(scene.zed)
    if config.keep_every_nth and config.keep_every_nth > 1:
        keep_indices = list(range(0, n, config.keep_every_nth))
    else:
        keep_indices = list(range(n))

    # 2. Train / test split — every Nth image goes to test.
    test_indices = set(keep_indices[::config.test_every_nth]) if config.test_every_nth > 1 else set()
    train_indices = [i for i in keep_indices if i not in test_indices]

    # 3. Write cameras.txt — one PINHOLE camera.
    fx, fy, cx, cy = zed_hd720_intrinsics()
    image_w, image_h = scene.zed.load_image(keep_indices[0]).shape[1], scene.zed.load_image(keep_indices[0]).shape[0]
    with open(os.path.join(sparse_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {image_w} {image_h} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

    # 4. Write images.txt + the actual PNG files.
    images_txt_lines = [
        "# Image list with two lines of data per image:\n",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)\n",
    ]
    image_records = []
    rng = np.random.default_rng(config.seed)
    pose_stream = scene.zed.pose_stream

    for image_id, idx in enumerate(keep_indices, start=1):
        # Pose at image's timestamp via SLERP interpolation of the dense
        # pose stream — more accurate than the per-image pose entry.
        t_img = float(scene.zed.image_timestamps[idx])
        pos, quat = interpolate_pose(pose_stream, t_img)

        qw, qx, qy, qz = _quaternion_w2c_from_body_quat(quat)
        tx, ty, tz = _translation_w2c_from_body(pos, quat)

        frame_num = int(scene.zed.image_frame_ids[idx])
        png_name = f"frame_{frame_num:06d}.png"
        png_path = os.path.join(images_dir, png_name)

        # Load + optional corruption.
        img = scene.zed.load_image(idx)
        if config.alpha < 1.0:
            seed_i = config.seed + frame_num
            img = corrupt(img, alpha=config.alpha, params=config.noise, seed=seed_i)
        Image.fromarray(img).save(png_path)

        images_txt_lines.append(
            f"{image_id} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f} "
            f"{tx:.6f} {ty:.6f} {tz:.6f} 1 {png_name}\n"
        )
        images_txt_lines.append("\n")  # empty 2D-feature line
        image_records.append({
            "image_id": image_id,
            "scene_image_idx": idx,
            "frame_num": frame_num,
            "filename": png_name,
            "t_image": t_img,
            "split": "test" if idx in test_indices else "train",
        })

    with open(os.path.join(sparse_dir, "images.txt"), "w") as f:
        f.writelines(images_txt_lines)

    # 5. Write points3D.txt.
    if anchor_positions is None and config.use_random_anchors:
        anchor_positions = rng.normal(size=(config.random_anchors_n, 3)) * 1.0
    if anchor_positions is None:
        anchor_positions = np.zeros((0, 3), dtype=np.float64)
    anchor_positions = np.asarray(anchor_positions, dtype=np.float64)

    r, g, b = config.anchor_color
    with open(os.path.join(sparse_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for pid, (x, y, z) in enumerate(anchor_positions, start=1):
            f.write(f"{pid} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 0.0\n")

    # 6. Write our own splits file (3DGS variants use --eval flag, but our
    # internal eval harness consumes this).
    splits = {
        "scene": scene.name,
        "alpha": config.alpha,
        "n_total": len(image_records),
        "n_train": sum(1 for r in image_records if r["split"] == "train"),
        "n_test": sum(1 for r in image_records if r["split"] == "test"),
        "test_image_names": [r["filename"] for r in image_records if r["split"] == "test"],
        "records": image_records,
        "n_points3D": int(anchor_positions.shape[0]),
        "image_size": [image_w, image_h],
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "config": {
            "alpha": config.alpha,
            "seed": config.seed,
            "test_every_nth": config.test_every_nth,
            "keep_every_nth": config.keep_every_nth,
        },
    }
    with open(os.path.join(out_dir, "splits.json"), "w") as f:
        json.dump(splits, f, indent=2)
    with open(os.path.join(out_dir, "test_image_names.txt"), "w") as f:
        f.write("\n".join(splits["test_image_names"]) + "\n")

    return splits
