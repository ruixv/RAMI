"""LIRA — Light-Invariant Radar Anchors.

Given a Scene and a target time, build a sparse set of world-frame anchor
points by:
  1. Picking all radar frames within an accumulation window around the target
     time.
  2. For each frame: ADC → RDAE cube → CFAR detection → radar-local 3D points.
  3. Transforming each frame's points to world frame using the pose
     interpolated at that radar's timestamp.
  4. Voxelizing the accumulated points at a fixed resolution (default 5 cm),
     and keeping one anchor per occupied voxel (the centroid of the points
     inside, with summary statistics as per-anchor features).

The result is the input to Phase 3's Scaffold-GS-style anchor MLP. For Phase
1 the anchors are evaluated against accumulated LiDAR ground truth via
`eval/lira_vs_lidar.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from data.radareyes import Scene
from data.radareyes.sync import interpolate_pose, nearest_index
from data.radareyes.transforms import SENSOR_OFFSETS, sensor_to_world

from .cfar import cube_to_detections, detections_to_points
from .rdae import AZI_CONFIG, ELE_CONFIG, RadarConfig, adc_to_rdae_cube


@dataclass(frozen=True)
class LIRAConfig:
    """Anchor-build configuration. Defaults are Phase 1 starting points."""

    voxel_size_m: float = 0.05
    accumulation_window_s: float = 2.0   # ± around target time
    cfar_pfa: float = 1e-3
    min_range_bin: int = 4
    max_range_bin: int = 120
    use_radar_azi: bool = True
    use_radar_ele: bool = False  # ele is high-rate but low-Doppler; off for now
    # World-frame bounding box for the voxel grid. None = auto-fit to data.
    bbox: Optional[tuple[float, float, float, float, float, float]] = None
    # Frame stride within the accumulation window; >1 sub-samples to save time.
    frame_stride: int = 1


@dataclass
class AnchorSet:
    """World-frame anchor cloud + per-anchor features for the Anchor MLP."""

    positions: np.ndarray         # (M, 3) float64, world frame
    intensity: np.ndarray         # (M,) mean peak power across contributors
    doppler: np.ndarray           # (M,) mean Doppler magnitude across contributors
    n_contributors: np.ndarray    # (M,) int64, number of source detections
    voxel_size_m: float
    bbox: tuple[float, float, float, float, float, float]
    source_radars: tuple[str, ...]

    def __len__(self) -> int:
        return self.positions.shape[0]

    def features(self) -> np.ndarray:
        """(M, F) per-anchor feature matrix to feed the Anchor MLP later.
        Phase 1 feature design:
            f0 — log(1 + intensity)
            f1 — Doppler magnitude (m/s)
            f2 — log(1 + n_contributors)
        Phase 2 may extend with sensor-of-origin one-hot, distance-to-frustum,
        etc. Keep the order stable so checkpoints survive.
        """
        return np.stack(
            [
                np.log1p(self.intensity).astype(np.float64),
                np.abs(self.doppler).astype(np.float64),
                np.log1p(self.n_contributors.astype(np.float64)),
            ],
            axis=1,
        )


def _accumulate_radar_world_points(
    scene: Scene, t_target: float, config: LIRAConfig
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Returns (P, 5) world-frame (x, y, z, intensity, doppler) and a tuple
    of source-radar names. P points are concatenated from all radars used."""

    chunks: list[np.ndarray] = []
    sources: list[str] = []

    plan: list[tuple[RadarConfig, str]] = []
    if config.use_radar_azi and scene.radar_azi is not None:
        plan.append((AZI_CONFIG, "azi"))
    if config.use_radar_ele and scene.radar_ele is not None:
        plan.append((ELE_CONFIG, "ele"))

    for cfg, _ in plan:
        stream = scene.radar_azi if cfg.name == "azi" else scene.radar_ele
        if stream is None:
            continue
        idx_lo = nearest_index(stream.timestamps, t_target - config.accumulation_window_s)
        idx_hi = nearest_index(stream.timestamps, t_target + config.accumulation_window_s)
        if idx_lo > idx_hi:
            idx_lo, idx_hi = idx_hi, idx_lo
        indices = list(range(idx_lo, idx_hi + 1, config.frame_stride))

        for i in indices:
            adc = stream.load_adc(i)
            cube = adc_to_rdae_cube(adc, cfg)
            detections = cube_to_detections(
                cube, cfg,
                pfa=config.cfar_pfa,
                min_range_bin=config.min_range_bin,
                max_range_bin=config.max_range_bin,
            )
            if detections.shape[0] == 0:
                continue
            pts_local = detections_to_points(detections, cfg)

            # Interpolate body pose at this radar's timestamp.
            t_radar = float(stream.timestamps[i])
            pose_pos, pose_quat = interpolate_pose(scene.zed.pose_stream, t_radar)
            offset = SENSOR_OFFSETS[f"radar_{cfg.name}"]
            pts_world = sensor_to_world(pts_local, pose_pos, pose_quat, offset)

            chunks.append(pts_world)
            sources.append(cfg.name)

    if not chunks:
        return np.zeros((0, 5), dtype=np.float64), tuple(sorted(set(sources)))

    return np.concatenate(chunks, axis=0), tuple(sorted(set(sources)))


def _voxelize(
    points: np.ndarray,
    voxel_size_m: float,
    bbox: Optional[tuple[float, float, float, float, float, float]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float, float, float, float]]:
    """Aggregate points into voxels. Returns per-voxel:
        centroids (M, 3) — centroid of the points in the voxel
        sums    (M, K)   — sum of any extra columns (intensity, doppler abs)
        counts  (M,) int — number of contributing detections per voxel
        bbox             — actual bbox used (input or auto-fit).
    `points` is (N, ≥3); columns 3+ are the "extra" features summed.
    """
    if points.shape[0] == 0:
        empty_bbox = bbox if bbox is not None else (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, max(points.shape[1] - 3, 0)), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
            empty_bbox,
        )

    xyz = points[:, :3]
    if bbox is None:
        lo = xyz.min(axis=0) - voxel_size_m * 0.5
        hi = xyz.max(axis=0) + voxel_size_m * 0.5
        bbox = (float(lo[0]), float(lo[1]), float(lo[2]),
                float(hi[0]), float(hi[1]), float(hi[2]))

    lo = np.array(bbox[:3], dtype=np.float64)
    hi = np.array(bbox[3:], dtype=np.float64)
    dims = np.maximum(np.ceil((hi - lo) / voxel_size_m).astype(np.int64), 1)

    # Voxel indices per point.
    idx = np.floor((xyz - lo) / voxel_size_m).astype(np.int64)
    in_bounds = np.all((idx >= 0) & (idx < dims[None, :]), axis=1)
    idx = idx[in_bounds]
    xyz = xyz[in_bounds]
    extra = points[in_bounds, 3:]

    # Encode voxel index as a single linear key.
    keys = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
    order = np.argsort(keys, kind="stable")
    keys_sorted = keys[order]
    xyz_sorted = xyz[order]
    extra_sorted = extra[order]

    unique_keys, starts = np.unique(keys_sorted, return_index=True)
    ends = np.append(starts[1:], keys_sorted.shape[0])

    # Aggregate per group.
    counts = (ends - starts).astype(np.int64)
    centroids = np.zeros((unique_keys.shape[0], 3), dtype=np.float64)
    sums = np.zeros((unique_keys.shape[0], extra_sorted.shape[1]), dtype=np.float64)
    for i, (s, e) in enumerate(zip(starts, ends)):
        centroids[i] = xyz_sorted[s:e].mean(axis=0)
        if extra_sorted.shape[1] > 0:
            sums[i] = extra_sorted[s:e].sum(axis=0)

    return centroids, sums, counts, bbox


def build_lira_anchors(
    scene: Scene, t_target: float, config: Optional[LIRAConfig] = None
) -> AnchorSet:
    """End-to-end Phase 1 LIRA pipeline.

    Returns an AnchorSet whose `.positions` are world-frame voxel centroids
    suitable for initializing Phase 3's Scaffold-GS-style anchor MLP.
    """
    if config is None:
        config = LIRAConfig()
    if scene.zed is None:
        raise RuntimeError(f"scene {scene.name!r} has no ZED pose stream; LIRA cannot run")

    pts_world, sources = _accumulate_radar_world_points(scene, t_target, config)
    # Points columns: (x, y, z, intensity, doppler).
    centroids, sums, counts, bbox = _voxelize(pts_world, config.voxel_size_m, config.bbox)
    if centroids.shape[0] == 0:
        intensity = np.zeros((0,), dtype=np.float64)
        doppler = np.zeros((0,), dtype=np.float64)
    else:
        intensity = sums[:, 0] / np.maximum(counts, 1)
        doppler = sums[:, 1] / np.maximum(counts, 1)
    return AnchorSet(
        positions=centroids,
        intensity=intensity,
        doppler=doppler,
        n_contributors=counts,
        voxel_size_m=config.voxel_size_m,
        bbox=bbox,
        source_radars=sources,
    )


def accumulate_lidar(
    scene: Scene, t_target: float, window_s: float = 2.0
) -> np.ndarray:
    """World-frame LiDAR ground-truth point cloud accumulated over the same
    window as LIRA. Returns (N, 4) array (x, y, z, intensity)."""
    if scene.lidar is None or scene.zed is None:
        raise RuntimeError("scene missing LiDAR or ZED pose stream")

    ts = scene.lidar.timestamps
    i_lo = nearest_index(ts, t_target - window_s)
    i_hi = nearest_index(ts, t_target + window_s)
    if i_lo > i_hi:
        i_lo, i_hi = i_hi, i_lo

    chunks: list[np.ndarray] = []
    for i in range(i_lo, i_hi + 1):
        pcd = scene.lidar.load_pcd(i)
        if pcd.shape[0] == 0:
            continue
        t = float(ts[i])
        pos, quat = interpolate_pose(scene.zed.pose_stream, t)
        chunks.append(sensor_to_world(pcd, pos, quat, SENSOR_OFFSETS["lidar"]))
    if not chunks:
        return np.zeros((0, 4), dtype=np.float64)
    return np.concatenate(chunks, axis=0)
