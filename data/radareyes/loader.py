"""RadarEyes per-scene loader.

Each scene under <RAMI_DATA_ROOT>/<scene_name>/ has heterogeneous
streams that are synchronized via per-stream timestamp.txt files. This module
exposes one Scene class that wraps the streams and provides typed access to
the raw data without doing any signal processing — RDAE / CFAR / fusion lives
in models/radar/ (Phase 1).

Conventions deduced from probing real scenes (see docs/datasets.md):

* timestamp.txt files: line 0 is "<sensor> start time: ..." header, line -1 is
  "<sensor> end time: ..." trailer; the middle N lines are float epoch seconds.
* pose.txt (Camera_ZED only): same header/trailer layout, middle lines are
  "[x, y, z, qx, qy, qz, qw]" strings.
* For ZED scenes the camera frame convention is (x, y, z) with no axis swap.
  For older T265-based captures the legacy code applied (x, -z, y); we filter
  to ZED-only scenes for this paper.
* Radar ADC frame indices start at 1, not 0 (e.g., frame_1.bin .. frame_774.bin).
  Position-in-sorted-list, not the integer in the filename, is what aligns to
  the timestamp list.
* ZED images are sparse (every 10th pose tick by default); pose stream is dense.
* del_frame.txt lists positional indices (post-sort) that the radar dropped
  during capture and must be removed from both the path list and the timestamp
  list before any downstream matching.
* Lidar_pcd/ has the per-frame .bin files; the timestamp file is at
  Lidar/timestamp.txt (one level up).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

NUM_ADC_SAMPLES = 128
NUM_RX = 4
NUM_TX = 3
# Chirps per frame differs between the two radars in the RadarEyes rig:
#   1843_azi (horizontal):  128 chirps × 10 Hz frame rate  → 3072 KB / frame
#   1843_ele (vertical):      8 chirps × 200 Hz frame rate →  192 KB / frame
# We auto-detect this from the first ADC file size on disk, so each
# RadarStream carries its own `chirps_per_frame` attribute.
NUM_CHIRPS_AZI = 128
NUM_CHIRPS_ELE = 8
COMPLEX128_BYTES = 16
BYTES_PER_CHIRP = NUM_ADC_SAMPLES * NUM_RX * NUM_TX * COMPLEX128_BYTES  # 24576

# Legacy alias preserved for callers that hardcoded the azi shape.
ADC_FRAME_SHAPE = (NUM_ADC_SAMPLES, NUM_CHIRPS_AZI, NUM_RX, NUM_TX)
ADC_FRAME_BYTES = BYTES_PER_CHIRP * NUM_CHIRPS_AZI  # 3145728


_FRAME_INT_RE = re.compile(r"(\d+)")


def _frame_index_from_name(path: str) -> int:
    """Extract the trailing integer from a frame_<N>.<ext> filename."""
    nums = _FRAME_INT_RE.findall(os.path.basename(path))
    if not nums:
        raise ValueError(f"no frame index parseable from {path!r}")
    return int(nums[-1])


def _list_frames_sorted(dir_path: str, ext: str) -> List[str]:
    if not os.path.isdir(dir_path):
        return []
    out = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.endswith(ext) and f.startswith("frame_")
    ]
    out.sort(key=_frame_index_from_name)
    return out


def _read_stripped_lines(path: str) -> List[str]:
    """Read timestamp.txt or pose.txt, strip the header (line 0) and trailer
    (line -1)."""
    with open(path, "r") as f:
        lines = f.read().splitlines()
    if len(lines) < 3:
        return []
    return lines[1:-1]


def _parse_timestamps(path: str) -> np.ndarray:
    raw = _read_stripped_lines(path)
    if not raw:
        return np.zeros((0,), dtype=np.float64)
    return np.array([float(s) for s in raw], dtype=np.float64)


def _parse_pose(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (positions Nx3 float64, quats Nx4 float64) with quat order
    (qx, qy, qz, qw)."""
    raw = _read_stripped_lines(path)
    positions = np.zeros((len(raw), 3), dtype=np.float64)
    quats = np.zeros((len(raw), 4), dtype=np.float64)
    for i, line in enumerate(raw):
        parts = [p.strip() for p in line.strip("[]").split(",")]
        if len(parts) != 7:
            raise ValueError(
                f"pose line {i} of {path!r} has {len(parts)} fields, expected 7"
            )
        vals = [float(p) for p in parts]
        positions[i] = vals[0:3]
        quats[i] = vals[3:7]
    return positions, quats


@dataclass
class PoseStream:
    """Dense 6-DoF pose stream. Quaternion convention is (qx, qy, qz, qw)."""

    positions: np.ndarray  # (N, 3) float64, world frame
    quats: np.ndarray      # (N, 4) float64, qx qy qz qw
    timestamps: np.ndarray  # (N,) float64, epoch seconds

    def __post_init__(self) -> None:
        n = self.positions.shape[0]
        if not (self.quats.shape[0] == self.timestamps.shape[0] == n):
            raise ValueError(
                f"pose stream length mismatch: pos={n} quat={self.quats.shape[0]} "
                f"ts={self.timestamps.shape[0]}"
            )
        if self.positions.shape[1] != 3 or self.quats.shape[1] != 4:
            raise ValueError("positions must be Nx3 and quats must be Nx4")

    def __len__(self) -> int:
        return self.positions.shape[0]


@dataclass
class RadarStream:
    """One of the two TI 1843 radars (azimuth or elevation arm).

    `chirps_per_frame` is auto-detected from the first ADC file size in
    Scene._build_radar — 128 for the azimuth (10 Hz) arm, 8 for the
    elevation (200 Hz) arm. Frame shape = (samples=128, chirps, rx=4, tx=3).
    """

    adc_paths: List[str]
    timestamps: np.ndarray   # (N,) float64
    name: str                # "azi" or "ele"
    chirps_per_frame: int    # 128 for azi, 8 for ele

    def __post_init__(self) -> None:
        if len(self.adc_paths) != len(self.timestamps):
            raise ValueError(
                f"radar {self.name}: {len(self.adc_paths)} paths vs "
                f"{len(self.timestamps)} timestamps"
            )

    def __len__(self) -> int:
        return len(self.adc_paths)

    @property
    def frame_shape(self) -> tuple[int, int, int, int]:
        return (NUM_ADC_SAMPLES, self.chirps_per_frame, NUM_RX, NUM_TX)

    @property
    def frame_bytes(self) -> int:
        return BYTES_PER_CHIRP * self.chirps_per_frame

    def load_adc(self, idx: int) -> np.ndarray:
        """Returns complex128 ADC tensor of shape (samples, chirps, rx, tx)."""
        path = self.adc_paths[idx]
        nbytes = os.path.getsize(path)
        if nbytes != self.frame_bytes:
            raise ValueError(
                f"ADC file {path!r} has {nbytes} bytes, expected "
                f"{self.frame_bytes} for {self.name} radar "
                f"(chirps_per_frame={self.chirps_per_frame})"
            )
        raw = np.fromfile(path, dtype=np.complex128)
        return raw.reshape(self.frame_shape)


@dataclass
class LidarStream:
    """Per-frame LiDAR point clouds, stored as (N, 4) float32 in Lidar_pcd/."""

    pcd_paths: List[str]
    timestamps: np.ndarray  # (N,) float64; from Lidar/timestamp.txt

    def __post_init__(self) -> None:
        if len(self.pcd_paths) != len(self.timestamps):
            raise ValueError(
                f"lidar: {len(self.pcd_paths)} pcds vs {len(self.timestamps)} "
                f"timestamps"
            )

    def __len__(self) -> int:
        return len(self.pcd_paths)

    def load_pcd(self, idx: int) -> np.ndarray:
        """Returns float32 array of shape (N, 4) with columns x, y, z, intensity."""
        path = self.pcd_paths[idx]
        arr = np.fromfile(path, dtype=np.float32)
        if arr.size % 4 != 0:
            raise ValueError(f"lidar pcd {path!r} size {arr.size} not divisible by 4")
        return arr.reshape(-1, 4)


@dataclass
class ZedStream:
    """ZED RGB images at sparse timestamps, plus the full dense pose stream.

    Two parallel arrays of timestamps are exposed:
      * image_timestamps: one entry per saved PNG, used when matching radar/lidar
        to the nearest image.
      * pose_stream.timestamps: the high-frequency pose updates the rig logged
        regardless of which frames were saved as PNG. Use this for accurate
        pose interpolation at arbitrary times.
    """

    image_paths: List[str]
    image_frame_ids: np.ndarray  # (M,) int64; the N in frame_N.png
    image_timestamps: np.ndarray  # (M,) float64; pose_stream.timestamps[image_frame_ids]
    pose_stream: PoseStream

    def __post_init__(self) -> None:
        if not (
            len(self.image_paths)
            == self.image_frame_ids.shape[0]
            == self.image_timestamps.shape[0]
        ):
            raise ValueError("ZED stream image arrays are not consistent in length")

    def __len__(self) -> int:
        return len(self.image_paths)

    def load_image(self, idx: int) -> np.ndarray:
        """Returns uint8 RGB array of shape (H, W, 3). Drops alpha if present."""
        # Defer the PIL import so the module is importable in environments
        # without PIL (e.g. radar-only test environments).
        from PIL import Image

        with Image.open(self.image_paths[idx]) as im:
            if im.mode != "RGB":
                im = im.convert("RGB")
            return np.asarray(im, dtype=np.uint8)


class Scene:
    """Multi-modal handle for a single RadarEyes capture.

    Construction is cheap: it scans directories, parses timestamp/pose files,
    and applies del_frame, but it does not load any frame data. Use the
    .load_*() methods on the per-modality stream attributes.
    """

    def __init__(self, scene_path: str, require_zed: bool = True) -> None:
        if not os.path.isdir(scene_path):
            raise FileNotFoundError(scene_path)
        self.path = os.path.abspath(scene_path)
        self.name = os.path.basename(self.path.rstrip("/"))

        zed_dir = os.path.join(self.path, "Camera_ZED")
        if require_zed and not os.path.isdir(zed_dir):
            raise FileNotFoundError(
                f"scene {self.name!r} has no Camera_ZED/; RadarSplat is ZED-only"
            )
        self.is_zed = os.path.isdir(zed_dir)

        self.zed: Optional[ZedStream] = self._build_zed(zed_dir) if self.is_zed else None
        self.radar_azi = self._build_radar(os.path.join(self.path, "1843_azi"), "azi")
        self.radar_ele = self._build_radar(os.path.join(self.path, "1843_ele"), "ele")
        self.lidar = self._build_lidar()

    # ------------------------------ stream builders ------------------------ #

    def _build_zed(self, zed_dir: str) -> Optional[ZedStream]:
        ts_path = os.path.join(zed_dir, "timestamp.txt")
        pose_path = os.path.join(zed_dir, "pose.txt")
        if not (os.path.isfile(ts_path) and os.path.isfile(pose_path)):
            return None

        all_ts = _parse_timestamps(ts_path)
        positions, quats = _parse_pose(pose_path)

        n = min(all_ts.shape[0], positions.shape[0], quats.shape[0])
        if n == 0:
            return None
        pose = PoseStream(positions[:n], quats[:n], all_ts[:n])

        image_paths = _list_frames_sorted(zed_dir, ".png")
        if not image_paths:
            return ZedStream(
                image_paths=[],
                image_frame_ids=np.zeros((0,), dtype=np.int64),
                image_timestamps=np.zeros((0,), dtype=np.float64),
                pose_stream=pose,
            )

        frame_ids = np.array(
            [_frame_index_from_name(p) for p in image_paths], dtype=np.int64
        )
        # Keep only images whose frame_id is within the pose stream length.
        valid = frame_ids < n
        if not valid.all():
            image_paths = [p for p, ok in zip(image_paths, valid) if ok]
            frame_ids = frame_ids[valid]
        image_timestamps = pose.timestamps[frame_ids]
        return ZedStream(image_paths, frame_ids, image_timestamps, pose)

    def _build_radar(self, radar_dir: str, name: str) -> Optional[RadarStream]:
        adc_dir = os.path.join(radar_dir, "ADC")
        ts_path = os.path.join(radar_dir, "timestamp.txt")
        del_path = os.path.join(radar_dir, "del_frame.txt")

        if not (os.path.isdir(adc_dir) and os.path.isfile(ts_path)):
            return None

        adc_paths = _list_frames_sorted(adc_dir, ".bin")
        timestamps = _parse_timestamps(ts_path).tolist()

        if not adc_paths or not timestamps:
            return None

        # Detect chirps_per_frame from the first ADC file size. 1843_azi is
        # 128 chirps (3072 KB), 1843_ele is 8 chirps (192 KB). Any other size
        # indicates an unexpected radar config and we raise.
        first_size = os.path.getsize(adc_paths[0])
        if first_size % BYTES_PER_CHIRP != 0:
            raise ValueError(
                f"radar {name}: first ADC file {adc_paths[0]!r} size "
                f"{first_size} not a multiple of {BYTES_PER_CHIRP} (one chirp)"
            )
        chirps_per_frame = first_size // BYTES_PER_CHIRP

        if os.path.isfile(del_path):
            with open(del_path) as f:
                del_indices = [int(s.strip()) for s in f if s.strip()]
            adc_paths, timestamps = _apply_del_frame(adc_paths, timestamps, del_indices)

        # Truncate to common length.
        n = min(len(adc_paths), len(timestamps))
        return RadarStream(
            adc_paths=list(adc_paths[:n]),
            timestamps=np.asarray(timestamps[:n], dtype=np.float64),
            name=name,
            chirps_per_frame=chirps_per_frame,
        )

    def _build_lidar(self) -> Optional[LidarStream]:
        pcd_dir = os.path.join(self.path, "Lidar_pcd")
        ts_path = os.path.join(self.path, "Lidar", "timestamp.txt")
        if not (os.path.isdir(pcd_dir) and os.path.isfile(ts_path)):
            return None

        pcd_paths = _list_frames_sorted(pcd_dir, ".bin")
        timestamps = _parse_timestamps(ts_path)
        if not pcd_paths or timestamps.shape[0] == 0:
            return None

        # The two streams sometimes differ by 1-2 entries due to capture-side
        # buffering quirks. Match positionally and truncate to the shorter side.
        n = min(len(pcd_paths), timestamps.shape[0])
        return LidarStream(
            pcd_paths=list(pcd_paths[:n]),
            timestamps=timestamps[:n].astype(np.float64),
        )

    # ------------------------------- summary ------------------------------- #

    def summary(self) -> dict:
        return {
            "name": self.name,
            "is_zed": self.is_zed,
            "n_zed_images": len(self.zed) if self.zed is not None else 0,
            "n_pose_samples": len(self.zed.pose_stream) if self.zed is not None else 0,
            "n_radar_azi": len(self.radar_azi) if self.radar_azi is not None else 0,
            "n_radar_ele": len(self.radar_ele) if self.radar_ele is not None else 0,
            "n_lidar": len(self.lidar) if self.lidar is not None else 0,
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"Scene({s['name']!r}: ZED={s['n_zed_images']}img/"
            f"{s['n_pose_samples']}pose, azi={s['n_radar_azi']}, "
            f"ele={s['n_radar_ele']}, lidar={s['n_lidar']})"
        )


def _apply_del_frame(
    paths: Sequence[str], timestamps: Sequence[float], del_indices: Sequence[int]
) -> tuple[List[str], List[float]]:
    """Delete the given positional indices from both lists, matching the legacy
    semantics in IPLab_mmwavePCD.ParseData.make_dataset (after each deletion
    the surviving list's indices shift left by one)."""
    paths = list(paths)
    timestamps = list(timestamps)
    deleted_so_far = 0
    for raw_idx in sorted(del_indices):
        adj = raw_idx - deleted_so_far
        if 0 <= adj < len(paths):
            del paths[adj]
            del timestamps[adj]
            deleted_so_far += 1
    return paths, timestamps
