"""Build LiDAR-raw and LiDAR+radar-fused 3DGS init clouds (Stage-7, 2026-06-10).

Target ranking experiment (user-requested):
  #3  RGB+LiDAR(raw cloud init)  >  RGB+Radar(DA-V2+radar-scale init)
  #2  RGB+LiDAR+Radar(fused init) >  RGB+LiDAR(raw cloud init)
Fusion mechanism: LiDAR cloud is accurate but covers only its hit regions
(~22% of pixels); the radar-anchored DA-V2 cloud covers the full camera
frustum. Fused init = LiDAR cloud + DA-V2 points that lie in LiDAR-BLIND
space (NN-distance > blind_dist from any LiDAR point) — radar fills the blind
regions at init granularity, where its coarse accuracy suffices.

Outputs:
  outputs/anchored_depth/<tag>_initcloud_lidarraw.npy
  outputs/anchored_depth/<tag>_initcloud_fuse.npy

Usage: python tools/gen_lidar_init_cloud.py --tag dark401a
"""
from __future__ import annotations
import argparse, json, os, sys

import numpy as np
from scipy.spatial import cKDTree

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from data.radareyes.sync import interpolate_pose
from data.radareyes.transforms import SENSOR_OFFSETS, sensor_to_world
from models.baselines.dn_splatter import _resolve_scene_from_splits


def _voxel_down(pts, voxel):
    vox = np.floor(pts / voxel).astype(np.int64)
    key = vox[:, 0] * 73856093 ^ vox[:, 1] * 19349663 ^ vox[:, 2] * 83492791
    order = np.argsort(key)
    key_s, pts_s = key[order], pts[order]
    _, start = np.unique(key_s, return_index=True)
    sums = np.add.reduceat(pts_s.astype(np.float64), start, axis=0)
    counts = np.diff(np.append(start, len(key_s)))[:, None]
    return (sums / counts).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-lidar-frames", type=int, default=60)
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--rmin", type=float, default=0.3)
    ap.add_argument("--rmax", type=float, default=20.0)
    ap.add_argument("--blind-dist", type=float, default=0.6,
                    help="DA-V2 points farther than this from any LiDAR point = blind fill")
    ap.add_argument("--strict-window", type=float, default=None,
                    help="Strict NVS protocol: keep only sweeps within +/-W s of a "
                         "TRAIN-view timestamp AND farther than W s from every "
                         "TEST-view timestamp; output gets the _strict suffix.")
    args = ap.parse_args()
    AD = "outputs/anchored_depth"
    ed = f"outputs/canonical_gt/{args.tag}"
    splits = json.load(open(os.path.join(ed, "splits.json")))
    scene = _resolve_scene_from_splits(ed)
    if scene.lidar is None:
        sys.exit(f"{args.tag}: no lidar stream")

    N = len(scene.lidar.timestamps)
    idxs = np.unique(np.linspace(0, N - 1, args.n_lidar_frames).astype(int))
    if args.strict_window is not None:
        # strict NVS protocol: sweep must be near a train view and away from
        # every test view (both windows = args.strict_window seconds)
        Wt = float(args.strict_window)
        recs = splits["records"]
        t_train = np.array([r["t_image"] for r in recs
                            if r.get("split", "train") != "test"], float)
        t_test = np.array([r["t_image"] for r in recs
                           if r.get("split", "train") == "test"], float)
        ts = np.asarray(scene.lidar.timestamps, float)
        near_train = np.abs(ts[:, None] - t_train[None, :]).min(1) <= Wt
        far_test = (np.abs(ts[:, None] - t_test[None, :]).min(1) > Wt
                    if len(t_test) else np.ones_like(near_train))
        allowed = np.where(near_train & far_test)[0]
        if len(allowed) == 0:
            sys.exit(f"{args.tag}: strict window {Wt}s leaves no sweeps")
        sel = np.unique(np.linspace(0, len(allowed) - 1,
                                    min(args.n_lidar_frames, len(allowed))).astype(int))
        idxs = allowed[sel]
        print(f"[strict] {args.tag}: {len(allowed)}/{N} sweeps allowed "
              f"(W={Wt}s), using {len(idxs)}")
    pts = []
    for j in idxs:
        pcd = scene.lidar.load_pcd(int(j))[:, :3]
        r = np.linalg.norm(pcd, axis=1)
        pcd = pcd[(r > args.rmin) & (r < args.rmax)]
        if pcd.shape[0] == 0:
            continue
        t = float(scene.lidar.timestamps[int(j)])
        pp, pq = interpolate_pose(scene.zed.pose_stream, t)
        pts.append(sensor_to_world(pcd, pp, pq, SENSOR_OFFSETS["lidar"])[:, :3])
    lidar = _voxel_down(np.concatenate(pts, 0).astype(np.float32), args.voxel)
    sfx = "_strict" if args.strict_window is not None else ""
    out_raw = os.path.join(AD, f"{args.tag}_initcloud_lidarraw{sfx}.npy")
    np.save(out_raw, lidar)

    dav2_path = os.path.join(AD, f"{args.tag}_initcloud.npy")
    dav2 = np.load(dav2_path)
    d, _ = cKDTree(lidar).query(dav2, k=1, workers=8)
    fill = dav2[d > args.blind_dist]
    fused = _voxel_down(np.concatenate([lidar, fill], 0), args.voxel)
    out_fuse = os.path.join(AD, f"{args.tag}_initcloud_fuse{sfx}.npy")
    np.save(out_fuse, fused)
    print(f"[{args.tag}] lidar {lidar.shape[0]} pts -> {out_raw}; "
          f"dav2 {dav2.shape[0]}, blind-fill {fill.shape[0]} "
          f"({fill.shape[0]/max(1,dav2.shape[0]):.0%} of dav2) -> fused {fused.shape[0]} -> {out_fuse}")


if __name__ == "__main__":
    main()
