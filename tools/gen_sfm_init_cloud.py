"""Known-pose feature-triangulation init cloud — the SfM control (2026-06-11).

Data-reality correction: the benchmark scenes are LIT indoor captures (night
timestamps, lights on). The decisive missing control for the radar-init claim
is therefore: can IMAGE FEATURES alone provide the init cloud under the same
known (LiDAR-odometry) poses? This isolates the init-cloud question from pose
estimation (fairer to vision than full COLMAP, whose pose stage could also
fail; poses are shared across all arms anyway).

Pipeline: SIFT detect+match (ratio test) on consecutive train-view pairs at
the same view stride as the radar cloud -> triangulate with known poses ->
cheirality + reprojection(<2px) + depth-range filters -> voxel downsample.

Output: outputs/anchored_depth/<tag>_initcloud_sfm.npy + match stats.
Usage: python tools/gen_sfm_init_cloud.py --tag n0630c
"""
from __future__ import annotations
import argparse, json, os, sys

import cv2
import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from data.radareyes.sync import interpolate_pose
from models.baselines.dn_splatter import _resolve_scene_from_splits
from tools.radar_arbiter_probe import _w2c_from_pose


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
    ap.add_argument("--stride-views", type=int, default=3)
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--reproj-px", type=float, default=2.0)
    ap.add_argument("--dmin", type=float, default=0.3)
    ap.add_argument("--dmax", type=float, default=20.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"outputs/anchored_depth/{args.tag}_initcloud_sfm.npy"

    ed = f"outputs/canonical_gt/{args.tag}"
    splits = json.load(open(os.path.join(ed, "splits.json")))
    scene = _resolve_scene_from_splits(ed)
    intr = splits["intrinsics"]
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]],
                  [0, 0, 1]], np.float64)
    recs = [r for r in splits["records"] if r.get("split", "train") != "test"]
    recs = recs[:: max(1, args.stride_views)]

    sift = cv2.SIFT_create()
    matcher = cv2.BFMatcher()
    pts_all, n_pairs, n_matches_tot = [], 0, 0
    per_pair_kept = []
    prev = None  # (gray, kps, des, w2c)
    for rec in recs:
        img = cv2.imread(os.path.join(ed, "images", rec["filename"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        zp, zq = interpolate_pose(scene.zed.pose_stream, float(rec["t_image"]))
        w2c = _w2c_from_pose(zp, zq)
        kps, des = sift.detectAndCompute(img, None)
        cur = (kps, des, w2c)
        if prev is not None and des is not None and prev[1] is not None \
                and len(kps) > 10 and len(prev[0]) > 10:
            n_pairs += 1
            raw = matcher.knnMatch(prev[1], des, k=2)
            good = [m for m, n in (p for p in raw if len(p) == 2)
                    if m.distance < args.ratio * n.distance]
            n_matches_tot += len(good)
            if len(good) >= 8:
                p1 = np.float64([prev[0][m.queryIdx].pt for m in good]).T  # (2,N)
                p2 = np.float64([kps[m.trainIdx].pt for m in good]).T
                P1 = K @ prev[2][:3, :]
                P2 = K @ w2c[:3, :]
                X = cv2.triangulatePoints(P1, P2, p1, p2)
                X = (X[:3] / np.clip(X[3], 1e-12, None)).T               # (N,3) world
                ok = np.ones(len(X), bool)
                for P, pp in ((P1, p1), (P2, p2)):
                    Xc = (P[:, :3] @ X.T + P[:, 3:4])                    # (3,N) image-space
                    z = Xc[2]
                    ok &= (z > args.dmin) & (z < args.dmax)
                    proj = Xc[:2] / np.clip(z, 1e-12, None)
                    ok &= (np.abs(proj - pp) < args.reproj_px).all(axis=0)
                per_pair_kept.append(int(ok.sum()))
                if ok.any():
                    pts_all.append(X[ok].astype(np.float32))
        prev = cur

    if not pts_all:
        print(f"[{args.tag}] SFM-INIT FAILED: 0 triangulated points "
              f"({n_pairs} pairs, {n_matches_tot} raw matches)")
        sys.exit(3)
    pts = np.concatenate(pts_all, 0)
    fused = _voxel_down(pts, args.voxel)
    np.save(out, fused)
    kept = np.array(per_pair_kept) if per_pair_kept else np.zeros(1)
    print(f"[{args.tag}] sfm cloud: {n_pairs} pairs, "
          f"{n_matches_tot/max(1,n_pairs):.0f} matches/pair, "
          f"kept/pair med {np.median(kept):.0f}, raw {pts.shape[0]} -> "
          f"voxel {fused.shape[0]} pts -> {out}")


if __name__ == "__main__":
    main()
