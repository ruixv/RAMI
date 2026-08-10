"""Build a 3DGS INITIALIZATION point cloud from radar-anchored foundation depth.

Stage-3 of the radar-positive search (2026-06-10). Lesson from Stage-2: the
anchored DA-V2 depth prior used as a LOSS helps/hurts unpredictably (benefit is
set by optimization dynamics, not geometric fidelity — F2room MAE 1.38m hurt
−0.91, n0630c MAE 3.82m helped +0.38; no geometry-judging gate can work).
INIT is the opposite kind of lever: it seeds the optimizer's basin without
fighting the photometric gradient afterwards, and the project's own diagnosis
says optimization dynamics (densification) is THE driver while the largest
single measured radar term was the (sparse) lira INIT (+0.44 dB). This builds
the dense version: unproject per-view radar-anchored DA-V2 metric depth from
TRAIN views only (no test leakage), fuse, voxel-downsample.

Usage:
  python tools/gen_depth_init_cloud.py --tag dark401a \
      [--prior-dir outputs/anchored_depth/dark401a_radarglobal] \
      [--stride-views 3] [--stride-px 8] [--voxel 0.05]
Output: outputs/anchored_depth/<tag>_initcloud.npy  (N,3 float32 world)
"""
from __future__ import annotations
import argparse, json, os, sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from data.radareyes.sync import interpolate_pose
from models.baselines.dn_splatter import _resolve_scene_from_splits
from tools.radar_arbiter_probe import _unproject, _verify_roundtrip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prior-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stride-views", type=int, default=3)
    ap.add_argument("--stride-px", type=int, default=8)
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--dmin", type=float, default=0.3)
    ap.add_argument("--dmax", type=float, default=20.0)
    ap.add_argument("--scale-factor", type=float, default=1.0,
                    help="Corrupt the metric scale (depth *= factor) — no-radar control.")
    ap.add_argument("--pose-noise-trans", type=float, default=0.0,
                    help="Per-view i.i.d. Gaussian translation noise (m) added to "
                         "the poses used to BUILD the seed cloud (odometry-drift robustness).")
    ap.add_argument("--pose-noise-rot", type=float, default=0.0,
                    help="Per-view i.i.d. Gaussian rotation noise (deg).")
    ap.add_argument("--pose-noise-seed", type=int, default=0)
    args = ap.parse_args()
    _png = np.random.RandomState(args.pose_noise_seed)

    def _perturb(zp, zq):
        if args.pose_noise_trans <= 0 and args.pose_noise_rot <= 0:
            return zp, zq
        zp = np.asarray(zp, np.float64) + _png.normal(0, args.pose_noise_trans, 3)
        if args.pose_noise_rot > 0:
            ax = _png.normal(0, 1, 3); ax /= (np.linalg.norm(ax) + 1e-9)
            th = np.deg2rad(_png.normal(0, args.pose_noise_rot)) / 2.0
            dq = np.array([ax[0]*np.sin(th), ax[1]*np.sin(th), ax[2]*np.sin(th), np.cos(th)])
            x0,y0,z0,w0 = zq; x1,y1,z1,w1 = dq   # Hamilton product zq*dq (qx,qy,qz,qw)
            zq = np.array([
                w0*x1 + x0*w1 + y0*z1 - z0*y1,
                w0*y1 - x0*z1 + y0*w1 + z0*x1,
                w0*z1 + x0*y1 - y0*x1 + z0*w1,
                w0*w1 - x0*x1 - y0*y1 - z0*z1])
        return zp, zq
    prior_dir = args.prior_dir or f"outputs/anchored_depth/{args.tag}_radarglobal"
    out = args.out or f"outputs/anchored_depth/{args.tag}_initcloud.npy"

    ed = f"outputs/canonical_gt/{args.tag}"
    splits = json.load(open(os.path.join(ed, "splits.json")))
    scene = _resolve_scene_from_splits(ed)
    intr = splits["intrinsics"]
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]],
                  [0, 0, 1]], np.float64)
    W, H = splits["image_size"]

    recs = [r for r in splits["records"]
            if r.get("split", "train") != "test"
            and os.path.exists(os.path.join(prior_dir, r["filename"] + ".npz"))]
    recs = recs[:: max(1, args.stride_views)]
    if not recs:
        sys.exit(f"no train records with priors in {prior_dir}")

    pts_all = []
    for ri, rec in enumerate(recs):
        t_img = float(rec["t_image"])
        zp, zq = interpolate_pose(scene.zed.pose_stream, t_img)
        zp, zq = _perturb(zp, zq)
        if ri == 0 and args.pose_noise_trans == 0 and args.pose_noise_rot == 0:
            _verify_roundtrip(K, zp, zq, W, H)
        dep = np.load(os.path.join(prior_dir, rec["filename"] + ".npz"))["depth"].astype(np.float64)
        if args.scale_factor != 1.0:
            dep = dep * args.scale_factor
        vv, uu = np.mgrid[0:H:args.stride_px, 0:W:args.stride_px]
        d = dep[vv, uu]
        ok = np.isfinite(d) & (d > args.dmin) & (d < args.dmax)
        if not ok.any():
            continue
        pw = _unproject(uu[ok].astype(np.float64), vv[ok].astype(np.float64),
                        d[ok], K, zp, zq)
        pts_all.append(pw.astype(np.float32))
    pts = np.concatenate(pts_all, 0)

    # voxel downsample (keep one point per voxel, mean position)
    vox = np.floor(pts / args.voxel).astype(np.int64)
    key = vox[:, 0] * 73856093 ^ vox[:, 1] * 19349663 ^ vox[:, 2] * 83492791
    order = np.argsort(key)
    key_s, pts_s = key[order], pts[order]
    uniq, start = np.unique(key_s, return_index=True)
    sums = np.add.reduceat(pts_s.astype(np.float64), start, axis=0)
    counts = np.diff(np.append(start, len(key_s)))[:, None]
    fused = (sums / counts).astype(np.float32)

    np.save(out, fused)
    print(f"[{args.tag}] {len(recs)} views -> {pts.shape[0]} raw -> "
          f"{fused.shape[0]} voxel({args.voxel}m) points -> {out}")


if __name__ == "__main__":
    main()
