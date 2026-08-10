"""Phase 2 baseline training entrypoint.

Usage:
    python training/baseline_train.py \\
        --scene 2023_06_04_08_07_29_B408_60s_z1 \\
        --baseline scaffold_gs \\
        --alpha 1.0 \\
        --out outputs/scaffold_gs_B408_a1

What this does today:
    1. Loads the named scene.
    2. Builds LIRA anchors (Phase 1 pipeline) at the median timestamp.
    3. Calls `data.colmap_export.export_scene` to write a COLMAP-format
       directory with the anchors as `points3D.txt` and the ZED frames
       (optionally α-corrupted).
    4. Calls `get_baseline(key)().train(...)`.

The baseline's `.train()` raises NotImplementedError until the gsplat env
is unblocked (docs/phase2_env.md). Use this driver as the entrypoint that
will Just Work once the env builds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.colmap_export import ExportConfig, export_scene  # noqa: E402
from data.radareyes import Scene  # noqa: E402
from models.baselines import get_baseline, list_baselines  # noqa: E402
from models.radar.lira import LIRAConfig, build_lira_anchors  # noqa: E402

_DEFAULT_DATASET_ROOT = os.environ.get("RAMI_DATA_ROOT", "data")


def resolve_scene_path(arg: str) -> str:
    if os.path.isdir(arg):
        return arg
    cand = os.path.join(_DEFAULT_DATASET_ROOT, arg)
    if os.path.isdir(cand):
        return cand
    raise FileNotFoundError(f"cannot resolve scene {arg!r}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True, help="Scene directory name or absolute path.")
    p.add_argument(
        "--baseline", required=True,
        help=f"Baseline key. Available: {[m.key for m in list_baselines()]}",
    )
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Darkness level (1.0 = clean).")
    p.add_argument("--out", required=True, help="Output run directory.")
    p.add_argument("--keep-every-nth", type=int, default=None,
                   help="Subsample ZED frames for fast iteration.")
    p.add_argument("--test-every-nth", type=int, default=8)
    p.add_argument("--lira-voxel", type=float, default=0.05)
    p.add_argument("--lira-window", type=float, default=2.0)
    p.add_argument("--lira-pfa", type=float, default=1e-2)
    p.add_argument("--init", choices=["lira", "random", "lira_oversample", "pointfile"], default="lira",
                   help="Point cloud init: lira (CFAR anchors), random (uniform in scene bbox), "
                        "lira_oversample (LIRA + per-anchor jittered duplicates), "
                        "pointfile (load (N,3) world points from --init-point-file)")
    p.add_argument("--init-n-random", type=int, default=10000,
                   help="Number of random anchors when --init random.")
    p.add_argument("--init-point-file", type=str, default=None,
                   help=".npy (N,3) world-frame init points for --init pointfile.")
    p.add_argument("--init-point-n", type=int, default=0,
                   help="If >0, random-subsample the pointfile cloud to this many points.")
    p.add_argument("--init-oversample-per-anchor", type=int, default=25,
                   help="Jittered duplicates per LIRA anchor when --init lira_oversample.")
    p.add_argument("--config", type=str, default=None,
                   help="Optional JSON file overriding baseline-specific hyperparams.")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    scene_path = resolve_scene_path(args.scene)
    scene = Scene(scene_path)
    print(scene)

    # 1. Build point-cloud init for 3DGS. Three options:
    #    - lira: CFAR + multi-frame accumulation around scene's median radar
    #      timestamp (the RadarSplat method, sparse ~400 anchors).
    #    - random: uniform sample of n points inside the scene's camera-pose
    #      bounding box. Pure control experiment to isolate "is sparse LIRA
    #      init the bottleneck?".
    #    - lira_oversample: LIRA anchors + k jittered duplicates each, to
    #      give 3DGS more seeds while preserving the radar-anchor spatial
    #      distribution.
    init_positions = None
    use_random = False
    if args.init == "lira" or args.init == "lira_oversample":
        t_target = float(np.median(scene.radar_azi.timestamps))
        lira_cfg = LIRAConfig(
            voxel_size_m=args.lira_voxel,
            accumulation_window_s=args.lira_window,
            cfar_pfa=args.lira_pfa,
        )
        print(f"Building LIRA at t={t_target:.3f} with {lira_cfg}")
        t0 = time.time()
        anchors = build_lira_anchors(scene, t_target, lira_cfg)
        print(f"LIRA: {len(anchors)} anchors in {time.time() - t0:.1f}s")
        init_positions = anchors.positions
        if args.init == "lira_oversample" and len(anchors) > 0:
            rng = np.random.default_rng(20260523)
            k = max(1, args.init_oversample_per_anchor)
            scale = max(args.lira_voxel * 1.5, 0.05)
            dup = np.repeat(anchors.positions, k, axis=0)
            jitter = rng.normal(scale=scale, size=dup.shape)
            init_positions = dup + jitter
            print(f"LIRA oversampled: {init_positions.shape[0]} points "
                  f"(k={k} per anchor, jitter σ={scale:.3f} m)")
    elif args.init == "random":
        use_random = True
    elif args.init == "pointfile":
        if not args.init_point_file:
            raise SystemExit("--init pointfile requires --init-point-file")
        init_positions = np.load(args.init_point_file).astype(np.float64)
        if args.init_point_n > 0 and init_positions.shape[0] > args.init_point_n:
            rng = np.random.default_rng(20260610)
            sel = rng.choice(init_positions.shape[0], args.init_point_n, replace=False)
            init_positions = init_positions[sel]
        print(f"pointfile init: {init_positions.shape[0]} points from {args.init_point_file}")

    # 2. Export COLMAP-format dataset.
    export_dir = os.path.join(args.out, f"colmap_export_a{args.alpha:g}")
    print(f"Exporting COLMAP dataset → {export_dir} (init={args.init})")
    ec = ExportConfig(
        alpha=args.alpha,
        test_every_nth=args.test_every_nth,
        keep_every_nth=args.keep_every_nth,
        use_random_anchors=use_random,
        random_anchors_n=args.init_n_random,
    )
    t0 = time.time()
    splits = export_scene(scene, export_dir, ec, anchor_positions=init_positions)
    print(f"Export: {splits['n_train']} train / {splits['n_test']} test "
          f"images, {splits['n_points3D']} points3D, in {time.time() - t0:.1f}s")

    # 3. Hand off to baseline.
    config = {}
    if args.config and os.path.isfile(args.config):
        with open(args.config) as f:
            config = json.load(f)
    baseline_cls = get_baseline(args.baseline)
    baseline = baseline_cls()
    print(f"Training baseline {baseline.meta.key} ({baseline.meta.venue}) on {export_dir}")
    train_out = os.path.join(args.out, "checkpoint")

    try:
        result = baseline.train(export_dir, train_out, config)
        print(f"Training done: {result.n_iters} iters in {result.seconds_elapsed:.1f}s "
              f"→ {result.checkpoint_dir}")
    except NotImplementedError as e:
        print(f"\n[stub] {e}")
        print("\nExport completed and is ready for an external trainer once env is fixed.")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
