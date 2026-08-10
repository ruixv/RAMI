"""Re-score a baseline run's renders against the CLEAN α=1.0 GT, regardless
of what darkness level the run was trained on.

This is the metric that matters for the paper's headline claim: under
low-light training, does the model recover the underlying clean scene?

If the run was trained on α=0.1 (dark, noisy) inputs, the model sees the
scene through that corruption. The standard metric (PSNR of render vs α=0.1
test image) measures whether the model fit the corruption; we instead want
PSNR of render vs **clean** (α=1.0) test image.

Usage:
    python tools/score_vs_clean.py \\
        --run-dir outputs/b1_alpha0.1_B408 \\
        --clean-gt-dir outputs/b1_v4_B408/colmap_export_a1/images

Writes metrics_vs_clean.json next to the run's metrics.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from eval.render_and_score import compute_image_pair  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--clean-gt-dir", required=True,
                   help="Directory holding the clean (α=1.0) GT PNGs, "
                        "typically <other-run>/colmap_export_a1/images/")
    p.add_argument("--out", default=None,
                   help="Defaults to <run>/metrics_vs_clean.json")
    args = p.parse_args()

    summary_path = os.path.join(args.run_dir, "renders_summary.json")
    if not os.path.isfile(summary_path):
        print(f"error: {summary_path} missing", file=sys.stderr)
        return 2
    with open(summary_path) as f:
        summary = json.load(f)

    test_names = summary["test_image_names"]
    renders_dir = summary["renders_dir"]
    print(f"Scoring {len(test_names)} renders vs CLEAN GT at {args.clean_gt_dir}")

    rows = []
    for name in test_names:
        pred = os.path.join(renders_dir, name)
        gt = os.path.join(args.clean_gt_dir, name)
        if not (os.path.isfile(pred) and os.path.isfile(gt)):
            continue
        m = compute_image_pair(pred, gt, with_lpips=True)
        rows.append(m)

    if not rows:
        print("error: no matching renders/GT found", file=sys.stderr)
        return 2

    psnrs = [r.psnr for r in rows if np.isfinite(r.psnr)]
    ssims = [r.ssim for r in rows]
    lpips = [r.lpips for r in rows if np.isfinite(r.lpips)]
    out = {
        "run_dir": args.run_dir,
        "clean_gt_dir": args.clean_gt_dir,
        "n_views": len(rows),
        "mean": {
            "psnr": float(np.mean(psnrs)) if psnrs else float("nan"),
            "ssim": float(np.mean(ssims)),
            "lpips": float(np.mean(lpips)) if lpips else float("nan"),
        },
        "per_view": [
            {"filename": r.filename, "psnr": r.psnr, "ssim": r.ssim, "lpips": r.lpips}
            for r in rows
        ],
    }
    out_path = args.out or os.path.join(args.run_dir, "metrics_vs_clean.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")
    print(f"=== vs CLEAN GT: PSNR={out['mean']['psnr']:.2f}  "
          f"SSIM={out['mean']['ssim']:.4f}  LPIPS={out['mean']['lpips']:.4f} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
