"""CLI to render a trained baseline's checkpoint at the held-out test views.

After `training/baseline_train.py` finishes, this script reads the splits.json
and the baseline checkpoint, renders the test images, and saves them under
<out>/renders/ so eval/render_and_score can grade them.

Usage:
    python eval/render_baseline.py \\
        --run-dir outputs/vanilla_3dgs_B408_a1 \\
        --baseline vanilla_3dgs
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

from PIL import Image  # noqa: E402

from models.baselines import get_baseline  # noqa: E402
from models.baselines._gsplat_common import (  # noqa: E402
    K_from_camera,
    colmap_image_to_w2c,
    parse_colmap_export,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True,
                   help="Directory containing colmap_export_* + checkpoint/")
    p.add_argument("--baseline", required=True)
    p.add_argument("--alpha", type=float, default=None,
                   help="Pick the colmap_export_a<alpha> subdir. Auto-detect if omitted.")
    p.add_argument("--checkpoint-dir", default=None,
                   help="Defaults to <run-dir>/checkpoint/checkpoints")
    args = p.parse_args()

    run_dir = args.run_dir
    # Find the export dir.
    if args.alpha is not None:
        export_dir = os.path.join(run_dir, f"colmap_export_a{args.alpha:g}")
    else:
        candidates = [d for d in os.listdir(run_dir)
                      if d.startswith("colmap_export_a") and os.path.isdir(os.path.join(run_dir, d))]
        if not candidates:
            print(f"error: no colmap_export_a* under {run_dir}", file=sys.stderr)
            return 2
        export_dir = os.path.join(run_dir, candidates[0])
    if not os.path.isdir(export_dir):
        print(f"error: export dir not found: {export_dir}", file=sys.stderr)
        return 2

    ckpt_dir = args.checkpoint_dir or os.path.join(run_dir, "checkpoint", "checkpoints")
    if not os.path.isdir(ckpt_dir):
        print(f"error: checkpoint dir not found: {ckpt_dir}", file=sys.stderr)
        return 2

    data = parse_colmap_export(export_dir)
    test_set = set(data.test_image_names)
    test_records = [im for im in data.images.values() if im.name in test_set]
    if not test_records:
        print(f"error: no test images registered for {export_dir}", file=sys.stderr)
        return 2

    cam = next(iter(data.cameras.values()))
    K = K_from_camera(cam)
    w2cs = np.stack([colmap_image_to_w2c(im) for im in test_records], axis=0)

    print(f"Rendering {len(test_records)} test views via baseline {args.baseline!r}...")
    baseline = get_baseline(args.baseline)()
    t0 = time.time()
    renders = baseline.render(ckpt_dir, w2cs, K, cam.width, cam.height)
    elapsed = time.time() - t0
    print(f"Rendered {renders.shape[0]} views in {elapsed:.1f}s ({elapsed / max(renders.shape[0], 1):.2f}s/view)")

    out_dir = os.path.join(run_dir, "renders")
    os.makedirs(out_dir, exist_ok=True)
    for rec, img in zip(test_records, renders):
        u8 = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(u8).save(os.path.join(out_dir, rec.name))
    print(f"Saved renders to {out_dir}")

    summary = {
        "run_dir": run_dir,
        "baseline": args.baseline,
        "n_test": len(test_records),
        "render_seconds_total": elapsed,
        "renders_dir": out_dir,
        "gt_dir": os.path.join(export_dir, "images"),
        "test_image_names": data.test_image_names,
    }
    with open(os.path.join(run_dir, "renders_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
