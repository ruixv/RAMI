#!/usr/bin/env python3
"""Export canonical clean GT (alpha=1.0) for a scene without training.

Replacement for tools/gen_canonical_gt.sh (which launched a throwaway MCMC
run just for its export side effect, and had a set-e/ls race that left
orphan trainings). Calls data.colmap_export.export_scene directly; no GPU.

Usage: python tools/export_canonical_gt.py --tag x30a
Output: outputs/canonical_gt/<tag>/{images,splits.json,test_image_names.txt}
"""
import argparse, os, shutil, sys, tempfile

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, _REPO)

from data.colmap_export import ExportConfig, export_scene  # noqa: E402
from data.radareyes import Scene  # noqa: E402
from tools.scene_registry import SCENES  # noqa: E402

ROOT = os.environ.get("RAMI_DATA_ROOT", "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dest", default=None)
    args = ap.parse_args()
    dest = args.dest or f"outputs/canonical_gt/{args.tag}"
    if os.path.isdir(os.path.join(dest, "images")) and \
            len(os.listdir(os.path.join(dest, "images"))) > 50:
        print(f"[{args.tag}] canonical_gt already exists, skip")
        return
    scene = Scene(os.path.join(ROOT, SCENES[args.tag]))
    tmp = tempfile.mkdtemp(prefix=f"gt_{args.tag}_", dir="/tmp")
    try:
        ec = ExportConfig(alpha=1.0)  # defaults: seed 20260523, test_every_nth 8
        splits = export_scene(scene, tmp, ec, anchor_positions=None)
        os.makedirs(dest, exist_ok=True)
        for item in ("images", "splits.json", "test_image_names.txt"):
            src = os.path.join(tmp, item)
            dst = os.path.join(dest, item)
            if not os.path.exists(src):
                continue
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        print(f"[{args.tag}] {splits['n_train']} train / {splits['n_test']} test "
              f"-> {dest}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
