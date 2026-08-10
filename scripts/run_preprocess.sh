#!/usr/bin/env bash
# Preprocessing for one scene: canonical GT export -> radar-anchored depth
# (DA-V2 + CFAR + train-frames-only weighted least-squares scale-shift fit)
# -> 5k RAMI seed cloud. Also builds the SfM and random control clouds.
#
# Usage: scripts/run_preprocess.sh <scene_tag> [gpu_id]
# Env:   RAMI_DATA_ROOT  root containing the raw capture directories (default ./data)
# Output: outputs/canonical_gt/<tag>/, outputs/anchored_depth/<tag>_initcloud_trainfit.npy,
#         outputs/anchored_depth/<tag>_initcloud_sfm.npy
set -euo pipefail
TAG="${1:?usage: run_preprocess.sh <scene_tag> [gpu_id]}"
GPU="${2:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

# 1. Canonical clean-GT export (CPU; images + deterministic train/test split)
python tools/export_canonical_gt.py --tag "$TAG"

# 2. Monocular depth + radar anchors + strict (train-frames-only) fit
python tools/gen_radar_anchored_depth.py --tag "$TAG" --mode global --st-modes trainfit

# 3. Back-project + fuse into the RAMI seed cloud
python tools/gen_depth_init_cloud.py --tag "$TAG" \
    --prior-dir "outputs/anchored_depth/${TAG}_trainfit" \
    --out "outputs/anchored_depth/${TAG}_initcloud_trainfit.npy"

# 4. SfM control cloud (known-pose SIFT triangulation, CPU)
python tools/gen_sfm_init_cloud.py --tag "$TAG"

echo "[preprocess] $TAG done."
