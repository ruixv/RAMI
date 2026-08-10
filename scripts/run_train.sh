#!/usr/bin/env bash
# Train one arm on one scene with the benchmark config (3DGS, 30k iters,
# light densification, 5k init points, alpha=1).
#
# Usage: scripts/run_train.sh <scene_tag> <arm> [gpu_id]
#   arm: rami   (RAMI seed cloud, strict fit)
#        sfm    (SfM triangulated cloud)
#        random (5k Gaussian-random points, sigma 1 m)
# Output: outputs/runs/<tag>_<arm>/
set -euo pipefail
TAG="${1:?usage: run_train.sh <scene_tag> <arm> [gpu_id]}"
ARM="${2:?arm: rami|sfm|random}"
GPU="${3:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

SCENE=$(python -c "from tools.scene_registry import SCENES; print(SCENES['$TAG'])")
OUT="outputs/runs/${TAG}_${ARM}"
CFG="${RAMI_TRAIN_CFG:-configs/train_strict_30k.json}"

case "$ARM" in
  rami)   INIT=(--init pointfile --init-point-file "outputs/anchored_depth/${TAG}_initcloud_trainfit.npy" --init-point-n 5000) ;;
  sfm)    INIT=(--init pointfile --init-point-file "outputs/anchored_depth/${TAG}_initcloud_sfm.npy" --init-point-n 5000) ;;
  random) INIT=(--init random --init-n-random 5000) ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac

python training/baseline_train.py --scene "$SCENE" --baseline radarsplat_v1 \
    --alpha 1.0 "${INIT[@]}" --out "$OUT" --config "$CFG"
echo "[train] $TAG/$ARM done -> $OUT"
