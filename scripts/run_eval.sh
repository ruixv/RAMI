#!/usr/bin/env bash
# Render the held-out views of a finished run and score them against the
# canonical clean GT, then print the aggregate table over all finished runs.
#
# Usage: scripts/run_eval.sh <scene_tag> <arm> [gpu_id]
# Output: outputs/runs/<tag>_<arm>/metrics_vs_clean.json + aggregate table
set -euo pipefail
TAG="${1:?usage: run_eval.sh <scene_tag> <arm> [gpu_id]}"
ARM="${2:?arm: rami|sfm|random}"
GPU="${3:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

RUN="outputs/runs/${TAG}_${ARM}"
python eval/render_baseline.py --run-dir "$RUN" --baseline radarsplat_v1
python tools/score_vs_clean.py --run-dir "$RUN" \
    --clean-gt-dir "outputs/canonical_gt/${TAG}/images" \
    --out "$RUN/metrics_vs_clean.json"
python tools/aggregate_results.py
