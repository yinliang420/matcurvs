#!/usr/bin/env bash
# Phase G: re-evaluate the best Phase C config under thickness-normalized IoU
# (skeletonize both GT and pred, dilate both to 5px, then IoU). This is the
# "shape-faithful reconstruction" metric — what an agent re-rasterizing the
# extracted polyline would actually compare against.
set -e
cd "$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-python} -u"
export PYTHONUNBUFFERED=1

KARGS=(--separator_mode v4 --k_source llm
       --k_csv experiments/results/iter/k_full_label_gemini.csv
       --backend gemini-3.1-pro-preview
       --morph_close_kernel 5 --min_cluster_area 200
       --m2f_ckpt ${KMEANS_ROOT:-.}/models/mask2foremer_q12_d3_h256_best.pt
       --m2f_threshold 0.25
       --m2f_scales 512,768,1024
       --flip_tta
       --thicken)

run() {
  local tag=$1; shift
  local log="experiments/results/iter/${tag}.log"
  echo "[$(date +%H:%M:%S)] ▶ ${tag}"
  $PY experiments/iter/run_iter_full.py --iter "${tag}" "${KARGS[@]}" "$@" \
      > "$log" 2>&1
  $PY -c "
import pandas as pd
df = pd.read_csv('experiments/results/iter/${tag}_full204_metrics.csv')
n_lines = int(df['k_gt'].sum()); n_hit = int(df['n_iou_ge_0_8'].sum())
print(f'[$(date +%H:%M:%S)] ✓ ${tag}  n={len(df)}  IoU_g={df.iou_global.mean():.3f}  IoU_m={df.iou_matched_mean.mean():.3f}  pct80={n_hit/max(1,n_lines):.3f}')
"
}

run phaseG_normalized --normalize_thickness
echo "[$(date +%H:%M:%S)] phaseG done."
