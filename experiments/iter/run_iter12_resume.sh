#!/usr/bin/env bash
# Resume iter12 pipeline after the Phase B + Phase C threshold sweep.
# Phase B winner: q12_d3 (IoU_m 0.515 vs v9_d6 0.510).
# Phase C threshold winner: 0.25 (IoU_g 0.658, IoU_m 0.521, cov_gt 0.775, cov_pred 0.820).
# Now: scales / flip / thicken / D / E / F.
set -e
cd "$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-python} -u"
export PYTHONUNBUFFERED=1

KARGS=(--separator_mode v4 --k_source llm
       --k_csv experiments/results/iter/k_full_label_gemini.csv
       --backend gemini-3.1-pro-preview
       --morph_close_kernel 5 --min_cluster_area 200
       --m2f_ckpt ${KMEANS_ROOT:-.}/models/mask2foremer_q12_d3_h256_best.pt
       --m2f_threshold 0.25)

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
print(f'[$(date +%H:%M:%S)] ✓ ${tag}  n={len(df)}  IoU_g={df.iou_global.mean():.3f}  IoU_m={df.iou_matched_mean.mean():.3f}  pct80={n_hit/max(1,n_lines):.3f}  cov_gt={df.coverage_gt.mean():.3f}  cov_pred={df.coverage_pred.mean():.3f}')
"
}

# ---- Phase C: scales + flip + thicken (composing on threshold 0.25) ----
run phaseC_scales512 --m2f_scales 512,768,1024
run phaseC_flip      --m2f_scales 512,768,1024 --flip_tta
run phaseC_thicken   --m2f_scales 512,768,1024 --flip_tta --thicken

# ---- Phase D: L4 query-NMS ----
run phaseD_qnms_l4   --m2f_scales 512,768,1024 --flip_tta --thicken --query_nms_mode l4_only
run phaseD_qnms_init --m2f_scales 512,768,1024 --flip_tta --thicken --query_nms_mode init

# ---- Phase E: scatter ----
run phaseE_scatter   --m2f_scales 512,768,1024 --flip_tta --thicken --query_nms_mode l4_only --scatter_branch strict

# ---- Phase F: per-line refinement ----
run phaseF_basic     --m2f_scales 512,768,1024 --flip_tta --thicken --query_nms_mode l4_only --scatter_branch strict --refine_mode basic
run phaseF_aggr      --m2f_scales 512,768,1024 --flip_tta --thicken --query_nms_mode l4_only --scatter_branch strict --refine_mode aggr
run phaseF_edge      --m2f_scales 512,768,1024 --flip_tta --thicken --query_nms_mode l4_only --scatter_branch strict --refine_mode edge

echo "[$(date +%H:%M:%S)] iter12 resume pipeline complete."
$PY experiments/iter/finalize_iter12.py
