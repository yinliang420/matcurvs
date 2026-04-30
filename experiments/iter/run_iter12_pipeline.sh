#!/usr/bin/env bash
# iter12 sequential pipeline: Phase B (control + 1 alt) → Phase C (recall sweep)
# → Phase D (query-NMS) → Phase E (scatter) → Phase F (refine).
# Each step is gated on the prior winner; orchestration in Python.
set -e
cd "$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-python} -u"
export PYTHONUNBUFFERED=1

KARGS=(--separator_mode v4 --k_source llm
       --k_csv experiments/results/iter/k_full_label_gemini.csv
       --backend gemini-3.1-pro-preview
       --morph_close_kernel 5 --min_cluster_area 200)

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

# ---- Phase B ----
# q12_d3 control already running externally — skip if its CSV exists
if [ ! -f experiments/results/iter/phaseB_q12_d3_full204_metrics.csv ]; then
  run phaseB_q12_d3 \
      --m2f_ckpt ${KMEANS_ROOT:-.}/models/mask2foremer_q12_d3_h256_best.pt
fi
run phaseB_v9_d6 \
    --m2f_ckpt ${MATDEPLOT_ROOT:-/path/to/MatDeplot}/codex_agent/line_models/v9_hr640_q10_d6_h256_best.pt

# Pick winner
WINNER_CKPT=$($PY -c "
import pandas as pd, glob
best = (None, -1.0)
for csv in sorted(glob.glob('experiments/results/iter/phaseB_*_full204_metrics.csv')):
    df = pd.read_csv(csv)
    n = int(df['k_gt'].sum()); h = int(df['n_iou_ge_0_8'].sum())
    pct80 = h / max(1, n)
    iou_m = float(df['iou_matched_mean'].mean())
    score = pct80 * 1.0 + iou_m * 0.5  # composite tie-breaker
    if score > best[1]: best = (csv, score)
print(best[0])
")
echo "[$(date +%H:%M:%S)] phase B winner: $WINNER_CKPT"
WINNER_CKPT_PATH=$(echo "$WINNER_CKPT" | grep -q "v9_d6" && \
  echo ${MATDEPLOT_ROOT:-/path/to/MatDeplot}/codex_agent/line_models/v9_hr640_q10_d6_h256_best.pt || \
  echo ${KMEANS_ROOT:-.}/models/mask2foremer_q12_d3_h256_best.pt)

CKPT_FLAG="--m2f_ckpt $WINNER_CKPT_PATH"

# ---- Phase C: recall push ----
# tighter sweep: one mode change at a time to compose
run phaseC_thr025  $CKPT_FLAG --m2f_threshold 0.25
run phaseC_thr030  $CKPT_FLAG --m2f_threshold 0.30
# pick best threshold (returns the float directly, not the suffix)
THR_VAL=$($PY -c "
import pandas as pd, glob
best = (None, -1.0)
for csv in sorted(glob.glob('experiments/results/iter/phaseC_thr*_full204_metrics.csv')):
    df = pd.read_csv(csv)
    if 'n_iou_ge_0_8' not in df.columns: continue
    n = int(df['k_gt'].sum()); h = int(df['n_iou_ge_0_8'].sum())
    pct80 = h / max(1, n)
    iou_m = float(df['iou_matched_mean'].mean())
    cov_pred = float(df['coverage_pred'].mean())
    if cov_pred < 0.85: continue
    suffix = csv.split('phaseC_thr')[1].split('_')[0]  # '025', '030', etc.
    thr_val = float('0.' + suffix.lstrip('0')) if suffix.lstrip('0') else 0.0
    # but '025' -> '0.25' (drop leading zero only one)
    if len(suffix) == 3 and suffix.startswith('0'):
        thr_val = float('0.' + suffix[1:])
    else:
        thr_val = float('0.' + suffix)
    score = pct80 * 1.0 + iou_m * 0.5
    if score > best[1]: best = (thr_val, score)
print(best[0] if best[0] is not None else 0.30)
")
echo "[$(date +%H:%M:%S)] phase C threshold winner: $THR_VAL"

# multiscale + flip + thicken on top of best threshold
run phaseC_scales512  $CKPT_FLAG --m2f_threshold $THR_VAL --m2f_scales 512,768,1024
run phaseC_flip       $CKPT_FLAG --m2f_threshold $THR_VAL --m2f_scales 512,768,1024 --flip_tta
run phaseC_thicken    $CKPT_FLAG --m2f_threshold $THR_VAL --m2f_scales 512,768,1024 --flip_tta --thicken

# ---- Phase D: L4 query-NMS ----
run phaseD_qnms_l4   $CKPT_FLAG --m2f_threshold $THR_VAL --m2f_scales 512,768,1024 --flip_tta --thicken --query_nms_mode l4_only
run phaseD_qnms_init $CKPT_FLAG --m2f_threshold $THR_VAL --m2f_scales 512,768,1024 --flip_tta --thicken --query_nms_mode init

# ---- Phase E: scatter ----
run phaseE_scatter  $CKPT_FLAG --m2f_threshold $THR_VAL --m2f_scales 512,768,1024 --flip_tta --thicken --query_nms_mode l4_only --scatter_branch strict

# ---- Phase F: per-line refinement ----
run phaseF_basic    $CKPT_FLAG --m2f_threshold $THR_VAL --m2f_scales 512,768,1024 --flip_tta --thicken --query_nms_mode l4_only --scatter_branch strict --refine_mode basic
run phaseF_aggr     $CKPT_FLAG --m2f_threshold $THR_VAL --m2f_scales 512,768,1024 --flip_tta --thicken --query_nms_mode l4_only --scatter_branch strict --refine_mode aggr
run phaseF_edge     $CKPT_FLAG --m2f_threshold $THR_VAL --m2f_scales 512,768,1024 --flip_tta --thicken --query_nms_mode l4_only --scatter_branch strict --refine_mode edge

# ---- Final aggregation ----
echo "[$(date +%H:%M:%S)] aggregating final results…"
$PY experiments/iter/aggregate_pct80.py \
    --csvs experiments/results/iter/phaseB_*_full204_metrics.csv \
           experiments/results/iter/phaseC_*_full204_metrics.csv \
           experiments/results/iter/phaseD_*_full204_metrics.csv \
           experiments/results/iter/phaseE_*_full204_metrics.csv \
           experiments/results/iter/phaseF_*_full204_metrics.csv \
    --out experiments/results/iter/iter12_final_pct80.csv --full
echo "[$(date +%H:%M:%S)] iter12 pipeline complete."
