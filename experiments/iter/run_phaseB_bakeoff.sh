#!/usr/bin/env bash
# Phase B: 5-ckpt Mask2Former bake-off on full 204 (sequential to avoid MPS contention).
# Logs: experiments/results/iter/phaseB_<tag>.log
set -e
cd "$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-python} -u"
export PYTHONUNBUFFERED=1
KARGS=(--separator_mode v4 --k_source llm
       --k_csv experiments/results/iter/k_full_label_gemini.csv
       --backend gemini-3.1-pro-preview
       --morph_close_kernel 5 --min_cluster_area 200)

run() {
  local tag=$1; local ckpt=$2
  local log="experiments/results/iter/phaseB_${tag}.log"
  echo "[$(date +%H:%M:%S)] ▶ phaseB_${tag}"
  $PY experiments/iter/run_iter_full.py --iter "phaseB_${tag}" \
      --m2f_ckpt "$ckpt" "${KARGS[@]}" > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] ✓ phaseB_${tag}"
}

run q12_d3      ${KMEANS_ROOT:-.}/models/mask2foremer_q12_d3_h256_best.pt
run v8q10       ${MATDEPLOT_ROOT:-/path/to/MatDeplot}/codex_agent/line_models/v8_hr640_bs2_q10_d3_h256_best.pt
run v8q15       ${MATDEPLOT_ROOT:-/path/to/MatDeplot}/codex_agent/line_models/v8_more_queries_q15_d3_h256_best.pt
run v9_d6       ${MATDEPLOT_ROOT:-/path/to/MatDeplot}/codex_agent/line_models/v9_hr640_q10_d6_h256_best.pt
run v11_raw40k  ${MATDEPLOT_ROOT:-/path/to/MatDeplot}/codex_agent/line_models/v11_raw40k_q12_d3_h256_best.pt

echo "[$(date +%H:%M:%S)] aggregating pct80…"
$PY experiments/iter/aggregate_pct80.py \
    --csvs experiments/results/iter/phaseB_*_full204_metrics.csv \
    --out experiments/results/iter/phaseB_pct80.csv --full
echo "[$(date +%H:%M:%S)] done."
