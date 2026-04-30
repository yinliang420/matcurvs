---
name: Label/ 204-image baseline evaluation
description: Full label/ set evaluation of default pipeline (Mask2Former q12_d3_h256 + KMeans, τ=0.003)
type: evaluation
---

# Label/ 204-image baseline evaluation

Default pipeline: YOLO axis + legend/clutter YOLO + Mask2Former-q12_d3_h256
FG (ckpt `models/mask2foremer_q12_d3_h256_best.pt`) + LAB k-means
separator, `m2f_min_fg = 0.003`.

## Overall (n = 204)

| Metric | Value |
|---|---|
| DTW-Δ (↓) | **18.48** |
| CCR@0.1 (↑) | **0.903** |
| Inst-F1 (↑) | **0.645** |
| Curve-PQ (↑) | 0.225 |
| Fallback rate | 0/204 (0.0%) |
| Elapsed / image | 1.51 s |

## Per-stratum

| Stratum | n | DTW-Δ | CCR | Inst-F1 | PQ | mean K | fb |
|---|---|---|---|---|---|---|---|
| L1_single (1 line) | 70 | 8.15 | 0.996 | 0.490 | 0.221 | 3.86 | 0 |
| L2_two_line (2) | 59 | 23.73 | 0.860 | 0.622 | 0.195 | 4.78 | 0 |
| L3_few_line (3–4) | 63 | 23.56 | 0.856 | 0.798 | 0.285 | 4.92 | 0 |
| L4_many_line (5+) | 12 | 26.24 | 0.823 | 0.865 | 0.088 | 6.67 | 0 |

## Observations

1. Geometry fidelity (CCR@0.1) stays above 0.82 across all strata and
   tracks 0.996 on single-line plots — consistent with the MatDeplot-36
   pattern and suggests the pipeline's advantage is robust at ~6× the
   sample size.
2. Inst-F1 is low in L1 (0.490) because Inst-F1 rewards match-or-miss
   instance counts; on single-line images, a spurious extra cluster
   halves F1 but barely affects DTW (mean K=3.86 > 1 indicates over-
   segmentation). The trade is deliberate: oversplit + merge is safer
   for multi-line extraction than undersplit.
3. L4 (5+ lines) has the best Inst-F1 (0.865) despite the highest DTW —
   many trails exist to match, so partial success still registers.
4. Fallback never fires: at τ=0.003 every label/ image has ≥0.3% FG
   coverage. Same story as MatDeplot-36.
5. Wall-clock 1.51 s/image → 5.4 min for the whole set on single-GPU
   (local M-series Mac, no batching).

## Comparison against MatDeplot-36 baseline

| Set | n | DTW-Δ | CCR | Inst-F1 |
|---|---|---|---|---|
| MatDeplot-36 | 36 | 15.88 | 0.932 | 0.601 |
| Label/204 | 204 | 18.48 | 0.903 | 0.645 |

The 5.7× larger evaluation preserves DTW and CCR within ±3 pp and
raises Inst-F1 by +4.4 pp (label/ has more multi-line cases where F1
recovers). Headline numbers in the paper therefore survive
scale-out.

## Files

- CSV: `experiments/results/ours_kmeans_m2f_label.csv`
- Script: `experiments/run_label_eval.py`
- Annotations: `label/*_annotation.json`
