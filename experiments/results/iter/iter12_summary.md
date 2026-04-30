# iter12 — full-204 + leakage-free n=90 results

## Best config (highest pct_lines_iou_ge_0.8 on n=90)

- **csv** = `phaseC_thr025_full204_metrics.csv`
- **pct80_hold90** = `0.0049`
- **pct80_full204** = `0.0044`
- **iou_m_hold90** = `0.4817`
- **iou_g_hold90** = `0.6484`
- **cov_gt_hold90** = `0.7582`
- **cov_pred_hold90** = `0.8243`
- **K_acc_hold90** = `0.8333`

## All phase CSVs ranked by score (pct80_hold90 + 0.5 × IoU_m_hold90)

| csv | pct80_hold90 | pct80_full204 | iou_m_hold90 | iou_g_hold90 | cov_gt_hold90 | cov_pred_hold90 | K_acc_hold90 | score_hold90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phaseC_thr025_full204_metrics.csv | 0.0049 | 0.0044 | 0.4817 | 0.6484 | 0.7582 | 0.8243 | 0.8333 | 0.24575 |
| phaseC_flip_full204_metrics.csv | 0.0049 | 0.011 | 0.481 | 0.6492 | 0.8306 | 0.7558 | 0.8333 | 0.24539999999999998 |
| phaseD_qnms_init_full204_metrics.csv | 0.0049 | 0.011 | 0.4809 | 0.649 | 0.8306 | 0.7555 | 0.8333 | 0.24534999999999998 |
| phaseF_basic_clean_full204_metrics.csv | 0.0049 | 0.011 | 0.4809 | 0.649 | 0.8306 | 0.7555 | 0.8333 | 0.24534999999999998 |
| phaseF_edge_clean_full204_metrics.csv | 0.0049 | 0.011 | 0.4809 | 0.649 | 0.8306 | 0.7555 | 0.8333 | 0.24534999999999998 |
| phaseC_thicken_full204_metrics.csv | 0.0049 | 0.011 | 0.4806 | 0.6487 | 0.8308 | 0.7551 | 0.8333 | 0.2452 |
| phaseC_thr030_full204_metrics.csv | 0.0049 | 0.0044 | 0.4791 | 0.6445 | 0.7487 | 0.8292 | 0.8333 | 0.24445 |
| phaseB_q12_d3_full204_metrics.csv | 0.0049 | 0.0022 | 0.4768 | 0.6408 | 0.7398 | 0.8336 | 0.8333 | 0.2433 |
| phaseC_scales512_full204_metrics.csv | 0.0049 | 0.011 | 0.4729 | 0.6415 | 0.8008 | 0.7703 | 0.8333 | 0.24134999999999998 |
| phaseD_qnms_l4_full204_metrics.csv | 0.0049 | 0.011 | 0.462 | 0.6468 | 0.8371 | 0.748 | 0.8333 | 0.2359 |
| phaseE_scatter_full204_metrics.csv | 0.0049 | 0.011 | 0.462 | 0.6468 | 0.8371 | 0.748 | 0.8333 | 0.2359 |
| phaseB_v9_d6_full204_metrics.csv | 0.0 | 0.0 | 0.4667 | 0.6247 | 0.7105 | 0.8356 | 0.8333 | 0.23335 |
| phaseG_norm5_full204_metrics.csv | 0.0 | 0.0022 | 0.3379 | 0.4344 | 0.6109 | 0.5967 | 0.8333 | 0.16895 |
| phaseG_norm5_refine_edge_full204_metrics.csv | 0.0 | 0.0022 | 0.3377 | 0.4342 | 0.611 | 0.5963 | 0.8333 | 0.16885 |

## Methodology

- Phase B: 2-ckpt Mask2Former bake-off (q12_d3 control + v9_hr640_q10_d6 alt).
- Phase C: m2f threshold sweep ∈ {0.25, 0.30}, then composing winning threshold with scales=(512,768,1024), flip TTA, thickness compensation (skeletonize→dilate-to-5px under LAB ΔE<12 gate).
- Phase D: per-query NMS branch when `K ≥ 5`, plus a query-init seeding mode for KMeans.
- Phase E: marker-only scatter detector (cc count ≥ 3K, mean cc area < 50 px, max cc longest axis < 40 px) → polyfit deg=3 per LAB cluster.
- Phase F: per-line LAB-gated dilation, three modes (basic ΔE=10 grow=4 / aggr ΔE=15 grow=8 / edge basic + Sobel mag>μ+σ gate).
- Each cell evaluates the full 204-image label set; metric reporting filters to the n=90 leakage-controlled holdout from `experiments/results/label_clean_holdout.txt`.
- New per-image columns in iter12 CSVs vs iter≤11: `iou_per_line` (JSON list), `matched_pairs` (JSON list of (gt_idx, pred_idx, iou)), `n_iou_ge_0_8` (count).
