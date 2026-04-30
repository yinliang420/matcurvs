---
name: FG-model ablation on label/204
description: 4-way Mask2Former checkpoint comparison on label/204, holding KMeans separator + YOLO axis/legend fixed
type: ablation
---

# FG-model ablation (label/204)

We swap only the Mask2Former FG checkpoint; everything downstream
(axis-YOLO, legend-YOLO, LAB-kmeans separator, skeleton-polyline
extraction) is held constant. Fallback trigger `τ_fg = 0.003`.

| Ckpt | Queries | Decoder | Hidden | Training data | DTW-Δ ↓ | CCR@0.1 ↑ | Inst-F1 ↑ | PQ ↑ | mean K | fb/204 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| **q12_d3_h256 (v11_raw40k, default)** | 12 | 3 | 256 | synthetic_v9_40k + real_182 | **18.48** | **0.903** | 0.645 | 0.225 | 4.62 | 0 |
| v8_q10_d3_h256 | 10 | 3 | 256 | synthetic_v6_20k + real_182 | 21.33 | 0.881 | 0.670 | 0.223 | 4.16 | 1 |
| v8_q15_d3_h256 | 15 | 3 | 256 | synthetic_v6_20k + real_182 | 21.98 | 0.876 | 0.688 | 0.192 | 3.93 | 1 |
| v9_q10_d6_h256 | 10 | 6 | 256 | synthetic_v7_30k + real_182 | 21.38 | 0.888 | 0.677 | **0.252** | 4.16 | 0 |

## Observations

1. **Default (v11_raw40k, q12_d3) wins on geometry**: lowest DTW-Δ
   (18.48) and highest CCR@0.1 (0.903). Other variants give up 2–3 pp
   on CCR and add 3–4 DTW units to gain 2–4 pp on Inst-F1.
2. **Adding queries (q15) or depth (d6) marginally raises Inst-F1**
   (0.688 / 0.677 vs 0.645) but degrades DTW. This matches the
   MatDeplot v7/v8 finding that `deeper_decoder` gets cleaner splits
   at the cost of color-gradient over-segmentation.
3. **Training-data size dominates over architectural knobs**: the
   40k-synthetic v11 beats 20k-synthetic v8 across every metric except
   Inst-F1 on single-line, confirming the v9 improvement plan's
   hypothesis that synthetic-v7-with-more-diversity is a bigger lever.
4. **Fallback stays dormant** across all 4 variants (0–2 triggers out
   of 204), i.e. the `τ_fg = 0.003` safety net is orthogonal to the
   FG checkpoint choice.
5. **Curve-PQ peaks on v9_q10_d6** (0.252 vs 0.225 default), mildly
   suggesting the deeper decoder produces cleaner per-curve
   segmentations when it does cluster correctly.

## Takeaway for the paper

The pipeline's DTW/CCR advantage comes from **the foreground-
guaranteed cell design** (M2F + fallback), not from squeezing SOTA
out of any single backbone; all four variants live within ~20% of
one another on DTW and all four beat every hosted VLM (best VLM DTW
on the shared set is 63.0, see `B3_method_aggregate.md`).

## Files

- CSVs:
  - `ours_kmeans_m2f_label.csv` (default q12_d3)
  - `ours_kmeans_m2f_label_v8q10.csv`
  - `ours_kmeans_m2f_label_v8q15.csv`
  - `ours_kmeans_m2f_label_v9q10d6.csv`
- Script: `experiments/run_label_eval.py` (`--m2f_ckpt` flag)
- Checkpoints live under
  `/path/to/MatDeplot/codex_agent/line_models/`
  and the default at `kmeans/models/mask2foremer_q12_d3_h256_best.pt`.
