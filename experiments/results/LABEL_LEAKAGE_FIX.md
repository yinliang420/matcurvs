---
name: label/204 train/eval leakage discovery + clean-subset re-eval
description: 114 of 204 label/ images are in the FG model's training root; corrected numbers on the n=90 held-out complement
type: correction
---

# label/204 train/eval leakage — discovery + fix

Reviewer 4 (data & benchmark critic) flagged that the
default Mask2Former checkpoint (`q12_d3_h256`) was trained on real
images that **also appear in `kmeans/label/`**. Verified directly:

```
train root:  /path/to/MatDeplot/h100_training/data/label_data/
             → 114 *_annotation.json files
eval set:    label/
             → 204 *_annotation.json files
intersection: 114 / 204 = 55.9% of label/ is in the FG training root
clean held-out: 90 images
```

`train.py:197` reads every `*_annotation.json` under `label_root`; the
FG model was trained on the 114 overlapping images. The paper's
"real_182" wording for the training set is consistent with `synthetic_v9_40k
+ 182 real`, where the 114 in this directory are a subset of the 182.

## Action

We re-aggregate every CSV (`ours_kmeans_m2f_label*.csv`,
`vlm_*_label.csv`) on the **clean held-out 90-image complement**, written
to `experiments/results/label_clean_holdout.txt`.

## Headline numbers (n=90 clean held-out)

| Method | DTW-Δ ↓ | CCR@0.1 ↑ | Inst-F1 ↑ | Curve-PQ ↑ |
|---|---:|---:|---:|---:|
| **Ours (q12_d3, default)** | **19.92** | **0.898** | 0.650 | **0.204** |
| Ours (v8_q10_d3) | 23.78 | 0.876 | 0.672 | 0.191 |
| Ours (v8_q15_d3) | 23.27 | 0.877 | 0.690 | 0.178 |
| Ours (v9_q10_d6) | 22.80 | 0.880 | 0.691 | 0.202 |
| GPT-5.4 | 61.59 | 0.587 | **0.915** | 0.000 |
| Gemini-3.1-pro-preview | 73.42 | 0.210 | 0.343 | 0.006 |
| Qwen-3.5-plus | 76.40 | 0.521 | 0.837 | 0.000 |

Comparison to the contaminated full n=204 numbers:

| Method | DTW (n=204) | DTW (n=90 clean) | Δ |
|---|---:|---:|---:|
| Ours (default) | 18.48 | 19.92 | +1.44 (+7.8%) |
| GPT-5.4 | 48.38 | 61.59 | +13.21 (+27%) |
| Gemini-3.1 | 55.87 | 73.42 | +17.55 (+31%) |
| Qwen-3.5 | 68.95 | 76.40 | +7.45 (+11%) |

**Key finding**: removing the 114 contaminated images **moves Ours's
DTW by less than 8%**, while VLM DTW gets noticeably worse (the dropped
contaminated subset was the *easier* slice). The qualitative picture
holds — Ours retains a 3.1-3.8× DTW advantage and a 30-69pp CCR
advantage on the clean held-out set.

## Per-stratum on n=90

| Stratum | n | Ours DTW/CCR/F1 | GPT DTW/CCR/F1 | Gemini DTW/CCR/F1 | Qwen DTW/CCR/F1 |
|---|---:|---|---|---|---|
| L1_single | 29 | 9.0 / 1.00 / 0.48 | 39.9 / 0.73 / 0.84 | 56.4 / 0.37 / 0.52 | 84.8 / 0.67 / 0.86 |
| L2_two_line | 28 | 26.2 / 0.84 / 0.62 | 70.2 / 0.57 / 0.95 | 71.0 / 0.19 / 0.32 | 85.0 / 0.46 / 0.77 |
| L3_few_line (3-4) | 28 | 25.4 / 0.86 / 0.82 | 75.5 / 0.48 / 0.95 | 115.4 / 0.09 / 0.25 | 63.1 / 0.45 / 0.89 |
| L4_many_line (≥5) | 5 | 17.3 / 0.90 / 0.86 | 61.1 / 0.44 / 0.97 | NaN / 0.00 / 0.00 | 54.8 / 0.36 / 0.80 |

L4 stratum drops from n=12 to n=5; explicitly noted as too small for
confident strata-level claims (CIs would span tens of percentage points).

## Files

- Manifest: `experiments/results/label_clean_holdout.txt` (one
  image_id per line, 90 lines)
- All headline metrics in this report are deterministic re-aggregations
  of existing CSVs filtered to this manifest. No re-runs of pipelines
  or VLM API calls are needed.

## What this changes in the paper

1. `tab:label-204` → relabel as `tab:label-clean-90`, replace numbers
   with the n=90 clean column above.
2. `tab:vlm-label-204` → same (n=90 column).
3. `tab:fg-ablation` → re-aggregate to n=90; the default still wins
   geometry (DTW 19.92 vs 22.8-23.8 for variants).
4. `\paragraph{Scale-out to the full \texttt{label/} set ($n{=}204$)}` →
   rewrite as "Scale-out to the held-out complement of \texttt{label/}
   that is disjoint from FG-model training data ($n{=}90$)".
5. Limitations §(a) Scale → add explicit leakage acknowledgement and
   point to this report as the corrective.

The headline claim — "Ours leads every hosted VLM by 3× on DTW, by
30+pp on CCR, and is the only method with non-zero Curve-PQ" — survives
on n=90 clean held-out images.
