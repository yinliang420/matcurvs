# B4 — Paired bootstrap 95% CI: ours_kmeans_m2f vs best VLM (Instance-F1)

## Setup

- Comparison: `ours_kmeans_m2f` vs **`vlm_qwen3.5-plus`** (best VLM by mean Inst-F1 on its coverage).
- Shared image set: intersection of the two methods' image_ids, n = **31** images.
- Metric: Instance-F1 (higher is better).
- Bootstrap: paired, 10000 resamples, percentile 95% CI, seed=20260424.
- Also computed: two-sided paired sign-flip permutation p-value, 20000 permutations.

## Results

| Quantity | Value |
|---|---|
| Mean Inst-F1 (ours_kmeans_m2f, shared set) | **0.615** |
| Mean Inst-F1 (vlm_qwen3.5-plus, shared set) | **0.645** |
| Mean paired Δ (ours − VLM) | **-0.031** |
| 95% bootstrap CI on Δ | **[-0.206, +0.157]** |
| Paired sign-flip permutation p-value (two-sided) | **0.7472** |

## Interpretation

- On the 31-image shared test set, `ours_kmeans_m2f` scores Inst-F1 0.615 vs `vlm_qwen3.5-plus`'s 0.645, a paired mean difference of -0.031.
- The 95% bootstrap CI includes zero.
- Paired sign-flip permutation p = 0.7472 (not significant at α=0.05).

## Reproduction

```bash
python experiments/scripts/b_consolidate.py
```