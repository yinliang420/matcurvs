# Curve-Count Source Reproducibility Capsule

This capsule exactly recomputes the AC curve-count-source comparison on the
leakage-clean L1 subset from frozen annotations and prediction masks. It does
not rerun the complete image-to-output pipeline.

The released comparison covers 90 leakage-clean panels, 203 manually annotated
curves, and six fixed curve-count conditions. The reported `LR0.5 ↑` is the
fraction of GT curves whose one-to-one matched prediction reaches pixel IoU at
least 0.5.

## What is included

- `annotations/`: 90 sanitized manual pixel-annotation files, one per panel.
- `predictions/`: 540 frozen prediction archives (`6 conditions × 90 panels`).
- `curve_counts/`: released per-panel selected-`K` CSVs for each condition.
- `splits/clean90.txt`: the leakage-clean 90-panel cohort used in the AC table.
- `splits/overlap114.txt`: the complementary overlap split retained as cohort
  provenance.
- `sources.csv`: source identifiers and image SHA-256 digests only.
- `condition_metadata.json`: condition labels, prompt hash, and provenance labels.
- `build_inputs.json`: hashes of the source manifests used to assemble the capsule.
- `evaluate.py`: standalone evaluator for exact regeneration of the AC table.
- `expected/`: canonical evaluator outputs for `--check-expected`.
- `MANIFEST.sha256`: integrity manifest for the public capsule.

Source figure images and raw hosted-model responses are not redistributed. The
evaluator begins from frozen prediction masks, so it verifies the released
metric table rather than the complete image-to-output pipeline.

## Recompute the released table

Use a new or empty output directory for each run.

From the repository root:

```bash
python reproducibility/ac_k_source/evaluate.py \
  --capsule reproducibility/ac_k_source \
  --output /tmp/matdeplot_ac_k_source_eval \
  --check-expected
```

Expected successful output:

```text
evaluation_ok panels=90 gt_curves=203 rows=1218 output_dir=/tmp/...
```

The regenerated `ac_k_source_table.csv` reports the six released conditions:
`gemini_submitted`, `gpt54`, `qwen_vl_max`, `qwen3_vl_plus`, `local_qwen25`,
and `silhouette`.

## Released AC table

| Condition | Hosted proprietary API needed to produce `K`? | `LR0.5 ↑` |
|---|:---:|---:|
| `gemini_submitted` | Yes | `93/203 = 45.81%` |
| `gpt54` | Yes | `91/203 = 44.83%` |
| `qwen_vl_max` | Yes | `96/203 = 47.29%` |
| `qwen3_vl_plus` | Yes | `94/203 = 46.31%` |
| `local_qwen25` | No | `93/203 = 45.81%` |
| `silhouette` | No | `69/203 = 33.99%` |

These numbers come from a controlled sensitivity check in which the downstream
masks and evaluator are fixed and only the source of `K` changes. Small
differences across hosted conditions should not be read as a reliable model
ordering.

## Integrity checks

To verify only the manifest:

```bash
python reproducibility/ac_k_source/evaluate.py \
  --capsule reproducibility/ac_k_source \
  --verify-only
```

## Licensing boundary

The released annotations and frozen prediction masks are author-created derived
artefacts for metric recomputation. Publisher, DOI, and figure-license fields in
`sources.csv` remain blank for withheld source images until redistribution
permission is confirmed. See `DATA_LICENSE.md` for details.
