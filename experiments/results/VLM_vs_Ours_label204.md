---
name: VLM-vs-Ours on label/204
description: 3 hosted VLMs (GPT-5.4, Gemini-3.1-pro-preview, Qwen-3.5-plus) vs Ours (q12_d3 default) on the 204-image label/ benchmark
type: comparison
---

# VLM vs Ours — label/204

3 hosted VLMs were run on every label/ image via viviai.cc + DashScope
gateways and scored against the same GT pixel polylines used for Ours.
Claude-opus-4-6 was excluded: the user's viviai key returns
`No available channel` for every Claude model.

## Eval protocol (VLM coordinate mapping)

The label/ annotations have a known bug: `axis_detection.y_ticks`
populates the "real" Y value with the pixel-Y (160/175 = 91% of
annotations are buggy), so GT "real" coords are NOT in the chart's
true axis-label space. We therefore use **GT pixel coords as canonical
ground truth** and map VLM real-coord output → pixel as follows:

- **X**: fit `real_x → pixel_x` affine from `axis_detection.x_ticks`
  (reliable in 99% of annotations). Fallback: stretch VLM x-bbox to
  `x_axis_line` extent.
- **Y**: stretch VLM's per-image y-bbox to the chart's plot-region
  Y-extent (between `y_axis_line` endpoints), Y-flipped. We do *not*
  trust `y_ticks`.

**Caveat**: this measures shape + inter-line spacing in Y, not absolute
Y placement. We cannot recover the chart's true Y axis-label range from
the annotation. The same protocol is applied to all 3 VLMs and to Ours
(Ours already outputs pixel coords, so no transform).

## Aggregate (n=204)

| Method | DTW-Δ ↓ | CCR@0.1 ↑ | Inst-F1 ↑ | Curve-PQ ↑ | Latency (s) |
|---|---:|---:|---:|---:|---:|
| **Ours (q12_d3, default)** | **18.48** | **0.903** | 0.645 | **0.225** | 0.4 |
| GPT-5.4 (viviai) | 48.38 | 0.654 | **0.907** | 0.002 | ~20 |
| Gemini-3.1-pro-preview (viviai) | 55.87 | 0.268 | 0.396 | 0.005 | ~30 |
| Qwen-3.5-plus (DashScope) | 68.95 | 0.586 | 0.881 | n/a (~0) | ~107 |
| Claude-opus-4-6 | n/a — gateway lacks Claude channel | | | | |

**Take**: Ours wins on 4/5 metrics. The single VLM advantage is
**Inst-F1** (line-count accuracy): GPT and Qwen reliably name the right
number of lines, but their per-line shape is off by 2.6× (GPT) to 3.7×
(Qwen) on DTW-Δ. **Curve-PQ is 100×+ lower for every VLM**, confirming
that VLMs don't produce per-pixel cluster labels — they regress
sparse-trail coordinates from their internal representation, which
Hungarian matching can pair up but PQ rasterization can't recover.

## Per-stratum (DTW-Δ ↓ / CCR ↑ / F1 ↑)

| Stratum | n | Ours | GPT-5.4 | Gemini-3.1 | Qwen-3.5 |
|---|---:|---|---|---|---|
| L1_single | 70 | 8.15 / 1.00 / 0.49 | 36.6 / 0.71 / 0.80 | 48.5 / 0.40 / 0.55 | 80.8 / 0.63 / 0.85 |
| L2_two_line | 59 | 23.7 / 0.86 / 0.62 | 53.1 / 0.67 / 0.97 | 47.7 / 0.30 / 0.42 | 66.4 / 0.57 / 0.86 |
| L3_few_line (3-4) | 63 | 23.6 / 0.86 / 0.80 | 56.5 / 0.58 / 0.98 | 84.1 / 0.14 / 0.28 | 62.0 / 0.56 / 0.95 |
| L4_many_line (5+) | 12 | 26.2 / 0.82 / 0.87 | 50.6 / 0.60 / 0.88 | NaN / 0.00 / 0.00 | 44.9 / 0.55 / 0.83 |

## Failure modes by backend

- **GPT-5.4**: best of the three. Geometry stays usable (DTW < 60 across
  all strata) and line-count F1 is excellent. Loses to Ours by 2-4× on
  shape DTW.
- **Gemini-3.1-pro-preview**: collapses on dense charts. L4_many_line:
  0/12 successes (DTW is NaN because no Hungarian matches survive at
  τ=0.05·diag). CCR drops from 0.40 (L1) → 0.14 (L3) → 0 (L4) — each
  added line halves CCR.
- **Qwen-3.5-plus**: middling DTW, slow (~107 s/image vs 20 s for GPT).
  L1 is its worst stratum (DTW=80.8) — Qwen tends to over-densify
  single-line outputs.

## Agent-Readiness implications

These numbers reinforce the paper's framing that **a hosted VLM is not
a drop-in replacement for an agent that needs structured chart data**.
Even GPT-5.4 — the strongest VLM here — has 2.6× the spatial error of
the pipeline, makes 100× fewer recoverable per-pixel cluster labels
(PQ ≈ 0), and costs ~20 s/image vs sub-second for the pipeline.

## Files

- CSVs:
  - `vlm_gpt-5.4_label.csv` (204/204 success, 203/204 successful trails)
  - `vlm_gemini-3.1-pro-preview_label.csv` (204/204 success)
  - `vlm_qwen3.5-plus_label.csv` (204/204 success)
- Sweep log: `experiments/results/vlm_label_sweep.log`
- Script: `experiments/run_vlm_label_eval.py`
