# Raman Peak-Recovery Pilot v2 (n=60)

## Setup
v1 pipeline + GT loading unchanged. Three fixes: (a) per-column **ridge trace** (smoothed-density argmax, falls back to median on near-uniform columns) replaces column-median; (b) smoothing 9 -> 3; (c) `find_peaks` prominence 0.05 -> 0.02. Sample: every 7th of 441 folders -> 60 images; v1's 30 are the even-indexed subset.

## Headline (60 images)
| Metric | Mean | Median |
|---|---|---|
| Recall | 0.488 | **0.500** |
| Precision | 0.354 | 0.375 |
| MAE (cm^-1) | 9.94 | 9.98 |
| Pearson r | 0.369 | 0.324 |

Aggregate recall 239/465 = **51.4%** (v1: 40.4%). 14/60 imgs >= 0.7 recall (v1: 1/30); 13/60 still <0.3.

## A/B on shared 30
| Median | v1 | v2 | delta |
|---|---|---|---|
| Recall | 0.400 | 0.500 | **+0.100** |
| Mean recall | 0.400 | 0.484 | +0.083 |
| MAE | 9.64 | 9.69 | +0.04 |
| Pearson r | 0.347 | 0.347 | 0.000 |

**15/30 improved, 15 unchanged, 0 regressed.** Clean monotone win. Combo only (shipped together). Likely contributions: lower prominence drives recall gain (recovers small bumps); window=3 protects peak shape (MAE held steady); ridge-trace ~no-op on clean masks.

## Failure analysis (recall<0.3)
Eyeballed `worst_52bf*`, `worst_82a9*`, `worst_c3ab*` in `diag_v2/`. Dominant residual failure is **axis x-mapping**, not peak-finding or segmentation. Panel 2 shows clean peaks at sensible relative positions but cm^-1 shifted by 50-150; bottom-right tick pattern matches GT laterally translated.

Root cause: YOLO axis_box left edge sits inside the chart, missing the first ~50-150 cm^-1, but we map xlsx `x_min` to that wrong edge. Extracted peaks at 280 cm^-1 should be 150 -> outside +-20 tol. Pearson depressed by same shift.

## Verdict
**MARGINAL** -- median sits exactly on the 0.50 boundary. Combo worked cleanly (zero regressions); residual sub-0.3 tail is an axis-mapping bug -- the suggested tick/text filter is the wrong next step.

Recommend: small axis-box left-edge fix (extend to leftmost mask col in ROI, or use y-axis line as left boundary), then v3 60-image rerun. If median > 0.7, scale to 1068; else M2F finetune is the real bottleneck. Do not scale yet.
