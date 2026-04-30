# Raman Peak-Recovery Pilot (n=30)

## Setup
30 images sampled uniformly across the 441 Raman folders (every 14th, first image). Pipeline: `pipeline.run_pipeline(n_lines=1, separator_mode='v4', morph_close_kernel=5, min_cluster_area=200)`. xlsx used as axis oracle (its min/max raman_shift and intensity define the linear pixel-to-real mapping). Predicted peaks via `scipy.signal.find_peaks(prominence=0.05*range, distance=10)` on the extracted intensity series; greedy matching to GT peaks within +-20 cm^-1.

## Headline numbers (all 30 ok)

| Metric                       | Mean   | Median |
|------------------------------|--------|--------|
| Peak position MAE (cm^-1)    | 10.12  | 9.64   |
| Peak recall                  | 0.40   | 0.40   |
| Peak precision               | 0.43   | 0.43   |
| Curve Pearson r              | 0.41   | 0.35   |
| Curve normalized RMSE        | 0.148  | 0.149  |

Aggregate recall = 95 / 235 GT peaks (40.4 %); aggregate precision = 95 / 212 predicted peaks (44.8 %).

Recall distribution: 5 imgs <=0.20, 10 in (0.2, 0.4], 12 in (0.4, 0.6], 2 in (0.6, 0.8], 0 above 0.8. Best image hit 0.75 recall (3/4 peaks); worst missed every peak (0/2).

## Failure modes
- 0 / 30 pipeline crashes, 0 / 30 missing xlsx, 0 / 30 axis-box detection failures (in this stratified sample).
- The dominant failure is "merged or missed peaks": find_peaks under-resolves clusters of close GT peaks because the extracted curve is smoothed and the M2F/KMeans output is noisier than xlsx ground truth (median Pearson 0.35).
- A few worst-case images (e.g. `8b6eee6c64dcd2e5/...`, recall 0.10, Pearson 0.00) show the extracted curve is essentially flat / decorrelated -- the cluster mask captured axis ticks or text instead of the curve.

## Honest assessment
At 40 % recall and ~10 cm^-1 MAE on matched peaks, our IoU=0.65 extraction is **not yet good enough for reliable Raman peak recovery**. Position accuracy on matched peaks (~10 cm^-1) is borderline acceptable for typical 4 cm^-1 instrument resolution, but missing 60 % of peaks means downstream phase-ID or vibrational-mode assignment will be unreliable. The shape of the recovered curve is also weak (median Pearson 0.35, NRMSE ~0.15), which is the upstream cause of the peak misses: small bumps get smoothed away by the median-row + moving-average step, and the prominence threshold then drops them.

## Recommendation
Do **not** scale the current pipeline to all 1068 images yet. The detector survives Raman charts (no crashes), but peak recovery is recall-limited. Cheap next steps before any large-scale run: (1) drop the column-median collapse and fit a 1-D ridge along the mask; (2) lower `prominence` to ~0.02 of range and reduce smoothing window from 9 to 3; (3) sanity-check axis-tick/text contamination of the cluster mask. If these push median recall above ~0.7 on a 60-image sample, then scale up. Otherwise the IoU=0.65 segmentation is the bottleneck and a Raman-specific finetune is required.
