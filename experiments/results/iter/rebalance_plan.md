# `label/` Rebalance Plan: Sourcing Multi-Line Charts

Date: 2026-04-25.  Scope: extend the 204-image `label/` test set so per-stratum statistics for K>=4 become defensible (target: >=30 images per L4_many_line stratum).

## 1. Verified K distribution (label/)

| Source folder       | Total | K=1       | K=2       | K=3       | K=4      | K=5     | K=6     |
|---------------------|-------|-----------|-----------|-----------|----------|---------|---------|
| battery_2           |   127 | 50 (39.4%) | 32 (25.2%) | 29 (22.8%) | 9 (7.1%) | 5 (3.9%) | 2 (1.6%) |
| battery_1           |    30 | 12 (40.0%) | 9 (30.0%)  | 8 (26.7%)  | 0        | 0       | 1 (3.3%) |
| battery_3           |    17 | 4 (23.5%)  | 10 (58.8%) | 2 (11.8%)  | 1 (5.9%) | 0       | 0       |
| label_line_figure   |    30 | 4 (13.3%)  | 8 (26.7%)  | 10 (33.3%) | 4 (13.3%) | 3 (10.0%) | 1 (3.3%) |
| **all**             | **204** | **70 (34.3%)** | **59 (28.9%)** | **49 (24.0%)** | **14 (6.9%)** | **8 (3.9%)** | **4 (2.0%)** |

K>=4 = 26 images.  Confirms the user's distribution exactly.

## 2. Sister-dataset inventory (raman + xafs)

Scanned 441 raman folders (1069 images) and 2217 xafs folders (2202 images) by reading every `chemical_formulas_*.json`.

| dataset | K=0 | K=1  | K=2 | K>=3 |
|---------|-----|------|-----|------|
| raman   | 0   | 1069 | 0   | 0   |
| xafs    | 320 | 1795 | 87  | **0**   |

- **raman is monolithically K=1.**
- **xafs has exactly 87 K=2 images** (all paired e.g. `<material>` + `Pb-foil` reference).  Inventory written to `/tmp/multi_k_inventory.json`.
- **Zero K>=3 images** exist in either sister dataset.

## 3. Multi-image folders are NOT split panels

Sampled the 5 largest raman folders (12-22 images each).  Every image inside a folder has a *unique* paper hash prefix (e.g. `0bc17c975e044f63` contains 22 images, 22 distinct hashes).  These are independent figures from different papers grouped only by chemistry topic, not subfigures of a single multi-panel original.  Hash-sharing scan over 200 folders found 0 cases.  Recombining sub-panels is therefore not a viable source of synthetic K>=2.

## 4. `.txt` sidecars carry only axis labels

Sample contents (5/6 inspected, all identical schema):
```json
{"x": "Raman shift (cm-1)", "y": "Intensity (a.u.)"}
{"x": "Photon energy / eV", "y": "Absorption coefficient (a.u.)"}
```
No captions, no hints about line counts.  Cannot be mined for multi-line discovery.

## 5. Sampling-bias finding (step 4)

`label/` is NOT drawn from raman/xafs at all -- 0 of 94 unique label hashes intersect the 441 raman or 2217 xafs hashes.  The image paths reveal four distinct upstream collections:

- `battery_1/2/3` (174/204 = 85.3%) -- a battery-paper corpus on annotator `Posai\\yinliang\\zilongW`'s machine.
- `label_line_figure` (30/204 = 14.7%) -- a separate `MLLM_Selected/label_line_figure` dump on drive E:.

So the test set is essentially "battery papers" with a small multi-line-rich supplement.  This is a *content* bias (battery domain dominates K=1 plots like CV, EIS Bode, single-cycle galvanostatic curves), not a labelling artefact.

## 6. Recommendation: path forward for L4 expansion

**Reaching 30 K>=4 from raman/xafs is infeasible.**  There are zero K>=3 images and only 87 K=2 images in either dataset.  The 87 xafs K=2 set is also a degenerate stratum (one sample line + one foil reference) -- adding them mostly enlarges K=2, not K>=4, and biases K=2 toward EXAFS spectra.

Concrete proposal, in priority order:

1. **Do not rebalance from raman/xafs.**  They contribute essentially no novel multi-line content.  Annotating the 87 xafs K=2 cases would shift K=2 from 28.9% -> 41.4% but leave K>=4 untouched -- the wrong axis.
2. **Mine the same upstream corpora** (`battery_1/2/3` and `label_line_figure`) for held-out images that the original annotator skipped.  The `label_line_figure` source is already 56.6% K>=3 and 26.7% K>=4 -- if more raw images exist on drive E:, a focused re-annotation of just that bucket is the cheapest path to +20 K>=4.
3. **Run a separate sourcing effort** targeting comparison-style figures (overlaid CV at multiple scan rates, multi-cycle plots, dQ/dV families).  Likely candidates: capacity-vs-cycle plots, rate-capability curves, multi-temperature impedance.  A paper search ("Figure 3a -d" style overlays) is the realistic source of K=5/6 examples.
4. **Pragmatic alternative** if (2)/(3) are blocked: report L4_many_line metrics with a wide bootstrap CI and an explicit caveat that n=26 is exploratory.  Merge K=4-6 into a single "L4+" stratum for headline tables; keep per-K break-down only as a supplementary appendix.

**Bottom line:** the 30-per-stratum target requires fresh annotation, not redistribution from raman/xafs.  Recommend pursuing (2)+(3) in parallel and committing to (4) for the next paper draft so reporting can proceed without waiting for new labels.
