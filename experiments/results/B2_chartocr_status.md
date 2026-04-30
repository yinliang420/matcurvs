# B2 — ChartOCR integration status

**Status: omitted — no public checkpoint available locally.**

## Summary

ChartOCR (Luo et al., 2021, *"ChartOCR: Data Extraction from Charts Images via
a Deep Hybrid Framework"*, WACV 2021) is listed in
`paper/neurips2026_intro/plan/MASTER_PLAN.md` as a desired
vision-decoder baseline. We investigated whether any usable artefact exists
on the local machine and conclude that the baseline cannot be run without
downloading model weights, which the master plan forbids for this subagent.

## Local search

We searched the two candidate project trees for any ChartOCR-related
adapter, weights, config, or data:

- `/path/to/MatDeplot/` — no files matching
  `chartocr*`, `ChartOCR*`, or `chart_ocr*` at any depth.
- `` — no files matching
  `chartocr*`, `ChartOCR*`, or `chart_ocr*` at any depth.

No pretrained weights were found and no adapter script is present.

## Why we do not download

Per the SUBAGENT_B task spec: "If no local weights: do NOT attempt to
download." Additionally:

- The original ChartOCR paper (Luo et al., 2021) did not release a public
  checkpoint. The CornerNet-based keypoint detector and grouping head
  would need to be re-trained on ChartOCR's proprietary chart data to
  obtain comparable numbers.
- Community re-implementations (e.g., chart-to-table replications) differ
  materially from the published method and are not a faithful baseline.
- Our paper already reports two modern vision-decoder baselines in the
  same family (DePlot and OneChart) plus four commercial VLMs, which
  exercise the same "end-to-end pixels → structured data" paradigm that
  ChartOCR represents.

## Paper citation

> Luo, J., Li, Z., Wang, J., & Lin, C.-Y. (2021).
> **ChartOCR: Data Extraction from Charts Images via a Deep Hybrid Framework.**
> In *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*,
> pp. 1917–1925.

## Paper-text handling

We will cite ChartOCR in §2 (Related Work) and in the Table-1 caption /
Limitations note as a missing comparison due to the absence of a public
checkpoint. This follows the failure-mode that was planned for in
`MASTER_PLAN.md`.
