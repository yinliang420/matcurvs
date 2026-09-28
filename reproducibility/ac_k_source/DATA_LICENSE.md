## Data and License Boundary

This capsule releases only the artefacts needed to recompute the AC clean-90
curve-count-source table:

- sanitized manual pixel annotations for 90 leakage-clean panels;
- frozen per-condition prediction masks and curve-count CSVs;
- source identifiers and image SHA-256 digests;
- the standalone evaluator and integrity manifest.

The capsule does **not** redistribute any source figure image. Every current
row in `sources.csv` is marked `image_redistribution_status=withheld`, meaning
the corresponding panel is represented only through author-created derived
annotations and frozen evaluator inputs.

Repository code remains under the repository root [LICENSE](../../LICENSE).
Unless a future version states otherwise on an item-by-item basis, the
author-created derived data in this directory is released under
Creative Commons Attribution 4.0 International (CC BY 4.0), with
attribution to the MatCurvs authors (see the repository README). This applies to
the sanitized annotation JSON files, frozen prediction archives, curve-count
CSVs, `sources.csv`, `condition_metadata.json`, `build_inputs.json`, the
expected evaluator outputs, and `MANIFEST.sha256`.

Third-party source figures are not part of that license grant. Their original
licenses remain with the rightsholders, and this capsule does not imply any
permission to redistribute withheld images. Reusers should not treat the
absence of a source image as evidence of source-image redistribution
permission.
