# Model Training

This directory contains the paper-aligned training entry points and selected
configurations for three YOLO detectors and the Mask2Former-style foreground
model. Supply the dataset roots, annotations, pretrained base weights, and
output directories required by each command. Model weights, training data,
split manifests, cached masks, logs, and run outputs are not included.

## Selected provenance

The YAML files under `training/yolo/configs/` and
`training/foreground/configs/` record the selected configurations.
`training/MANIFEST.json` and `models/MANIFEST.json` record the corresponding
checkpoint names and SHA-256 digests.

| Component | Selected configuration | Checkpoint SHA-256 |
| --- | --- | --- |
| Axis detector | YOLO11x, 100 epochs, batch 16, 640 px, seed 0 | `cfec499d888fdf28f4475b2f2a040fc6a3275d2b0c34fabee6d9ebe816512569` |
| Legend/clutter detector | YOLO11x, 500 epochs, batch 32, 640 px, seed 0 | `9e1bbdfcabeff45a0faa19eb276b97a96d9ba3f96b0f561eae48832df0403310` |
| Subfigure detector | YOLO11x, 200 epochs, batch 32, 640 px, seed 0 | `ae40f5e7fab0a10d8b56537e301f90ff81b3204ebd2535359f00b5d98cb37742` |
| Foreground model | MiT-B3, hidden dim 256, 12 queries, 3 decoder layers, seed 42 | `847aba64be1507c9061c98756d396721a8ff1e47f0d1a9bac41e3af02eee17a9` |

Exact checkpoint reproduction also requires the original training data,
upstream base weights, and a compatible software environment.

## Run the selected configurations

Replace every `path/to/...` argument with the corresponding local path.

```bash
python -m training.yolo.train \
  --recipe training/yolo/configs/axis.yaml \
  --dataset path/to/axis.dataset.yaml \
  --base-model path/to/yolo11x.pt \
  --output path/to/axis_runs \
  --device 0

python -m training.yolo.train \
  --recipe training/yolo/configs/legend_clutter.yaml \
  --dataset path/to/legend_clutter.dataset.yaml \
  --base-model path/to/yolo11x.pt \
  --output path/to/legend_clutter_runs \
  --device 0

python -m training.yolo.train \
  --recipe training/yolo/configs/subfigure.yaml \
  --dataset path/to/subfigure.dataset.yaml \
  --base-model path/to/yolo11x.pt \
  --output path/to/subfigure_runs \
  --device 0

python -m training.foreground.train \
  --config training/foreground/configs/paper_q12_d3_h256.yaml \
  --synthetic-dir path/to/synthetic_chart_root \
  --real-dir path/to/real_chart_root \
  --axis-model path/to/axis_yolo11_best.pt \
  --legend-model path/to/legend_yolo11_best.pt \
  --output-dir path/to/foreground_run \
  --device cuda:0
```

The YOLO commands require an upstream `yolo11x.pt` base model supplied by the
user. The foreground command requires user-supplied axis and legend checkpoints
only for the real-mask preprocessing path selected by
`paper_q12_d3_h256.yaml`. Those weights are dependencies, not release
artifacts.

## Input contracts

### YOLO dataset YAML

Each `training/yolo/datasets/*.dataset.yaml.example` file shows the required
schema:

- `train`: relative or absolute image root for the training split.
- `val`: relative or absolute image root for the validation split.
- `nc`: number of classes for the selected role.
- `names`: class names matching the selected recipe exactly.

The on-disk layout follows standard YOLO detection conventions with
`images/{train,val}` and `labels/{train,val}` plus normalized
`class x_center y_center width height` rows.

### Foreground data

The foreground loader expects two user-supplied roots:

- A synthetic chart root with image/annotation pairs that can be rasterized to
  binary foreground masks.
- A real chart root following the released coordinate-annotation schema used by
  the runtime pipeline.

The loader constructs deterministic splits in code. No training samples, masks,
split manifests, or serialized dataset exports are bundled in this release.

## Objective and threshold distinctions

The released foreground trainer exposes the selected V11 binary-mask-only
objective. The model still contains objectness and auxiliary heads, but the
selected criterion supervises the union of query masks rather than those heads.

Several similarly named quantities in the paper pipeline are intentionally
distinct:

- `line_width=3` is the training-time raster width used when building binary
  masks for the released foreground recipe.
- `binary_threshold=0.5` is the checkpoint-selection threshold used during
  released foreground validation.
- `foreground.threshold=0.25` is the paper inference-time foreground threshold
  used by the extraction pipeline, not by the training loop.
- Evaluator raster `thickness=5` belongs to the released metric code and is
  separate from both training rasterization and inference thresholding.

## Outputs and scope

The commands write checkpoints, metrics, and framework metadata only to the
user-supplied output directories; these files are not tracked by the repository.
Because the training datasets, pretrained base weights, and selected checkpoint
weights are not included, this release supports implementation inspection and
configuration reuse, not turnkey reproduction of the trained checkpoints.

The repository code is MIT licensed. Users remain responsible for the licenses
and permissions governing any data or pretrained weights they supply.
