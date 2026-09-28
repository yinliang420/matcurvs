# MatCurvs

**MatCurvs: Article Real-Coordinate Curve Extraction for Agent-Ready Materials Reasoning**

**Accepted at NeurIPS 2026 — Main Conference**

Liang Yin · Zhan'ao Yao · Jiahui Shi · Songlin Yu · Jianjun Liu

[Dataset](https://huggingface.co/datasets/yinliang22/matcurvs) · [Reproducibility](reproducibility/ac_k_source/README.md) · [Training](training/README.md)

MatCurvs turns materials-science charts into numerical curve data for quantitative analysis. This repository provides **MatDeplot**, the extraction pipeline, together with model-training recipes and a reproducibility capsule. MatDeplot detects axes and legends, segments curve pixels, separates individual curves, and exports pixel and calibrated coordinates as JSON and Excel files.

## Dataset

**Data address: [huggingface.co/datasets/yinliang22/matcurvs](https://huggingface.co/datasets/yinliang22/matcurvs)**

The benchmark covers XRD, Raman, XPS, XAFS, EIS, GCD, and CV charts.

| Level | Contents | Scale reported in the paper |
|---|---|---|
| L0 | Classified scientific subfigures | 1,375,165 subfigures from 55,763 articles |
| L1 | Manual per-curve pixel annotations | 204 panels; 90-panel leakage-clean evaluation subset |
| L2 | Screened real-coordinate curves | 8,026 curves from 3,817 panels |

See the dataset page for files, access, and licensing. The in-repository capsule below can be evaluated independently of the full dataset and model weights.

## Installation

Use Python 3.10 or later and run commands from the repository root.

```bash
git clone https://github.com/yinliang420/matcurvs.git
cd matcurvs
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

PaddleOCR and its PaddlePaddle runtime are optional dependencies for reading axis ticks. MinerU is needed only for PDF input. The local Qwen curve-count runner has a separate GPU environment described below.

## Reproduce the released evaluation

The capsule contains **90 panels, 203 manually annotated curves, and six curve-count conditions**, including annotations, frozen prediction masks, count CSVs, splits, expected results, and an integrity manifest. It recomputes the released curve-count comparison without API calls, source images, or model weights.

For evaluation alone, the minimal dependencies are:

```bash
python -m pip install numpy scipy opencv-python
```

Run the evaluator with a new output directory:

```bash
python reproducibility/ac_k_source/evaluate.py \
  --capsule reproducibility/ac_k_source \
  --output outputs/clean90 \
  --check-expected
```

Success prints `evaluation_ok panels=90 gt_curves=203 rows=1218` followed by the output path. Results include `ac_k_source_table.csv`.

| Curve-count source | Recovered curves at IoU ≥ 0.5 | LR0.5 |
|---|---:|---:|
| Gemini, submitted configuration | 93 / 203 | 45.81% |
| GPT-5.4 | 91 / 203 | 44.83% |
| Qwen-VL-Max | 96 / 203 | 47.29% |
| Qwen3-VL-Plus | 94 / 203 | 46.31% |
| Local Qwen2.5-VL-7B-Instruct | 93 / 203 | 45.81% |
| Silhouette selection | 69 / 203 | 33.99% |

LR0.5 measures the fraction of ground-truth curves with a one-to-one matched prediction reaching pixel IoU ≥ 0.5; ground-truth polylines are rasterized at 5-pixel thickness. These results concern the fixed clean-90 comparison. Recomputing them from frozen masks does not rerun image extraction or the full reasoning benchmark. See the [capsule documentation](reproducibility/ac_k_source/README.md) for the protocol and data provenance.

## Extract curves from a chart

Provide the trained checkpoints in a model directory:

```text
model_directory/
├── axis_yolo11_best.pt
├── legend_yolo11_best.pt
├── mask2foremer_q12_d3_h256_best.pt
└── subfigure_yolo11x_best.pt          # document workflow only
```

Checkpoint filenames, sizes, and SHA-256 hashes are listed in [models/MANIFEST.json](models/MANIFEST.json). Weights are not bundled in this repository; extraction requires separately supplied checkpoints. The [training guide](training/README.md) provides the selected recipes and required input formats.

```bash
python scripts/run_chart.py \
  --image path/to/chart.png \
  --models path/to/model_directory \
  --output outputs/chart \
  --curve-count-source silhouette
```

Outputs are `extraction.json` and one `curve_001.xlsx`, `curve_002.xlsx`, … file per curve. Excel columns are `pixel_x`, `pixel_y`, `x`, and `y`. Add `--pixel-only` to skip axis calibration.

The [paper configuration](configs/paper.json) specifies foreground segmentation and LAB-XY KMeans separation. The submitted curve-count condition uses `gemini-3.1-pro-preview`; `silhouette` is a local alternative, not the headline configuration. Cached counts can be supplied with `--curve-count-source cached --curve-count-cache path/to/counts.csv`. Live Gemini counting uses `--curve-count-source gemini` and `GEMINI_API_KEY` or `GOOGLE_API_KEY` (optionally `GEMINI_BASE_URL`).

Axis detection can attempt CV+VLM recovery using `DASHSCOPE_API_KEY` when YOLO fails. Calibration metadata records OCR availability and fallback use. If synthetic ticks are used, `metadata.calibration.synthetic_ticks_used` is true: those coordinates are a pixel-derived fallback, not recovered physical axis units. See [failure and fallback behavior](docs/FAILURE_AND_FALLBACKS.md).

### Local Qwen curve counting

Use a compatible CUDA environment and the checkpoint recorded in [models/QWEN2_5_VL_7B_INSTRUCT.json](models/QWEN2_5_VL_7B_INSTRUCT.json).

```bash
python -m pip install -r requirements-qwen25.txt

python scripts/predict_qwen25_counts.py \
  --model-path path/to/Qwen2.5-VL-7B-Instruct \
  --image path/to/chart.png \
  --output-csv outputs/qwen_counts.csv

python scripts/run_chart.py \
  --image path/to/chart.png \
  --models path/to/model_directory \
  --output outputs/chart_qwen \
  --curve-count-source cached \
  --curve-count-cache outputs/qwen_counts.csv
```

The runner uses the shared prompt and greedy decoding, and records checkpoint hashes, input hashes, generation settings, and dependency versions alongside the count CSV. `--resume` checks that metadata before reusing results. FlashAttention-2 matches the recorded condition; `--attn-implementation sdpa` is a portability option.

## Process documents

With MinerU installed and a DashScope key configured:

```bash
export DASHSCOPE_API_KEY="your-key"
python scripts/run_document.py \
  --input path/to/article.pdf \
  --models path/to/model_directory \
  --output outputs/article \
  --curve-count-source silhouette
```

The same entry point accepts figure directories. For a directory of known line charts, add `--assume-line-charts` to bypass VLM routing. Add `--judge qwen-vl-max` for an optional extraction-quality check, or inspect an existing extraction:

```bash
python scripts/judge_extraction.py \
  --image path/to/chart.png \
  --extraction outputs/chart/extraction.json \
  --output outputs/chart/judge_verdict.json
```

Judge admission checks curve count, shape, and axis-range plausibility. It is distinct from ground-truth evaluation and does not establish extraction accuracy.

## Code and training

| Path | Purpose |
|---|---|
| `src/matdeplot/` | Detection, segmentation, curve separation, calibration, and export |
| `scripts/` | Chart, document, local Qwen, and quality-check entry points |
| `configs/paper.json` | Selected extraction configuration |
| `models/` | Checkpoint metadata |
| `training/` | YOLO and foreground-model training code and recipes |
| `reproducibility/ac_k_source/` | Clean-90 evaluation inputs, evaluator, and expected outputs |
| `tests/` | Extraction and training regression tests |

See [training/README.md](training/README.md) for training commands, data formats, and configuration details. Training requires the corresponding datasets and base weights; these are not included in the Git repository.

Run the regression tests after installation:

```bash
python -m unittest discover -s tests
```

## Citation

```bibtex
@inproceedings{yin2026matcurvs,
  title     = {{MatCurvs}: Article Real-Coordinate Curve Extraction for Agent-Ready Materials Reasoning},
  author    = {Yin, Liang and Yao, Zhan'ao and Shi, Jiahui and Yu, Songlin and Liu, Jianjun},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

Code is released under the [MIT License](LICENSE). The capsule's author-created annotations and frozen predictions are covered by its [data license](reproducibility/ac_k_source/DATA_LICENSE.md). Source figure images are not redistributed in the capsule; consult the dataset page for the full dataset's terms.
