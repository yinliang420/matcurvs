# MatCurvs

Article-scale real-coordinate curve extraction for agent-ready materials reasoning.
NeurIPS 2026 (Datasets & Benchmarks track).

MatCurvs pairs (i) **MatDeplot**, an end-to-end pipeline that extracts per-curve
real-coordinate (x, y) trajectories from rendered scientific line charts, with
(ii) a **two-tier benchmark** (L1 manual pixel polylines on 204 chart panels,
L2 model-curated and human-audited curves at article scale) and (iii) a
**MatCurvs-Reasoning (MR)** benchmark that asks vision-language and
text-only LLMs quantitative questions over the extracted curves.

## Repository structure

```
.
├── pipeline.py                # canonical end-to-end extraction (axis YOLO -> Mask2Former -> KMeans)
├── mask2former_line.py        # MiT-B3 + 12-query Mask2Former line segmenter
├── Axis_yolo_extraction.py    # axis line detector (YOLOv11)
├── Axis_extraction.py         # legacy Hough/heuristic axis utilities
├── ModalLLMResponse.py        # OpenAI-compat clients for DashScope multimodal LLMs
├── label/                     # 204 manual pixel-polyline annotations (the L1 set)
├── experiments/               # benchmark builders, runners, scorers, iteration history
│   ├── mr_task_builder.py     # build the MR benchmark (rigorous + scaled)
│   ├── mr_runner.py           # run a VLM/LLM over MR (m1/m2/m3 reasoning modes)
│   ├── mr_scorer.py           # score MR outputs vs ground-truth answers
│   ├── MR_BENCHMARK_README.md # MR protocol / reproduction instructions
│   ├── iter/                  # per-iteration eval scripts (iter0..iter12)
│   └── results/               # per-iteration metrics CSVs (no large viz outputs)
├── subfigures_and_class/      # subfigure detection + chart-type classification training
├── paper/neurips2026_intro/   # LaTeX sources (figures and PDF released with the paper)
├── requirements.txt
├── .env.example               # template for vendor API keys + path overrides
└── LICENSE                    # MIT
```

The L1 set ships in this repo at `label/` (204 chart panels, each
`<sha>_<n>.jpg` paired with `<sha>_<n>_annotation.json`). The much larger L2
curated set lives on HuggingFace (link below) — not in the GitHub repo.

## Setup

```bash
# 1. Create an environment (python >= 3.9 recommended)
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Provide credentials (only needed for the LLM-K predictor and MR runner)
cp .env.example .env
# then edit .env and fill in any keys you have access to

# 4. Pull model checkpoints (not in git)
#   - Mask2Former line model: place at  models/mask2foremer_q12_d3_h256_best.pt
#   - YOLO axis weights:      place at  yolo/axis_lines_detect/best.pt
#   - YOLO legend weights:    place at  yolo/yolov11n_lenged_clutter_detector/weights/best.pt
#   These are released via HuggingFace and the GitHub release tag.
```

If your repo checkout sits at an unusual path, set `KMEANS_ROOT=/abs/path/to/checkout`
in `.env`. All scripts default to inferring it from `__file__`, so most users
don't need to touch this.

## Reproducing the headline numbers

The MR (MatCurvs-Reasoning) benchmark is the cleanest one-command reproduction:

```bash
# 1. Build the rigorous task set (writes mr_set_rigorous.jsonl)
python experiments/mr_task_builder.py --set rigorous

# 2. Run a VLM in image-only / image+data / text-only mode
python experiments/mr_runner.py --set rigorous --method m1 --model qwen-vl-max --workers 8
python experiments/mr_runner.py --set rigorous --method m2 --model gpt-5.4   --workers 4
python experiments/mr_runner.py --set rigorous --method m3 --model qwen3-max --workers 8

# 3. Score the answers against ground truth
python experiments/mr_scorer.py --set rigorous
```

See `experiments/MR_BENCHMARK_README.md` for the full protocol, the question
taxonomy (A1/A2/A3 difficulty levels, FWHM / argmax / y-at-x / etc. tasks), and
the per-model leaderboard.

For the **MatDeplot pipeline** itself (extraction quality, the IoU comparison
to Deplot/ChartOCR baselines on the L1-204 set), see `experiments/iter/` —
each `run_iter*.py` and `phase*` shell script pins one configuration and dumps
a `<iter>_full204_metrics.csv` into `experiments/results/iter/`. The current
best baseline is `iter12` (`v4 separator + multi-scale TTA + LAB-gated
dilation`).

A minimal smoke test on a handful of L1 charts (writes a 3-panel viz):

```bash
python pipeline.py --input_dir label --output_dir pipeline_out --limit 15
```

## Dataset on HuggingFace

The L2 curated set (model-extracted, then human-audited curves at article
scale) is too large to ship in git. It's hosted at:

- **L2 dataset:**  `https://huggingface.co/datasets/<your-namespace>/matcurvs-l2`
- **Model checkpoints:**  `https://huggingface.co/<your-namespace>/matcurvs-checkpoints`

(Replace `<your-namespace>` with the deposit URL once published.)

## Citation

If you use this code or the MatCurvs benchmark, please cite the NeurIPS 2026
paper:

```bibtex
@inproceedings{matcurvs2026,
  title     = {MatCurvs: Article-scale Real-coordinate Curve Extraction for
               Agent-ready Materials Reasoning},
  author    = {<author list TBD>},
  booktitle = {Advances in Neural Information Processing Systems (Datasets and
               Benchmarks Track)},
  year      = {2026}
}
```

## License

- **Code (this repository):** MIT — see [LICENSE](LICENSE).
- **MatCurvs-L1 / L2 benchmark data:** CC-BY-4.0, distributed via HuggingFace
  with full attribution to the source articles.

The third-party model checkpoints (Mask2Former, YOLOv11) inherit their
upstream licenses; please consult those projects for terms.
