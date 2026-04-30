# MatCurvs-Reasoning Benchmark

**Version**: 0.2 (2026-04-30, frozen for NeurIPS 2026 D&B submission)

A benchmark of agentic numerical-reasoning questions over real materials-science
line plots. Each question has a deterministic, scipy/numpy-computable
ground-truth answer derived from the panel's underlying real-coordinate
$(x,y)$ curves.

**Source**: All 3,817 panels in this benchmark are sampled exclusively from
the L2 tier of MatCurvs (i.e. semi-automatic MatDeplot extractions that
passed VLM-judge verification). External raman/xafs corpora that were
extracted by earlier or different pipelines are *deliberately excluded* so
that every panel here is provably derived from the same MatCurvs pipeline.

## Why this benchmark

Hosted vision–language models can describe a scientific chart in fluent prose
but cannot read its numerical values at instrument resolution. We claim that
once an extraction pipeline (e.g., MatDeplot) recovers each curve as
real-coordinate $(x,y)$ data, an agent armed with `numpy.interp` and
`scipy.signal.find_peaks` can answer numerical questions that VLMs cannot.
**MatCurvs-Reasoning measures that gap.**

## Composition

| | Rigorous set | Scaled set |
|---|---|---|
| Source | L1 manual pixel polylines | L2 VLM-judge-verified MatDeplot output |
| Panels | 49 | 3,817 |
| Questions | 188 | 14,740 |
| GT source | independent manual annotation | curated extraction pipeline |
| Role | head-line numbers across multiple VLMs | scale-out, single-VLM evaluation |

The 49 rigorous panels are the **L1 ∩ L2** intersection: panels for which we
have both (a) hand-traced pixel polylines (L1) and (b) MatDeplot output that
passed VLM-judge curation (L2). For these panels the rigorous-set GT is
computed from L1's manual real-coordinate points, so the comparison does not
use MatDeplot-derived data at any step. The scaled set covers all 3,817
L2-accepted panels and is intended for measuring VLM error at scale; on this
set the GT is computed from L2 directly.

## Coverage

```
Rigorous set
  panels                    : 49
  questions                 : 188
  q/panel distribution      : 3 generic + 0–2 family-specific + 0–1 C-Stack
  family distribution       : XRD 19 / Raman 9 / XPS 10 / XAFS 2 / generic 9

Scaled set
  panels                    : 3,817 (all L2-accepted)
  questions                 : 14,740
  q/panel coverage          : 100% (every panel has ≥3 questions)
  family distribution       :
    XRD     1,357   (35.6%)
    XPS     1,131   (29.6%)
    Raman     666   (17.4%)
    EIS       476   (12.5%)
    XAFS       89   ( 2.3%)
    generic    68   ( 1.8%)
    GCD/CV/Cycling 30 (0.8%)
```

## Schema

Each line of the JSONL is a single question item. Required fields:

| Field | Type | Notes |
|---|---|---|
| `set` | str | `"rigorous"` or `"scaled"` |
| `item_id` | str | Globally unique question ID |
| `panel` | str | Panel filename stem (links rigorous and scaled sets) |
| `battery`, `sha` | str | Source-paper identifiers (within MatCurvs corpus) |
| `image_path` | str | Absolute path to the chart .jpg |
| `l2_xlsx_paths` | list[str] | (scaled) Paths to MatDeplot extraction Excel(s) |
| `l1_json_path` | str | (rigorous) Path to manual annotation JSON |
| `fmt` | str | L2 format: `"A"` (one xlsx per curve) or `"B"` (paired-cols) |
| `n_l2_curves` / `n_l1_curves` | int | Number of curves in the source data |
| `main_curve_label` | str | Label of the primary curve used for A1/A2/A3 |
| `family` | str | Materials-science chart family (see below) |
| `x_label`, `y_label` | str | Axis labels (empty if source `.txt` not available) |
| `level` | str | Question category (see below) |
| `task` | str | Specific task identifier |
| `question` | str | Natural-language prompt — feed to VLM as-is |
| `gt_answer` | float \| list[float] | Ground-truth answer |
| `gt_source` | str | `"L1_manual"` or `"L2_curated"` |
| (task-specific) | various | E.g., `x_query`, `range_a/b`, `peak_x/y`, `baseline_lower/upper` |

## Question taxonomy

### Cat A — Generic numerical readout (3 questions per panel, all panels)

| Level | Task | What it measures | GT formula |
|---|---|---|---|
| A1 | `y_at_x` | Numerical interpolation precision | `np.interp(x₀, x, y)` on main curve |
| A2 | `argmax_in_range` | Local extremum localization | `argmax y` over a randomly chosen sub-range $[a, b]$ |
| A3 | `fwhm_main_peak` (preferred) | Half-width feature extraction | `scipy.signal.find_peaks` + linear interpolation at half-prominence |
| A3 | `y_amplitude` (fallback) | Peak-to-peak amplitude | `max(y) − min(y)` (used when no detectable peak) |

### Cat B — Per-family domain features (additional questions on selected families)

| Family | Task | GT formula |
|---|---|---|
| Raman | `raman_top3_peaks` | Top-3 peaks ranked by `find_peaks` prominence (ascending x) |
| Raman | `raman_id_ig_ratio` | $y(1350)/y(1580)$ via interpolation (only if x-range covers both) |
| EIS | `eis_high_freq_intercept` | $\min(Z')$ in the dataset (leftmost Nyquist point) |

### Cat C — Multi-curve post-processing (XRD/Raman/XPS/XAFS multi-curve panels)

| Task | What it measures | GT formula |
|---|---|---|
| `stacked_intensity_offset` | Whether the agent can quantify the constant intensity offset added between vertically stacked spectra (a publication convention) | $\text{baseline}(y_{\text{upper}}) - \text{baseline}(y_{\text{lower}})$, where $\text{baseline}(y) = \text{median}\bigl(y \le P_{25}(y)\bigr)$ |

This question is the strongest single piece of evidence that **a pipeline +
agent can post-process extracted curves in ways a VLM cannot**: VLMs can guess
"about 5 units" by visual inspection but cannot compute the offset to
instrument resolution. With the extracted (x, y), an agent's answer is exact
up to the extraction error.

## Family detection

Family labels are inferred from L0 (subfigure classification) using a
3-level priority:

1. **L0 reasoning text** (free-form LLM explanation) — pattern-match for
   `xrd`, `x-ray diffraction`, `raman shift`, `nyquist`, `electrochemical
   impedance`, `xanes`/`exafs`, `binding energy`, `galvanostatic`, etc.
2. **L0 `second_level` taxonomy** + x-range heuristic (`Structural
   Characterization` + x ∈ [4°, 90°] → XRD; same + x ∈ [80, 4500] cm⁻¹ →
   Raman; etc.).
3. **Pure x-range fallback** if no other signal.

Panels with no clear signal in any layer are tagged `generic` (1.8% of L2);
they receive only the 3 Cat-A questions, no family-specific extras.

Family-detection accuracy is not 100% — it inherits ~94.5% per-class accuracy
from the upstream L0 classifier, plus an additional fallback heuristic layer.
For the benchmark this is acceptable because the GT formulas (Cat A and
C-Stack) do not depend on family being correct; only the family-specific
Cat B tasks (Raman / EIS) require correct family routing.

## How to use

### As a VLM benchmark
For each item `it`:
```python
prompt = it["question"]
image  = it["image_path"]
answer_text = vlm(image, prompt)
answer_num  = parse_number(answer_text)        # caller's responsibility
err = abs(answer_num - it["gt_answer"])         # for scalar tasks
```
Aggregate by `level`, `task`, or `family`. Report MAE in the chart's physical
units (no normalization — answers are physically meaningful).

### As an agent + tool benchmark
For each item `it`, the agent loads the structured curves directly:
```python
xlsx = it["l2_xlsx_paths"]                     # (scaled set)
curves = parse_curves(xlsx, it["fmt"])
ans = scipy_or_numpy_compute(curves, it["task"], it.get("x_query"), ...)
```
For the rigorous set, `parse_l1_json(it["l1_json_path"])` is the agent's
input.

### Comparing the two
The paper's headline pairs:
- **VLM-only** vs **MatDeplot-output + scipy-agent**, on `rigorous` (49 panels,
  3 VLM families) — a strict GT-from-L1 comparison.
- **VLM-only** vs **same agent**, on `scaled` (3,817 panels, single VLM) — a
  scale claim that the VLM error is uniformly larger than the agent error
  even on the L2 GT.

## Reproducibility

The benchmark file is regenerated by:
```bash
python \
    experiments/mr_task_builder.py
```
The script is deterministic (seeded RNGs in `mr_task_builder.py:RNG`,
`build_rigorous_set:rng_local`, `build_scaled_set:rng_local`). Outputs:
```
experiments/mr_set_rigorous.jsonl
experiments/mr_set_scaled.jsonl
```

Inputs required:
- L1 annotations: `label/<stem>_annotation.json` (204 files)
- L2 acceptance log: `data_base_L2/judgment_log.jsonl`
- L2 accepted artifacts: `data_base_L2/<safe_battery>__<sha>/`
- L0 classification: `subfigures_and_class/classbattery_*.json` (55 files)

## Limitations

1. **GT for the scaled set comes from L2**, which is itself a
   semi-automatic curation of MatDeplot output. Method 2 (MatDeplot + agent)
   on the scaled set therefore measures *agent + extraction-error*, not
   agent error alone. The rigorous set (L1 GT) is the strict comparison.
2. **Family detection is heuristic** (~94% upstream accuracy, plus regex
   patterns on the LLM-generated reasoning text). Cat-B questions on
   misclassified family panels will look ill-formed; we report family
   detection accuracy alongside performance numbers.
3. **`y_amplitude` (A3 fallback)** is a strictly easier question than
   `fwhm_main_peak`; reports should subdivide A3 by these two tasks.
4. **`raman_id_ig_ratio`** requires the x-range to cover both 1350 and 1580
   cm⁻¹. ~70% of Raman panels qualify; the rest get only `raman_top3_peaks`
   as the family-specific question.
5. **Image paths are absolute** to the original release host. Downstream
   users must rebase these via a `data_root` argument in their loader.

## Versioning policy

Benchmark version `0.1` is frozen for the paper submission. Future updates:
- `0.2`: add Cat C (curve count, ranking) if reviewers request
- `0.3`: replace the "all-L2" GT path with manual-verification GT on a
  larger subset

## Citation

To be added on submission acceptance.
