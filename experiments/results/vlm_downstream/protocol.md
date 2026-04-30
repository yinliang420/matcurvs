# VLM Downstream Evaluation Protocol

**Version**: v1 (pilot)
**Date**: 2026-04-25
**Owner**: chart-extraction project (`.`)
**Goal**: Show that polyline curves extracted by our pipeline (IoU=0.65 ceiling) provide **scientific-reasoning value** to a VLM, even when the extraction is imperfect — under a more forgiving evaluation than peak detection (recall=40%, currently too low).

---

## 0. Setup and shared infrastructure

### 0.1 Model
- **Single VLM**: `gemini-3.1-pro-preview` (already used for K-prediction; see `experiments/results/iter/k_full_label_gemini.csv`).
- **Backend**: Vivi gateway, OpenAI-compatible endpoint. Reuse the call pattern in `experiments/iter/llm_k_predictor.py:117-126`:
  ```python
  client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.viviai.cc/v1", timeout=60)
  resp = client.chat.completions.create(
      model="gemini-3.1-pro-preview",
      messages=[{"role":"user","content":[
          {"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{img_b64}"}},
          {"type":"text","text": prompt_text},
      ]}],
  )
  ```
- **Single multimodal call per question**. No tool use, no agentic loops.
- **Generation params**: `temperature=0` if accepted by gateway, else default. `max_tokens=256`.

### 0.2 Conditions

Both conditions consume the *same* image and the *same* question. They differ only by what's in the text prompt.

- **Condition A (Image-only baseline)**: image + question + parsing instruction.
- **Condition B (Image + Ours)**: image + question + parsing instruction + **structured curve(s)** extracted by our pipeline, serialized as JSON.

We do NOT compare against the GT-curve ceiling here — the goal is to show A < B with our IoU=0.65 extractor.

### 0.3 Curve serialization (Condition B)

For every line we extracted (one per `cluster_mask` from `pipeline.run_pipeline(...)`), produce a JSON block:

```json
{
  "axis_units": {"x": "<x_unit>", "y": "<y_unit>"},
  "x_range": [x_min, x_max],
  "y_range": [y_min, y_max],
  "curves": [
    {"id": "curve_1", "label": "<legend_text or null>", "n_points": 50,
     "points": [[x1,y1],[x2,y2], ..., [x50,y50]]},
    {"id": "curve_2", ...}
  ]
}
```

- **Resampling**: each cluster mask is converted to a single-valued curve via the existing `experiments/iter/reconstruct.py` workflow (median-y per x-bin), then resampled to **N=50 evenly spaced x positions** spanning the cluster's x extent.
- **Real units**: x and y converted from pixel to data-unit using the axis tick mapping (per-image `axis_range` available in `label/<id>_annotation.json` or the fixed file `label/axis_range_fixed/<id>_axis.json`; for raman/xafs, taken from the xlsx-supplied true curve when running the pilot, or from a separate `Axis_scale_extraction` pass for production).
- **Rounding**: 4 significant figures on x, 4 on y. Drop NaN points.
- **No legend label** is emitted unless we have one (we currently don't extract legend text reliably — leave `"label": null`).
- **Order of curves**: by descending pixel area of the cluster mask (matches how `pipeline.py:135` already sorts them).

The serialization function will live at `experiments/iter/curve_serializer.py` (see Section 6).

### 0.4 Pipeline extraction call

Use existing entrypoint:

```python
from pipeline import run_pipeline
pr = run_pipeline(image_path, n_lines=K_pred, separator_mode="v4",
                  morph_close_kernel=3, min_cluster_area=200)
# pr.cluster_masks: list[np.ndarray] — bool mask per detected curve
# pr.axis_lines, pr.legend_boxes, pr.m2f_fg_frac
```

`K_pred` comes from the gemini K-predictor we already have (`experiments/iter/llm_k_predictor.py`).

If `pr.ok=False` or `len(pr.cluster_masks)==0`, the image is **dropped from the eval** (logged with reason). We are NOT trying to model a fallback "no curve" condition for B in the pilot.

---

## 1. Per-dataset task choice

Pick one task per dataset. Each is chosen to be (a) more forgiving than peak detection at the level of **slope / location / count / category** rather than exact y-values at exact x-values, but (b) still scientifically meaningful.

### 1.1 `label/` (battery CV/Bode plots, 204 imgs, K=1–6)

**Task: Numerical y-lookup at sampled x positions.**

> Q: "What is the value of y at x = X*?  Reply with strict JSON: {\"y\": <number>}."

This is the most natural interrogation of a polyline. It is more forgiving than peak detection because:
- The VLM can give a *close* answer that scores well under ±10% tolerance even if it gets the exact pixel wrong.
- Every image yields multiple Q (no need for the chart to have peaks at all).
- Battery domain has wide y dynamic range, so Condition B's numerical info is genuinely useful.

We avoid line-matching ambiguity by **always asking about the longest curve** (the largest cluster_mask). For the pilot we also run a separate Q-set on the **shortest** curve, to test whether Condition B's per-line resolution beats Condition A.

### 1.2 `raman/` (1069 imgs, K=1, peak labels available)

**Task: Peak-presence in a window (yes/no).**

> Q: "Is there a Raman peak in the window [W_low, W_high] cm⁻¹? Reply with strict JSON: {\"peak\": \"yes\"|\"no\"}."

GT peaks are explicit (per-line `peaks[i].raman_shift_cm1`). For each image we sample windows: half are positive (some GT peak falls inside), half are negative (no GT peak inside, drawn from the gap regions).

This is more forgiving than peak counting / peak-localization because the VLM only has to commit to a binary call within a known x window. It IS still scientifically meaningful — chemists routinely check "does this spectrum have a peak around 1000 cm⁻¹?" when matching spectra to references.

### 1.3 `xafs/` (2202 imgs, no peak labels)

**Task: Edge-position classification (multiple choice).**

> Q: "Where does the absorption edge (the largest rising step) occur in this XAFS spectrum?
>     A) below 8000 eV   B) 8000–9000 eV   C) 9000–11000 eV   D) above 11000 eV.
>     Reply with strict JSON: {\"choice\": \"A\"|\"B\"|\"C\"|\"D\"}."

GT comes from the xlsx true curve: compute the x position of the largest first-derivative spike and bucket it into the four ranges. Buckets are chosen to cover common K-edges (Cr, Mn, Fe, Cu, Zn, Mo, Ag, etc.) so most images fall into one of B/C and we still get diversity. We re-bucket per image based on the actual edge energy distribution after sampling — not all four buckets need to be balanced; we just need ≥30 images per non-empty bucket.

This is much more forgiving than peak detection: the VLM only has to commit to one of 4 categories, and the "correct" bucket has a wide tolerance (~1000 eV). Yet getting it right is non-trivial without seeing the x-axis numerals clearly — exactly the regime where Condition B's serialized x-axis info should help.

We use **edge position** rather than chemical-formula classification because the formula label is not always derivable from the curve alone (depends on chemistry context).

---

## 2. Question generation

All Q+A pairs are pre-generated offline once, saved to JSONL, and reused for both A and B. Format:

```json
{"image_id": "<id>", "qid": "<id>_q03", "task": "label_y_lookup",
 "x_query": 42.7, "answer_gt": 218.4,
 "answer_format": "numeric", "metadata": {...}}
```

### 2.1 `label/` (numeric)

For each image with `n_gt_lines=K`:
- Load `label/<id>_annotation.json` → use `lines.Line_1.points[*].real` (the longest line by point count) as ground-truth polyline `P_gt`.
- Compute `x_min, x_max` from `P_gt`.
- Sample **N=8 query x positions** uniformly on `[x_min + 0.05*Δx, x_max - 0.05*Δx]` (avoid endpoints; many curves spike there).
- For each `x*`, linearly interpolate y on `P_gt` → `y*` is the GT answer.
- Skip x positions where the GT polyline has a vertical gap > 5 pixels (multivalued — common in CV plots) by checking that consecutive points within 5 pixels of `x*` differ by less than 30% of total y range.

**Total**: 204 imgs × 8 Qs ≈ 1632 Qs (we'll subsample, see Section 5).

### 2.2 `raman/` (yes/no)

Per image (K=1 always):
- Load the json metadata; extract GT peak list `peaks_gt = [p.raman_shift_cm1 for p in ...]`.
- Load xlsx true curve for `x_min, x_max`.
- Generate **N=6 windows per image**: 3 positive, 3 negative.
  - **Positive** window: pick GT peak at random; window = `[peak − 25, peak + 25]` cm⁻¹.
  - **Negative** window: 50-cm⁻¹-wide window where no GT peak lies within ±50 cm⁻¹ (gap regions). If <3 such gaps exist, generate fewer (logged).
- Answer: `"yes"` for positive, `"no"` for negative.

To prevent the VLM from gaming via window-width bias we use the same width for both classes.

**Total**: ~1069 × 6 = 6414 Qs (we'll subsample to 100 imgs ≈ 600 Qs).

### 2.3 `xafs/` (multiple choice, A–D)

Per image (most K=1, 87 K=2 — restrict to K=1 for pilot to keep edge well-defined):
- Load xlsx true curve `(x_i, y_i)`.
- Compute `dy/dx` via central differences.
- `x_edge = x[argmax|dy/dx|]`. Answer bucket: A/B/C/D as in Section 1.3.
- One question per image (a single edge per spectrum).

We add a sanity filter: if the max derivative is < 3× the median absolute derivative, the curve does not have a clear edge — drop the image. Logged.

**Total**: up to 2202 Qs after K=1 filter and edge-existence filter; we'll subsample to 100 imgs.

### 2.4 Answer parser

JSON-first, regex fallback. For each task the parser is small and explicit:

| task | parse | extract |
|---|---|---|
| `label_y_lookup` | `json.loads → d["y"]` | float |
| `raman_peak_window` | `d["peak"]` | str ∈ {yes,no} |
| `xafs_edge_bucket` | `d["choice"]` | str ∈ {A,B,C,D} |

If JSON parsing fails, regex-extract the first matching token (`yes|no`, `A|B|C|D`, or numeric). If still unparseable, mark the question as `parse_fail` and exclude from accuracy denominator (but report rate of parse_fail per condition — important for honesty: B may shift the parse-fail rate).

---

## 3. Prompt templates

All prompts begin with a 1-line task framing, then the question, then the strict-JSON instruction. Prompts are intentionally minimal — no chain-of-thought, no examples (zero-shot).

### 3.1 Condition A (Image-only)

**`label_y_lookup`**:
```
You are reading a scientific line chart. The image is provided.
QUESTION: What is the value of y on the LONGEST plotted curve at x = {x_query}?
The chart's x-axis is labeled "{x_label}" and its y-axis is "{y_label}".
The visible x range is approximately [{x_min}, {x_max}] and y range [{y_min}, {y_max}].
Reply with strict JSON, no extra text:
{"y": <number>}
```

**`raman_peak_window`**:
```
You are reading a Raman spectrum. The image is provided.
QUESTION: Is there a peak (a clear local maximum in intensity) anywhere in the
range [{w_low}, {w_high}] cm^-1?
Reply with strict JSON, no extra text:
{"peak": "yes" or "no"}
```

**`xafs_edge_bucket`**:
```
You are reading an XAFS absorption spectrum. The image is provided.
QUESTION: Where does the absorption edge (the largest rising step in y as x
increases) occur?
A) below 8000 eV   B) 8000-9000 eV   C) 9000-11000 eV   D) above 11000 eV
Reply with strict JSON, no extra text:
{"choice": "A"|"B"|"C"|"D"}
```

### 3.2 Condition B (Image + Ours)

For all three tasks, append after the question and before the JSON instruction:

```
We have also algorithmically extracted the curve(s) from this chart. The
extraction is approximate (~65% pixel-IoU vs ground truth) but axis units
are correct. Use it as a hint, but trust the image when they disagree.
EXTRACTED CURVES (JSON):
{<curve serialization from Section 0.3>}
```

The full B-prompt for `label_y_lookup` thus looks like:
```
You are reading a scientific line chart. The image is provided.
QUESTION: What is the value of y on the LONGEST plotted curve at x = 42.7?
The chart's x-axis is labeled "Time" and its y-axis is "Capacity".
The visible x range is approximately [0, 100] and y range [0, 300].

We have also algorithmically extracted the curve(s) from this chart. The
extraction is approximate (~65% pixel-IoU vs ground truth) but axis units
are correct. Use it as a hint, but trust the image when they disagree.
EXTRACTED CURVES (JSON):
{"axis_units":{"x":"Time","y":"Capacity"},"x_range":[5,85],
 "y_range":[17,293],"curves":[{"id":"curve_1","label":null,"n_points":50,
 "points":[[5.0,293.0],[6.6,288.4],...]}, ...]}

Reply with strict JSON, no extra text:
{"y": <number>}
```

### 3.3 Note on order
Always: **image first, then text**. Most multimodal Geminis condition stronger on the image when it appears before the text in the message; we want both A and B to share that.

---

## 4. Metrics

### 4.1 Per-question correctness

| task | correctness rule |
|---|---|
| `label_y_lookup` | `\|y_pred − y_gt\| / \|y_gt\| ≤ 0.10` (10% relative error). For |y_gt|<1, use abs error ≤ 0.05 of `(y_max−y_min)` |
| `raman_peak_window` | `pred_label == gt_label` |
| `xafs_edge_bucket` | `pred_choice == gt_choice` |

Also compute (where applicable):
- **MAE** and **RMSE** in real units for `label_y_lookup`.
- **Per-question latency** and **token usage** (for cost report).

### 4.2 Per-image / per-dataset aggregation

- **Image-level accuracy** = mean of per-question correctness within image.
- **Dataset-level accuracy** = mean over all questions in that dataset.

### 4.3 Effect size: B − A

Primary metric: **absolute accuracy gap `Δ = acc_B − acc_A`** at the dataset level. Report:
- Point estimate.
- **Bootstrap 95% CI** over **images** (not questions — questions within an image are not independent). 10000 resamples; cluster bootstrap (resample images, keep all their questions). Implement with `scipy.stats.bootstrap` or hand-written numpy.
- **Paired test**: Wilcoxon signed-rank on per-image (acc_B_image − acc_A_image), one-sided alt = "B > A". Report p-value but treat the bootstrap CI as the headline claim.

### 4.4 Stratified breakdowns

Slice the gap Δ by:

| dataset | strata |
|---|---|
| label | K (1–6 from `n_lines`); chart family inferred from filename or skip if unavailable |
| raman | window position bucket (low <500 cm⁻¹, mid 500–1500, high >1500), peak intensity from json (strong/medium/weak) |
| xafs | gt edge bucket (A/B/C/D); chemical-formula transition-metal family if extractable from `chemical_formulas` |

Strata with n<20 are reported but flagged "underpowered."

### 4.5 Honesty checks

Also report (separately, NOT folded into the headline number):
- **Parse-fail rate per condition.** If B's parse-fail rate is materially higher (>2 absolute pp), discuss possible "VLM choked on the long context" effects.
- **Latency / input-token delta**: how much more expensive is B vs A?
- **Cases where B *worsens* A**: per-image (acc_B − acc_A) histogram, isolating images where B made things worse. Inspect 10 worst manually.

---

## 5. Sample size + power

### 5.1 Pilot sizing (cost-aware)

Pilot: **100 images per dataset**, total 300 images.

| dataset | imgs | Q per img | total Q | calls (A+B) |
|---|---|---|---|---|
| label | 100 | 8 | 800 | 1600 |
| raman | 100 | 6 | 600 | 1200 |
| xafs | 100 | 1 | 100 | 200 |
| **total** | 300 | — | 1500 | **3000 calls** |

Estimated cost at gemini-3.1-pro-preview pricing (~1.5k input tokens image-only baseline + ~3k for B with 50 points × 2 curves average; ~250 output tokens):
- A calls: ~1.5k × $0.005/1k input + 0.25k × $0.015/1k output ≈ $0.011/call → ~$16
- B calls: ~3k × $0.005 + 0.25k × $0.015 ≈ $0.019/call → ~$28
- **Pilot total ≈ $45.** Under the $50 ceiling.

### 5.2 Detectable effect

For a paired binary outcome, MDE at 80% power, α=0.05 (two-sided), with ~600 questions and bootstrap-cluster CI on 100 images:

- If A baseline accuracy is ~50%, the MDE on Δ is roughly **±5 absolute pp** on label/raman, **±10 pp** on xafs (only 100 questions).
- xafs is intentionally underpowered in the pilot — we accept it as a feasibility check.

If pilot shows positive trend (Δ ≥ 3 pp on at least one task with CI excluding 0 or close), scale up to 200 imgs/dataset for the headline number.

### 5.3 Stopping rule

Stop pilot and reconsider if:
- Parse-fail rate in B > 15% (the curve serialization is breaking the model).
- Δ < −5 pp on any dataset (B actively *hurts* — implies extracted curve quality is too low and we may need to fix the pipeline before continuing).

---

## 6. Implementation plan

### 6.1 File layout

```
experiments/
├── iter/
│   ├── curve_serializer.py        # NEW: cluster_masks + axis -> JSON block
│   ├── vlm_qa.py                  # NEW: orchestrator (loads QA jsonl, runs A+B, caches)
│   ├── qa_generators/             # NEW
│   │   ├── gen_label_qa.py        # produce label/qa_pairs.jsonl
│   │   ├── gen_raman_qa.py
│   │   └── gen_xafs_qa.py
│   ├── llm_k_predictor.py         # existing — reuse for K_pred when needed
│   └── reconstruct.py             # existing — reuse for mask -> polyline
└── results/
    └── vlm_downstream/
        ├── protocol.md            # THIS FILE
        ├── qa/
        │   ├── label_qa.jsonl
        │   ├── raman_qa.jsonl
        │   └── xafs_qa.jsonl
        ├── extracted_curves/
        │   └── <image_id>.json    # cached pipeline outputs
        ├── responses/
        │   └── <task>_<cond>/<qid>.json   # one file per call
        ├── label_results.csv      # per-question (qid, cond, pred, gt, correct, latency, tokens)
        ├── raman_results.csv
        ├── xafs_results.csv
        └── summary.md             # auto-generated headline numbers + plots
```

### 6.2 Caching strategy

- **Per-question, per-condition cache**: response saved to `responses/<task>_<cond>/<qid>.json`. If the file exists, skip the call (key includes prompt hash to invalidate on prompt change).
- **Per-image extracted curve cache**: `extracted_curves/<image_id>.json` saved once; reused across all questions about that image.
- **Failure retries**: on transient API error (timeout, 5xx, rate-limit), retry up to 3 times with exponential backoff (1s, 4s, 16s). Persistent failures recorded with `error` field — do NOT silently skip.
- **Idempotency**: `vlm_qa.py` always safe to re-run; only fills gaps.
- **Concurrency**: process images in parallel up to 8 threads (single API key, gateway permits this). Per-thread retries.

### 6.3 Honest failure modes

When can B be **worse** than A?

1. **Garbage extraction.** If `pr.cluster_masks` is empty or wrong (we already log this at extraction time), B's prompt contains a curve that has nothing to do with the image. Mitigation: drop images where `pr.ok=False` from BOTH conditions before running the eval (so we're not biasing A). Risk: this also drops "hard" images, slightly inflating both A and B; we'll log how many.
2. **Wrong line picked.** For multi-curve images (label, K≥2), our `cluster_masks[0]` (largest by area) may not be the longest by x-extent; if the question targets "the longest curve" and we serialize all curves, the VLM may anchor to the wrong one. Mitigation: in B, label `"id":"curve_1"` as `"longest_by_area"`; instruct in the question "use the longest curve in the image".
3. **Axis-unit error.** Our axis extraction has its own error rate (~10%). When axis is wrong, B's serialized x/y values mislead the VLM. Mitigation: include the **GT-fixed `axis_range_new`** for the pilot to isolate "extraction quality" from "axis quality." Then run a second cell of the pilot with our predicted axis to measure the combined effect (this is the realistic deployment number).
4. **Long-context distraction.** 50-point JSON × 6 curves = ~700 numbers in the prompt; the VLM may be swayed by stale numbers and lose the visual cue. Mitigation: limit `n_points` to 30 if K ≥ 4 (reducing context bloat).
5. **Information leak.** B contains the answer for trivially-extractable questions ("what's y at x=42?"). For the pilot we accept this — it's actually the *point*: if our extraction encodes the answer, B should win, and that win is real scientific value. We make this explicit in the writeup. To avoid the trivial-leak critique we can also report results on the **shortest-curve y-lookup** (where extraction is noisier and the leak is weaker) as a secondary number.

### 6.4 Order of execution

1. Run `gen_*_qa.py` to produce QA jsonl files. Verify by hand on 5 random Qs per dataset.
2. Run pipeline extraction on all 300 images, cache curves. Verify the JSON serialization on 5 random imgs.
3. Run `vlm_qa.py --condition A --task label`. Check parse rate, sample 10 outputs.
4. Run B on label.
5. Repeat for raman, xafs.
6. Compute summary, write `summary.md`.

### 6.5 Reproducibility

- Random seed 42 throughout (sampling images, sampling Qs).
- `temperature=0`. We accept gemini-3.1-pro-preview is not perfectly deterministic at temp=0 via Vivi gateway; we run **2 repetitions** of each call and average accuracy if budget allows. For the pilot, single-shot is fine and we note this as a limitation.
- Log full prompt + raw response per call (for audit).

---

## 7. Honest predictions (gut check)

| dataset | task | predicted A | predicted B | predicted Δ (B−A) | Prob Δ ≥ +5 pp |
|---|---|---|---|---|---|
| label | y-lookup | 25% | 55% | **+30 pp** | 90% |
| raman | peak-window | 65% | 75% | **+10 pp** | 70% |
| xafs | edge-bucket | 70% | 80% | **+10 pp** | 60% |

**Reasoning**:

- **label is easiest for B to win.** The y-lookup answer is essentially in the serialized curve (information leak), and Gemini is mediocre at reading axis numerals from low-res charts. A baseline at 25% is my guess (a coin-flip on order-of-magnitude with some axis-reading skill); B should be near "interpolate the JSON" accuracy, capped by axis-extraction error and the 65% IoU of the curve itself.
- **raman is harder.** A is already decent (Gemini can see clear peaks); B helps mainly when a peak is small / near the noise floor / off-axis-tick. We expect a meaningful but smaller bump.
- **xafs is the hardest test of this protocol.** A is also decent — even a rough x-axis read narrows you to two buckets. B helps disambiguate "which side of 9000 eV." Our sample size limits the Δ we can resolve.

**Headline claim**: probability that **at least one** dataset shows B − A ≥ 5 pp with bootstrap 95% CI excluding 0 ≈ **85–90%**.

**Failure scenarios** to watch:
- If A on `label_y_lookup` is already 60% (Gemini reads charts better than I think) → Δ shrinks.
- If our extracted curves are systematically axis-shifted → B underperforms A. Run sanity-check eval on 10 imgs first.

---

## 8. Out of scope (explicitly)

- No multi-step / agentic VLM (single call only).
- No comparison to other VLMs in the headline (we have iter results suggesting Gemini is mid-pack; switching VLMs is a follow-up).
- No GT-curve ceiling number in the pilot (planned follow-up: "If we had perfect extraction, what would Δ be?").
- No cross-dataset transfer ("does the curve format help on charts the pipeline wasn't tuned for?").
- No human eval. Auto-judging only via the parsers in 2.4.
