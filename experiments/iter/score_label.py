"""Score label/ y-lookup eval.

Reads:
  - experiments/results/vlm_downstream/label_qa.json
  - experiments/results/vlm_downstream/llm_cache/<qid>__A.json, ...__B.json

Computes:
  - per-question correctness (±10% relative; abs threshold for tiny |y_gt|)
  - aggregate acc_A, acc_B, Δ
  - cluster bootstrap CI on Δ (resample images, B=10000)
  - per-K stratum breakdown
  - parse-fail rate per condition
  - 10 worst regressions

Writes:
  - label_results.csv
  - label_summary.md
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np
import os

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[2]))
VLM_DIR = KMEANS_ROOT / "experiments" / "results" / "vlm_downstream"
QA_PATH = VLM_DIR / "label_qa.json"
LLM_CACHE_DIR = VLM_DIR / "llm_cache"
RESULTS_CSV = VLM_DIR / "label_results.csv"
SUMMARY_MD = VLM_DIR / "label_summary.md"

REL_TOL = 0.10
ABS_FALLBACK_FRAC = 0.05  # for |y_gt|<1, abs ≤ 0.05*(y_max−y_min)
N_BOOT = 10000
SEED = 42


def correctness(y_pred: float | None, y_gt: float, y_min: float, y_max: float) -> int | None:
    if y_pred is None or not math.isfinite(y_pred):
        return None
    if not math.isfinite(y_gt):
        return None
    if abs(y_gt) >= 1.0:
        rel = abs(y_pred - y_gt) / max(abs(y_gt), 1e-9)
        return int(rel <= REL_TOL)
    # tiny |y_gt|: use abs error tolerance
    span = max(y_max - y_min, 1e-9)
    return int(abs(y_pred - y_gt) <= ABS_FALLBACK_FRAC * span)


def relerr(y_pred: float | None, y_gt: float) -> float | None:
    if y_pred is None or not math.isfinite(y_pred):
        return None
    return abs(y_pred - y_gt) / max(abs(y_gt), 1e-9)


def load_response(qid: str, cond: str) -> dict | None:
    p = LLM_CACHE_DIR / f"{qid}__{cond}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def main() -> None:
    if not QA_PATH.exists():
        print(f"QA missing: {QA_PATH}")
        return
    qa = json.loads(QA_PATH.read_text())

    rows: list[dict] = []
    parse_fail_A = 0; parse_fail_B = 0
    missing_A = 0; missing_B = 0

    for image_id, qs in qa.items():
        for q in qs:
            qid = q["qid"]
            ra = load_response(qid, "A")
            rb = load_response(qid, "B")
            if ra is None: missing_A += 1
            if rb is None: missing_B += 1
            ya = ra.get("y_pred") if ra else None
            yb = rb.get("y_pred") if rb else None
            pa = (ra.get("parse_method") == "fail") if ra else False
            pb = (rb.get("parse_method") == "fail") if rb else False
            parse_fail_A += int(pa)
            parse_fail_B += int(pb)

            ymin = q["y_min_curve"]; ymax = q["y_max_curve"]
            ca = correctness(ya, q["y_gt"], ymin, ymax)
            cb = correctness(yb, q["y_gt"], ymin, ymax)
            ea = relerr(ya, q["y_gt"]); eb = relerr(yb, q["y_gt"])

            rows.append({
                "image_id": image_id,
                "qid": qid,
                "x_star": q["x_star"],
                "y_gt": q["y_gt"],
                "y_pred_A": ya, "y_pred_B": yb,
                "correct_A": ca, "correct_B": cb,
                "abs_relerr_A": ea, "abs_relerr_B": eb,
                "K": q["n_lines"],
                "parse_A": ra.get("parse_method") if ra else "missing",
                "parse_B": rb.get("parse_method") if rb else "missing",
                "lat_A": ra.get("latency_s") if ra else None,
                "lat_B": rb.get("latency_s") if rb else None,
                "err_A": ra.get("error") if ra else "missing_response",
                "err_B": rb.get("error") if rb else "missing_response",
            })

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with RESULTS_CSV.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # aggregate
    valid_A = [r["correct_A"] for r in rows if r["correct_A"] is not None]
    valid_B = [r["correct_B"] for r in rows if r["correct_B"] is not None]
    n_q_total = len(rows)
    acc_A = mean(valid_A) if valid_A else 0.0
    acc_B = mean(valid_B) if valid_B else 0.0
    delta = acc_B - acc_A

    # cluster bootstrap by image
    rng = np.random.default_rng(SEED)
    by_img: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_img[r["image_id"]].append(r)
    img_ids = sorted(by_img.keys())

    def acc_for(samp_imgs: list[str], cond: str) -> float:
        # each sample: pool questions then compute mean correctness (None excluded)
        vals = []
        for iid in samp_imgs:
            for r in by_img[iid]:
                v = r["correct_A"] if cond == "A" else r["correct_B"]
                if v is not None:
                    vals.append(v)
        return mean(vals) if vals else 0.0

    deltas = []
    n_imgs = len(img_ids)
    for _ in range(N_BOOT):
        samp_idx = rng.integers(0, n_imgs, size=n_imgs)
        samp_imgs = [img_ids[i] for i in samp_idx]
        a = acc_for(samp_imgs, "A")
        b = acc_for(samp_imgs, "B")
        deltas.append(b - a)
    deltas_arr = np.array(deltas)
    ci_lo, ci_hi = np.percentile(deltas_arr, [2.5, 97.5])

    # paired Wilcoxon (per-image accuracy delta)
    paired = []
    for iid in img_ids:
        ra = [r["correct_A"] for r in by_img[iid] if r["correct_A"] is not None]
        rb = [r["correct_B"] for r in by_img[iid] if r["correct_B"] is not None]
        if not ra or not rb:
            continue
        paired.append((mean(rb) - mean(ra), iid))
    paired_diffs = [p[0] for p in paired]
    try:
        from scipy.stats import wilcoxon
        if any(d != 0 for d in paired_diffs):
            stat, pval = wilcoxon(paired_diffs, alternative="greater")
            wpval = float(pval)
        else:
            wpval = float("nan")
    except Exception:
        wpval = float("nan")

    # per-K breakdown
    per_k: dict[int, dict] = {}
    for K in sorted({r["K"] for r in rows}):
        subset = [r for r in rows if r["K"] == K]
        a = [r["correct_A"] for r in subset if r["correct_A"] is not None]
        b = [r["correct_B"] for r in subset if r["correct_B"] is not None]
        if not a or not b:
            continue
        per_k[K] = {
            "n_q": len(subset),
            "n_imgs": len({r["image_id"] for r in subset}),
            "acc_A": mean(a), "acc_B": mean(b),
            "delta": mean(b) - mean(a),
        }

    # 10 worst regressions per image
    worst_imgs = sorted(paired, key=lambda x: x[0])[:10]

    # latencies
    lat_A = [r["lat_A"] for r in rows if r.get("lat_A") is not None]
    lat_B = [r["lat_B"] for r in rows if r.get("lat_B") is not None]

    # write summary.md
    lines = []
    lines.append("# Label/ y-lookup VLM Downstream Summary")
    lines.append("")
    lines.append(f"**Model**: gemini-3.1-pro-preview  •  ±10% rel-error tolerance")
    lines.append(f"**Images**: {len(img_ids)}  •  **Questions**: {n_q_total}")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- **acc_A (image-only baseline)** = {acc_A:.3f}  (n_valid={len(valid_A)})")
    lines.append(f"- **acc_B (image + extracted curve)** = {acc_B:.3f}  (n_valid={len(valid_B)})")
    lines.append(f"- **Δ = acc_B − acc_A** = **{delta:+.3f}**")
    lines.append(f"- **95% bootstrap CI on Δ** = [{ci_lo:+.3f}, {ci_hi:+.3f}]  ({N_BOOT} resamples, cluster by image)")
    if not math.isnan(wpval):
        lines.append(f"- Wilcoxon paired (one-sided, B>A) p = {wpval:.4g}  (n={len(paired_diffs)} paired imgs)")
    lines.append("")
    lines.append("## Per-K breakdown")
    lines.append("")
    lines.append("| K | n_imgs | n_q | acc_A | acc_B | Δ |")
    lines.append("|---|---|---|---|---|---|")
    for K, s in sorted(per_k.items()):
        lines.append(f"| {K} | {s['n_imgs']} | {s['n_q']} | {s['acc_A']:.3f} | {s['acc_B']:.3f} | {s['delta']:+.3f} |")
    lines.append("")
    lines.append("## Failure modes & honesty checks")
    lines.append("")
    lines.append(f"- Parse-fail count: A={parse_fail_A}, B={parse_fail_B}  (rates {parse_fail_A/max(n_q_total,1):.3f} / {parse_fail_B/max(n_q_total,1):.3f})")
    lines.append(f"- Missing responses: A={missing_A}, B={missing_B}")
    if lat_A and lat_B:
        lines.append(f"- Median latency: A={median(lat_A):.2f}s  B={median(lat_B):.2f}s")
    lines.append(f"- Per-image B−A median = {median([p[0] for p in paired]):+.3f}")
    lines.append("")
    lines.append("### 10 worst image-level regressions (B−A)")
    lines.append("")
    lines.append("| image_id | per-img Δ |")
    lines.append("|---|---|")
    for d, iid in worst_imgs:
        lines.append(f"| {iid[:30]} | {d:+.3f} |")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    if ci_lo > 0:
        verdict = "**B > A with 95% CI excluding 0** — extracted curve provides measurable scientific-reasoning value on numerical y-lookup."
    elif delta > 0.05:
        verdict = "**B > A point estimate** but 95% CI overlaps 0 — directional support; expand sample to confirm."
    elif delta > 0:
        verdict = "Marginal positive Δ; not statistically distinguishable from null."
    else:
        verdict = "**B does not improve A** — re-examine extraction quality or task formulation."
    lines.append(verdict)
    lines.append("")
    SUMMARY_MD.write_text("\n".join(lines))

    print(f"Wrote {RESULTS_CSV}  ({n_q_total} rows)")
    print(f"Wrote {SUMMARY_MD}")
    print(f"acc_A={acc_A:.3f}  acc_B={acc_B:.3f}  Δ={delta:+.3f}  CI=[{ci_lo:+.3f},{ci_hi:+.3f}]")


if __name__ == "__main__":
    main()
