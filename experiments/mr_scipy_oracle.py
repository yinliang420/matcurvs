"""
M4 oracle: re-apply each task's GT formula to the L2 extracted (x,y) data.

For rigorous items (GT is from L1 manual annotation), M4 measures the
extraction-error ceiling — i.e., how close MatDeplot's extracted curves
are to the manual L1 reference.

For scaled items (GT is itself computed from L2 (x,y)), M4 is trivially
equal to GT and serves only as a sanity check.

Output schema matches mr_runner.py for use by mr_scorer.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import os

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[1]))
EXP_DIR = KMEANS_ROOT / "experiments"
ANS_DIR = EXP_DIR / "mr_answers"

sys.path.insert(0, str(EXP_DIR))
from mr_runner import parse_curves_from_paths


def pick_main_curve(curves):
    return max(curves, key=lambda c: len(c[1])) if curves else None


def find_curve(curves, label):
    for c in curves:
        if c[0] == label:
            return c
    return None


def fwhm_main_peak(x, y):
    if len(x) < 10:
        return None
    if np.nanmax(y) - np.nanmin(y) <= 0:
        return None
    peaks, props = find_peaks(y, prominence=(np.nanmax(y) - np.nanmin(y)) * 0.05)
    if len(peaks) == 0:
        return None
    pk = int(np.argmax(props["prominences"]))
    main = peaks[pk]
    half_height = y[main] - props["prominences"][pk] / 2.0
    li = main
    while li > 0 and y[li] > half_height:
        li -= 1
    ri = main
    while ri < len(y) - 1 and y[ri] > half_height:
        ri += 1
    if li == 0 or ri == len(y) - 1:
        return None
    x_left = float(np.interp(half_height, [y[li], y[li + 1]], [x[li], x[li + 1]]))
    x_right = float(np.interp(half_height, [y[ri], y[ri - 1]], [x[ri], x[ri - 1]]))
    fwhm = x_right - x_left
    return float(fwhm) if (np.isfinite(fwhm) and fwhm > 0) else None


def compute_answer(it: dict, curves: list) -> dict:
    task = it["task"]
    label = it.get("curve_label") or it.get("main_curve_label") or (curves[0][0] if curves else None)
    main = find_curve(curves, label) or pick_main_curve(curves)
    if main is None:
        return {"answer": None, "error": "no_curve"}

    _, x, y = main
    try:
        if task == "y_at_x":
            x0 = float(it["x_query"])
            return {"answer": float(np.interp(x0, x, y))}
        if task == "argmax_in_range":
            a, b = float(it["range_a"]), float(it["range_b"])
            mask = (x >= a) & (x <= b)
            if mask.sum() < 1:
                return {"answer": None, "error": "no_points_in_range"}
            return {"answer": float(x[mask][int(np.argmax(y[mask]))])}
        if task == "fwhm_main_peak":
            v = fwhm_main_peak(x, y)
            return {"answer": v, "error": None if v is not None else "no_peak"}
        if task == "y_amplitude":
            return {"answer": float(np.nanmax(y) - np.nanmin(y))}
        if task == "raman_top3_peaks":
            if len(x) < 20:
                return {"answer": None, "error": "too_short"}
            peaks, props = find_peaks(y, prominence=(np.nanmax(y) - np.nanmin(y)) * 0.05)
            if len(peaks) == 0:
                return {"answer": None, "error": "no_peaks"}
            order = np.argsort(-props["prominences"])
            top = peaks[order][:3]
            return {"answer": sorted([float(x[i]) for i in top])}
        if task == "raman_id_ig_ratio":
            yD = float(np.interp(1350.0, x, y))
            yG = float(np.interp(1580.0, x, y))
            if yG <= 0:
                return {"answer": None, "error": "yG<=0"}
            return {"answer": yD / yG}
        if task == "eis_high_freq_intercept":
            return {"answer": float(np.nanmin(x))}
        if task == "stacked_intensity_offset":
            if len(curves) < 2:
                return {"answer": None, "error": "single_curve"}
            cs = sorted(curves, key=lambda c: float(np.nanmedian(c[2])))
            (_, _, y_lo), (_, _, y_hi) = cs[0], cs[1]
            def baseline(yy):
                thr = np.nanpercentile(yy, 25)
                bot = yy[yy <= thr]
                return float(np.nanmedian(bot)) if len(bot) > 0 else float(np.nanmin(yy))
            return {"answer": baseline(y_hi) - baseline(y_lo)}
    except Exception as e:
        return {"answer": None, "error": f"{type(e).__name__}: {e}"}
    return {"answer": None, "error": f"unknown_task:{task}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["rigorous", "scaled"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    set_path = EXP_DIR / f"mr_set_{args.set}.jsonl"
    items = [json.loads(l) for l in open(set_path)]
    if args.limit:
        items = items[: args.limit]

    out_path = ANS_DIR / f"{args.set}_m4_scipy.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    n_ok = n_err = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            xlsx = it.get("l2_xlsx_paths") or []
            fmt = it.get("fmt") or "A"
            curves = parse_curves_from_paths(xlsx, fmt)
            res = compute_answer(it, curves)
            res["item_id"] = it["item_id"]
            res["model"] = "matdeplot+scipy"
            res["method"] = "m4"
            res["gt_answer"] = it["gt_answer"]
            res["task"] = it["task"]
            res["family"] = it.get("family", "")
            res["level"] = it["level"]
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            if res.get("answer") is None:
                n_err += 1
            else:
                n_ok += 1
    print(f"[done] ok={n_ok} err={n_err} | output -> {out_path}")


if __name__ == "__main__":
    main()
