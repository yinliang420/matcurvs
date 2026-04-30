"""Collapse per-cell CSVs into summary markdown tables for A2/A6."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import os

RESULTS_DIR = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[1])) / "experiments" / "results"


def _df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = df.astype(object).where(pd.notnull(df), "").values.tolist()
    lines = ["| " + " | ".join(map(str, cols)) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(lines)


def summary_row(df: pd.DataFrame) -> dict:
    fb_frac = float(df.get("used_fallback", pd.Series([False])).mean())
    return {
        "n": len(df),
        "DTW": round(float(df["dtw_delta"].mean()), 3),
        "CCR": round(float(df["ccr_at_tau"].mean()), 3),
        "InstF1": round(float(df["instance_f1"].mean()), 3),
        "PQ": round(float(df["curve_pq"].mean()), 3),
        "fb_frac": round(fb_frac, 3),
    }


def a2_summary() -> str:
    cells = {
        "thr0.001": "ours_kmeans_m2f_thr0001",
        "thr0.003 (default)": "ours_kmeans_m2f",
        "thr0.01": "ours_kmeans_m2f_thr001",
        "thr0.03": "ours_kmeans_m2f_thr003",
        "thr0.1": "ours_kmeans_m2f_thr01",
        "thr0.3": "ours_kmeans_m2f_thr03",
        "thr0.99 (always fallback)": "ours_kmeans_m2f_fallback",
    }
    rows = []
    for label, cell in cells.items():
        p = RESULTS_DIR / f"{cell}.csv"
        if not p.exists():
            rows.append({"threshold": label, "status": "MISSING"})
            continue
        df = pd.read_csv(p)
        s = summary_row(df)
        s["threshold"] = label
        rows.append(s)
    out = pd.DataFrame(rows)
    cols = ["threshold", "n", "DTW", "CCR", "InstF1", "PQ", "fb_frac"]
    out = out[[c for c in cols if c in out.columns]]
    md = "## A2 Fallback-trigger threshold sweep\n\n"
    md += _df_to_md(out) + "\n"
    return md


def a6_summary() -> str:
    cells = {
        "clean (default)": "ours_kmeans_m2f",
        "rotation ±5°": "ours_robust_rot5",
        "JPEG q=30": "ours_robust_jpeg30",
        "Gaussian σ=5": "ours_robust_noise",
    }
    rows = []
    for label, cell in cells.items():
        p = RESULTS_DIR / f"{cell}.csv"
        if not p.exists():
            rows.append({"perturbation": label, "status": "MISSING"})
            continue
        df = pd.read_csv(p)
        s = summary_row(df)
        s["perturbation"] = label
        rows.append(s)
    out = pd.DataFrame(rows)
    cols = ["perturbation", "n", "DTW", "CCR", "InstF1", "PQ"]
    out = out[[c for c in cols if c in out.columns]]
    md = "## A6 Robustness under perturbation\n\n"
    md += _df_to_md(out) + "\n"
    return md


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["a2", "a6", "all"], default="all")
    args = ap.parse_args()
    if args.which in ("a2", "all"):
        md = a2_summary()
        (RESULTS_DIR / "A2_threshold_sweep.md").write_text(md)
        print(md)
    if args.which in ("a6", "all"):
        md = a6_summary()
        (RESULTS_DIR / "A6_robustness.md").write_text(md)
        print(md)
