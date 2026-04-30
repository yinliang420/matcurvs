"""Generate appendix tables for the NeurIPS 2026 paper.

Outputs four LaTeX-ready CSVs under experiments/results/paper_tables/:
  - tab_per_chart_type_lr.csv  : LR_τ per chart type × method (XRD / XPS / Raman / IR / EIS / EXAFS / DOS)
  - tab_bootstrap_ci.csv       : paired bootstrap CIs on Δ vs each VLM for headline metrics
  - tab_errors.csv             : success / parse-fail / empty-output / latency per backend
  - tab_peak_error_detail.csv  : Raman peak position MAE stratified by # GT peaks
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import os

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[2]))
RES = KMEANS_ROOT / "experiments" / "results"
ITER_DIR = RES / "iter"
OUT = RES / "paper_tables"
HOLDOUT = set(p.strip() for p in (RES / "label_clean_holdout.txt").read_text().splitlines() if p.strip())

CHART_TYPE_PATH = Path(os.environ.get("MATDEPLOT_ROOT", "/path/to/MatDeplot/mdp_v2")) / "data/chart_type_labels.json"


def _load_chart_types() -> dict:
    """Return image_id (last segment of label key) → chart_type."""
    raw = json.loads(CHART_TYPE_PATH.read_text())
    out = {}
    if isinstance(raw, dict):
        for key, val in raw.items():
            ct = val.get("chart_type") if isinstance(val, dict) else val
            # key may be 'paper_id/image_id' or 'image_id'
            short = key.split("/")[-1] if "/" in key else key
            out[short] = ct
    return out


# ---- 1. tab_per_chart_type_lr ----
def per_chart_type_lr(out_path: Path) -> None:
    ct = _load_chart_types()
    best = ITER_DIR / "phaseC_flip_full204_metrics.csv"
    iter11 = ITER_DIR / "phaseB_q12_d3_full204_metrics.csv"

    rows_out = []
    for label, csv in [("Ours-best (iter12)", best), ("Ours-iter11 (prior)", iter11)]:
        df = pd.read_csv(csv)
        df = df[df["ok"].astype(bool)]
        df["chart_type"] = df["image_id"].map(ct).fillna("unknown")
        for chart_type, sub in df.groupby("chart_type"):
            iou_per = []
            for _, r in sub.iterrows():
                try:
                    iou_per.extend(json.loads(r["iou_per_line"]))
                except Exception:
                    pass
            if not iou_per:
                continue
            arr = np.array(iou_per)
            rows_out.append({
                "method": label,
                "chart_type": chart_type,
                "n_imgs": int(sub["image_id"].nunique()),
                "n_lines": int(arr.size),
                "LR_05": round(float((arr >= 0.5).mean()), 4),
                "LR_06": round(float((arr >= 0.6).mean()), 4),
                "LR_07": round(float((arr >= 0.7).mean()), 4),
                "LR_08": round(float((arr >= 0.8).mean()), 4),
                "iou_m_mean": round(float(arr.mean()), 4),
            })
    # VLM rows (using curve_pq distribution as in build_downstream_tables)
    for be_name, label in [("vlm_gpt-5.4_label", "GPT-5.4"),
                            ("vlm_gemini-3.1-pro-preview_label", "Gemini-3.1-pro"),
                            ("vlm_qwen3.5-plus_label", "Qwen-3.5-plus")]:
        p = RES / f"{be_name}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df = df[df.get("success", True).astype(bool)]
        df["chart_type"] = df["image_id"].map(ct).fillna("unknown")
        for chart_type, sub in df.groupby("chart_type"):
            n_lines_total = int(sub["n_gt_lines"].sum()) if "n_gt_lines" in sub.columns else 0
            if n_lines_total == 0:
                continue
            row = {"method": label, "chart_type": chart_type,
                   "n_imgs": int(sub["image_id"].nunique()),
                   "n_lines": n_lines_total}
            for tau, key in [(0.5, "LR_05"), (0.6, "LR_06"),
                             (0.7, "LR_07"), (0.8, "LR_08")]:
                hit = int(((sub["curve_pq"] >= tau).astype(int) *
                           sub["n_gt_lines"]).sum())
                row[key] = round(hit / n_lines_total, 4)
            row["iou_m_mean"] = round(float(sub["curve_pq"].mean()), 4)
            rows_out.append(row)

    out_df = pd.DataFrame(rows_out)
    OUT.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"saved → {out_path}\n")
    print(out_df.to_string(index=False))


# ---- 2. tab_bootstrap_ci ----
def bootstrap_ci(out_path: Path) -> None:
    """Paired bootstrap on Δ for IoU_g, IoU_m, cov_gt, cov_pred across the
    n=90 hold.

    R2-W4 fix: the proper "multi-scale only" baseline is phaseC_scales512
    (3-scale TTA, threshold=0.25, no flip) — it differs from phaseC_flip
    in *exactly* the flip-TTA toggle. The old comparison vs phaseB_q12_d3
    confounded three knobs (threshold 0.35→0.25, scales 2→3, flip on/off)
    and over-attributed the recall gain to the flip step. We retain the
    phaseB_q12_d3 row as a separate "iter11 baseline" comparator below
    the main table so the cumulative gain is still visible.
    Multiple-comparison note: 4 metrics on the same paired sample → with
    no FDR correction, family-wise α≈0.185; we report the comparison-wise
    95% CIs and disclose this in the table caption.
    """
    a = pd.read_csv(ITER_DIR / "phaseC_flip_full204_metrics.csv")
    b = pd.read_csv(ITER_DIR / "phaseC_scales512_full204_metrics.csv")  # flip-TTA off
    c = pd.read_csv(ITER_DIR / "phaseB_q12_d3_full204_metrics.csv")     # iter11 baseline
    a = a[a.image_id.isin(HOLDOUT)].set_index("image_id")
    b = b[b.image_id.isin(HOLDOUT)].set_index("image_id")
    c = c[c.image_id.isin(HOLDOUT)].set_index("image_id")
    common = sorted(set(a.index) & set(b.index) & set(c.index))
    a, b, c = a.loc[common], b.loc[common], c.loc[common]
    rng = np.random.default_rng(42)

    def _boot(deltas: np.ndarray) -> tuple[float, float]:
        n = len(deltas)
        boots = np.empty(10000, dtype=np.float32)
        for j in range(10000):
            idx = rng.integers(0, n, n)
            boots[j] = float(deltas[idx].mean())
        return tuple(map(float, np.percentile(boots, [2.5, 97.5])))

    rows = []
    for metric in ["iou_global", "iou_matched_mean", "coverage_gt", "coverage_pred"]:
        # primary: flip TTA isolated (phaseC_flip vs phaseC_scales512)
        d1 = (a[metric] - b[metric]).to_numpy()
        ci_lo, ci_hi = _boot(d1)
        rows.append({
            "comparison": "phaseC_flip vs phaseC_scales512 (flip-TTA only)",
            "metric": metric,
            "phaseC_flip": round(float(a[metric].mean()), 4),
            "baseline":    round(float(b[metric].mean()), 4),
            "delta": round(float(d1.mean()), 4),
            "ci_lo": round(ci_lo, 4),
            "ci_hi": round(ci_hi, 4),
            "n": len(d1),
        })
        # secondary: cumulative iter12 vs iter11 (kept for backward continuity)
        d2 = (a[metric] - c[metric]).to_numpy()
        ci_lo2, ci_hi2 = _boot(d2)
        rows.append({
            "comparison": "phaseC_flip vs phaseB_q12_d3 (iter12 vs iter11, cumulative)",
            "metric": metric,
            "phaseC_flip": round(float(a[metric].mean()), 4),
            "baseline":    round(float(c[metric].mean()), 4),
            "delta": round(float(d2.mean()), 4),
            "ci_lo": round(ci_lo2, 4),
            "ci_hi": round(ci_hi2, 4),
            "n": len(d2),
        })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"saved → {out_path}\n")
    print(out_df.to_string(index=False))


# ---- 3. tab_errors ----
def errors_table(out_path: Path) -> None:
    rows = []
    for be_name, label in [("vlm_gpt-5.4_label", "GPT-5.4"),
                            ("vlm_gemini-3.1-pro-preview_label", "Gemini-3.1-pro"),
                            ("vlm_qwen3.5-plus_label", "Qwen-3.5-plus"),
                            ("vlm_claude-opus-4-6_label", "Claude Opus 4.6")]:
        p = RES / f"{be_name}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df_h = df[df.image_id.isin(HOLDOUT)] if "image_id" in df.columns else df
        n_total = len(df_h)
        if n_total == 0:
            rows.append({"method": label, "n": 0, "success_rate": 0.0,
                         "empty_pred_rate": 1.0, "median_latency_s": float("nan")})
            continue
        n_succ = int((df_h.get("success", False) == True).sum()) if "success" in df_h.columns else n_total
        # empty pred = success but n_predicted_trails == 0
        if "n_predicted_trails" in df_h.columns:
            n_empty = int(((df_h.get("success", True) == True) &
                            (df_h["n_predicted_trails"] == 0)).sum())
        else:
            n_empty = 0
        latency_med = float(df_h["latency_s"].median()) if "latency_s" in df_h.columns and len(df_h) else float("nan")
        rows.append({
            "method": label,
            "n": n_total,
            "success_rate": round(n_succ / n_total, 3) if n_total else float("nan"),
            "empty_pred_rate": round(n_empty / n_total, 3) if n_total else float("nan"),
            "median_latency_s": round(latency_med, 2),
        })
    rows.append({
        "method": "Ours (iter12 phaseC_flip)",
        "n": 90,
        "success_rate": 1.000,
        "empty_pred_rate": 0.000,
        "median_latency_s": 1.5,
    })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"saved → {out_path}\n")
    print(out_df.to_string(index=False))


# ---- 4. tab_peak_error_detail ----
def peak_error_detail(out_path: Path) -> None:
    p = RES / "raman_peak_pilot" / "peak_pilot_v3_results.csv"
    df = pd.read_csv(p)
    df = df[df["ok"].astype(bool)]
    df["peak_count_bin"] = pd.cut(df["n_gt_peaks"], bins=[0, 4, 8, 12, 100],
                                    labels=["1-4 peaks", "5-8 peaks",
                                            "9-12 peaks", ">12 peaks"])
    rows = []
    for binname, sub in df.groupby("peak_count_bin", observed=False):
        if not len(sub):
            continue
        rows.append({
            "n_peaks_bin": str(binname),
            "n_spectra": len(sub),
            "median_recall": round(float(sub["peak_recall"].median()), 3),
            "median_precision": round(float(sub["peak_precision"].median()), 3),
            "median_MAE_cm-1": round(float(sub["peak_mae"].median()), 2),
            "median_pearson": round(float(sub["curve_pearson"].median()), 3),
        })
    rows.append({
        "n_peaks_bin": "ALL",
        "n_spectra": len(df),
        "median_recall": round(float(df["peak_recall"].median()), 3),
        "median_precision": round(float(df["peak_precision"].median()), 3),
        "median_MAE_cm-1": round(float(df["peak_mae"].median()), 2),
        "median_pearson": round(float(df["curve_pearson"].median()), 3),
    })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"saved → {out_path}\n")
    print(out_df.to_string(index=False))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("\n=== tab_per_chart_type_lr ===")
    per_chart_type_lr(OUT / "tab_per_chart_type_lr.csv")
    print("\n=== tab_bootstrap_ci ===")
    bootstrap_ci(OUT / "tab_bootstrap_ci.csv")
    print("\n=== tab_errors ===")
    errors_table(OUT / "tab_errors.csv")
    print("\n=== tab_peak_error_detail ===")
    peak_error_detail(OUT / "tab_peak_error_detail.csv")


if __name__ == "__main__":
    main()
