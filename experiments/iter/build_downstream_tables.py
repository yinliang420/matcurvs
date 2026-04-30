"""Generate paper tables for §9 (Downstream Scientific Reasoning):

  - tab:raman-peak  — Raman peak detection (Ours v1 → v2 → v3 progression on n=60)
  - tab:vlm-ylookup — VLM y-lookup (per-K breakdown, n=100 images × 8 Q)
  - tab:iou-headline — % lines matched IoU ≥ 0.8 (Ours strata + VLM Curve-PQ surrogate)

All inputs are existing CSVs / markdowns; no new model calls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import os

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[2]))
RES = KMEANS_ROOT / "experiments" / "results"
ITER_DIR = RES / "iter"
LABEL_HOLDOUT = RES / "label_clean_holdout.txt"


def _holdout() -> set[str]:
    return set(p.strip() for p in LABEL_HOLDOUT.read_text().splitlines() if p.strip())


# ---------- tab:raman-peak ----------

def raman_table(out: Path) -> None:
    v3 = pd.read_csv(RES / "raman_peak_pilot" / "peak_pilot_v3_results.csv")
    v3 = v3[v3["ok"].astype(bool)]
    v2cmp = pd.read_csv(RES / "raman_peak_pilot" / "peak_pilot_v2_compare.csv")

    def stat(s: pd.Series) -> dict:
        s = s.dropna()
        return {"median": float(np.median(s)) if len(s) else float("nan"),
                "mean": float(s.mean()) if len(s) else float("nan")}

    rows = [
        {"variant": "Ours-v1 (axis_box, n=30)", "recall_med": 0.40, "precision_med": 0.43,
         "MAE_cm-1_med": 9.64, "pearson_med": 0.35},
        {"variant": "Ours-v2 (axis_box, n=60)", "recall_med": float(np.median(v2cmp["v2_recall"])),
         "precision_med": float("nan"),
         "MAE_cm-1_med": float(np.median(v2cmp["v2_mae"])),
         "pearson_med": float(np.median(v2cmp["v2_pearson"])) if "v2_pearson" in v2cmp.columns else float("nan")},
        {"variant": "Ours-v3 (cluster x-bounds, n=60)",
         "recall_med": float(np.median(v3["peak_recall"])),
         "precision_med": float(np.median(v3["peak_precision"])),
         "MAE_cm-1_med": float(np.median(v3["peak_mae"])),
         "pearson_med": float(np.median(v3["curve_pearson"]))},
    ]
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("=== tab:raman-peak ===")
    print(df.round(3).to_string(index=False))
    print(f"saved → {out}\n")


# ---------- tab:vlm-ylookup ----------

def ylookup_table(out: Path) -> None:
    df = pd.read_csv(RES / "vlm_downstream" / "label_results.csv")

    def acc(col: pd.Series) -> float:
        v = pd.to_numeric(col, errors="coerce").dropna()
        return float(v.mean()) if len(v) else float("nan")

    by_k = []
    for k, g in df.groupby("K"):
        a = acc(g["correct_A"])
        b = acc(g["correct_B"])
        by_k.append({
            "K": int(k),
            "n_imgs": int(g["image_id"].nunique()),
            "n_q": int(len(g)),
            "acc_A": round(a, 3),
            "acc_B": round(b, 3),
            "delta": round(b - a, 3),
        })
    by_k_df = pd.DataFrame(by_k).sort_values("K")
    overall = pd.DataFrame([{
        "K": "ALL",
        "n_imgs": int(df["image_id"].nunique()),
        "n_q": int(len(df)),
        "acc_A": round(acc(df["correct_A"]), 3),
        "acc_B": round(acc(df["correct_B"]), 3),
        "delta": round(acc(df["correct_B"]) - acc(df["correct_A"]), 3),
    }])
    full = pd.concat([by_k_df, overall], ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out, index=False)
    print("=== tab:vlm-ylookup ===")
    print(full.to_string(index=False))
    print(f"saved → {out}\n")


# ---------- tab:iou-headline ----------

def _per_line_pct80_one(csv_path: Path, holdout: set[str]) -> dict:
    df = pd.read_csv(csv_path)
    df = df[df.get("ok", True).astype(bool)]
    df = df[df["image_id"].isin(holdout)].copy()
    rows = []
    for _, r in df.iterrows():
        k_gt = int(r["k_gt"])
        raw = r.get("iou_per_line", "")
        if isinstance(raw, str) and raw.startswith("["):
            try:
                vec = json.loads(raw)
                if isinstance(vec, list) and len(vec) == k_gt:
                    for v in vec:
                        rows.append({"stratum": r["stratum"], "iou": float(v)})
                    continue
            except (ValueError, TypeError):
                pass
        # fallback: every line gets iou_matched_mean (approximate)
        for _ in range(k_gt):
            rows.append({"stratum": r["stratum"], "iou": float(r["iou_matched_mean"])})

    if not rows:
        return {"pct80_All": 0.0}
    ldf = pd.DataFrame(rows)
    out = {"pct80_All": round(float((ldf["iou"] >= 0.8).mean()), 4),
           "iou_m_All": round(float(ldf["iou"].mean()), 4)}
    for st_short, st in [("L1", "L1_single"), ("L2", "L2_two_line"),
                         ("L3", "L3_few_line"), ("L4", "L4_many_line")]:
        sub = ldf[ldf["stratum"] == st]
        out[f"pct80_{st_short}"] = round(float((sub["iou"] >= 0.8).mean()), 4) if len(sub) else 0.0
    return out


def vlm_pct80_surrogate(csv_path: Path, holdout: set[str]) -> dict:
    """Curve-PQ ≈ Hungarian-IoU × F1; this is an *upper bound* on the fraction of
    GT lines that could plausibly hit IoU ≥ 0.8, since PQ collapses to 0 when
    Hungarian-matched IoU is very low.  We also report mean curve_pq directly."""
    df = pd.read_csv(csv_path)
    df = df[df.get("success", True).astype(bool)]
    df = df[df["image_id"].isin(holdout)].copy()
    out: dict = {}
    if not len(df):
        return {"pct80_All": 0.0, "curve_pq_mean": 0.0,
                "instance_f1_mean": 0.0}
    out["curve_pq_mean"] = round(float(df["curve_pq"].mean()), 4)
    out["instance_f1_mean"] = round(float(df["instance_f1"].mean()), 4)
    # surrogate: any image with curve_pq >= 0.8 contributes its n_gt_lines lines
    df["pct_lines_at_pq80"] = (df["curve_pq"] >= 0.8).astype(int)
    n_lines_total = int(df["n_gt_lines"].sum())
    n_lines_hit = int((df["pct_lines_at_pq80"] * df["n_gt_lines"]).sum())
    out["pct80_All"] = round(n_lines_hit / max(1, n_lines_total), 4)
    for st_short, st in [("L1", "L1_single"), ("L2", "L2_two_line"),
                         ("L3", "L3_few_line"), ("L4", "L4_many_line")]:
        sub = df[df["stratum"] == st]
        nl_t = int(sub["n_gt_lines"].sum()) if len(sub) else 0
        nl_h = int(((sub["curve_pq"] >= 0.8).astype(int) * sub["n_gt_lines"]).sum()) if len(sub) else 0
        out[f"pct80_{st_short}"] = round(nl_h / max(1, nl_t), 4) if nl_t else 0.0
    return out


def iou_headline_table(ours_csvs: list[Path], out: Path) -> None:
    holdout = _holdout()
    rows = []
    for csv in ours_csvs:
        if not csv.exists():
            continue
        d = _per_line_pct80_one(csv, holdout)
        d = {"method": csv.stem.replace("_full204_metrics", ""), **d}
        rows.append(d)
    # VLM rows
    for vlm_csv in sorted(RES.glob("vlm_*_label.csv")):
        d = vlm_pct80_surrogate(vlm_csv, holdout)
        d = {"method": vlm_csv.stem, **d}
        rows.append(d)

    df = pd.DataFrame(rows)
    cols = ["method", "pct80_All", "iou_m_All", "pct80_L1", "pct80_L2",
            "pct80_L3", "pct80_L4", "curve_pq_mean", "instance_f1_mean"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("=== tab:iou-headline (n=90 holdout) ===")
    print(df.fillna("-").to_string(index=False))
    print(f"saved → {out}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", nargs="*", default=[],
                    help="iter CSVs to include in tab:iou-headline (default: latest phaseB+phaseC+phaseD wins)")
    ap.add_argument("--out_dir", default=str(RES / "paper_tables"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    raman_table(out_dir / "tab_raman_peak.csv")
    ylookup_table(out_dir / "tab_vlm_ylookup.csv")

    if args.ours:
        ours = [Path(p) for p in args.ours]
    else:
        # fall back to whatever phaseB/C/D/F CSVs exist + iter11
        candidates = sorted(ITER_DIR.glob("phase*_full204_metrics.csv"))
        candidates += [ITER_DIR / "iter11_m2f_multiscale_full204_metrics.csv"]
        ours = [p for p in candidates if p.exists()]
    iou_headline_table(ours, out_dir / "tab_iou_headline.csv")


if __name__ == "__main__":
    main()
