"""Aggregate pct_lines_iou_ge_0.8 for any iter*_full204_metrics.csv produced by
``run_iter_full.py`` (after the iter12+ patch that exports ``iou_per_line``).

Usage::

    python experiments/iter/aggregate_pct80.py \
        --csvs experiments/results/iter/iter11*.csv \
              experiments/results/iter/phaseB_*.csv \
        --out experiments/results/iter/pct80_summary.csv

If the CSV was generated *before* the per-line export patch (no
``iou_per_line`` column), each image is treated as having ``iou_matched_mean``
applied to every GT line — that's a coarse approximation and will be flagged in
the output as ``approx=True``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import os

LABEL_HOLDOUT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[2])) / "experiments" / "results" / "label_clean_holdout.txt"


def _per_line_vec(row, k_gt: int) -> tuple[list[float], bool]:
    raw = row.get("iou_per_line", None)
    if isinstance(raw, str) and raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and len(parsed) == k_gt:
                return [float(v) for v in parsed], False
        except (ValueError, TypeError):
            pass
    return [float(row.get("iou_matched_mean", 0.0))] * k_gt, True


def aggregate_one(csv_path: Path, holdout: set[str] | None) -> dict:
    df = pd.read_csv(csv_path)
    df = df[df["ok"].astype(bool)] if "ok" in df.columns else df
    if holdout is not None:
        df = df[df["image_id"].isin(holdout)].copy()

    rows = []
    approx_any = False
    for _, r in df.iterrows():
        k_gt = int(r["k_gt"])
        per, approx = _per_line_vec(r, k_gt)
        approx_any = approx_any or approx
        for v in per:
            rows.append({"stratum": r["stratum"], "iou": float(v)})
    if not rows:
        return {"csv": csv_path.name, "approx": approx_any, "n_lines": 0,
                "pct80_all": 0.0, "iou_m_all": 0.0,
                "pct80_L1": 0.0, "pct80_L2": 0.0, "pct80_L3": 0.0, "pct80_L4": 0.0}
    ldf = pd.DataFrame(rows)
    pct80 = (ldf["iou"] >= 0.8).mean()
    iou_m = ldf["iou"].mean()
    by = ldf.groupby("stratum")["iou"].apply(lambda s: float((s >= 0.8).mean())).to_dict()
    iou_m_by = ldf.groupby("stratum")["iou"].mean().to_dict()
    n_by = ldf.groupby("stratum")["iou"].count().to_dict()
    return {
        "csv": csv_path.name,
        "approx": approx_any,
        "n_images": int(df["image_id"].nunique()),
        "n_lines": len(ldf),
        "pct80_all": round(float(pct80), 4),
        "iou_m_all": round(float(iou_m), 4),
        "pct80_L1": round(float(by.get("L1_single", 0.0)), 4),
        "pct80_L2": round(float(by.get("L2_two_line", 0.0)), 4),
        "pct80_L3": round(float(by.get("L3_few_line", 0.0)), 4),
        "pct80_L4": round(float(by.get("L4_many_line", 0.0)), 4),
        "iou_m_L1": round(float(iou_m_by.get("L1_single", 0.0)), 4),
        "iou_m_L2": round(float(iou_m_by.get("L2_two_line", 0.0)), 4),
        "iou_m_L3": round(float(iou_m_by.get("L3_few_line", 0.0)), 4),
        "iou_m_L4": round(float(iou_m_by.get("L4_many_line", 0.0)), 4),
        "n_L1": int(n_by.get("L1_single", 0)),
        "n_L2": int(n_by.get("L2_two_line", 0)),
        "n_L3": int(n_by.get("L3_few_line", 0)),
        "n_L4": int(n_by.get("L4_many_line", 0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True,
                    help="iter*_full204_metrics.csv files (globs supported)")
    ap.add_argument("--out", default="experiments/results/iter/pct80_summary.csv")
    ap.add_argument("--holdout", default=None,
                    help="path to holdout txt; default = label_clean_holdout.txt; pass 'none' to skip filtering")
    ap.add_argument("--full", action="store_true",
                    help="don't filter to holdout (full 204 view)")
    args = ap.parse_args()

    holdout: set[str] | None
    if args.full:
        holdout = None
    else:
        path = Path(args.holdout) if args.holdout else LABEL_HOLDOUT
        if str(path) == "none":
            holdout = None
        else:
            holdout = set(p.strip() for p in path.read_text().splitlines() if p.strip())

    paths: list[Path] = []
    for p in args.csvs:
        paths += [Path(x) for x in sorted(Path().glob(p))] if "*" in p else [Path(p)]
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise SystemExit("no matching CSVs")

    out = pd.DataFrame([aggregate_one(p, holdout) for p in paths])
    out = out.sort_values("pct80_all", ascending=False)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"saved → {out_path}")
    cols_show = ["csv", "n_images", "n_lines", "pct80_all", "iou_m_all",
                 "pct80_L1", "pct80_L2", "pct80_L3", "pct80_L4", "approx"]
    print(out[cols_show].to_string(index=False))


if __name__ == "__main__":
    main()
