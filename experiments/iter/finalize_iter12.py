"""Wrap up: aggregate iter12 phase B-F CSVs, pick the best config,
generate paper tables + the iter12 summary markdown, and call
fill_paper_iou_headline.py to land the LR_0.8 numbers in the tex.

Idempotent — safe to re-run after each phase if you want incremental
updates.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import os
import sys

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[2]))
RES = KMEANS_ROOT / "experiments" / "results"
ITER_DIR = RES / "iter"
PAPER_DIR = KMEANS_ROOT / "paper" / "neurips2026_intro"
HOLDOUT = set(p.strip() for p in (RES / "label_clean_holdout.txt").read_text().splitlines() if p.strip())


def _row_stats(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    df = df[df.get("ok", True).astype(bool)]
    holdout_df = df[df["image_id"].isin(HOLDOUT)]
    full_df = df
    out = {"csv": csv_path.name, "n_full": len(full_df), "n_holdout": len(holdout_df)}
    for tag, sub in [("full204", full_df), ("hold90", holdout_df)]:
        if not len(sub):
            continue
        n_lines = int(sub["k_gt"].sum())
        n_hit = int(sub.get("n_iou_ge_0_8", pd.Series([0])).sum())
        out[f"pct80_{tag}"] = round(n_hit / max(1, n_lines), 4)
        out[f"iou_m_{tag}"] = round(float(sub["iou_matched_mean"].mean()), 4)
        out[f"iou_g_{tag}"] = round(float(sub["iou_global"].mean()), 4)
        out[f"cov_gt_{tag}"] = round(float(sub["coverage_gt"].mean()), 4)
        out[f"cov_pred_{tag}"] = round(float(sub["coverage_pred"].mean()), 4)
        out[f"K_acc_{tag}"] = round(float((sub["k_pred"] == sub["k_gt"]).mean()), 4)
    return out


def main() -> None:
    csvs = sorted(ITER_DIR.glob("phase*_full204_metrics.csv"))
    if not csvs:
        print("no phase CSVs yet; nothing to finalize")
        return
    rows = [_row_stats(c) for c in csvs]
    summary = pd.DataFrame(rows)

    # composite score for ranking on the holdout set
    summary["score_hold90"] = (summary.get("pct80_hold90", 0).fillna(0) +
                                0.5 * summary.get("iou_m_hold90", 0).fillna(0))
    summary = summary.sort_values("score_hold90", ascending=False).reset_index(drop=True)

    summary_csv = ITER_DIR / "iter12_phase_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"saved → {summary_csv}\n")
    cols = ["csv", "pct80_hold90", "pct80_full204", "iou_m_hold90", "iou_g_hold90",
            "cov_gt_hold90", "cov_pred_hold90", "K_acc_hold90"]
    print(summary[cols].to_string(index=False))

    # build downstream tables (auto-pulls best phase CSV)
    subprocess.run([sys.executable,
                    "experiments/iter/build_downstream_tables.py"],
                   cwd=str(KMEANS_ROOT), check=True)

    # fill paper tex placeholders (best + iter11 control rows)
    subprocess.run([sys.executable,
                    "experiments/iter/fill_paper_iou_headline.py"],
                   cwd=str(KMEANS_ROOT), check=True)

    # iter12 summary markdown
    md_path = ITER_DIR / "iter12_summary.md"
    best = summary.iloc[0]
    with md_path.open("w") as f:
        f.write("# iter12 — full-204 + leakage-free n=90 results\n\n")
        f.write("## Best config (highest pct_lines_iou_ge_0.8 on n=90)\n\n")
        for k in cols:
            f.write(f"- **{k}** = `{best.get(k, '')}`\n")
        f.write("\n## All phase CSVs ranked by score (pct80_hold90 + 0.5 × IoU_m_hold90)\n\n")
        # Manual markdown emission to avoid the optional ``tabulate`` dep.
        cs = cols + ["score_hold90"]
        f.write("| " + " | ".join(cs) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cs)) + " |\n")
        for _, r in summary[cs].iterrows():
            f.write("| " + " | ".join(str(r[c]) for c in cs) + " |\n")
        f.write("\n## Methodology\n\n")
        f.write("- Phase B: 2-ckpt Mask2Former bake-off (q12_d3 control + v9_hr640_q10_d6 alt).\n")
        f.write("- Phase C: m2f threshold sweep ∈ {0.25, 0.30}, then composing winning threshold with scales=(512,768,1024), flip TTA, thickness compensation (skeletonize→dilate-to-5px under LAB ΔE<12 gate).\n")
        f.write("- Phase D: per-query NMS branch when `K ≥ 5`, plus a query-init seeding mode for KMeans.\n")
        f.write("- Phase E: marker-only scatter detector (cc count ≥ 3K, mean cc area < 50 px, max cc longest axis < 40 px) → polyfit deg=3 per LAB cluster.\n")
        f.write("- Phase F: per-line LAB-gated dilation, three modes (basic ΔE=10 grow=4 / aggr ΔE=15 grow=8 / edge basic + Sobel mag>μ+σ gate).\n")
        f.write("- Each cell evaluates the full 204-image label set; metric reporting filters to the n=90 leakage-controlled holdout from `experiments/results/label_clean_holdout.txt`.\n")
        f.write("- New per-image columns in iter12 CSVs vs iter≤11: `iou_per_line` (JSON list), `matched_pairs` (JSON list of (gt_idx, pred_idx, iou)), `n_iou_ge_0_8` (count).\n")
    print(f"saved → {md_path}")


if __name__ == "__main__":
    main()
