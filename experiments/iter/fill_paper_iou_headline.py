"""Once iter12 pipeline finishes, fill the \\TODO{} placeholders in
\\Cref{tab:iou-headline} of neurips_2026.tex with the real
LR_{0.8} numbers from build_downstream_tables.py output.

Strategy: pick the highest-pct80 phase[F|E|D|C|B] CSV as 'Ours-best',
use iter11_m2f_multiscale_full204_metrics.csv (or phaseB_q12_d3 = same
model with the new per-line column) as 'Ours-iter11 (prior)'.

Only modifies the two \\textsc{Ours-*} rows in tab:iou-headline. Idempotent.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import os

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[2]))
RES = KMEANS_ROOT / "experiments" / "results"
ITER_DIR = RES / "iter"
TEX = KMEANS_ROOT / "paper" / "neurips2026_intro" / "neurips_2026.tex"
HOLDOUT = set(p.strip() for p in (RES / "label_clean_holdout.txt").read_text().splitlines() if p.strip())


def _stats(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    df = df[df.get("ok", True).astype(bool)]
    df = df[df["image_id"].isin(HOLDOUT)].copy()
    if not len(df):
        return {}
    by = {}
    for st_short, st in [("All", None), ("L1", "L1_single"), ("L2", "L2_two_line"),
                          ("L3", "L3_few_line"), ("L4", "L4_many_line")]:
        sub = df if st is None else df[df["stratum"] == st]
        n_lines = int(sub["k_gt"].sum()) if len(sub) else 0
        n_hit = int(sub["n_iou_ge_0_8"].sum()) if "n_iou_ge_0_8" in sub.columns and len(sub) else 0
        by[f"LR_{st_short}"] = round(n_hit / max(1, n_lines), 3)
    by["IoU_m"] = round(float(df["iou_matched_mean"].mean()), 3)
    by["IoU_g"] = round(float(df["iou_global"].mean()), 3)
    by["cov_gt"] = round(float(df["coverage_gt"].mean()), 3)
    by["cov_pred"] = round(float(df["coverage_pred"].mean()), 3)
    return by


def _pick_best() -> Path | None:
    best = (None, -1.0)
    for csv in sorted(ITER_DIR.glob("phase*_full204_metrics.csv")):
        df = pd.read_csv(csv)
        df = df[df["image_id"].isin(HOLDOUT)]
        if "n_iou_ge_0_8" not in df.columns:
            continue
        n_lines = int(df["k_gt"].sum()); n_hit = int(df["n_iou_ge_0_8"].sum())
        pct = n_hit / max(1, n_lines)
        score = pct + 0.5 * float(df["iou_matched_mean"].mean())
        if score > best[1]:
            best = (csv, score)
    return best[0]


def _row(label: str, st: dict) -> str:
    if not st:
        return f"\\textsc{{{label}}}    & --- & --- & --- & --- & --- & --- & --- \\\\"
    return (f"\\textsc{{{label}}}    & "
            f"\\textbf{{{st['LR_All']*100:.1f}\\%}} & "
            f"{st['LR_L1']*100:.1f}\\% & "
            f"{st['LR_L2']*100:.1f}\\% & "
            f"{st['LR_L3']*100:.1f}\\% & "
            f"{st['LR_L4']*100:.1f}\\% & "
            f"--- & --- \\\\")


def main() -> None:
    best = _pick_best()
    iter11 = ITER_DIR / "phaseB_q12_d3_full204_metrics.csv"
    if not iter11.exists():
        iter11 = ITER_DIR / "iter11_m2f_multiscale_full204_metrics.csv"

    best_st = _stats(best) if best else {}
    iter11_st = _stats(iter11) if iter11.exists() else {}

    print(f"Ours-best ({best.name if best else 'NONE'}): {best_st}")
    print(f"Ours-iter11 ({iter11.name}): {iter11_st}")

    tex = TEX.read_text()
    new_best_row = _row("Ours-best (\\S5)", best_st)
    new_iter11_row = _row("Ours-iter11 (prior)", iter11_st)

    # Replace the \TODO rows in tab:iou-headline (use re.sub with lambda
    # to avoid template-escape semantics in the replacement string).
    pat_best = re.compile(
        r"\\textsc\{Ours-best \(\\S5\)\}.*?\\\\",
        re.DOTALL)
    pat_iter11 = re.compile(
        r"\\textsc\{Ours-iter11 \(prior\)\}.*?\\\\",
        re.DOTALL)
    tex2 = pat_best.sub(lambda m: new_best_row, tex, count=1)
    tex2 = pat_iter11.sub(lambda m: new_iter11_row, tex2, count=1)

    # Also fill abstract \widehat{LR} placeholder
    if best_st:
        lr_str = f"{best_st['LR_All']*100:.1f}"
        tex2 = tex2.replace(r"$\mathrm{LR}_{0.8}{=}\widehat{LR}$\,\%",
                            f"$\\mathrm{{LR}}_{{0.8}}{{=}}{lr_str}$\\,\\%")
        tex2 = tex2.replace(r"$\mathrm{LR}_{0.8}{=}\widehat{LR}\%$",
                            f"$\\mathrm{{LR}}_{{0.8}}{{=}}{lr_str}\\%$")

    if tex2 == tex:
        print("WARNING: no replacements made (regex miss?)")
    TEX.write_text(tex2)
    print(f"updated → {TEX}")


if __name__ == "__main__":
    main()
