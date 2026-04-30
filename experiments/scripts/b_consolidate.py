#!/usr/bin/env python
"""Subagent B — Consolidate baseline CSV and produce deliverables B1/B3/B4."""
from __future__ import annotations

import os
import sys
import json
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[2])) / "experiments" / "results"
MATDEPLOT_SNAP = Path(os.environ.get("MATDEPLOT_ROOT", "/path/to/MatDeplot/mdp_v2")) / "paper/neurips_2026/results_snapshot"


# ----- Step 1: Inspect method list -----
def main():
    src = pd.read_csv(MATDEPLOT_SNAP / "per_method_metrics.csv")
    print("== Source per_method_metrics.csv ==", file=sys.stderr)
    print("methods:", sorted(src['method'].unique()), file=sys.stderr)
    print("families:", sorted(src['family'].unique()), file=sys.stderr)
    print("rows:", len(src), file=sys.stderr)
    print(src['method'].value_counts(), file=sys.stderr)
    print("", file=sys.stderr)

    # Methods we want to keep from the cache (all of them — VLMs + 6 M*_mock + 6 M*_v11b)
    wanted_methods = set(src['method'].unique())
    # Spec said: "filter out methods we don't report". In this snapshot everything is
    # relevant (4 VLMs + 6 oracle-FG M*_mock + 6 V11b M*_v11b). No filter.
    src = src[src['method'].isin(wanted_methods)].copy()

    # Columns of the consolidated CSV
    keep_cols = [
        "method", "family", "image_id", "dtw_delta", "ccr_at_tau",
        "instance_f1", "curve_pq", "n_predicted_trails", "n_gt_trails", "stratum",
    ]

    src_out = src[keep_cols].copy()

    # Ours main
    ours = pd.read_csv(ROOT / "ours_kmeans_m2f.csv")
    ours_out = ours.rename(columns={})[
        ["image_id", "dtw_delta", "ccr_at_tau", "instance_f1",
         "curve_pq", "n_predicted_trails", "n_gt_trails", "stratum"]
    ].copy()
    ours_out.insert(0, "method", "ours_kmeans_m2f")
    ours_out.insert(1, "family", "ours_pipeline")
    ours_out = ours_out[keep_cols]

    # Ours fallback
    fb_path = ROOT / "ours_kmeans_m2f_fallback.csv"
    if fb_path.exists():
        fb = pd.read_csv(fb_path)
        fb_out = fb[["image_id", "dtw_delta", "ccr_at_tau", "instance_f1",
                     "curve_pq", "n_predicted_trails", "n_gt_trails", "stratum"]].copy()
        fb_out.insert(0, "method", "ours_kmeans_m2f_fallback")
        fb_out.insert(1, "family", "ours_pipeline")
        fb_out = fb_out[keep_cols]
    else:
        fb_out = pd.DataFrame(columns=keep_cols)

    consolidated = pd.concat([src_out, ours_out, fb_out], ignore_index=True)
    out_path = ROOT / "all_baselines.csv"
    consolidated.to_csv(out_path, index=False)
    print(f"[B1] wrote {out_path} with {len(consolidated)} rows, "
          f"{consolidated['method'].nunique()} methods", file=sys.stderr)

    # ----- Step B3: aggregate by method × (stratum | ALL), sorted by instance_f1 desc -----
    agg_rows = []
    df_all = consolidated

    def agg_block(df_block, stratum_label):
        for (method, family), sub in df_block.groupby(["method", "family"]):
            n = len(sub)
            # DTW can have NaN (e.g. vlm_claude-opus-4-6 some strata), use mean skipna
            dtw = sub["dtw_delta"].mean(skipna=True)
            ccr = sub["ccr_at_tau"].mean(skipna=True)
            f1 = sub["instance_f1"].mean(skipna=True)
            pq = sub["curve_pq"].mean(skipna=True)
            agg_rows.append({
                "method": method, "family": family, "stratum": stratum_label,
                "n": n, "dtw": dtw, "ccr": ccr, "f1": f1, "pq": pq,
            })

    agg_block(df_all, "ALL")
    for stratum in ["L1_single", "L2_multi_no_overlap", "L3_multi_overlap",
                    "L4_same_color_overlap", "L5_dense_scatter"]:
        agg_block(df_all[df_all["stratum"] == stratum], stratum)

    agg = pd.DataFrame(agg_rows)

    def fmt(v, digits=3):
        if pd.isna(v):
            return "nan"
        return f"{v:.{digits}f}"

    lines = ["# B3 — Method-level aggregate table", ""]
    lines.append("Computed from `experiments/results/all_baselines.csv`. "
                 "Cells show mean over the image set; n is the sample size the method covered in that stratum. "
                 "Sorted by Inst-F1 descending within each stratum.")
    lines.append("")

    for stratum in ["ALL", "L1_single", "L2_multi_no_overlap",
                    "L3_multi_overlap", "L4_same_color_overlap", "L5_dense_scatter"]:
        sub = agg[agg["stratum"] == stratum].sort_values("f1", ascending=False)
        if sub.empty:
            continue
        lines.append(f"## {stratum}")
        lines.append("")
        lines.append("| method | family | n | DTW-Δ | CCR@0.1 | Inst-F1 | Curve-PQ |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['method']} | {r['family']} | {int(r['n'])} | "
                f"{fmt(r['dtw'], 2)} | {fmt(r['ccr'])} | "
                f"{fmt(r['f1'])} | {fmt(r['pq'])} |"
            )
        lines.append("")

    (ROOT / "B3_method_aggregate.md").write_text("\n".join(lines))
    print(f"[B3] wrote {ROOT / 'B3_method_aggregate.md'}", file=sys.stderr)

    # ----- Step B4: paired bootstrap CI: ours_kmeans_m2f vs best VLM (Inst-F1) on shared image IDs -----
    vlm_methods = [m for m in df_all["method"].unique() if m.startswith("vlm_")]
    print(f"VLM methods available: {vlm_methods}", file=sys.stderr)

    # Determine best VLM by ALL Inst-F1 (mean over whatever images it covered)
    vlm_scores = agg[(agg["stratum"] == "ALL") & (agg["method"].isin(vlm_methods))]
    vlm_scores = vlm_scores.sort_values("f1", ascending=False)
    print("VLM ranking by Inst-F1 (ALL):", file=sys.stderr)
    print(vlm_scores[["method", "n", "f1"]].to_string(index=False), file=sys.stderr)
    best_vlm = vlm_scores.iloc[0]["method"]
    best_vlm_f1_reported = vlm_scores.iloc[0]["f1"]
    print(f"Best VLM: {best_vlm} (mean F1={best_vlm_f1_reported:.3f})", file=sys.stderr)

    ours_df = df_all[df_all["method"] == "ours_kmeans_m2f"][["image_id", "instance_f1"]]
    best_df = df_all[df_all["method"] == best_vlm][["image_id", "instance_f1"]]

    merged = ours_df.merge(best_df, on="image_id", suffixes=("_ours", "_vlm"))
    print(f"Shared images ours vs {best_vlm}: {len(merged)}", file=sys.stderr)

    # Diff series
    diffs = (merged["instance_f1_ours"] - merged["instance_f1_vlm"]).to_numpy()
    n = len(diffs)
    mean_diff = float(np.mean(diffs))

    # Paired bootstrap 95% CI (percentile method), 10000 iterations, fixed seed
    rng = np.random.default_rng(20260424)
    B = 10000
    boot = np.empty(B)
    idx_range = np.arange(n)
    for b in range(B):
        sel = rng.choice(idx_range, size=n, replace=True)
        boot[b] = diffs[sel].mean()
    lo = float(np.percentile(boot, 2.5))
    hi = float(np.percentile(boot, 97.5))

    # Also: paired-permutation / sign test p-value (two-sided, sign-flip)
    P = 20000
    obs_abs = abs(mean_diff)
    count_geq = 0
    signs = rng.choice([-1.0, 1.0], size=(P, n))
    perm_means = (signs * diffs).mean(axis=1)
    count_geq = int(np.sum(np.abs(perm_means) >= obs_abs))
    perm_p = (count_geq + 1) / (P + 1)

    ours_mean = float(merged["instance_f1_ours"].mean())
    vlm_mean = float(merged["instance_f1_vlm"].mean())

    lines = [
        "# B4 — Paired bootstrap 95% CI: ours_kmeans_m2f vs best VLM (Instance-F1)",
        "",
        "## Setup",
        "",
        f"- Comparison: `ours_kmeans_m2f` vs **`{best_vlm}`** (best VLM by mean Inst-F1 on its coverage).",
        f"- Shared image set: intersection of the two methods' image_ids, n = **{n}** images.",
        f"- Metric: Instance-F1 (higher is better).",
        f"- Bootstrap: paired, {B} resamples, percentile 95% CI, seed=20260424.",
        f"- Also computed: two-sided paired sign-flip permutation p-value, {P} permutations.",
        "",
        "## Results",
        "",
        f"| Quantity | Value |",
        f"|---|---|",
        f"| Mean Inst-F1 (ours_kmeans_m2f, shared set) | **{ours_mean:.3f}** |",
        f"| Mean Inst-F1 ({best_vlm}, shared set) | **{vlm_mean:.3f}** |",
        f"| Mean paired Δ (ours − VLM) | **{mean_diff:+.3f}** |",
        f"| 95% bootstrap CI on Δ | **[{lo:+.3f}, {hi:+.3f}]** |",
        f"| Paired sign-flip permutation p-value (two-sided) | **{perm_p:.4f}** |",
        "",
        "## Interpretation",
        "",
        f"- On the {n}-image shared test set, `ours_kmeans_m2f` scores Inst-F1 "
        f"{ours_mean:.3f} vs `{best_vlm}`'s {vlm_mean:.3f}, a paired mean difference of "
        f"{mean_diff:+.3f}.",
        f"- The 95% bootstrap CI {'excludes' if (lo > 0 or hi < 0) else 'includes'} zero.",
        f"- Paired sign-flip permutation p = {perm_p:.4f} "
        f"({'significant' if perm_p < 0.05 else 'not significant'} at α=0.05).",
        "",
        "## Reproduction",
        "",
        "```bash",
        f"{sys.executable} {Path(os.environ.get('KMEANS_ROOT', Path(__file__).resolve().parents[2])) / 'experiments' / 'scripts' / 'b_consolidate.py'}",
        "```",
    ]
    (ROOT / "B4_bootstrap_ci.md").write_text("\n".join(lines))
    print(f"[B4] wrote {ROOT / 'B4_bootstrap_ci.md'}", file=sys.stderr)

    # Returned useful values for the subagent report
    return {
        "n_rows": len(consolidated),
        "n_methods": int(consolidated['method'].nunique()),
        "best_vlm": best_vlm,
        "ours_mean_shared": ours_mean,
        "vlm_mean_shared": vlm_mean,
        "mean_diff": mean_diff,
        "ci_lo": lo,
        "ci_hi": hi,
        "perm_p": perm_p,
        "n_shared": n,
    }


if __name__ == "__main__":
    r = main()
    print(json.dumps(r, indent=2))
