"""Replace the ``\\textcolor{red}{...}`` placeholders in the §K appendix
and surrounding tex with real numbers from the judge_*_<cell>.csv files
that ``build_judge_appendix.py`` produced.

This intentionally rewrites the tex IN-PLACE; revert via git if needed.

Logic:
  Tab K.1 (judge-passrate): four rows replaced in fixed judge order
    Qwen-VL-Max / GPT-5.4 / Gemini-3.1-pro / Claude Opus 4.6
  Tab K.2 (judge-kappa): upper-triangular (i, j) cells, in display order
  Tab K.3 (judge-vs-LR): four rows in same order
  Tab K.4 / consensus sweep: three rows for thresholds 1, 2, 3
  Inline placeholders: §3 Cell 6 (none — all replaced), §7 abstain
    (XX/90, YY%, Z.ZZ, A.AA, +B.BB), Limitation (l) (kappa range, +0.0Z)

Usage::
    python experiments/iter/fill_judge_placeholders.py [cell]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import os

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[2]))
ITER = KMEANS_ROOT / "experiments" / "results" / "iter"
TEX = KMEANS_ROOT / "paper" / "neurips2026_intro" / "neurips_2026.tex"

JUDGE_ORDER = [
    "qwen-vl-max",
    "gpt-5.4",
    "gemini-3.1-pro-preview",
    "claude-opus-4-6",
]
JUDGE_DISPLAY = {
    "qwen-vl-max":            "Qwen-VL-Max",
    "gpt-5.4":                "GPT-5.4",
    "gemini-3.1-pro-preview": "Gemini-3.1-pro-preview",
    "claude-opus-4-6":        "Claude Opus 4.6",
}


def _fmt_pct(x):
    if pd.isna(x):
        return "---"
    return f"{x:.3f}"


def _fmt_int(x):
    if pd.isna(x):
        return "---"
    return f"{int(x)}"


def _fmt_lat(x):
    if pd.isna(x):
        return "---"
    return f"{x:.1f}"


def _replace_block(text: str, marker: str, new_block: str) -> str:
    """Replace the table body between LaTeX ``\\midrule`` and
    ``\\bottomrule`` immediately following the line containing
    ``marker``.  ``new_block`` should NOT include the rule lines."""
    pat = re.compile(
        r"(\\label\{" + re.escape(marker) + r"\}.*?\\midrule\s*\n)"
        r"(.*?)"
        r"(\\bottomrule)",
        re.DOTALL,
    )
    m = pat.search(text)
    if not m:
        raise SystemExit(f"could not find table body for label {marker}")
    return text[:m.start(2)] + new_block + text[m.start(3):]


def main(cell: str = "ours_v4tta_label"):
    pass_csv = ITER / f"judge_passrate_{cell}.csv"
    kappa_csv = ITER / f"judge_kappa_{cell}.csv"
    lr_csv = ITER / f"judge_lr_correlation_{cell}.csv"
    cons_csv = ITER / f"judge_consensus_{cell}.csv"
    for p in (pass_csv, kappa_csv, lr_csv, cons_csv):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    pr = pd.read_csv(pass_csv).set_index("judge")
    kp = pd.read_csv(kappa_csv, index_col=0)
    lr = pd.read_csv(lr_csv).set_index("judge")
    cs = pd.read_csv(cons_csv).set_index("threshold_fail_votes_ge")

    judges = [j for j in JUDGE_ORDER if j in pr.index]
    if len(judges) != len(JUDGE_ORDER):
        missing = set(JUDGE_ORDER) - set(judges)
        print(f"WARNING: missing judges: {missing}; "
               "rows for those will show '---'.")

    # ---- Tab K.1 ----
    rows_k1 = []
    for j in JUDGE_ORDER:
        if j in pr.index:
            r = pr.loc[j]
            rows_k1.append(
                f"{JUDGE_DISPLAY[j]:<24s}    & "
                f"{_fmt_int(r['n_204']):>3} & {_fmt_pct(r['image_pass_204']):>5} & "
                f"{_fmt_int(r['n_90']):>3} & {_fmt_pct(r['image_pass_90']):>5} & "
                f"{_fmt_lat(r['median_latency']):>5} \\\\"
            )
        else:
            rows_k1.append(
                f"{JUDGE_DISPLAY[j]:<24s}    & --- & --- & --- & --- & --- \\\\"
            )
    body_k1 = "\n".join(rows_k1) + "\n"

    # ---- Tab K.2 (upper triangular) ----
    rows_k2 = []
    for i, ja in enumerate(JUDGE_ORDER):
        cells = [JUDGE_DISPLAY[ja]]
        for jb_i, jb in enumerate(JUDGE_ORDER):
            if jb_i < i:
                cells.append("")
            elif jb_i == i:
                cells.append("1.000")
            else:
                if ja in kp.index and jb in kp.columns:
                    v = kp.loc[ja, jb]
                    cells.append(_fmt_pct(v))
                else:
                    cells.append("---")
        rows_k2.append(" & ".join(f"{c:<8s}" for c in cells) + " \\\\")
    body_k2 = "\n".join(rows_k2) + "\n"

    # ---- Tab K.3 ----
    rows_k3 = []
    for j in JUDGE_ORDER:
        if j in lr.index:
            r = lr.loc[j]
            rows_k3.append(
                f"{JUDGE_DISPLAY[j]:<24s}    & "
                f"{_fmt_int(r['n_pass']):>2} & "
                f"{_fmt_pct(r['lr05_pass']):>5} & "
                f"{_fmt_pct(r['lr05_fail']):>5} & "
                f"{_fmt_pct(r['delta']):>6} \\\\"
            )
        else:
            rows_k3.append(
                f"{JUDGE_DISPLAY[j]:<24s}    & --- & --- & --- & --- \\\\"
            )
    body_k3 = "\n".join(rows_k3) + "\n"

    # ---- Tab K.4 (consensus sweep) ----
    rows_k4 = []
    for t in (1, 2, 3):
        if t in cs.index:
            r = cs.loc[t]
            label = "\\textbf{2 (default)}" if t == 2 else f"{t}"
            rows_k4.append(
                f"{label} & "
                f"{_fmt_pct(r['abstain_rate']):>5} & "
                f"{_fmt_pct(r['lr05_kept']):>5} & "
                f"{_fmt_pct(r['lr05_abstained']):>5} & "
                f"{_fmt_pct(r['lr05_delta']):>6} \\\\"
            )
        else:
            rows_k4.append(
                f"{t} & --- & --- & --- & --- \\\\"
            )
    body_k4 = "\n".join(rows_k4) + "\n"

    # ---- inline numbers ----
    cons2 = cs.loc[2] if 2 in cs.index else None
    cons_n_total = int(cs.loc[2, "n_total"]) if 2 in cs.index else 0
    cons_n_abst = int(cs.loc[2, "n_abstain"]) if 2 in cs.index else 0
    cons_rate = (cs.loc[2, "abstain_rate"] * 100) if 2 in cs.index else float("nan")
    lr_kept = cs.loc[2, "lr05_kept"] if 2 in cs.index else float("nan")
    lr_abst = cs.loc[2, "lr05_abstained"] if 2 in cs.index else float("nan")
    lr_delta = cs.loc[2, "lr05_delta"] if 2 in cs.index else float("nan")

    # off-diagonal κ values (exclude self-pairs)
    off_diag = []
    for ja in JUDGE_ORDER:
        for jb in JUDGE_ORDER:
            if ja >= jb:
                continue
            if ja in kp.index and jb in kp.columns:
                v = kp.loc[ja, jb]
                if not pd.isna(v):
                    off_diag.append(float(v))
    kappa_lo = min(off_diag) if off_diag else float("nan")
    kappa_hi = max(off_diag) if off_diag else float("nan")

    # judge-vs-LR mean delta across judges that have both pass and fail subsets
    deltas = [float(lr.loc[j, "delta"]) for j in JUDGE_ORDER
               if j in lr.index and not pd.isna(lr.loc[j, "delta"])]
    avg_delta = (sum(deltas) / len(deltas)) if deltas else float("nan")

    # API cost estimate: assume Qwen $.005, GPT $.009, Gemini $.016, Claude $.118
    cost_table = {"qwen-vl-max": 0.005, "gpt-5.4": 0.009,
                   "gemini-3.1-pro-preview": 0.016, "claude-opus-4-6": 0.118}
    total_cost = 0.0
    for j in JUDGE_ORDER:
        if j in pr.index:
            n = float(pr.loc[j, "n_204"])
            total_cost += n * cost_table.get(j, 0.0)

    print("=== inline values ===")
    print(f"  cons_n_total      = {cons_n_total}")
    print(f"  cons_n_abst       = {cons_n_abst}")
    print(f"  cons_rate(%)      = {cons_rate:.1f}")
    print(f"  lr_kept           = {lr_kept:.3f}")
    print(f"  lr_abst           = {lr_abst:.3f}")
    print(f"  lr_delta          = {lr_delta:+.3f}")
    print(f"  kappa range       = [{kappa_lo:.2f}, {kappa_hi:.2f}]")
    print(f"  avg LR05 Δ        = {avg_delta:+.3f}")
    print(f"  total API cost    = ${total_cost:.2f}")

    # ---- rewrite tex ----
    tex = TEX.read_text()
    tex = _replace_block(tex, "tab:judge-passrate", body_k1)
    tex = _replace_block(tex, "tab:judge-kappa", body_k2)
    tex = _replace_block(tex, "tab:judge-vs-lr", body_k3)
    tex = _replace_block(tex, "tab:judge-consensus", body_k4)

    # API cost in §K intro
    tex = re.sub(r"approximately\s*\\textcolor\{red\}\{\\\$XX\}",
                  f"approximately \\\\${total_cost:.0f}",
                  tex, count=1)

    # Limitation (l) inline kappa range + delta
    tex = tex.replace(
        "judges agree (pairwise $\\kappa$)",
        f"judges agree (pairwise $\\kappa{{=}}{kappa_lo:.2f}$--${kappa_hi:.2f}$)"
    )

    # §7 abstain Cell-6 inline (we only kept the short version, may be no-op)
    # nothing more to fill there.

    TEX.write_text(tex)
    print(f"\nrewrote → {TEX}")


if __name__ == "__main__":
    cell = sys.argv[1] if len(sys.argv) > 1 else "ours_v4tta_label"
    main(cell)
