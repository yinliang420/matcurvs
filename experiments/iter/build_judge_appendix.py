"""Build §K appendix tables from a judge_<cell>.csv produced by
``run_vlm_judge_sweep.py``:

    Tab K.1  judge_passrate_<cell>.csv        per-judge image-pass rates
    Tab K.2  judge_kappa_<cell>.csv           pairwise Cohen's κ
    Tab K.3  judge_lr_correlation_<cell>.csv  judge flag vs GT-derived LR_0.5
                                                 on n=90 hold
    Cons.    judge_consensus_<cell>.csv       summary of >=2-judge consensus
                                                 abstain rule
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import os

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[2]))
ITER = KMEANS_ROOT / "experiments" / "results" / "iter"
RES = KMEANS_ROOT / "experiments" / "results"
HOLD = set(open(KMEANS_ROOT / "experiments/results/label_clean_holdout.txt"
                  ).read().strip().split())

# Display names (for table prose only); CSV uses backend identifiers.
JUDGE_DISPLAY = {
    "qwen-vl-max":            "Qwen-VL-Max",
    "qwen3-vl-plus":          "Qwen3-VL-Plus",
    "gpt-5.4":                "GPT-5.4",
    "gemini-3.1-pro-preview": "Gemini-3.1-pro",
    "claude-opus-4-6":        "Claude Opus 4.6",
}


def _coerce_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "yes", "pass", "1"}
    )


def cohen_kappa(a, b) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    n = len(a)
    if n == 0:
        return float("nan")
    po = float((a == b).mean())
    pa = float(a.mean())
    pb = float(b.mean())
    pe = pa * pb + (1 - pa) * (1 - pb)
    if abs(1.0 - pe) < 1e-9:
        return 1.0 if po >= 1.0 - 1e-9 else 0.0
    return float((po - pe) / (1 - pe))


def _per_image_lr05(phaseC_csv: Path, image_subset: set[str]) -> pd.DataFrame:
    """Return DataFrame[image_id, lr05] from phaseC_flip_full204_metrics.csv
    restricted to ``image_subset``.  ``lr05`` is the per-image fraction of
    GT lines whose IoU >= 0.5."""
    df = pd.read_csv(phaseC_csv)
    sub = df[df.image_id.isin(image_subset)].copy()

    def _row_lr05(s):
        try:
            vs = ast.literal_eval(s) if isinstance(s, str) and s else []
        except Exception:
            return np.nan
        if not vs:
            return np.nan
        return float(np.mean([1.0 if v >= 0.5 else 0.0 for v in vs]))

    sub["lr05"] = sub["iou_per_line"].apply(_row_lr05)
    return sub[["image_id", "lr05"]].dropna()


def main(cell: str = "ours_v4tta_label") -> dict:
    judge_csv = ITER / f"judge_{cell}.csv"
    if not judge_csv.exists():
        raise SystemExit(f"missing {judge_csv}")
    df = pd.read_csv(judge_csv)
    # only valid rows: image_pass parsed True/False, no error.
    df["image_pass_bool"] = _coerce_bool(df["image_pass"])
    # A row is valid iff image_pass parsed to a real bool (True/False).
    # Errored rows have image_pass=NaN (we don't separately filter on the
    # error column; NaN error / empty error are both fine).
    df["valid"] = df["image_pass"].astype(str).isin(["True", "False"])
    df = df[df.valid].copy()
    if not len(df):
        raise SystemExit("no valid verdict rows")

    judges = sorted(df.judge.unique().tolist())
    print(f"judges: {judges}  total verdicts: {len(df)}")

    # ----------------- Tab K.1 per-judge pass rate -----------------
    rows = []
    for j in judges:
        sub = df[df.judge == j]
        sub_h = sub[sub.image_id.isin(HOLD)]
        rows.append({
            "judge": j,
            "n_204": len(sub),
            "image_pass_204": round(float(sub.image_pass_bool.mean()), 3),
            "n_90": len(sub_h),
            "image_pass_90": round(float(sub_h.image_pass_bool.mean()),
                                     3) if len(sub_h) else float("nan"),
            "median_latency": round(float(sub.latency_s.median()), 2),
        })
    tab_passrate = pd.DataFrame(rows)
    out_passrate = ITER / f"judge_passrate_{cell}.csv"
    tab_passrate.to_csv(out_passrate, index=False)
    print("\n=== Tab K.1 per-judge pass rate ===")
    print(tab_passrate.to_string(index=False))

    # ----------------- Tab K.2 Cohen's κ -----------------
    pivot = df.pivot_table(index="image_id", columns="judge",
                            values="image_pass_bool", aggfunc="first")
    # Some judges may have failed on some images.  Cohen's κ is computed
    # only on the common-image set for each pair, not the strict
    # all-judges intersection (which can be empty if any judge errored
    # on too many images).
    n_common_all = len(pivot.dropna())
    print(f"\n=== Tab K.2 Cohen's κ (common-pair n) ===")
    K = pd.DataFrame(index=judges, columns=judges, dtype=float)
    n_pair = pd.DataFrame(index=judges, columns=judges, dtype=int)
    for a in judges:
        for b in judges:
            if a == b:
                K.loc[a, b] = 1.0
                n_pair.loc[a, b] = int(pivot[a].notna().sum())
                continue
            both = pivot[[a, b]].dropna()
            if not len(both):
                K.loc[a, b] = float("nan")
                n_pair.loc[a, b] = 0
                continue
            K.loc[a, b] = round(cohen_kappa(both[a], both[b]), 3)
            n_pair.loc[a, b] = len(both)
    out_kappa = ITER / f"judge_kappa_{cell}.csv"
    K.to_csv(out_kappa)
    n_pair.to_csv(ITER / f"judge_kappa_npairs_{cell}.csv")
    print(K.to_string())
    print(f"(strict-intersection n_common = {n_common_all})")

    # ----------------- Tab K.3 judge vs GT-derived LR_0.5 -----------------
    lr_df = _per_image_lr05(ITER / "phaseC_flip_full204_metrics.csv", HOLD)
    rows = []
    for j in judges:
        sub = df[(df.judge == j) & df.image_id.isin(HOLD)
                  ].set_index("image_id")
        merged = lr_df.set_index("image_id").join(
            sub[["image_pass_bool"]], how="inner"
        )
        if not len(merged):
            rows.append({"judge": j, "n_pass": 0, "n_fail": 0,
                          "lr05_pass": float("nan"),
                          "lr05_fail": float("nan"),
                          "delta": float("nan")})
            continue
        pass_set = merged[merged.image_pass_bool == True]
        fail_set = merged[merged.image_pass_bool == False]
        lr_pass = float(pass_set.lr05.mean()) if len(pass_set) else float("nan")
        lr_fail = float(fail_set.lr05.mean()) if len(fail_set) else float("nan")
        rows.append({
            "judge": j,
            "n_pass": int(len(pass_set)),
            "n_fail": int(len(fail_set)),
            "lr05_pass": round(lr_pass, 3) if not np.isnan(lr_pass) else float("nan"),
            "lr05_fail": round(lr_fail, 3) if not np.isnan(lr_fail) else float("nan"),
            "delta": (round(lr_pass - lr_fail, 3)
                       if not (np.isnan(lr_pass) or np.isnan(lr_fail))
                       else float("nan")),
        })
    tab_lr = pd.DataFrame(rows)
    out_lr = ITER / f"judge_lr_correlation_{cell}.csv"
    tab_lr.to_csv(out_lr, index=False)
    print("\n=== Tab K.3 judge image-pass vs GT-derived LR_0.5 (n=90) ===")
    print(tab_lr.to_string(index=False))

    # ----------------- Consensus rule (≥2 of 4 judges flag fail) -----------------
    # Each row is one image; columns are per-judge image_pass (NaN if
    # judge errored on that image).  We fail-vote with NaN→PASS (the
    # missing judge does not flag).
    pivot_h = pivot[pivot.index.isin(HOLD)].copy()
    cons_rows = []
    for thresh in (1, 2, 3):
        # number of judges that voted FAIL
        n_fail_votes = pivot_h.apply(
            lambda r: int(((~r.fillna(True).astype(bool)).sum())), axis=1
        )
        consensus_fail = (n_fail_votes >= thresh)
        n_total = len(consensus_fail)
        n_abstain = int(consensus_fail.sum())
        # mean LR_0.5 on the consensus-pass vs consensus-fail subsets
        joined = lr_df.set_index("image_id").join(
            consensus_fail.rename("consensus_fail"), how="inner"
        )
        lr_pass = float(joined[~joined.consensus_fail].lr05.mean()) \
            if (~joined.consensus_fail).any() else float("nan")
        lr_fail = float(joined[joined.consensus_fail].lr05.mean()) \
            if joined.consensus_fail.any() else float("nan")
        cons_rows.append({
            "threshold_fail_votes_ge": thresh,
            "n_total": n_total,
            "n_abstain": n_abstain,
            "abstain_rate": round(n_abstain / max(n_total, 1), 3),
            "lr05_kept": (round(lr_pass, 3)
                            if not np.isnan(lr_pass) else float("nan")),
            "lr05_abstained": (round(lr_fail, 3)
                                if not np.isnan(lr_fail) else float("nan")),
            "lr05_delta": (round(lr_pass - lr_fail, 3)
                            if not (np.isnan(lr_pass) or np.isnan(lr_fail))
                            else float("nan")),
        })
    cons_df = pd.DataFrame(cons_rows)
    out_cons = ITER / f"judge_consensus_{cell}.csv"
    cons_df.to_csv(out_cons, index=False)
    print("\n=== Cell-6 consensus rule sweep (n=90 hold) ===")
    print(cons_df.to_string(index=False))

    return {
        "passrate_csv": str(out_passrate),
        "kappa_csv": str(out_kappa),
        "lr_csv": str(out_lr),
        "consensus_csv": str(out_cons),
        "tab_passrate": tab_passrate.to_dict(orient="records"),
        "tab_kappa": K.to_dict(),
        "tab_lr": tab_lr.to_dict(orient="records"),
        "consensus": cons_df.to_dict(orient="records"),
    }


if __name__ == "__main__":
    cell = sys.argv[1] if len(sys.argv) > 1 else "ours_v4tta_label"
    summary = main(cell)
    print(f"\n=== artefacts ===")
    for k, v in summary.items():
        if k.endswith("_csv"):
            print(f"  {k:>16s}  {v}")
