"""
MatCurvs-Reasoning scorer.

Reads all answer JSONLs in experiments/mr_answers/<set>_<method>_<model>.jsonl
and produces a scoring report.

Per-item metric:
  - scalar GT: abs_err = |answer - gt|
  - list GT (e.g. raman_top3_peaks): mean abs distance after sorted-pair match
  - failure-to-parse → counted as "no_answer"

Aggregates: mean / median / IQR of abs_err, plus pass rate
(within-tolerance fraction; tolerance per task family).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np
import os

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[1]))
EXP_DIR = KMEANS_ROOT / "experiments"
ANS_DIR = EXP_DIR / "mr_answers"


def normalize_answer(a):
    if a is None:
        return None
    if isinstance(a, (int, float)):
        return [float(a)]
    if isinstance(a, list):
        out = []
        for v in a:
            try:
                out.append(float(v))
            except Exception:
                pass
        return out if out else None
    if isinstance(a, str):
        # try to parse "1.2, 3.4" or "[1.2, 3.4]"
        s = a.strip().strip("[]()").replace(";", ",")
        parts = [p.strip() for p in s.split(",") if p.strip()]
        out = []
        for p in parts:
            try:
                out.append(float(p))
            except Exception:
                pass
        return out if out else None
    return None


def err_for(answer, gt):
    a = normalize_answer(answer)
    g = normalize_answer(gt)
    if a is None or g is None or len(a) == 0 or len(g) == 0:
        return None
    if len(g) == 1:
        return abs(a[0] - g[0])
    # list: pair sorted, mean abs diff (truncate to min length)
    a_sorted = sorted(a); g_sorted = sorted(g)
    n = min(len(a_sorted), len(g_sorted))
    if n == 0:
        return None
    return float(np.mean([abs(a_sorted[i] - g_sorted[i]) for i in range(n)]))


def relative_err_for(answer, gt):
    """Relative error normalized by |gt| (or 1 if |gt|<1) for cross-task aggregation."""
    err = err_for(answer, gt)
    if err is None:
        return None
    g = normalize_answer(gt)
    if g is None or not g:
        return None
    scale = max(abs(np.mean(g)), 1.0)
    return err / scale


def load_answers(set_name: str) -> dict:
    """Returns {(method, model): [rows]}"""
    out = collections.defaultdict(list)
    for fp in sorted(ANS_DIR.glob(f"{set_name}_*_*.jsonl")):
        # filename pattern: {set}_{method}_{model}.jsonl
        stem = fp.stem
        try:
            _, method, model = stem.split("_", 2)
        except ValueError:
            continue
        for line in fp.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            r["_method"] = method
            r["_model"] = model
            out[(method, model)].append(r)
    return out


def summarize(rows: list, key_fn=lambda r: ()) -> dict:
    """Group rows by key_fn and compute MAE / median / pass rate."""
    groups = collections.defaultdict(list)
    for r in rows:
        gt = r.get("gt_answer")
        ans = r.get("answer")
        rel = relative_err_for(ans, gt)
        groups[key_fn(r)].append({
            "abs_err": err_for(ans, gt),
            "rel_err": rel,
            "no_answer": ans is None,
        })
    summary = {}
    for k, items in groups.items():
        n = len(items)
        no_ans = sum(1 for d in items if d["no_answer"])
        rels = [d["rel_err"] for d in items if d["rel_err"] is not None]
        abss = [d["abs_err"] for d in items if d["abs_err"] is not None]
        # within-instr pass: rel_err < 0.05 (≈ within 5% of GT)
        passed = sum(1 for d in items if (d["rel_err"] is not None and d["rel_err"] < 0.05))
        summary[k] = {
            "n": n,
            "no_answer": no_ans,
            "mean_rel_err": float(np.mean(rels)) if rels else None,
            "median_rel_err": float(np.median(rels)) if rels else None,
            "p25_rel_err": float(np.percentile(rels, 25)) if rels else None,
            "p75_rel_err": float(np.percentile(rels, 75)) if rels else None,
            "mean_abs_err": float(np.mean(abss)) if abss else None,
            "median_abs_err": float(np.median(abss)) if abss else None,
            "pass_rate_5pct": passed / n if n else None,
        }
    return summary


def fmt_pct(x):
    return "—" if x is None else f"{x*100:.1f}%"


def fmt_num(x, p=3):
    if x is None: return "—"
    if abs(x) >= 1000 or (abs(x) < 0.01 and x != 0):
        return f"{x:.3e}"
    return f"{x:.{p}g}"


def print_overall_table(set_name: str, by_method: dict):
    print(f"\n{'='*90}")
    print(f"{set_name.upper()} — overall by method × model (rel-err normalized by |GT|)")
    print('='*90)
    print(f"{'Method':<6} {'Model':<28} {'N':>4} {'noAns':>5} {'meanRel':>9} {'medRel':>8} {'pass≤5%':>9}")
    print('-'*90)
    for (method, model), rows in sorted(by_method.items()):
        s = summarize(rows)[()]
        print(f"{method:<6} {model:<28} {s['n']:>4d} {s['no_answer']:>5d} "
              f"{fmt_pct(s['mean_rel_err']):>9} {fmt_pct(s['median_rel_err']):>8} "
              f"{fmt_pct(s['pass_rate_5pct']):>9}")


def print_per_level(set_name: str, by_method: dict):
    print(f"\n{set_name.upper()} — per-level (rel-err median / pass-rate)")
    levels = sorted({r["level"] for rows in by_method.values() for r in rows if "level" in r})
    methods = sorted(by_method.keys())
    print(f"{'Level':<10} | " + " | ".join(f"{m[0]+'+'+m[1][:14]:<22}" for m in methods))
    for lvl in levels:
        cells = []
        for m in methods:
            rows = [r for r in by_method[m] if r.get("level") == lvl]
            s = summarize(rows)[()] if rows else {"median_rel_err": None, "pass_rate_5pct": None, "n": 0}
            cells.append(f"med={fmt_pct(s.get('median_rel_err')):<7} ({s.get('n',0):>3d})")
        print(f"{lvl:<10} | " + " | ".join(f"{c:<22}" for c in cells))


def print_per_family(set_name: str, by_method: dict):
    print(f"\n{set_name.upper()} — per-family (median rel-err)")
    fams = sorted({r.get("family", "") for rows in by_method.values() for r in rows})
    methods = sorted(by_method.keys())
    header = f"{'Family':<10} | " + " | ".join(f"{m[0]+'+'+m[1][:16]:<24}" for m in methods)
    print(header)
    print('-'*len(header))
    for fam in fams:
        cells = []
        for m in methods:
            rows = [r for r in by_method[m] if r.get("family", "") == fam]
            s = summarize(rows)[()] if rows else {"median_rel_err": None, "n": 0}
            cells.append(f"{fmt_pct(s.get('median_rel_err')):<10} (n={s.get('n',0):>3d})")
        print(f"{fam:<10} | " + " | ".join(f"{c:<24}" for c in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["rigorous", "scaled"], default="rigorous")
    args = ap.parse_args()

    by_method = load_answers(args.set)
    if not by_method:
        print(f"no answer files found in {ANS_DIR} for set={args.set}")
        return

    print(f"Loaded {sum(len(v) for v in by_method.values()):,} answers across {len(by_method)} method-model combos")
    print_overall_table(args.set, by_method)
    print_per_level(args.set, by_method)
    print_per_family(args.set, by_method)


if __name__ == "__main__":
    main()
