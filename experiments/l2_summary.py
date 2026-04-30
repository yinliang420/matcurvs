"""Summarize judgment_log.jsonl from l2_curate.py: overall + per-format + per-battery + failure breakdown."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path,
                    default=Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[1])) / "data_base_L2" / "judgment_log.jsonl")
    args = ap.parse_args()

    if not args.log.exists():
        print(f"log not found: {args.log}")
        return
    rows = []
    with open(args.log, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    n = len(rows)
    if n == 0:
        print("no rows in log")
        return

    n_good = sum(1 for r in rows if r.get("decision") == "good")
    n_bad = sum(1 for r in rows if r.get("decision") == "bad")
    n_err = sum(1 for r in rows if r.get("error"))
    elapsed = sum(r.get("elapsed_s", 0) for r in rows)

    print("=" * 60)
    print(f"Overall: {n:,} panels judged")
    print(f"  ACCEPTED (good): {n_good:,} ({n_good/n*100:.2f}%)")
    print(f"  REJECTED (bad):  {n_bad:,} ({n_bad/n*100:.2f}%)")
    print(f"  errored:         {n_err:,} ({n_err/n*100:.2f}%)")
    print(f"  total LLM elapsed (sum of per-panel): {elapsed/60:.1f} min")
    print()

    print("By format:")
    by_fmt = collections.defaultdict(lambda: {"n": 0, "good": 0})
    for r in rows:
        f = r.get("fmt", "?")
        by_fmt[f]["n"] += 1
        if r.get("decision") == "good":
            by_fmt[f]["good"] += 1
    for fmt, d in sorted(by_fmt.items()):
        rate = d["good"] / d["n"] * 100 if d["n"] else 0
        print(f"  {fmt}: {d['good']:,}/{d['n']:,}  ({rate:.2f}%)")
    print()

    print("By battery (top 15 by accept count):")
    by_bat = collections.defaultdict(lambda: {"n": 0, "good": 0})
    for r in rows:
        b = r.get("battery", "?")
        by_bat[b]["n"] += 1
        if r.get("decision") == "good":
            by_bat[b]["good"] += 1
    sorted_bat = sorted(by_bat.items(), key=lambda kv: -kv[1]["good"])
    for bat, d in sorted_bat[:15]:
        rate = d["good"] / d["n"] * 100 if d["n"] else 0
        print(f"  {bat:25s} {d['good']:5,}/{d['n']:5,}  ({rate:5.2f}%)")
    print()

    print("By number of curves (n_curves bucket):")
    by_nc = collections.defaultdict(lambda: {"n": 0, "good": 0})
    for r in rows:
        nc = r.get("n_curves", 0)
        bucket = nc if nc <= 6 else 7
        by_nc[bucket]["n"] += 1
        if r.get("decision") == "good":
            by_nc[bucket]["good"] += 1
    for nc in sorted(by_nc):
        d = by_nc[nc]
        rate = d["good"] / d["n"] * 100 if d["n"] else 0
        label = f"≥{nc}" if nc == 7 else str(nc)
        print(f"  n_curves={label:4s}: {d['good']:5,}/{d['n']:5,}  ({rate:5.2f}%)")
    print()

    print("Top rejection failure modes (over rejected only):")
    fails = collections.Counter()
    for r in rows:
        if r.get("decision") != "bad":
            continue
        if not r.get("curve_count_match"):
            fails["count_mismatch"] += 1
        elif not r.get("shape_match"):
            fails["shape_mismatch"] += 1
        elif not r.get("range_plausible"):
            fails["range_implausible"] += 1
        elif not r.get("noise_free"):
            fails["noisy"] += 1
        else:
            fails["other_or_error"] += 1
    for k, v in fails.most_common():
        print(f"  {k:18s} {v:6,}  ({v/max(n_bad,1)*100:.2f}% of rejects)")


if __name__ == "__main__":
    main()
