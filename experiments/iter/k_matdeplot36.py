"""K predictor for MatDeplot-36 image_ids (paper_id/sha_N format).

The label/ K predictor (k_full_label.py) operates on bare sha_N IDs from
the kmeans/label/ directory.  MatDeplot-36 uses image_ids of the form
``<paper_id>/<sha>_<n>`` and the JPEGs live under
``MATDEPLOT_ROOT/data/raw/<paper_id>/<sha>_<n>.jpg``.  This script
mirrors k_full_label.py for that path layout and writes one K cache
suitable for re-running run_eval.py with --k_csv.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import os

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[2]))
MATDEPLOT_ROOT = Path(os.environ.get("MATDEPLOT_ROOT", "/path/to/MatDeplot/mdp_v2"))
sys.path.insert(0, str(KMEANS_ROOT))

from experiments.iter.llm_k_predictor import predict_k                   # noqa: E402
from experiments.kmeans_matdeplot_adapter import get_test_image_paths    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qwen3.5-plus")
    ap.add_argument("--out", default=str(
        KMEANS_ROOT / "experiments/results/iter/k_matdeplot36.csv"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    paths = get_test_image_paths()
    gt = pd.read_parquet(MATDEPLOT_ROOT / "data/gt_unified.parquet")
    gt_by_id = {iid: gt[gt.image_id == iid].iloc[0]
                  for iid in {p[0] for p in paths}
                  if len(gt[gt.image_id == iid])}
    items = []
    for image_id, image_path in paths:
        if image_id not in gt_by_id:
            continue
        # n_gt_lines from gt parquet ('n_trails' or similar)
        try:
            row = gt_by_id[image_id]
            n_gt = int(row.get("n_trails", 0) or 0)
            if n_gt <= 0:
                # try counting non-null trails fields
                cols = [c for c in row.index if c.startswith("trail_")]
                n_gt = sum(1 for c in cols if pd.notna(row[c]))
        except Exception:
            n_gt = 0
        items.append((image_id, image_path, n_gt))

    print(f"running {args.backend} K predictor on {len(items)} "
          f"MatDeplot-36 images, workers={args.workers}")

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut_to = {ex.submit(predict_k, p, args.backend): (iid, n) for iid, p, n in items}
        for fut in as_completed(fut_to):
            iid, n_gt = fut_to[fut]
            try:
                r = fut.result()
                rows.append({
                    "image_id": iid,
                    "n_gt_lines": n_gt,
                    f"{args.backend}_k": r.k_pred,
                    f"{args.backend}_lat": r.latency_s,
                    f"{args.backend}_err": r.error or "",
                })
                print(f"  {iid[:36]:>36s}  K_gt={n_gt} K_pred={r.k_pred} "
                       f"t={r.latency_s:.1f}s err={r.error or '-'}")
            except Exception as e:
                rows.append({
                    "image_id": iid,
                    "n_gt_lines": n_gt,
                    f"{args.backend}_k": None,
                    f"{args.backend}_lat": 0.0,
                    f"{args.backend}_err": f"{type(e).__name__}: {e}",
                })

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    n_ok = int((df[f"{args.backend}_k"].notna()).sum())
    if n_ok:
        acc = float((df[f"{args.backend}_k"] == df.n_gt_lines).mean())
        print(f"\nsaved → {out_path}  (n_ok={n_ok}/{len(df)} accuracy={acc:.3f})")


if __name__ == "__main__":
    main()
