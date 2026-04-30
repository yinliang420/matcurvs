"""在 subset12 上跑所有 LLM 后端,产出 K 预测准确率表。

输出:
    experiments/results/iter/k_predictions_subset12.csv
        列: image_id, stratum, k_gt, <backend>_k, <backend>_lat, <backend>_err
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import os

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(KMEANS_ROOT))

from experiments.iter.llm_k_predictor import (  # noqa: E402
    BACKENDS, predict_k,
)

LABEL_DIR = KMEANS_ROOT / "label"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="experiments/results/iter_subset12.csv")
    ap.add_argument("--out",
                    default="experiments/results/iter/k_predictions_subset12.csv")
    ap.add_argument("--backends", default=",".join(BACKENDS.keys()),
                    help="comma-separated backend keys")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    subset = pd.read_csv(args.subset)
    backends = args.backends.split(",")
    print(f"images: {len(subset)}, backends: {backends}")

    rows = []
    for _, row in subset.iterrows():
        img_id = row["image_id"]
        img_path = LABEL_DIR / f"{img_id}.jpg"
        if not img_path.exists():
            print(f"  MISSING {img_path}")
            continue
        rec = {
            "image_id": img_id,
            "stratum": row["stratum"],
            "k_gt": int(row["n_gt_lines"]),
        }
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(predict_k, str(img_path), be): be for be in backends}
            for fut in as_completed(futs):
                be = futs[fut]
                try:
                    r = fut.result()
                    rec[f"{be}_k"] = r.k_pred
                    rec[f"{be}_lat"] = round(r.latency_s, 2)
                    rec[f"{be}_err"] = r.error or ""
                except Exception as e:
                    rec[f"{be}_k"] = None
                    rec[f"{be}_lat"] = 0.0
                    rec[f"{be}_err"] = f"{type(e).__name__}:{e}"
        rows.append(rec)
        # print live row
        ks = " ".join(f"{be[:6]}={rec.get(f'{be}_k')}" for be in backends)
        print(f"  [{img_id[:24]}…] gt={rec['k_gt']:>2}  {ks}")

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nsaved → {out}")

    # accuracy summary
    print("\n=== K accuracy on subset12 ===")
    for be in backends:
        col = f"{be}_k"
        if col not in df.columns:
            continue
        ok = (df[col].astype("Int64") == df["k_gt"]).sum()
        within1 = (df[col].astype("Int64") - df["k_gt"]).abs().le(1).sum()
        n = df[col].notna().sum()
        lat = df[f"{be}_lat"].mean()
        print(f"  {be:30s}  exact={ok}/{len(df)}  ±1={within1}/{len(df)}  "
              f"valid={n}/{len(df)}  mean_lat={lat:.1f}s")


if __name__ == "__main__":
    main()
