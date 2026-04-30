"""并行跑全 label/204 的 K 预测,使用 qwen3.5-plus(子集上准确率最高)。

输出: experiments/results/iter/k_full_label_qwen.csv
列: image_id, n_gt_lines, qwen3.5-plus_k, qwen3.5-plus_lat, qwen3.5-plus_err
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

from experiments.iter.llm_k_predictor import predict_k  # noqa: E402

LABEL_DIR = KMEANS_ROOT / "label"


def gt_k(ann_path: Path) -> int:
    return len(json.loads(ann_path.read_text())["lines"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qwen3.5-plus")
    ap.add_argument("--out",
                    default="experiments/results/iter/k_full_label_qwen.csv")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    anns = sorted(LABEL_DIR.glob("*_annotation.json"))
    if args.limit:
        anns = anns[:args.limit]
    items = [(a.stem.replace("_annotation", ""), gt_k(a)) for a in anns]
    print(f"label set: {len(items)} images, backend={args.backend}, "
          f"workers={args.workers}")

    rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut_to_id = {}
        for img_id, k_gt in items:
            img_path = LABEL_DIR / f"{img_id}.jpg"
            if not img_path.exists():
                continue
            fut = ex.submit(predict_k, str(img_path), args.backend)
            fut_to_id[fut] = (img_id, k_gt)
        for fut in as_completed(fut_to_id):
            img_id, k_gt = fut_to_id[fut]
            try:
                r = fut.result()
                rows.append({
                    "image_id": img_id,
                    "n_gt_lines": k_gt,
                    f"{args.backend}_k": r.k_pred,
                    f"{args.backend}_lat": round(r.latency_s, 2),
                    f"{args.backend}_err": r.error or "",
                })
            except Exception as e:
                rows.append({
                    "image_id": img_id, "n_gt_lines": k_gt,
                    f"{args.backend}_k": None,
                    f"{args.backend}_lat": 0.0,
                    f"{args.backend}_err": f"{type(e).__name__}:{e}",
                })
            done += 1
            if done % 10 == 0:
                ok = sum(1 for r in rows if r[f"{args.backend}_k"] is not None)
                print(f"  {done}/{len(items)}  valid={ok}")

    df = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nsaved → {out}")

    # accuracy
    col = f"{args.backend}_k"
    if col in df:
        valid = df[col].notna().sum()
        exact = ((df[col].astype("Int64") == df["n_gt_lines"]).sum())
        within1 = ((df[col].astype("Int64") - df["n_gt_lines"]).abs().le(1).sum())
        print(f"valid {valid}/{len(df)}  exact {exact}/{len(df)}  "
              f"±1 {within1}/{len(df)}  mean_lat {df[f'{args.backend}_lat'].mean():.1f}s")


if __name__ == "__main__":
    main()
