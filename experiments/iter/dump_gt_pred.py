"""为代表性子集导出 GT 图 和 抽取重构图(iter7 最终配置)两个独立 jpg。

输出:
    experiments/results/iter/viz/gt_vs_pred/<stratum>__<image_id>__gt.jpg
    experiments/results/iter/viz/gt_vs_pred/<stratum>__<image_id>__pred.jpg
    experiments/results/iter/viz/gt_vs_pred/<stratum>__<image_id>__orig.jpg  (原图副本, 便于对照)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import os

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(KMEANS_ROOT))

from pipeline import run_pipeline  # noqa: E402
from experiments.iter.reconstruct import (  # noqa: E402
    draw_reconstruction, draw_gt,
)

LABEL_DIR = KMEANS_ROOT / "label"


def stratified_pick(df: pd.DataFrame, n_per: int = 4) -> pd.DataFrame:
    """每 stratum 选 n_per 张代表 (low/mid/mid/high IoU)。"""
    picks = []
    for s, g in df.groupby("stratum"):
        if len(g) <= n_per:
            picks.append(g)
            continue
        gs = g.sort_values("iou_global").reset_index(drop=True)
        n = len(gs)
        idxs = sorted({int(i * (n - 1) / (n_per - 1)) for i in range(n_per)})
        picks.append(gs.iloc[idxs])
    return pd.concat(picks, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv",
                    default="experiments/results/iter/iter7_v4_gemini_full204_metrics.csv")
    ap.add_argument("--k_csv",
                    default="experiments/results/iter/k_full_label_gemini.csv")
    ap.add_argument("--backend", default="gemini-3.1-pro-preview")
    ap.add_argument("--n_per_stratum", type=int, default=4)
    ap.add_argument("--out_dir",
                    default="experiments/results/iter/viz/gt_vs_pred")
    ap.add_argument("--separator_mode", default="v4")
    ap.add_argument("--morph_close_kernel", type=int, default=5)
    ap.add_argument("--min_cluster_area", type=int, default=200)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(args.metrics_csv)
    k_df = pd.read_csv(args.k_csv).set_index("image_id")
    picks = stratified_pick(metrics, n_per=args.n_per_stratum)
    print(f"picked {len(picks)} representative images")

    n_done = 0
    for _, row in picks.iterrows():
        img_id = row["image_id"]
        ann = LABEL_DIR / f"{img_id}_annotation.json"
        img_path = LABEL_DIR / f"{img_id}.jpg"
        if not (img_path.exists() and ann.exists()):
            continue
        d = json.loads(ann.read_text())
        orig = cv2.imread(str(img_path))
        gt_img = draw_gt(d["lines"], orig.shape, thickness=3)

        col = f"{args.backend}_k"
        k_use = (int(k_df.at[img_id, col])
                 if img_id in k_df.index and not pd.isna(k_df.at[img_id, col])
                 else None)
        pr = run_pipeline(str(img_path), n_lines=k_use,
                          separator_mode=args.separator_mode,
                          morph_close_kernel=args.morph_close_kernel,
                          min_cluster_area=args.min_cluster_area)
        pred_img = draw_reconstruction(orig.shape,
                                       list(pr.cluster_masks or []),
                                       thickness=3)

        stem = f"{row['stratum']}__{img_id}"
        cv2.imwrite(str(out_dir / f"{stem}__orig.jpg"), orig)
        cv2.imwrite(str(out_dir / f"{stem}__gt.jpg"), gt_img)
        cv2.imwrite(str(out_dir / f"{stem}__pred.jpg"), pred_img)
        n_done += 1
        print(f"  [{n_done:>2}] {row['stratum']:<14s} {img_id[:24]}…  "
              f"K={pr.n_clusters}/{int(row['k_gt'])}  "
              f"IoU_g={row['iou_global']:.3f}")

    print(f"\nsaved {n_done * 3} files (orig + gt + pred per image) → {out_dir}")


if __name__ == "__main__":
    main()
