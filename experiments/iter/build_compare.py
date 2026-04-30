"""把多轮迭代的重构图按图片横向拼起来:

    [原图 | GT 标注 | iter0 | iter1 | iter2 | ... ]

每张图一个 jpg 在 viz/compare/<image_id>.jpg。
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

from experiments.iter.reconstruct import (  # noqa: E402
    horizontal_panel, draw_gt,
)

LABEL_DIR = KMEANS_ROOT / "label"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="experiments/results/iter_subset12.csv")
    ap.add_argument("--iters", required=True,
                    help="comma-separated iter names, in display order")
    ap.add_argument("--out_dir",
                    default="experiments/results/iter/viz/compare")
    ap.add_argument("--metrics_dir",
                    default="experiments/results/iter")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    iters = [s.strip() for s in args.iters.split(",")]

    # load metrics csvs to annotate captions with IoU
    metrics = {}
    for it in iters:
        p = Path(args.metrics_dir) / f"{it}_metrics.csv"
        if p.exists():
            metrics[it] = pd.read_csv(p).set_index("image_id")

    subset = pd.read_csv(args.subset)
    n_done = 0
    for _, row in subset.iterrows():
        img_id = row["image_id"]
        img_path = LABEL_DIR / f"{img_id}.jpg"
        ann_path = LABEL_DIR / f"{img_id}_annotation.json"
        if not img_path.exists():
            continue
        orig = cv2.imread(str(img_path))
        d = json.loads(ann_path.read_text())
        gt_img = draw_gt(d["lines"], orig.shape, thickness=3)

        panels = [
            (f"original  L={row['stratum'].split('_')[0]}  K_gt={int(row['n_gt_lines'])}",
             orig),
            (f"GT lines (n={int(row['n_gt_lines'])})", gt_img),
        ]
        for it in iters:
            recon_path = (Path(args.metrics_dir) / "viz" / it /
                          f"{img_id}.jpg")
            if not recon_path.exists():
                continue
            recon = cv2.imread(str(recon_path))
            cap = it
            if it in metrics and img_id in metrics[it].index:
                m = metrics[it].loc[img_id]
                cap = (f"{it}  K={int(m['k_pred'])}  "
                       f"IoU_g={m['iou_global']:.3f}  "
                       f"IoU_m={m['iou_matched_mean']:.3f}")
            panels.append((cap, recon))

        out = horizontal_panel(panels)
        cv2.imwrite(str(out_dir / f"{row['stratum']}__{img_id}.jpg"), out)
        n_done += 1
        print(f"  [{n_done:>2}] {row['stratum']:<14s} {img_id[:24]}…  -> "
              f"{out.shape[1]}x{out.shape[0]}")

    print(f"\nsaved {n_done} compare images → {out_dir}")


if __name__ == "__main__":
    main()
