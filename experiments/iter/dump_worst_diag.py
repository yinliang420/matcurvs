"""Worst-case 诊断:从 iter7 metrics 抽 IoU 最差的 N 张,渲染 4-panel
(orig | GT | m2f 前景 mask | cluster recon),给 reviewer agent 当输入。

输出:
    experiments/results/iter/viz/worst_diag/<rank>__<stratum>__<image_id>.jpg
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
    draw_reconstruction, draw_gt, horizontal_panel,
)

LABEL_DIR = KMEANS_ROOT / "label"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv",
                    default="experiments/results/iter/iter7_v4_gemini_full204_metrics.csv")
    ap.add_argument("--k_csv",
                    default="experiments/results/iter/k_full_label_gemini.csv")
    ap.add_argument("--backend", default="gemini-3.1-pro-preview")
    ap.add_argument("--top_n", type=int, default=24)
    ap.add_argument("--per_stratum_n", type=int, default=3,
                    help="每个 stratum 最差的 N 张")
    ap.add_argument("--out_dir",
                    default="experiments/results/iter/viz/worst_diag")
    ap.add_argument("--separator_mode", default="v4")
    ap.add_argument("--morph_close_kernel", type=int, default=5)
    ap.add_argument("--min_cluster_area", type=int, default=200)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(args.metrics_csv).sort_values("iou_global")
    k_df = pd.read_csv(args.k_csv).set_index("image_id")

    picks = []
    # 全局最差 top_n
    picks.append(metrics.head(args.top_n).assign(_pick="global"))
    # 每个 stratum 最差 N
    for s, g in metrics.groupby("stratum"):
        picks.append(g.head(args.per_stratum_n).assign(_pick=f"stratum:{s}"))
    picks = pd.concat(picks).drop_duplicates("image_id").reset_index(drop=True)
    print(f"Picked {len(picks)} unique worst images")

    n_done = 0
    for rank, (_, row) in enumerate(picks.iterrows(), 1):
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

        # m2f 前景 mask 渲染:白底 + 黑色二值
        m2f_vis = np.full(orig.shape, 255, dtype=np.uint8)
        if pr.final_mask is not None:
            m2f_vis[pr.final_mask > 0] = (60, 60, 60)
        pred_recon = draw_reconstruction(orig.shape,
                                         list(pr.cluster_masks or []),
                                         thickness=3)

        cap0 = (f"orig  {row['stratum']}  "
                f"K_gt={int(row['k_gt'])} K_pred={int(row['k_pred'])}")
        cap1 = f"GT (n={int(row['k_gt'])})"
        cap2 = f"m2f fg (frac={pr.m2f_fg_frac:.3f})"
        cap3 = (f"pred recon  IoU_g={row['iou_global']:.3f}  "
                f"covGT={row['coverage_gt']:.2f}  covPR={row['coverage_pred']:.2f}")
        panel = horizontal_panel([
            (cap0, orig), (cap1, gt_img),
            (cap2, m2f_vis), (cap3, pred_recon),
        ])
        out_path = out_dir / f"{rank:02d}__{row['stratum']}__{img_id[:24]}.jpg"
        cv2.imwrite(str(out_path), panel)
        n_done += 1
        print(f"  [{n_done:>2}] {row['stratum']:<14s} {img_id[:24]}…  "
              f"IoU_g={row['iou_global']:.3f}  cov_gt={row['coverage_gt']:.2f}  "
              f"cov_pred={row['coverage_pred']:.2f}  m2f_fg={pr.m2f_fg_frac:.3f}")

    print(f"\nsaved {n_done} diag images → {out_dir}")


if __name__ == "__main__":
    main()
