"""为 full 204 跑出来的 iter0/iter4 metrics 选一个分层代表子集,
渲染重构图并拼成对比图。

由于 run_iter_full.py 没有保存 viz, 这里需要重新跑 pipeline 来重建
mask 然后画图(只针对采样到的代表图,不会爆磁盘)。

输出:
    experiments/results/iter/viz/compare_full/<stratum>__<image_id>.jpg
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

from pipeline import run_pipeline, PipelineResult  # noqa: E402
from experiments.iter.reconstruct import (  # noqa: E402
    draw_reconstruction, draw_gt, horizontal_panel,
)

LABEL_DIR = KMEANS_ROOT / "label"


def stratified_pick(df: pd.DataFrame, n_per: int = 3) -> pd.DataFrame:
    """每 stratum 选 n_per 张代表(low/mid/high IoU_global)。"""
    picks = []
    for s, g in df.groupby("stratum"):
        if len(g) <= n_per:
            picks.append(g)
            continue
        g_sorted = g.sort_values("iou_global").reset_index(drop=True)
        n = len(g_sorted)
        idxs = sorted(set([0, n // 2, n - 1] + [n_per_idx for n_per_idx in
                                                 range(0, n, max(1, n // n_per))]))[:n_per]
        picks.append(g_sorted.iloc[idxs])
    return pd.concat(picks, ignore_index=True)


def run_one(image_id: str, mode: str, k_use: int | None,
            morph_close: int = 0, min_area: int = 0) -> tuple[np.ndarray, PipelineResult]:
    img_path = LABEL_DIR / f"{image_id}.jpg"
    pr = run_pipeline(str(img_path), n_lines=k_use,
                      separator_mode=mode,
                      morph_close_kernel=morph_close,
                      min_cluster_area=min_area)
    img = cv2.imread(str(img_path))
    recon = draw_reconstruction(img.shape, list(pr.cluster_masks or []),
                                thickness=3)
    return recon, pr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter0_csv", required=True,
                    help="full204 iter0 metrics csv")
    ap.add_argument("--iter4_csv", required=True,
                    help="full204 iter4 metrics csv")
    ap.add_argument("--k_csv", required=True)
    ap.add_argument("--backend", default="qwen3.5-plus")
    ap.add_argument("--n_per_stratum", type=int, default=3)
    ap.add_argument("--out_dir",
                    default="experiments/results/iter/viz/compare_full")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d0 = pd.read_csv(args.iter0_csv).set_index("image_id")
    d4 = pd.read_csv(args.iter4_csv).set_index("image_id")
    k_df = pd.read_csv(args.k_csv).set_index("image_id")

    common = sorted(set(d0.index) & set(d4.index))
    base = d4.loc[common].reset_index().copy()
    base["stratum"] = base["stratum"]
    picks = stratified_pick(base, n_per=args.n_per_stratum)
    print(f"picked {len(picks)} representative images")

    n_done = 0
    for _, row in picks.iterrows():
        img_id = row["image_id"]
        ann = LABEL_DIR / f"{img_id}_annotation.json"
        img_path = LABEL_DIR / f"{img_id}.jpg"
        if not img_path.exists() or not ann.exists():
            continue
        d = json.loads(ann.read_text())
        orig = cv2.imread(str(img_path))
        gt_img = draw_gt(d["lines"], orig.shape, thickness=3)

        # iter0: heuristic K, v1
        recon0, _ = run_one(img_id, "v1", None)
        # iter4: LLM K, v4 + morph
        col = f"{args.backend}_k"
        k_use = (int(k_df.at[img_id, col])
                 if img_id in k_df.index and not pd.isna(k_df.at[img_id, col])
                 else None)
        recon4, _ = run_one(img_id, "v4", k_use, morph_close=5, min_area=200)

        m0 = d0.loc[img_id]
        m4 = d4.loc[img_id]
        panels = [
            (f"original  L={row['stratum'].split('_')[0]}  K_gt={int(row['k_gt'])}",
             orig),
            (f"GT lines (n={int(row['k_gt'])})", gt_img),
            (f"iter0_heuristic  K={int(m0['k_pred'])}  "
             f"IoU_g={m0['iou_global']:.3f}  IoU_m={m0['iou_matched_mean']:.3f}",
             recon0),
            (f"iter4_v4_qwen   K={int(m4['k_pred'])}  "
             f"IoU_g={m4['iou_global']:.3f}  IoU_m={m4['iou_matched_mean']:.3f}",
             recon4),
        ]
        out = horizontal_panel(panels)
        cv2.imwrite(str(out_dir / f"{row['stratum']}__{img_id}.jpg"), out)
        n_done += 1
        print(f"  [{n_done:>2}] {row['stratum']:<14s} {img_id[:24]}…  "
              f"IoU_g {m0['iou_global']:.2f}→{m4['iou_global']:.2f}")

    print(f"\nsaved {n_done} compare images → {out_dir}")


if __name__ == "__main__":
    main()
