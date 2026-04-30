"""跑一轮迭代:在指定子集上跑 pipeline + IoU + 重构图。

输入:
    --subset <csv>     列至少含 image_id, n_gt_lines, stratum
    --iter   <name>    iter 名 (如 iter0_heuristic, iter1_qwen-vl-max)
    --k_source heuristic | gt_oracle | llm
    --k_csv  <csv>     当 k_source=llm 时, 从该 CSV 读 K(列 <backend>_k)
    --backend <name>   k_source=llm 时指定哪个 backend

输出:
    experiments/results/iter/<iter>_metrics.csv     ← 每张图 IoU 详情
    experiments/results/iter/viz/<iter>/<image_id>.jpg  ← 重构图(本轮)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import os

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(KMEANS_ROOT))

from pipeline import run_pipeline, PipelineResult  # noqa: E402
from experiments.iter.iou_metric import (  # noqa: E402
    load_gt_masks, compute_iou,
)
from experiments.iter.reconstruct import draw_reconstruction  # noqa: E402

LABEL_DIR = KMEANS_ROOT / "label"


def get_k(image_id: str, k_source: str,
          k_gt: int, k_csv: pd.DataFrame | None,
          backend: str | None) -> tuple[int | None, str]:
    """返回 (k, source_str)。"""
    if k_source == "heuristic":
        return None, "heuristic"
    if k_source == "gt_oracle":
        return int(k_gt), f"gt_oracle({k_gt})"
    if k_source == "llm":
        if k_csv is None or backend is None:
            raise ValueError("llm 需要 --k_csv 和 --backend")
        col = f"{backend}_k"
        if col not in k_csv.columns:
            raise ValueError(f"missing column {col}")
        row = k_csv[k_csv["image_id"] == image_id]
        if row.empty:
            return None, f"{backend}(missing→heuristic)"
        v = row.iloc[0][col]
        if pd.isna(v):
            return None, f"{backend}(nan→heuristic)"
        return int(v), f"{backend}({int(v)})"
    raise ValueError(f"bad k_source: {k_source}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="experiments/results/iter_subset12.csv")
    ap.add_argument("--iter", required=True)
    ap.add_argument("--k_source", choices=["heuristic", "gt_oracle", "llm"],
                    default="heuristic")
    ap.add_argument("--k_csv",
                    default="experiments/results/iter/k_predictions_subset12.csv")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--m2f_min_fg", type=float, default=0.003)
    ap.add_argument("--gt_thickness", type=int, default=5)
    ap.add_argument("--out_dir", default="experiments/results/iter")
    ap.add_argument("--separator_mode", default="v1",
                    choices=["v1", "v2", "v3", "v4"])
    ap.add_argument("--normalize_thickness", action="store_true",
                    help="skeletonize+dilate both gt/pred to same thickness")
    ap.add_argument("--morph_close_kernel", type=int, default=0)
    ap.add_argument("--min_cluster_area", type=int, default=0)
    args = ap.parse_args()

    subset = pd.read_csv(args.subset)
    k_csv = None
    if args.k_source == "llm":
        k_csv = pd.read_csv(args.k_csv)

    out_dir = Path(args.out_dir)
    viz_dir = out_dir / "viz" / args.iter
    viz_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    t0 = time.time()
    for i, row in subset.iterrows():
        image_id = row["image_id"]
        img_path = LABEL_DIR / f"{image_id}.jpg"
        ann_path = LABEL_DIR / f"{image_id}_annotation.json"
        if not img_path.exists() or not ann_path.exists():
            print(f"  MISS {image_id}")
            continue
        k_gt = int(row["n_gt_lines"])
        k_use, k_src = get_k(image_id, args.k_source, k_gt, k_csv, args.backend)

        t_im = time.time()
        try:
            pr = run_pipeline(str(img_path), n_lines=k_use,
                              m2f_min_fg=args.m2f_min_fg,
                              separator_mode=args.separator_mode,
                              morph_close_kernel=args.morph_close_kernel,
                              min_cluster_area=args.min_cluster_area)
        except Exception as e:
            pr = PipelineResult(image_path=str(img_path), ok=False,
                                used_fallback=False, error=str(e))
        elapsed = time.time() - t_im

        # GT masks
        gt_masks, (gw, gh) = load_gt_masks(ann_path, thickness=args.gt_thickness)
        # Pred masks (bool)
        pred_masks = [m for m in (pr.cluster_masks or [])]
        # 重新对齐到 GT 图像尺寸 (理论上一致, 但保险起见)
        if pred_masks and pred_masks[0].shape != (gh, gw):
            pred_masks = [cv2.resize(m.astype(np.uint8), (gw, gh),
                                     interpolation=cv2.INTER_NEAREST) > 0
                          for m in pred_masks]

        ir = compute_iou(image_id, gt_masks, pred_masks,
                         normalize=args.normalize_thickness,
                         thickness=args.gt_thickness)

        # 写重构图
        img = cv2.imread(str(img_path))
        recon = draw_reconstruction(img.shape, pred_masks, thickness=3)
        cv2.imwrite(str(viz_dir / f"{image_id}.jpg"), recon)

        rec = {
            "image_id": image_id,
            "stratum": row["stratum"],
            "k_gt": k_gt,
            "k_pred": pr.n_clusters,
            "k_source": k_src,
            "iou_global": round(ir.iou_global, 4),
            "iou_matched_mean": round(ir.iou_matched_mean, 4),
            "coverage_gt": round(ir.coverage_gt, 4),
            "coverage_pred": round(ir.coverage_pred, 4),
            "n_pred": ir.n_pred,
            "fallback": pr.used_fallback,
            "m2f_fg": round(pr.m2f_fg_frac, 4),
            "elapsed_s": round(elapsed, 2),
            "ok": pr.ok,
        }
        rows.append(rec)
        print(f"  [{i+1:>2}/{len(subset)}] {image_id[:24]}…  k_src={k_src:>20s}  "
              f"IoU_g={ir.iou_global:.3f}  IoU_m={ir.iou_matched_mean:.3f}  "
              f"K={pr.n_clusters}/{k_gt}  fb={pr.used_fallback}")

    dt = time.time() - t0
    df = pd.DataFrame(rows)
    metrics_path = out_dir / f"{args.iter}_metrics.csv"
    df.to_csv(metrics_path, index=False)
    print(f"\nsaved → {metrics_path}  ({dt:.1f}s)")

    print("\n=== summary by stratum ===")
    grp = df.groupby("stratum").agg(
        n=("image_id", "count"),
        iou_global=("iou_global", "mean"),
        iou_matched=("iou_matched_mean", "mean"),
        cov_gt=("coverage_gt", "mean"),
        cov_pred=("coverage_pred", "mean"),
    ).round(3)
    print(grp.to_string())
    print(f"\nALL: IoU_global={df['iou_global'].mean():.3f}  "
          f"IoU_matched={df['iou_matched_mean'].mean():.3f}  "
          f"K_acc(exact)={(df['k_pred'] == df['k_gt']).mean():.2f}")


if __name__ == "__main__":
    main()
