"""在全 label/204 上跑某一轮迭代,产出 IoU CSV(不渲染重构图,太费空间)。

输出:
    experiments/results/iter/<iter>_full204_metrics.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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

LABEL_DIR = KMEANS_ROOT / "label"


def stratum_for(n: int) -> str:
    if n == 1: return "L1_single"
    if n == 2: return "L2_two_line"
    if n <= 4: return "L3_few_line"
    return "L4_many_line"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", required=True)
    ap.add_argument("--separator_mode", default="v1",
                    choices=["v1", "v2", "v3", "v4"])
    ap.add_argument("--k_source", choices=["heuristic", "llm", "gt_oracle"],
                    default="heuristic")
    ap.add_argument("--k_csv",
                    default="experiments/results/iter/k_full_label_qwen.csv")
    ap.add_argument("--backend", default="qwen3.5-plus")
    ap.add_argument("--m2f_min_fg", type=float, default=0.003)
    ap.add_argument("--gt_thickness", type=int, default=5)
    ap.add_argument("--morph_close_kernel", type=int, default=0)
    ap.add_argument("--min_cluster_area", type=int, default=0)
    ap.add_argument("--out_dir", default="experiments/results/iter")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--normalize_thickness", action="store_true")
    # iter12+ knobs
    ap.add_argument("--m2f_ckpt", default=None,
                    help="absolute path to a Mask2Former checkpoint (default: pipeline default)")
    ap.add_argument("--m2f_threshold", type=float, default=0.35,
                    help="binary threshold on m2f sigmoid prob (was hardcoded 0.35 in pipeline.py)")
    ap.add_argument("--m2f_scales", default="640,1024",
                    help="comma-sep multiscale TTA sizes; e.g. 640,1024 or 512,768,1024")
    ap.add_argument("--flip_tta", action="store_true",
                    help="add horizontal flip TTA in get_line_prob_multiscale")
    ap.add_argument("--thicken", action="store_true",
                    help="skeletonize+dilate-to-5px each cluster_mask after clustering")
    ap.add_argument("--query_nms_mode", default="off",
                    choices=["off", "l4_only", "all", "init"])
    ap.add_argument("--scatter_branch", default="off",
                    choices=["off", "on", "strict"])
    ap.add_argument("--refine_mode", default="off",
                    choices=["off", "basic", "aggr", "edge"])
    args = ap.parse_args()
    scales = tuple(int(s) for s in args.m2f_scales.split(",") if s.strip())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    k_csv = None
    if args.k_source == "llm":
        k_csv = pd.read_csv(args.k_csv).set_index("image_id")

    anns = sorted(LABEL_DIR.glob("*_annotation.json"))
    if args.limit:
        anns = anns[:args.limit]

    rows = []
    t0 = time.time()
    for i, ann in enumerate(anns):
        image_id = ann.stem.replace("_annotation", "")
        img_path = LABEL_DIR / f"{image_id}.jpg"
        if not img_path.exists():
            continue
        gt_dict = json.loads(ann.read_text())
        k_gt = len(gt_dict["lines"])
        # K source
        k_use = None
        k_src = "heuristic"
        if args.k_source == "gt_oracle":
            k_use = k_gt
            k_src = f"gt_oracle({k_gt})"
        elif args.k_source == "llm":
            col = f"{args.backend}_k"
            if image_id in k_csv.index and not pd.isna(k_csv.at[image_id, col]):
                k_use = int(k_csv.at[image_id, col])
                k_src = f"{args.backend}({k_use})"
            else:
                k_src = f"{args.backend}(missing→heuristic)"

        try:
            pr = run_pipeline(str(img_path), n_lines=k_use,
                              m2f_min_fg=args.m2f_min_fg,
                              separator_mode=args.separator_mode,
                              morph_close_kernel=args.morph_close_kernel,
                              min_cluster_area=args.min_cluster_area,
                              m2f_ckpt=args.m2f_ckpt,
                              m2f_threshold=args.m2f_threshold,
                              m2f_scales=scales,
                              flip_tta=args.flip_tta,
                              thicken=args.thicken,
                              query_nms_mode=args.query_nms_mode,
                              scatter_branch=args.scatter_branch,
                              refine_mode=args.refine_mode)
        except Exception as e:
            pr = PipelineResult(image_path=str(img_path), ok=False,
                                used_fallback=False, error=str(e))

        gt_masks, (gw, gh) = load_gt_masks(ann, thickness=args.gt_thickness)
        pred_masks = list(pr.cluster_masks or [])
        if pred_masks and pred_masks[0].shape != (gh, gw):
            pred_masks = [cv2.resize(m.astype(np.uint8), (gw, gh),
                                     interpolation=cv2.INTER_NEAREST) > 0
                          for m in pred_masks]

        ir = compute_iou(image_id, gt_masks, pred_masks,
                         normalize=args.normalize_thickness,
                         thickness=args.gt_thickness)
        per_line = [round(float(v), 4) for v in (ir.iou_per_line or [])]
        n_iou_ge_08 = int(sum(1 for v in per_line if v >= 0.8))
        rows.append({
            "image_id": image_id,
            "stratum": stratum_for(k_gt),
            "k_gt": k_gt,
            "k_pred": pr.n_clusters,
            "k_source": k_src,
            "iou_global": round(ir.iou_global, 4),
            "iou_matched_mean": round(ir.iou_matched_mean, 4),
            "coverage_gt": round(ir.coverage_gt, 4),
            "coverage_pred": round(ir.coverage_pred, 4),
            "n_pred": ir.n_pred,
            "n_iou_ge_0_8": n_iou_ge_08,
            "iou_per_line": json.dumps(per_line),
            "matched_pairs": json.dumps([(int(g), int(p), round(float(v), 4))
                                          for g, p, v in (ir.matched_pairs or [])]),
            "fallback": pr.used_fallback,
            "ok": pr.ok,
        })
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(anns)}] partial IoU_g="
                  f"{np.mean([r['iou_global'] for r in rows]):.3f}  "
                  f"K_acc={np.mean([r['k_pred']==r['k_gt'] for r in rows]):.2f}")

    df = pd.DataFrame(rows)
    p = out_dir / f"{args.iter}_full204_metrics.csv"
    df.to_csv(p, index=False)
    dt = time.time() - t0
    print(f"\n{args.iter}  n={len(df)}  ({dt:.0f}s)")
    print("=== per stratum ===")
    def _pct80(sub: pd.DataFrame) -> float:
        n_lines = int(sub["k_gt"].sum())
        n_hit = int(sub["n_iou_ge_0_8"].sum())
        return n_hit / n_lines if n_lines else 0.0
    print(df.groupby("stratum").apply(lambda s: pd.Series({
        "n": len(s),
        "iou_global": round(float(s["iou_global"].mean()), 3),
        "iou_matched": round(float(s["iou_matched_mean"].mean()), 3),
        "cov_gt": round(float(s["coverage_gt"].mean()), 3),
        "K_acc": round(float((s["k_pred"] == s["k_gt"]).mean()), 3),
        "pct80": round(_pct80(s), 3),
    })).to_string())
    print(f"\nALL  IoU_g={df['iou_global'].mean():.3f}  "
          f"IoU_m={df['iou_matched_mean'].mean():.3f}  "
          f"K_acc={(df['k_pred'] == df['k_gt']).mean():.2f}  "
          f"pct80={_pct80(df):.3f}")
    print(f"saved → {p}")


if __name__ == "__main__":
    main()
