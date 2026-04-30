"""POC: end-to-end VLM-supervised cluster-merge on a small sample.

Steps for each test image:
    1. Load GT polylines from label/<id>_annotation.json.
    2. Call Qwen-VL → curve metadata (K + colour + anchors).
    3. Run pipeline.run_pipeline(...) with two flavours:
         A. baseline pixel-mode: n_lines = K_VLM (default behaviour).
         B. over-cluster + merge:
              n_lines = K_VLM + buffer (default buffer=2),
              then merge cluster_masks with vlm_cluster_merge.
    4. Compute Inst-F1 for both A and B against GT.
    5. Print per-image and aggregate deltas.

Outputs:
    - prints A/B/Δ per image
    - writes ``experiments/results/iter/poc_vlm_merge.csv``
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
MATDEPLOT_ROOT = Path(os.environ.get("MATDEPLOT_ROOT", "/path/to/MatDeplot/mdp_v2"))
sys.path.insert(0, str(KMEANS_ROOT))
sys.path.insert(0, str(MATDEPLOT_ROOT))

from pipeline import run_pipeline, PipelineResult  # noqa: E402
from skimage.morphology import skeletonize  # noqa: E402

from experiments.iter.llm_curves_predictor import predict_curves, CurvesPred  # noqa: E402
from experiments.iter.vlm_cluster_merge import (  # noqa: E402
    merge_clusters_with_vlm, merge_redundant_clusters, vlm_anchor_assign,
)

from matdeplot.eval.metrics import instance_pr  # noqa: E402

LABEL_DIR = KMEANS_ROOT / "label"


def mask_to_polyline(mask: np.ndarray, sample_every: int = 2) -> np.ndarray:
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if mask.sum() < 10:
        return np.zeros((0, 2), dtype=float)
    sk = skeletonize(mask)
    ys, xs = np.where(sk)
    if len(xs) < 5:
        ys, xs = np.where(mask)
        if len(xs) < 5:
            return np.zeros((0, 2), dtype=float)
    df = pd.DataFrame({"x": xs, "y": ys}).groupby("x")["y"].mean().reset_index()
    df = df.sort_values("x").reset_index(drop=True)
    if sample_every > 1 and len(df) > 4 * sample_every:
        df = df.iloc[::sample_every].reset_index(drop=True)
    return df[["x", "y"]].to_numpy(dtype=float)


def load_gt_trails(ann_path: Path):
    d = json.loads(ann_path.read_text())
    w, h = d["image_size"]
    trails = []
    for _, ln in d["lines"].items():
        pts = ln["points"]
        arr = np.array([[p["pixel"][0], p["pixel"][1]] for p in pts], dtype=float)
        trails.append(arr)
    return trails, (w, h)


def plot_box_from_axis_lines(axis_lines, fallback_shape):
    """Return (x0, y0, x1, y1) for the inside-plot box."""
    if axis_lines is None or axis_lines[0] is None or axis_lines[1] is None:
        h, w = fallback_shape[:2]
        return (0, 0, w, h)
    x_axis, y_axis = axis_lines
    x_coords = [x_axis[0], x_axis[2], y_axis[0], y_axis[2]]
    y_coords = [x_axis[1], x_axis[3], y_axis[1], y_axis[3]]
    return (int(min(x_coords)), int(min(y_coords)),
            int(max(x_coords)), int(max(y_coords)))


def inst_f1(pred_trails, gt_trails) -> tuple[float, int, int, int]:
    if not pred_trails and not gt_trails:
        return 1.0, 0, 0, 0
    res = instance_pr(pred_trails, gt_trails, distance_threshold=0.3)
    return res["f1"], res["tp"], res["fp"], res["fn"]


def trails_from_pipeline(r: PipelineResult) -> list[np.ndarray]:
    out = []
    for m in r.cluster_masks:
        arr = mask_to_polyline(m)
        if arr.shape[0] >= 10:
            out.append(arr)
    return out


def trails_from_masks(masks) -> list[np.ndarray]:
    out = []
    for m in masks:
        arr = mask_to_polyline(m)
        if arr.shape[0] >= 10:
            out.append(arr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="number of images to test")
    ap.add_argument("--backend", default="qwen-vl-max",
                    choices=["qwen-vl-max", "qwen3-vl-plus"])
    ap.add_argument("--buffer", type=int, default=2,
                    help="over-cluster by K_VLM + buffer")
    ap.add_argument("--out_csv",
                    default=str(KMEANS_ROOT /
                                "experiments/results/iter/poc_vlm_merge.csv"))
    ap.add_argument("--shuffle_seed", type=int, default=42)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--hold_only", action="store_true",
                    help="restrict to n=90 leakage-clean hold images")
    ap.add_argument("--hard_only", action="store_true",
                    help="restrict to images where baseline pixel-mode F1 < 0.7"
                          " (uses cached ours_v4tta_label.csv)")
    args = ap.parse_args()

    anns = sorted(LABEL_DIR.glob("*_annotation.json"))
    if args.hold_only:
        hold = set(open(KMEANS_ROOT /
                         "experiments/results/label_clean_holdout.txt"
                         ).read().strip().split())
        anns = [a for a in anns if a.stem.replace("_annotation", "") in hold]
    if args.hard_only:
        cache = pd.read_csv(KMEANS_ROOT /
                             "experiments/results/ours_v4tta_label.csv")
        hard_ids = set(cache[cache.instance_f1 < 0.7].image_id.tolist())
        anns = [a for a in anns if a.stem.replace("_annotation", "") in hard_ids]
    rng = np.random.default_rng(args.shuffle_seed)
    perm = rng.permutation(len(anns))
    anns = [anns[i] for i in perm][args.start:args.start + args.n]
    print(f"sampling {len(anns)} images (seed={args.shuffle_seed}, start={args.start}; "
          f"hold_only={args.hold_only}, hard_only={args.hard_only})")

    rows = []
    for i, ann in enumerate(anns):
        image_id = ann.stem.replace("_annotation", "")
        img_path = LABEL_DIR / f"{image_id}.jpg"
        if not img_path.exists():
            continue
        gt_trails, _ = load_gt_trails(ann)
        K_gt = len(gt_trails)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        # 1. VLM call
        t0 = time.monotonic()
        cp = predict_curves(img_path, backend=args.backend)
        t_vlm = time.monotonic() - t0
        K_vlm = cp.k_pred or len(cp.curves) or K_gt
        if K_vlm <= 0:
            K_vlm = K_gt

        # 2. Baseline (A): pixel-mode pipeline at K_VLM
        rA = run_pipeline(
            str(img_path), n_lines=K_vlm,
            separator_mode="v4", morph_close_kernel=5, min_cluster_area=200,
            m2f_threshold=0.25, m2f_scales=(512, 768, 1024), flip_tta=True,
        )
        trails_A = trails_from_pipeline(rA)
        f1A, tpA, fpA, fnA = inst_f1(trails_A, gt_trails)

        # 3. Anchor-assigned (B): re-cluster the same FG mask used in A
        #    by ASSIGNING each foreground pixel to the nearest VLM-anchor
        #    polyline (replaces 5D LAB-xy k-means with VLM-supervised
        #    assignment).  Skips when VLM gave no anchors.
        plot_box = plot_box_from_axis_lines(rA.axis_lines, img_bgr.shape)
        if cp.curves and any(len(c.anchors) for c in cp.curves):
            fg = (rA.final_mask > 0)
            vlm_masks = vlm_anchor_assign(img_bgr, fg, plot_box, cp.curves)
            trails_B = trails_from_masks(vlm_masks)
        else:
            trails_B = trails_from_pipeline(rA)
        f1B, tpB, fpB, fnB = inst_f1(trails_B, gt_trails)

        delta = f1B - f1A
        print(f"  [{i+1:>3}/{len(anns)}] {image_id[:32]}.. "
              f"K_gt={K_gt} K_vlm={K_vlm}  A.f1={f1A:.3f}({tpA},{fpA},{fnA}) "
              f"B.f1={f1B:.3f}({tpB},{fpB},{fnB})  Δ={delta:+.3f} "
              f"(VLM err={cp.error}, t={t_vlm:.1f}s)")
        rows.append({
            "image_id": image_id, "K_gt": K_gt, "K_vlm": K_vlm,
            "f1_A": f1A, "f1_B": f1B, "delta": delta,
            "tp_A": tpA, "fp_A": fpA, "fn_A": fnA,
            "tp_B": tpB, "fp_B": fpB, "fn_B": fnB,
            "vlm_err": cp.error, "vlm_lat": cp.latency_s,
            "n_clusters_A": len(rA.cluster_masks),
            "n_clusters_B": len(trails_B),
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"\nsaved → {args.out_csv}")
    print(f"\n=== summary on n={len(df)} images ===")
    print(f"Inst-F1  baseline(A) = {df.f1_A.mean():.3f}")
    print(f"Inst-F1  with merge(B) = {df.f1_B.mean():.3f}")
    print(f"Δ = {df.delta.mean():+.3f}  (median {df.delta.median():+.3f})")
    print(f"images where B improved: {(df.delta > 0).sum()}/{len(df)}")
    print(f"images where B regressed: {(df.delta < 0).sum()}/{len(df)}")


if __name__ == "__main__":
    main()
