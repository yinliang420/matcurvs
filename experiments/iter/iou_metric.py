"""IoU 指标:衡量"重建图复原原图标注线条"的程度。

两类:
    1. 整图前景 IoU  ―― ⋃(GT 线 mask)  vs  ⋃(pred cluster mask)
    2. 逐线匹配 IoU ―― 对每条 GT 线 (rasterized to mask), 用 Hungarian
       与 pred cluster_masks 做 1-1 匹配, 报均值 + 完整 cost matrix

GT 用 ``label/*_annotation.json`` 里的 ``lines.{Line_i}.points[].pixel`` polyline
配合 thickness 光栅化成 mask。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import os


def rasterize_polyline(points: np.ndarray, h: int, w: int,
                       thickness: int = 5) -> np.ndarray:
    """``points`` (N,2) pixel coords → bool mask of shape (h, w)."""
    if points is None or len(points) < 2:
        return np.zeros((h, w), dtype=bool)
    pts = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
    canvas = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(canvas, [pts], isClosed=False, color=255,
                  thickness=thickness, lineType=cv2.LINE_AA)
    return canvas > 0


def load_gt_masks(ann_path: Path, thickness: int = 5
                   ) -> tuple[list[np.ndarray], tuple[int, int]]:
    d = json.loads(Path(ann_path).read_text())
    w, h = d["image_size"]
    masks = []
    for _, ln in d["lines"].items():
        pts = np.array([[p["pixel"][0], p["pixel"][1]]
                        for p in ln["points"]], dtype=float)
        masks.append(rasterize_polyline(pts, h, w, thickness=thickness))
    return masks, (w, h)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def hungarian_match(gt_masks: list[np.ndarray],
                    pred_masks: list[np.ndarray]) -> tuple[list[tuple[int,int,float]], np.ndarray]:
    """返回 [(gt_idx, pred_idx, iou), ...] 配对 + 完整 IoU 矩阵 (n_gt, n_pred)."""
    n_g, n_p = len(gt_masks), len(pred_masks)
    if n_g == 0 or n_p == 0:
        return [], np.zeros((n_g, n_p))
    M = np.zeros((n_g, n_p), dtype=float)
    for i, g in enumerate(gt_masks):
        for j, p in enumerate(pred_masks):
            M[i, j] = iou(g, p)
    # Hungarian 在 cost = 1-IoU 上找最小
    cost = 1.0 - M
    rows, cols = linear_sum_assignment(cost)
    pairs = [(int(r), int(c), float(M[r, c])) for r, c in zip(rows, cols)]
    return pairs, M


@dataclass
class IoUResult:
    image_id: str
    n_gt: int
    n_pred: int
    iou_global: float        # 整图前景 IoU
    iou_matched_mean: float  # 1-1 匹配均值
    iou_per_line: list[float] = field(default_factory=list)  # 与 gt 顺序对齐
    matched_pairs: list[tuple[int, int, float]] = field(default_factory=list)
    coverage_gt: float = 0.0     # |GT∩Pred| / |GT|       (recall)
    coverage_pred: float = 0.0   # |GT∩Pred| / |Pred|     (precision)


def normalize_thickness(mask: np.ndarray, thickness: int = 5) -> np.ndarray:
    """骨架化后膨胀到统一线宽,消除"GT 5px vs pred 1-many px"的厚度偏差。"""
    from skimage.morphology import skeletonize
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if mask.sum() < 5:
        return mask
    sk = skeletonize(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (thickness, thickness))
    return cv2.dilate(sk.astype(np.uint8), kernel, iterations=1) > 0


def compute_iou(image_id: str, gt_masks: list[np.ndarray],
                pred_masks: list[np.ndarray],
                normalize: bool = False,
                thickness: int = 5) -> IoUResult:
    if normalize:
        gt_masks = [normalize_thickness(m, thickness) for m in gt_masks]
        pred_masks = [normalize_thickness(m, thickness) for m in pred_masks]
    if len(gt_masks) == 0:
        return IoUResult(image_id=image_id, n_gt=0, n_pred=len(pred_masks),
                         iou_global=0.0, iou_matched_mean=0.0)
    h, w = gt_masks[0].shape
    gt_union = np.zeros((h, w), dtype=bool)
    for m in gt_masks:
        gt_union |= m
    if len(pred_masks) == 0:
        return IoUResult(image_id=image_id, n_gt=len(gt_masks), n_pred=0,
                         iou_global=0.0, iou_matched_mean=0.0,
                         iou_per_line=[0.0] * len(gt_masks),
                         coverage_gt=0.0, coverage_pred=0.0)
    pred_union = np.zeros((h, w), dtype=bool)
    for m in pred_masks:
        # m can be 0/255 mask; ensure bool
        pred_union |= (m.astype(bool) if m.dtype != bool else m)

    inter = int(np.logical_and(gt_union, pred_union).sum())
    g_area = int(gt_union.sum())
    p_area = int(pred_union.sum())
    iou_g = inter / max(1, g_area + p_area - inter)
    cov_gt = inter / max(1, g_area)
    cov_pred = inter / max(1, p_area)

    pairs, _M = hungarian_match(gt_masks, [m.astype(bool) for m in pred_masks])
    per_line = [0.0] * len(gt_masks)
    for gi, _pi, v in pairs:
        per_line[gi] = v
    iou_match_mean = float(np.mean(per_line)) if per_line else 0.0

    return IoUResult(
        image_id=image_id, n_gt=len(gt_masks), n_pred=len(pred_masks),
        iou_global=float(iou_g), iou_matched_mean=iou_match_mean,
        iou_per_line=per_line, matched_pairs=pairs,
        coverage_gt=float(cov_gt), coverage_pred=float(cov_pred),
    )


if __name__ == "__main__":
    # quick smoke test on one annotation
    KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[2]))
    ann = (KMEANS_ROOT / "label" /
           "510aa6c30e12fcc5830a9a0db5aed0c0e0d4e550757711618da0f36a92bbad74_4_annotation.json")
    gt_masks, (w, h) = load_gt_masks(ann)
    print(f"loaded {len(gt_masks)} gt masks, image {w}x{h}, "
          f"GT fg frac = {sum(int(m.sum()) for m in gt_masks) / (w * h):.4f}")
    # 自身 IoU 应为 1
    r = compute_iou("self", gt_masks, gt_masks)
    print(f"self-IoU global={r.iou_global:.3f}  matched_mean={r.iou_matched_mean:.3f}")
