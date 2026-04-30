"""VLM-supervised cluster merge.

After ``pipeline.run_pipeline`` produces over-clustered cluster masks
(by passing ``n_lines = K_VLM + buffer``), we use the VLM's per-curve
descriptors (color name, x-range, y-anchors) to merge clusters that
correspond to the same GT curve.

Algorithm:
1. For each cluster, compute (a) mean LAB colour, (b) skeleton polyline
   resampled to anchors at x = 10%, 50%, 90% of the cluster's x-extent.
2. Build a cost matrix [N_cluster x K_VLM] combining
   (i) LAB colour distance (with "unknown" → penalty 0 contribution),
   (ii) anchor mean Euclidean distance in normalised plot coords.
3. Hard-assign each cluster to its argmin-cost VLM curve.
4. Merge cluster masks per assignment.

Returns merged ``cluster_masks`` and per-merged-curve metadata
(VLM colour / style / anchors copied through).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import List, Tuple, Optional

import cv2
import numpy as np
from skimage.morphology import skeletonize
import os

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(KMEANS_ROOT))

from experiments.iter.llm_curves_predictor import CurveDesc  # noqa: E402

# Standard CSS-ish RGB anchors for the colour vocabulary the VLM emits.
_COLOR_ANCHORS_RGB = {
    "red":     (228,  26,  28),
    "blue":    ( 55, 126, 184),
    "green":   ( 77, 175,  74),
    "orange":  (255, 127,   0),
    "purple":  (152,  78, 163),
    "black":   (  0,   0,   0),
    "brown":   (166,  86,  40),
    "cyan":    (  0, 200, 220),
    "gray":    (128, 128, 128),
    "pink":    (247, 129, 191),
    "yellow":  (255, 255,  51),
    "white":   (240, 240, 240),
    "unknown": (None,),
}


def _rgb_to_lab(rgb: Tuple[int, int, int]) -> np.ndarray:
    arr = np.array([[list(rgb)]], dtype=np.uint8)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    return lab  # (3,)


_COLOR_ANCHORS_LAB = {
    name: (_rgb_to_lab(rgb) if name != "unknown" else None)
    for name, rgb in _COLOR_ANCHORS_RGB.items()
}


def _cluster_mean_lab(img_bgr: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """Mean LAB of pixels under the boolean cluster mask."""
    if mask.sum() < 5:
        return None
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab[mask].mean(axis=0)  # (3,)


def _cluster_anchors_norm(mask: np.ndarray, plot_box: tuple[int, int, int, int]
                          ) -> np.ndarray | None:
    """Return three (x, y) anchors normalized to [0, 1] inside the plot box.

    plot_box = (x0, y0, x1, y1) in pixel coordinates of the plot area.
    Anchors are taken at the cluster's 10/50/90 percentile x-positions.
    y is mapped so 0 = bottom of plot, 1 = top.
    """
    if mask.sum() < 10:
        return None
    sk = skeletonize(mask.astype(bool))
    ys, xs = np.where(sk)
    if len(xs) < 5:
        ys, xs = np.where(mask)
    if len(xs) < 5:
        return None
    x0, y0, x1, y1 = plot_box
    pw = max(1, x1 - x0)
    ph = max(1, y1 - y0)
    # Per-x mean-y so we don't accidentally pick top/bottom of multi-y mask
    by_x = {}
    for x, y in zip(xs, ys):
        by_x.setdefault(int(x), []).append(int(y))
    sx = np.array(sorted(by_x), dtype=int)
    sy = np.array([float(np.mean(by_x[int(x)])) for x in sx], dtype=float)
    if len(sx) < 3:
        return None
    qs = np.percentile(np.arange(len(sx)), [10, 50, 90]).astype(int)
    out = []
    for q in qs:
        x_pix, y_pix = sx[q], sy[q]
        nx = (x_pix - x0) / pw
        ny = 1.0 - (y_pix - y0) / ph   # invert: 0 = bottom of plot
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        out.append((nx, ny))
    return np.array(out, dtype=float)  # (3, 2)


def _color_dist(cluster_lab: np.ndarray | None, vlm_color: str) -> float:
    """Return a [0, 1]-ish penalty for colour mismatch."""
    if cluster_lab is None or vlm_color == "unknown":
        return 0.0  # no information — neutral
    anchor = _COLOR_ANCHORS_LAB.get(vlm_color)
    if anchor is None:
        return 0.0
    # ΔE distance in LAB; typical max around ~150
    de = float(np.linalg.norm(cluster_lab - anchor))
    # Normalise to [0, 1] with soft cap at 50
    return min(1.0, de / 50.0)


def _anchor_dist(cluster_anchors: np.ndarray | None,
                 vlm_anchors: list[tuple[float, float]]) -> float:
    """Mean Euclidean distance between matched x-percentile anchors."""
    if cluster_anchors is None or len(cluster_anchors) < 1:
        return 1.0  # max penalty if cluster yields no anchors
    if not vlm_anchors:
        return 0.5
    va = np.asarray(vlm_anchors, dtype=float)
    if va.shape[0] >= 3:
        # Pair them in order (we asked for left-to-right anchors)
        m = min(cluster_anchors.shape[0], va.shape[0])
        d = np.linalg.norm(cluster_anchors[:m] - va[:m], axis=1).mean()
    else:
        # Take the closest VLM anchor for each cluster anchor
        d = float(np.mean([
            np.min(np.linalg.norm(va - ca, axis=1))
            for ca in cluster_anchors
        ]))
    return float(min(1.0, d))


@dataclass
class MergeResult:
    merged_masks: list[np.ndarray]      # one per emitted curve
    cluster_to_vlm: list[int]           # for each input cluster: VLM idx
    used_anchors: bool
    used_colors: bool


def vlm_anchor_assign(img_bgr: np.ndarray,
                       fg_mask: np.ndarray,
                       plot_box: tuple[int, int, int, int],
                       vlm_curves: list[CurveDesc],
                       color_weight: float = 5.0,
                       use_color: bool = True
                       ) -> list[np.ndarray]:
    """Replace LAB-xy k-means with VLM-anchor-supervised pixel assignment.

    For each foreground pixel, compute its distance to every VLM curve
    (Euclidean in normalised plot coordinates between the pixel and the
    nearest point of the virtual polyline through that curve's anchors).
    Add a small colour penalty if VLM colour is informative.  Assign
    pixel to the argmin.  This handles the same-colour stacked-curves
    case where k-means in 5D LAB-xy fails because spatial-x dominates.

    Returns a list of cluster masks aligned with ``vlm_curves``.
    """
    K = len(vlm_curves)
    if K == 0 or fg_mask.sum() < 5:
        return [fg_mask.astype(bool)]

    H, W = fg_mask.shape
    x0, y0, x1, y1 = plot_box
    pw = max(1, x1 - x0)
    ph = max(1, y1 - y0)

    ys, xs = np.where(fg_mask.astype(bool))
    if len(xs) == 0:
        return [np.zeros_like(fg_mask, dtype=bool) for _ in range(K)]

    # Normalise pixel coords into the plot box.
    nx = np.clip((xs - x0) / pw, 0.0, 1.0)
    ny = 1.0 - np.clip((ys - y0) / ph, 0.0, 1.0)   # 0 = bottom

    # Build virtual polylines from VLM anchors (sorted by x; clip to x_lo, x_hi).
    polylines = []
    for vc in vlm_curves:
        if not vc.anchors:
            polylines.append(None)
            continue
        anc = sorted(vc.anchors, key=lambda p: p[0])
        polylines.append(np.array(anc, dtype=float))   # (M, 2)

    def _dist_to_polyline(pts_xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
        """Mean Euclidean distance from each (x, y) point in ``pts_xy``
        to the piecewise-linear polyline through ``poly``.

        For each point we find the nearest segment of the polyline and
        compute its perpendicular distance.
        """
        if poly is None or len(poly) == 0:
            return np.full(len(pts_xy), 1.0, dtype=float)
        # If single anchor, just point distance.
        if len(poly) == 1:
            return np.sqrt(((pts_xy - poly[0]) ** 2).sum(axis=1))
        # Segment distances.
        a = poly[:-1]                  # (M-1, 2) start
        b = poly[1:]                   # (M-1, 2) end
        ab = b - a                     # (M-1, 2)
        ab_len2 = (ab * ab).sum(axis=1) + 1e-12     # (M-1,)
        # For each point compute distance to each segment.
        best = np.full(len(pts_xy), np.inf, dtype=float)
        for k in range(len(a)):
            ap = pts_xy - a[k]                       # (N, 2)
            t = (ap @ ab[k]) / ab_len2[k]            # (N,)
            t = np.clip(t, 0.0, 1.0)
            proj = a[k] + np.outer(t, ab[k])         # (N, 2)
            d = np.sqrt(((pts_xy - proj) ** 2).sum(axis=1))
            best = np.minimum(best, d)
        return best

    pts = np.stack([nx, ny], axis=1)  # (N, 2)
    geom_costs = np.stack([_dist_to_polyline(pts, p) for p in polylines], axis=1)
    # Outside x-range: penalty = +large.
    for j, vc in enumerate(vlm_curves):
        out_of_range = (nx < vc.x_lo - 0.05) | (nx > vc.x_hi + 0.05)
        geom_costs[out_of_range, j] += 0.3

    if use_color:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        pixel_lab = lab[ys, xs]                               # (N, 3)
        color_costs = np.zeros((len(xs), K), dtype=float)
        for j, vc in enumerate(vlm_curves):
            anchor = _COLOR_ANCHORS_LAB.get(vc.color)
            if anchor is None or vc.color == "unknown":
                color_costs[:, j] = 0.05  # mild flat penalty
            else:
                de = np.linalg.norm(pixel_lab - anchor, axis=1) / 50.0  # ~[0,3]
                color_costs[:, j] = np.clip(de, 0.0, 1.0)
        cost = geom_costs + (color_weight / 100.0) * color_costs
    else:
        cost = geom_costs

    assign = cost.argmin(axis=1)

    masks = []
    for j in range(K):
        m = np.zeros_like(fg_mask, dtype=bool)
        sel = (assign == j)
        m[ys[sel], xs[sel]] = True
        masks.append(m)
    return masks


def merge_redundant_clusters(img_bgr: np.ndarray,
                              cluster_masks: list[np.ndarray],
                              plot_box: tuple[int, int, int, int],
                              vlm_curves: list[CurveDesc],
                              iou_threshold: float = 0.20,
                              cost_threshold: float = 0.30
                              ) -> MergeResult:
    """Conservative merge: never reduce below K_VLM unless cluster pairs
    are *clearly* redundant.

    A pair (i, j) is merged iff:
      - their masks share IoU ≥ ``iou_threshold`` (they overlap
        substantially in pixel space), OR
      - they map to the same VLM curve under the cost matrix AND that
        cost is below ``cost_threshold``.

    Pass-through if no merges fire.  This guarantees that on inputs
    where k-means already produces non-redundant clusters, the merge
    step is identity (no regression risk).
    """
    n_cl = len(cluster_masks)
    if n_cl <= 1 or not vlm_curves:
        return MergeResult(merged_masks=list(cluster_masks),
                           cluster_to_vlm=list(range(n_cl)),
                           used_anchors=False, used_colors=False)

    K_vlm = len(vlm_curves)
    # Per-cluster summaries
    cluster_anchors = [_cluster_anchors_norm(cm, plot_box) for cm in cluster_masks]
    cluster_labs = [_cluster_mean_lab(img_bgr, cm) for cm in cluster_masks]

    # Cost matrix to find each cluster's best VLM match
    cost = np.zeros((n_cl, K_vlm), dtype=float)
    for i in range(n_cl):
        for j, vc in enumerate(vlm_curves):
            cost[i, j] = (0.5 * _color_dist(cluster_labs[i], vc.color)
                           + 1.0 * _anchor_dist(cluster_anchors[i], vc.anchors))
    best_vlm = cost.argmin(axis=1)
    best_cost = cost.min(axis=1)

    # Pairwise mask IoU
    def _iou(a, b):
        inter = int((a & b).sum())
        union = int((a | b).sum())
        return inter / max(union, 1)

    # Build undirected graph of clusters to merge
    parent = list(range(n_cl))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n_cl):
        for j in range(i + 1, n_cl):
            iou = _iou(cluster_masks[i].astype(bool),
                        cluster_masks[j].astype(bool))
            same_vlm = (best_vlm[i] == best_vlm[j])
            confident_pair = (best_cost[i] < cost_threshold and
                               best_cost[j] < cost_threshold)
            if iou >= iou_threshold:
                union(i, j)
            elif same_vlm and confident_pair:
                union(i, j)

    # Aggregate by component
    groups: dict[int, list[int]] = {}
    for i in range(n_cl):
        groups.setdefault(find(i), []).append(i)
    merged: list[np.ndarray] = []
    cluster_to_vlm: list[int] = [0] * n_cl
    for gid, members in groups.items():
        m = cluster_masks[members[0]].astype(bool).copy()
        for k in members[1:]:
            m |= cluster_masks[k].astype(bool)
        # label this group with its dominant VLM idx
        gid_vlm = int(best_vlm[members[0]])
        merged.append(m)
        for k in members:
            cluster_to_vlm[k] = gid_vlm
    return MergeResult(merged_masks=merged,
                        cluster_to_vlm=cluster_to_vlm,
                        used_anchors=any(len(v.anchors) > 0 for v in vlm_curves),
                        used_colors=any(v.color != "unknown" for v in vlm_curves))


def merge_clusters_with_vlm(img_bgr: np.ndarray,
                             cluster_masks: list[np.ndarray],
                             plot_box: tuple[int, int, int, int],
                             vlm_curves: list[CurveDesc],
                             color_weight: float = 0.5,
                             anchor_weight: float = 1.0
                             ) -> MergeResult:
    """Merge over-clustered ``cluster_masks`` into ``len(vlm_curves)`` curves.

    plot_box is the inner-plot pixel box (x0, y0, x1, y1) used to
    normalise both cluster and VLM anchors into [0, 1]^2.
    """
    K_vlm = len(vlm_curves)
    if K_vlm == 0 or not cluster_masks:
        return MergeResult(merged_masks=list(cluster_masks),
                           cluster_to_vlm=list(range(len(cluster_masks))),
                           used_anchors=False, used_colors=False)

    n_cl = len(cluster_masks)
    color_costs = np.zeros((n_cl, K_vlm), dtype=float)
    anchor_costs = np.zeros((n_cl, K_vlm), dtype=float)
    cluster_anchors = []
    cluster_labs = []
    for cm in cluster_masks:
        cluster_labs.append(_cluster_mean_lab(img_bgr, cm))
        cluster_anchors.append(_cluster_anchors_norm(cm, plot_box))

    used_colors = any(v.color != "unknown" for v in vlm_curves)
    used_anchors = any(len(v.anchors) > 0 for v in vlm_curves)
    for i in range(n_cl):
        for j, vc in enumerate(vlm_curves):
            color_costs[i, j] = _color_dist(cluster_labs[i], vc.color)
            anchor_costs[i, j] = _anchor_dist(cluster_anchors[i], vc.anchors)

    cost = color_weight * color_costs + anchor_weight * anchor_costs

    # Hard assignment: each cluster → argmin VLM curve.
    assign = cost.argmin(axis=1)

    # Merge cluster masks by assignment.  We always emit exactly K_vlm
    # masks; if no cluster picked a particular VLM curve, that mask is
    # empty (downstream filters drop empties).
    merged: list[np.ndarray] = []
    for j in range(K_vlm):
        idxs = [i for i in range(n_cl) if int(assign[i]) == j]
        if not idxs:
            merged.append(np.zeros_like(cluster_masks[0], dtype=bool))
            continue
        m = cluster_masks[idxs[0]].astype(bool).copy()
        for k in idxs[1:]:
            m |= cluster_masks[k].astype(bool)
        merged.append(m)
    return MergeResult(merged_masks=merged,
                        cluster_to_vlm=assign.tolist(),
                        used_anchors=used_anchors,
                        used_colors=used_colors)
