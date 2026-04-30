"""端到端线条提取管道。

串联四个模型 / 步骤:
    1. YOLO 坐标轴检测    → axis_lines           (Axis_yolo_extraction.detect_axes_yolo)
    2. YOLO 图例/杂图检测 → legend boxes         (yolo/.../yolov11n_lenged_clutter_detector)
    3. Mask2Former        → 所有线条/散点的二值掩膜 (mask2former_line.get_line_binary_mask)
    4. KMeans 聚类        → 按颜色分离出每条曲线的子掩膜

回退机制: 若 Mask2Former 输出的前景比例低于 ``m2f_min_fg`` (默认 0.3%),
认为模型在这张图上失效 (典型情形为纯散点图, 比如 mask_vis_0.jpg),
自动回退到仅用 "YOLO 轴+图例+OCR 文本" mask 后的 KMeans 聚类。

只做"能否跑通"的冒烟测试, 不调用多模态 LLM, 聚类数 K 由调用方给定或用 heuristic。
"""
from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from Axis_yolo_extraction import detect_axes_yolo  # noqa: E402
from mask2former_line import (  # noqa: E402
    get_line_binary_mask, get_line_prob_multiscale, get_line_per_query,
)
from skimage.morphology import skeletonize  # noqa: E402

ROOT = Path(__file__).resolve().parent
LEGEND_MODEL_PATH = ROOT / "yolo" / "yolov11n_lenged_clutter_detector" / "weights" / "best.pt"
AXIS_MODEL_PATH = ROOT / "yolo" / "axis_lines_detect" / "best.pt"
MASK2FORMER_CKPT = ROOT / "models" / "mask2foremer_q12_d3_h256_best.pt"


@dataclass
class PipelineResult:
    image_path: str
    ok: bool
    used_fallback: bool
    axis_lines: Optional[list] = None          # [x_axis, y_axis]
    legend_boxes: list = field(default_factory=list)
    m2f_fg_frac: float = 0.0
    final_mask: Optional[np.ndarray] = None     # 最终用于聚类的二值 mask (H, W)
    cluster_masks: list = field(default_factory=list)  # 每条线的 bool mask
    n_clusters: int = 0
    # Cell-5 calibration / abstention (one entry per cluster, aligned with
    # cluster_masks).  ``curve_confidences`` is the per-curve max of the
    # Mask2Former soft mask F(u,v) over the cluster pixels, falling back to
    # the per-cluster silhouette score when used_fallback=True.
    # ``curve_abstain[i] = curve_confidences[i] < tau_abs``.
    curve_confidences: list = field(default_factory=list)
    curve_abstain: list = field(default_factory=list)
    tau_abs: float = 0.0
    error: Optional[str] = None


def _load_legend_model():
    from ultralytics import YOLO
    if not LEGEND_MODEL_PATH.exists():
        raise FileNotFoundError(f"legend YOLO 模型缺失: {LEGEND_MODEL_PATH}")
    return YOLO(str(LEGEND_MODEL_PATH))


_LEGEND_MODEL = None


def detect_legend_boxes(image_path: str, conf: float = 0.3) -> list[dict]:
    """返回 [{'top_left': [x1,y1], 'bottom_right': [x2,y2]}, ...]。"""
    global _LEGEND_MODEL
    if _LEGEND_MODEL is None:
        _LEGEND_MODEL = _load_legend_model()
    res = _LEGEND_MODEL.predict(source=image_path, conf=conf, verbose=False)[0]
    out = []
    for box in res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        out.append({"top_left": [x1, y1], "bottom_right": [x2, y2]})
    return out


def build_ignore_mask(img_shape, axis_lines, legend_boxes, pad: int = 4) -> np.ndarray:
    """返回 uint8 掩膜 (255 = 屏蔽区域, 0 = 有效区域)。"""
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if axis_lines is not None and axis_lines[0] is not None and axis_lines[1] is not None:
        x_axis, y_axis = axis_lines
        x1 = int(min(x_axis[0], x_axis[2], y_axis[0], y_axis[2])) + pad
        y1 = int(min(x_axis[1], x_axis[3], y_axis[1], y_axis[3])) + pad
        x2 = int(max(x_axis[0], x_axis[2], y_axis[0], y_axis[2])) - pad
        y2 = int(max(x_axis[1], x_axis[3], y_axis[1], y_axis[3])) - pad
        # 内部保留, 外部屏蔽
        mask[:] = 255
        mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 0
    for box in legend_boxes:
        tl = box["top_left"]; br = box["bottom_right"]
        mask[max(0, tl[1]):min(h, br[1]), max(0, tl[0]):min(w, br[0])] = 255
    return mask


def apply_mask_whiten(img: np.ndarray, ignore_mask: np.ndarray) -> np.ndarray:
    out = img.copy()
    out[ignore_mask > 0] = [255, 255, 255]
    return out


def kmeans_on_pixels(img_bgr: np.ndarray, pixel_mask: np.ndarray, n_clusters: int,
                     random_state: int = 42,
                     return_centroids: bool = False,
                     xy_weight: float = 0.5) -> list[np.ndarray]:
    """iter9: 6D KMeans (LAB + xy). xy_weight ∈ [0,1]: 0=pure LAB (legacy),
    0.5=balanced (waterfall fix), 1=pure spatial."""
    ys, xs = np.where(pixel_mask)
    if ys.size < max(50, n_clusters * 10):
        return []
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    pixels = lab[ys, xs].astype(np.float32)                    # (N,3)
    H, W = pixel_mask.shape
    xy = np.stack([xs / max(W, 1), ys / max(H, 1)],
                  axis=1).astype(np.float32)                    # (N,2) in [0,1]
    lab_scaled = StandardScaler().fit_transform(pixels)
    xy_scaled = StandardScaler().fit_transform(xy) * xy_weight
    feats = np.concatenate([lab_scaled, xy_scaled], axis=1)    # (N,5)
    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=random_state)
    labels = km.fit_predict(feats)
    out = []
    for i in range(n_clusters):
        m = np.zeros(pixel_mask.shape, dtype=bool)
        m[ys[labels == i], xs[labels == i]] = True
        if return_centroids:
            cent_lab = pixels[labels == i].mean(axis=0) if (labels == i).any() else np.zeros(3)
            out.append((m, cent_lab))
        else:
            out.append(m)
    if return_centroids:
        out.sort(key=lambda mc: -int(mc[0].sum()))
    else:
        out.sort(key=lambda m: -int(m.sum()))
    return out


def is_grayscale_chart(img_bgr: np.ndarray, threshold: float = 6.0) -> bool:
    """判断是不是黑白图(各通道几乎相同)。"""
    diff = np.abs(img_bgr.astype(np.int16) - img_bgr.mean(axis=2, keepdims=True))
    return float(diff.mean()) < threshold


def cluster_by_y(pixel_mask: np.ndarray, n_clusters: int,
                 random_state: int = 42) -> list[np.ndarray]:
    """灰度多线场景:仅按 y 坐标聚类(适合 stacked 多条曲线水平铺开)。"""
    ys, xs = np.where(pixel_mask)
    if ys.size < max(50, n_clusters * 10):
        return []
    feats = ys.reshape(-1, 1).astype(np.float32)
    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=random_state)
    labels = km.fit_predict(feats)
    out = []
    for i in range(n_clusters):
        m = np.zeros(pixel_mask.shape, dtype=bool)
        m[ys[labels == i], xs[labels == i]] = True
        out.append(m)
    out.sort(key=lambda m: -int(m.sum()))
    return out


def heuristic_cluster_k(img_bgr: np.ndarray, pixel_mask: np.ndarray,
                        k_min: int = 1, k_max: int = 8) -> int:
    """在缺少 LLM 时用启发式: 前景有效像素 / 1500 近似线条数, 裁剪到 [k_min, k_max]."""
    n_fg = int(pixel_mask.sum())
    if n_fg < 500:
        return k_min
    guess = max(k_min, min(k_max, round(np.sqrt(n_fg / 400.0))))
    return int(guess)


# ---------------- iter12 phase C/D/E/F helpers ----------------

def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def _query_nms(probs: np.ndarray, obj: np.ndarray, valid_region: np.ndarray,
               threshold: float = 0.35, iou_thr: float = 0.5,
               keep_top_k: Optional[int] = None) -> list[np.ndarray]:
    """phase D: per-query mask NMS. probs (Q,H,W), obj (Q,)."""
    if probs.size == 0:
        return []
    bins = [(probs[i] >= threshold) & valid_region for i in range(probs.shape[0])]
    order = np.argsort(-obj)
    kept: list[int] = []
    for i in order:
        if int(bins[i].sum()) < 50:
            continue
        ok = True
        for j in kept:
            if _mask_iou(bins[i], bins[j]) > iou_thr:
                ok = False
                break
        if ok:
            kept.append(int(i))
        if keep_top_k is not None and len(kept) >= keep_top_k:
            break
    return [bins[i] for i in kept]


def _is_scatter_mask(pixel_mask: np.ndarray, k: int,
                     strict: bool) -> bool:
    """phase E: connected-component test for marker-only scatter."""
    if pixel_mask.dtype != np.uint8:
        pm = pixel_mask.astype(np.uint8)
    else:
        pm = pixel_mask
    n_lbl, _lbls, stats, _cents = cv2.connectedComponentsWithStats(pm, connectivity=8)
    if n_lbl <= 1:
        return False
    areas = stats[1:, cv2.CC_STAT_AREA]
    widths = stats[1:, cv2.CC_STAT_WIDTH]
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    longest = np.maximum(widths, heights)
    n_cc = len(areas)
    if n_cc < max(3, k * 3):
        return False
    if float(areas.mean()) >= 50.0:
        return False
    if strict:
        if float(longest.max()) >= 40.0:
            return False
    return True


def _scatter_branch(img_bgr: np.ndarray, pixel_mask: np.ndarray,
                    valid_region: np.ndarray, k: int) -> list[np.ndarray]:
    """phase E: cc centroids → KMeans LAB groups → polyfit deg=3 → rasterize 5px."""
    if pixel_mask.dtype != np.uint8:
        pm = pixel_mask.astype(np.uint8)
    else:
        pm = pixel_mask
    n_lbl, lbls, stats, cents = cv2.connectedComponentsWithStats(pm, connectivity=8)
    if n_lbl <= 1:
        return []
    centers = []
    colors = []
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    for i in range(1, n_lbl):
        if stats[i, cv2.CC_STAT_AREA] < 3:
            continue
        cy, cx = cents[i, 1], cents[i, 0]
        centers.append((cx, cy))
        comp = lbls == i
        colors.append(img_lab[comp].mean(axis=0))
    if len(centers) < max(2, k):
        return []
    pts = np.array(centers, dtype=np.float32)
    cols = np.array(colors, dtype=np.float32)
    cols_s = StandardScaler().fit_transform(cols)
    km = KMeans(n_clusters=k, n_init=5, random_state=42)
    labels = km.fit_predict(cols_s)
    H, W = pixel_mask.shape
    out = []
    for ci in range(k):
        sel = labels == ci
        if int(sel.sum()) < 3:
            out.append(np.zeros((H, W), dtype=bool))
            continue
        sub = pts[sel]
        sub = sub[np.argsort(sub[:, 0])]  # sort by x
        try:
            coeffs = np.polyfit(sub[:, 0], sub[:, 1], deg=min(3, len(sub) - 1))
        except (np.linalg.LinAlgError, ValueError):
            out.append(np.zeros((H, W), dtype=bool))
            continue
        x_vals = np.arange(int(sub[:, 0].min()), int(sub[:, 0].max()) + 1)
        y_vals = np.polyval(coeffs, x_vals)
        canvas = np.zeros((H, W), dtype=np.uint8)
        poly_pts = np.stack([x_vals, y_vals], axis=1).astype(np.int32)
        poly_pts = poly_pts.reshape(-1, 1, 2)
        cv2.polylines(canvas, [poly_pts], isClosed=False, color=255,
                      thickness=5, lineType=cv2.LINE_AA)
        canvas &= valid_region.astype(np.uint8) * 255
        out.append(canvas > 0)
    return out


def _thicken_to_5px(cluster_masks: list[np.ndarray],
                     img_lab: np.ndarray,
                     valid_region: np.ndarray,
                     delta_e: float = 12.0) -> list[np.ndarray]:
    """phase C: per-cluster skeletonize → dilate to thickness=5, LAB ΔE-gated."""
    out = []
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for cm in cluster_masks:
        if int(cm.sum()) < 8:
            out.append(cm)
            continue
        sk = skeletonize(cm.astype(bool))
        thick = cv2.dilate(sk.astype(np.uint8), kernel5, iterations=1) > 0
        ring = thick & (~cm.astype(bool)) & valid_region
        if not ring.any():
            out.append(cm.astype(bool) | thick)
            continue
        mean_lab = img_lab[cm.astype(bool)].mean(axis=0)
        ring_lab = img_lab[ring]
        delta = np.linalg.norm(ring_lab - mean_lab, axis=1)
        keep = delta < delta_e
        rys, rxs = np.where(ring)
        keep_mask = np.zeros_like(cm, dtype=bool)
        keep_mask[rys[keep], rxs[keep]] = True
        out.append(cm.astype(bool) | keep_mask)
    return out


def _refine_per_line(cluster_masks: list[np.ndarray], img_bgr: np.ndarray,
                     valid_region: np.ndarray,
                     mode: str = "basic") -> list[np.ndarray]:
    """phase F: iterative LAB-/edge-gated dilation per cluster.
    mode: basic (ΔE=10, grow=4), aggr (ΔE=15, grow=8), edge (basic + Sobel gate)."""
    if mode == "off":
        return cluster_masks
    cfg = {
        "basic": dict(delta_e=10.0, n_iter=4, edge_gate=False),
        "aggr": dict(delta_e=15.0, n_iter=8, edge_gate=False),
        "edge": dict(delta_e=10.0, n_iter=4, edge_gate=True),
    }.get(mode)
    if cfg is None:
        return cluster_masks
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    edge_mask = None
    if cfg["edge_gate"]:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        thr = float(mag.mean() + mag.std())
        edge_mask = mag > thr
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out = []
    for cm in cluster_masks:
        cur = cm.astype(bool).copy()
        if int(cur.sum()) < 20:
            out.append(cur)
            continue
        mean_lab = img_lab[cur].mean(axis=0)
        for _ in range(cfg["n_iter"]):
            grown = cv2.dilate(cur.astype(np.uint8), kernel3, iterations=1) > 0
            ring = grown & (~cur) & valid_region
            if not ring.any():
                break
            ring_lab = img_lab[ring]
            delta = np.linalg.norm(ring_lab - mean_lab, axis=1)
            keep = delta < cfg["delta_e"]
            if edge_mask is not None:
                rys, rxs = np.where(ring)
                edge_keep = edge_mask[rys, rxs]
                keep = keep & edge_keep
            if not keep.any():
                break
            rys, rxs = np.where(ring)
            grow_mask = np.zeros_like(cur)
            grow_mask[rys[keep], rxs[keep]] = True
            new_cur = cur | grow_mask
            if int((new_cur ^ cur).sum()) / max(1, int(cur.sum())) < 0.02:
                cur = new_cur
                break
            cur = new_cur
        out.append(cur)
    return out


def run_pipeline(
    image_path: str,
    n_lines: Optional[int] = None,
    m2f_min_fg: float = 0.003,
    axis_conf: float = 0.5,
    legend_conf: float = 0.3,
    m2f_ckpt: Optional[str] = None,
    separator_mode: str = "v1",
    morph_close_kernel: int = 0,
    min_cluster_area: int = 0,
    # iter12+ knobs (defaults preserve iter11 behavior)
    m2f_threshold: float = 0.35,
    m2f_scales: tuple = (640, 1024),
    flip_tta: bool = False,
    thicken: bool = False,
    query_nms_mode: str = "off",   # off / l4_only / all / init
    scatter_branch: str = "off",   # off / on / strict
    refine_mode: str = "off",      # off / basic / aggr / edge
    tau_abs: float = 0.30,         # Cell-5 abstention threshold on per-curve max-F
) -> PipelineResult:
    """``separator_mode``:
        - 'v1': 旧逻辑, kmeans(K+1) 后丢最大簇当背景 (默认, 保兼容)
        - 'v2': kmeans(K) 直接保留, 不做"丢最大"
        - 'v3': v2 + 灰度图自动切到 cluster_by_y
        - 'v4': v3 + Mask2Former mask morph-close + 长度过滤碎片

    iter12 phase C/D/E/F knobs:
        - m2f_threshold/m2f_scales/flip_tta: 上游 m2f 召回 sweep
        - thicken: 每条 cluster_mask skeletonize → dilate 到 thickness=5
        - query_nms_mode: per-query NMS 替代 LAB-KMeans (L4 same-color overlap)
        - scatter_branch: marker-only 散点检测, 切换到 polyfit
        - refine_mode: Hungarian 后 LAB-/edge-gated 邻域生长
    """
    img = cv2.imread(image_path)
    if img is None:
        return PipelineResult(image_path=image_path, ok=False, used_fallback=False,
                              error=f"cv2.imread returned None for {image_path}")

    # --- Step 1: 轴 ---
    try:
        x_axis, y_axis = detect_axes_yolo(image_path, model_path=str(AXIS_MODEL_PATH),
                                          conf_threshold=axis_conf)
        axis_lines = [x_axis, y_axis] if (x_axis and y_axis) else None
    except Exception as e:
        return PipelineResult(image_path=image_path, ok=False, used_fallback=False,
                              error=f"axis YOLO failed: {e}")

    # --- Step 2: 图例/杂图 ---
    try:
        legend_boxes = detect_legend_boxes(image_path, conf=legend_conf)
    except Exception as e:
        legend_boxes = []
        print(f"[{Path(image_path).name}] 图例检测失败: {e}")

    ignore_mask = build_ignore_mask(img.shape, axis_lines, legend_boxes)
    valid_region = ignore_mask == 0

    # --- Step 3: Mask2Former ---
    try:
        _ckpt = m2f_ckpt if m2f_ckpt else str(MASK2FORMER_CKPT)
        # iter11: multi-scale TTA prob max; iter12 phase C: configurable scales + flip TTA
        m2f_prob = get_line_prob_multiscale(image_path, ckpt_path=_ckpt,
                                             scales=tuple(m2f_scales),
                                             flip_tta=flip_tta)
        # iter8: 阈值 0.5 -> 0.35, 提升 FG 召回率; iter12 phase C: configurable
        m2f_bin = (m2f_prob >= float(m2f_threshold)) & valid_region
        m2f_fg_frac = float(m2f_bin.mean())
    except Exception as e:
        return PipelineResult(image_path=image_path, ok=False, used_fallback=False,
                              axis_lines=axis_lines, legend_boxes=legend_boxes,
                              error=f"mask2former failed: {e}\n{traceback.format_exc()}")

    used_fallback = m2f_fg_frac < m2f_min_fg

    # --- Step 4: 聚类 ---
    if not used_fallback:
        pixel_mask = m2f_bin
    else:
        # 回退: 在轴内 / 非白色 / 非图例 的像素里做颜色聚类
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        non_white = gray < 240       # 过滤大面积白背景
        pixel_mask = valid_region & non_white

    k = n_lines if n_lines else heuristic_cluster_k(img, pixel_mask)
    k = max(1, k)

    # iter4: 形态学清理 FG mask, 减少碎片
    if separator_mode in ("v4",) and morph_close_kernel > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_close_kernel, morph_close_kernel))
        pm_uint = pixel_mask.astype(np.uint8) * 255
        pm_uint = cv2.morphologyEx(pm_uint, cv2.MORPH_CLOSE, kernel)
        # iter8: 3x3 椭圆 dilate, 增厚细线减少断裂
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        pm_uint = cv2.dilate(pm_uint, dilate_kernel, iterations=1)
        pixel_mask = pm_uint > 0

    # iter12 phase E: marker-only scatter branch
    cluster_masks: list[np.ndarray]
    scatter_used = False
    if scatter_branch in ("on", "strict"):
        if _is_scatter_mask(pixel_mask, k=k, strict=(scatter_branch == "strict")):
            cluster_masks = _scatter_branch(img, pixel_mask, valid_region, k)
            if cluster_masks:
                scatter_used = True

    # iter12 phase D: per-query NMS replacing LAB-KMeans
    qnms_used = False
    if not scatter_used and query_nms_mode != "off":
        gate_l4_only = (query_nms_mode == "l4_only" and k >= 5)
        if query_nms_mode in ("all", "init") or gate_l4_only:
            try:
                _ckpt2 = m2f_ckpt if m2f_ckpt else str(MASK2FORMER_CKPT)
                probs_q, obj_q = get_line_per_query(image_path, ckpt_path=_ckpt2,
                                                     infer_size=int(m2f_scales[-1]),
                                                     obj_threshold=0.3)
                q_masks = _query_nms(probs_q, obj_q, valid_region,
                                     threshold=float(m2f_threshold),
                                     iou_thr=0.5, keep_top_k=k)
                if query_nms_mode == "init" and q_masks:
                    # use query masks as KMeans initial centroids approximation:
                    # fall through to KMeans but seed with query union
                    union_qm = np.zeros_like(pixel_mask, dtype=bool)
                    for qm in q_masks:
                        union_qm |= qm
                    pixel_mask = pixel_mask | union_qm
                elif q_masks and (query_nms_mode in ("all",) or gate_l4_only):
                    cluster_masks = q_masks
                    if k > len(cluster_masks):
                        # didn't reach target K → fall back to kmeans on residual
                        residual = pixel_mask & ~np.any(np.stack(cluster_masks, 0), axis=0)
                        extra = kmeans_on_pixels(img, residual, n_clusters=k - len(cluster_masks))
                        cluster_masks = cluster_masks + extra
                    qnms_used = True
            except Exception as e:
                print(f"[query_nms] {Path(image_path).name}: skipped ({e})")

    if not scatter_used and not qnms_used:
        if separator_mode == "v1":
            # 旧逻辑: 多聚 1 簇丢最大
            cluster_masks = kmeans_on_pixels(img, pixel_mask, n_clusters=k + 1)
            if cluster_masks:
                cluster_masks = cluster_masks[1:]
        elif separator_mode in ("v2", "v3", "v4"):
            # v3/v4: 灰度图走纯空间聚类, 否则 LAB
            gray_mode = (separator_mode in ("v3", "v4")
                         and is_grayscale_chart(img))
            if gray_mode and k >= 2:
                cluster_masks = cluster_by_y(pixel_mask, n_clusters=k)
            else:
                cluster_masks = kmeans_on_pixels(img, pixel_mask, n_clusters=k)
        else:
            raise ValueError(f"unknown separator_mode: {separator_mode}")

    # iter4 长度过滤
    if separator_mode == "v4" and min_cluster_area > 0:
        cluster_masks = [m for m in cluster_masks
                         if int(m.sum()) >= min_cluster_area]

    # iter10: per-cluster LAB-gated dilation
    # m2f 边缘 anti-alias 损失 1-2px, GT 用 5px 厚度光栅化, 这里把每条
    # cluster 向外 2px 膨胀, 但只保留 LAB ΔE < 12 的颜色相邻像素 → 不串色.
    if separator_mode == "v4" and cluster_masks:
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        restore_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        restored = []
        for cm in cluster_masks:
            if int(cm.sum()) < 20:
                restored.append(cm)
                continue
            mean_lab = img_lab[cm].mean(axis=0)  # (3,)
            grown = cv2.dilate(cm.astype(np.uint8), restore_kernel,
                               iterations=1) > 0
            ring = grown & (~cm) & valid_region
            if not ring.any():
                restored.append(cm)
                continue
            ring_lab = img_lab[ring]                  # (N,3)
            delta = np.linalg.norm(ring_lab - mean_lab, axis=1)
            keep_idx = delta < 12.0
            ring_keep = np.zeros_like(cm)
            rys, rxs = np.where(ring)
            ring_keep[rys[keep_idx], rxs[keep_idx]] = True
            restored.append(cm | ring_keep)
        cluster_masks = restored

    # iter12 phase C: skeleton+dilate to thickness 5 (LAB-gated)
    if thicken and cluster_masks:
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        cluster_masks = _thicken_to_5px(cluster_masks, img_lab, valid_region,
                                         delta_e=12.0)

    # iter12 phase F: per-line LAB+edge refinement
    if refine_mode != "off" and cluster_masks:
        cluster_masks = _refine_per_line(cluster_masks, img, valid_region,
                                          mode=refine_mode)

    # Cell-5: per-curve confidence + abstention (R1-W3 fix; previously the
    # paper described this Cell but the runtime never emitted it).
    # Default branch: c_i = mean over cluster pixels of the Mask2Former
    # soft mask F(u,v) — saturates less than max-F and so discriminates
    # between well-anchored clusters and spurious carve-outs of low-prob
    # residual pixels.  Fallback branch: F is unreliable (M2F collapsed),
    # so we use a silhouette-style coherence proxy
    #     c_i = |area_i| / (|area_i| + |ring_5px|)
    # which lives on the same [0, 1] axis as F.
    curve_confidences: list[float] = []
    curve_abstain: list[bool] = []
    for cm in cluster_masks:
        if cm.sum() == 0:
            curve_confidences.append(0.0)
            curve_abstain.append(True)
            continue
        if not used_fallback:
            c = float(m2f_prob[cm].mean())
        else:
            cm_u8 = cm.astype(np.uint8)
            ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            ring = (cv2.dilate(cm_u8, ring_kernel) > 0) & ~cm
            inner = float(cm.sum())
            outer = float(ring.sum())
            c = inner / (inner + max(outer, 1.0))
        curve_confidences.append(c)
        curve_abstain.append(c < float(tau_abs))

    return PipelineResult(
        image_path=image_path,
        ok=True,
        used_fallback=used_fallback,
        axis_lines=axis_lines,
        legend_boxes=legend_boxes,
        m2f_fg_frac=m2f_fg_frac,
        final_mask=pixel_mask.astype(np.uint8) * 255,
        cluster_masks=cluster_masks,
        n_clusters=len(cluster_masks),
        curve_confidences=curve_confidences,
        curve_abstain=curve_abstain,
        tau_abs=float(tau_abs),
    )


def _palette(n: int) -> list[tuple[int, int, int]]:
    rng = np.random.default_rng(123)
    base = [(230, 30, 30), (30, 180, 30), (30, 100, 230), (240, 180, 30),
            (170, 60, 220), (30, 190, 200), (255, 120, 0), (120, 80, 40)]
    while len(base) < n:
        base.append(tuple(int(v) for v in rng.integers(30, 230, size=3)))
    return base[:n]


def visualize(result: PipelineResult, save_path: Path) -> None:
    img = cv2.imread(result.image_path)
    if img is None:
        return
    h, w = img.shape[:2]
    mask_img = np.zeros_like(img)
    if result.final_mask is not None:
        mask_img[result.final_mask > 0] = (255, 255, 255)

    overlay = img.copy()
    if result.cluster_masks:
        colors = _palette(len(result.cluster_masks))
        for color, m in zip(colors, result.cluster_masks):
            overlay[m] = color

    # 在 overlay 上画 axis_lines
    if result.axis_lines is not None:
        for line in result.axis_lines:
            if line:
                p1 = (int(line[0]), int(line[1]))
                p2 = (int(line[2]), int(line[3]))
                cv2.line(overlay, p1, p2, (0, 255, 255), 2)
    for box in result.legend_boxes:
        cv2.rectangle(overlay, tuple(box["top_left"]), tuple(box["bottom_right"]),
                      (255, 0, 255), 2)

    tag = "FALLBACK(KMeans)" if result.used_fallback else "Mask2Former"
    caption = f"{tag} | m2f_fg={result.m2f_fg_frac:.4f} | K={result.n_clusters}"
    bar = np.full((26, w * 3, 3), 240, dtype=np.uint8)
    cv2.putText(bar, caption, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1)
    side = np.hstack([img, mask_img, overlay])
    out = np.vstack([bar, side])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="label")
    ap.add_argument("--output_dir", default="pipeline_out")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--n_lines", type=int, default=0, help="0 = heuristic")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    imgs = sorted(list(in_dir.glob("*.jpg")) + list(in_dir.glob("*.png")))
    if args.limit > 0:
        imgs = imgs[: args.limit]
    print(f"处理 {len(imgs)} 张图 → {out_dir}")

    fallback_cnt = 0
    ok_cnt = 0
    err_cnt = 0
    for i, p in enumerate(imgs):
        r = run_pipeline(str(p), n_lines=args.n_lines or None)
        if r.ok:
            ok_cnt += 1
            if r.used_fallback:
                fallback_cnt += 1
            visualize(r, out_dir / f"{p.stem}.jpg")
        else:
            err_cnt += 1
        axis_ok = r.axis_lines is not None
        print(f"[{i+1:>3}/{len(imgs)}] {p.name}: ok={r.ok} axis={axis_ok} "
              f"legend_n={len(r.legend_boxes)} m2f_fg={r.m2f_fg_frac:.4f} "
              f"fallback={r.used_fallback} K={r.n_clusters}"
              + (f" err={r.error[:60]}" if r.error else ""))

    print(f"\nsummary: ok={ok_cnt}, err={err_cnt}, "
          f"fallback_triggered={fallback_cnt}/{ok_cnt} "
          f"({100.0*fallback_cnt/max(ok_cnt,1):.1f}%)")


if __name__ == "__main__":
    main()
