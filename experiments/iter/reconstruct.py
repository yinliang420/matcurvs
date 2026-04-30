"""把 PipelineResult.cluster_masks 渲染成"线条重建图":

    白底画布 + 每条 cluster mask 用骨架化 + 不同颜色绘制 → 形成一个
    "我们能从原图抽出来的所有线" 的纯净图。

这张图回答的是: "如果把它喂给 LLM/agent, 它能不能从重建图判断曲线
形状/数量/颜色?" 是 agent-readiness 的可视化代理。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from skimage.morphology import skeletonize

PALETTE = [
    (220, 30, 30),    # red
    (30, 130, 220),   # blue
    (30, 180, 70),    # green
    (240, 170, 30),   # orange
    (170, 60, 220),   # purple
    (30, 200, 200),   # cyan
    (255, 110, 0),    # deep orange
    (120, 80, 40),    # brown
    (200, 30, 130),   # pink
    (80, 80, 80),     # grey
    (130, 200, 30),   # lime
    (30, 100, 130),   # navy
]


def palette(n: int) -> list[tuple[int, int, int]]:
    rng = np.random.default_rng(123)
    base = list(PALETTE)
    while len(base) < n:
        base.append(tuple(int(v) for v in rng.integers(30, 230, size=3)))
    return base[:n]


def draw_reconstruction(img_shape: tuple[int, int],
                        cluster_masks: list[np.ndarray],
                        thickness: int = 3,
                        bg: tuple[int, int, int] = (255, 255, 255),
                        skeleton: bool = True) -> np.ndarray:
    """返回 BGR 重构图。"""
    h, w = img_shape[:2]
    canvas = np.full((h, w, 3), bg, dtype=np.uint8)
    if not cluster_masks:
        return canvas
    colors = palette(len(cluster_masks))
    for color, m in zip(colors, cluster_masks):
        m = m.astype(bool) if m.dtype != bool else m
        if m.sum() < 5:
            continue
        if skeleton:
            sk = skeletonize(m)
            # 加粗骨架 → 线宽
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (thickness, thickness))
            sk = cv2.dilate(sk.astype(np.uint8), kernel, iterations=1) > 0
            canvas[sk] = (color[2], color[1], color[0])  # palette is RGB → BGR
        else:
            canvas[m] = (color[2], color[1], color[0])
    return canvas


def draw_gt(ann_lines: dict, img_shape: tuple[int, int],
            thickness: int = 3) -> np.ndarray:
    """从 annotation lines 字段画 GT polylines."""
    h, w = img_shape[:2]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    n = len(ann_lines)
    colors = palette(n)
    for (color, (_, ln)) in zip(colors, ann_lines.items()):
        pts = np.array([[p["pixel"][0], p["pixel"][1]]
                        for p in ln["points"]], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=False,
                      color=(color[2], color[1], color[0]),
                      thickness=thickness, lineType=cv2.LINE_AA)
    return canvas


def horizontal_panel(panels: list[tuple[str, np.ndarray]],
                     bar_h: int = 28, font_scale: float = 0.55) -> np.ndarray:
    """[(caption, BGR_image), ...] → 横向拼接, 每块顶部一条标题栏。"""
    if not panels:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    # 统一高度 = max
    target_h = max(im.shape[0] for _, im in panels)
    resized = []
    for cap, im in panels:
        h, w = im.shape[:2]
        if h != target_h:
            new_w = int(w * target_h / h)
            im = cv2.resize(im, (new_w, target_h),
                            interpolation=cv2.INTER_AREA)
        resized.append((cap, im))
    images = [im for _, im in resized]
    captions = [cap for cap, _ in resized]
    body = np.hstack(images)
    bar = np.full((bar_h, body.shape[1], 3), 245, dtype=np.uint8)
    x = 0
    for cap, im in zip(captions, images):
        cv2.putText(bar, cap, (x + 6, bar_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (10, 10, 10), 1,
                    cv2.LINE_AA)
        x += im.shape[1]
        # vertical separator
        cv2.line(bar, (x - 1, 0), (x - 1, bar_h - 1), (180, 180, 180), 1)
    return np.vstack([bar, body])
