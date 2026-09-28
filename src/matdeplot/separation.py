"""Paper-aligned foreground preparation and per-curve separation."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .config import SeparationConfig


def prepare_pixel_mask(
    image_bgr: np.ndarray,
    foreground_mask: np.ndarray,
    valid_region: np.ndarray,
    used_fallback: bool,
    config: SeparationConfig,
) -> np.ndarray:
    """Select extraction pixels and apply the v4 pre-clustering morphology."""

    if used_fallback:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        pixel_mask = valid_region & (gray < 240)
    else:
        pixel_mask = foreground_mask.astype(bool) & valid_region

    close_size = int(config.morph_close_kernel)
    if close_size > 0:
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_size, close_size)
        )
        work = cv2.morphologyEx(
            pixel_mask.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            close_kernel,
        )
        dilation_size = int(config.dilation_kernel)
        if dilation_size > 0:
            dilation_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (dilation_size, dilation_size),
            )
            work = cv2.dilate(work, dilation_kernel, iterations=1)
        pixel_mask = work > 0
    return pixel_mask


def pixel_features(
    image_bgr: np.ndarray,
    pixel_mask: np.ndarray,
    spatial_weight: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return standardized 5D LAB-XY features and their pixel coordinates."""

    ys, xs = np.where(pixel_mask)
    if ys.size == 0:
        return np.empty((0, 5), dtype=np.float32), ys, xs

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    colors = lab[ys, xs].astype(np.float32)
    height, width = pixel_mask.shape
    coordinates = np.stack(
        [xs / max(width, 1), ys / max(height, 1)],
        axis=1,
    ).astype(np.float32)

    color_features = StandardScaler().fit_transform(colors)
    spatial_features = (
        StandardScaler().fit_transform(coordinates) * float(spatial_weight)
    )
    return (
        np.concatenate([color_features, spatial_features], axis=1),
        ys,
        xs,
    )


def is_grayscale_chart(image_bgr: np.ndarray, threshold: float = 6.0) -> bool:
    channel_mean = image_bgr.mean(axis=2, keepdims=True)
    difference = np.abs(image_bgr.astype(np.int16) - channel_mean)
    return float(difference.mean()) < threshold


def _cluster_by_y(
    pixel_mask: np.ndarray,
    n_clusters: int,
    n_init: int,
    random_state: int,
) -> List[np.ndarray]:
    ys, xs = np.where(pixel_mask)
    if ys.size < max(50, n_clusters * 10):
        return []
    labels = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        random_state=random_state,
    ).fit_predict(ys.reshape(-1, 1).astype(np.float32))
    masks: List[np.ndarray] = []
    for cluster_index in range(n_clusters):
        mask = np.zeros(pixel_mask.shape, dtype=bool)
        selected = labels == cluster_index
        mask[ys[selected], xs[selected]] = True
        masks.append(mask)
    return sorted(masks, key=lambda mask: -int(mask.sum()))


def separate_curves(
    image_bgr: np.ndarray,
    pixel_mask: np.ndarray,
    n_curves: int,
    valid_region: np.ndarray,
    config: SeparationConfig,
) -> List[np.ndarray]:
    """Split eligible foreground pixels into paper-configured curve masks."""

    n_curves = max(1, int(n_curves))
    if is_grayscale_chart(image_bgr) and n_curves >= 2:
        masks = _cluster_by_y(
            pixel_mask,
            n_clusters=n_curves,
            n_init=config.kmeans_n_init,
            random_state=config.random_state,
        )
    else:
        features, ys, xs = pixel_features(
            image_bgr,
            pixel_mask,
            config.spatial_weight,
        )
        if len(features) < max(50, n_curves * 10):
            return []
        labels = KMeans(
            n_clusters=n_curves,
            n_init=config.kmeans_n_init,
            random_state=config.random_state,
        ).fit_predict(features)
        masks = []
        for cluster_index in range(n_curves):
            mask = np.zeros(pixel_mask.shape, dtype=bool)
            selected = labels == cluster_index
            mask[ys[selected], xs[selected]] = True
            masks.append(mask)
        masks.sort(key=lambda mask: -int(mask.sum()))

    masks = [
        mask
        for mask in masks
        if int(mask.sum()) >= int(config.min_cluster_area)
    ]
    return restore_antialiased_edges(
        image_bgr,
        masks,
        valid_region,
        delta_e_threshold=config.delta_e_threshold,
    )


def restore_antialiased_edges(
    image_bgr: np.ndarray,
    masks: List[np.ndarray],
    valid_region: np.ndarray,
    delta_e_threshold: float,
) -> List[np.ndarray]:
    """Grow each cluster by two pixels while retaining color-compatible edges."""

    if not masks:
        return []
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    restored: List[np.ndarray] = []
    for mask in masks:
        current = mask.astype(bool)
        if int(current.sum()) < 20:
            restored.append(current)
            continue
        mean_color = image_lab[current].mean(axis=0)
        grown = cv2.dilate(current.astype(np.uint8), kernel, iterations=1) > 0
        ring = grown & ~current & valid_region
        if not ring.any():
            restored.append(current)
            continue
        ring_y, ring_x = np.where(ring)
        color_distance = np.linalg.norm(
            image_lab[ring] - mean_color,
            axis=1,
        )
        keep = color_distance < float(delta_e_threshold)
        compatible = np.zeros_like(current)
        compatible[ring_y[keep], ring_x[keep]] = True
        restored.append(current | compatible)
    return restored
