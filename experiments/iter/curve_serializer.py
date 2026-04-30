"""Curve serialization for VLM downstream eval (label/ y-lookup task).

Convert ``pipeline.run_pipeline()`` `cluster_masks` (pixel-level bool arrays) into a
JSON block of real-unit polylines suitable for use in Condition B prompt.

Pipeline of operations:
  1. Load annotation -> recover axis_box (x/y axis pixel positions)
  2. Load fixed sidecar (label/axis_range_fixed/<id>_axis.json) for real-world
     [x_min, x_max] x [y_min, y_max] derived from GT polylines
  3. Linear pixel<->real map between axis_box and the real range
  4. Per cluster_mask: median-y per x-bin -> single-valued curve (px coords)
  5. Resample to N=50 evenly-spaced x positions on cluster's x extent
  6. Convert px -> real
  7. Serialize JSON {axis_units, x_range, y_range, curves:[{points: [[x,y]]}]}
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import os


KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[2]))
LABEL_DIR = KMEANS_ROOT / "label"
SIDECAR_DIR = LABEL_DIR / "axis_range_fixed"


@dataclass
class AxisMapping:
    """Linear pixel <-> real coordinate mapping for one chart.

    Internally stores affine slopes/intercepts:  real = slope * px + intercept
    so we work for either sign of slope (e.g. when image y increases downward
    AND chart y also increases downward, both are positive).
    """
    x_pix_min: float; x_pix_max: float
    y_pix_top: float; y_pix_bot: float   # top = small px; bot = large px
    x_real_min: float; x_real_max: float  # min/max of real x (regardless of slope)
    y_real_min: float; y_real_max: float
    ax: float = 1.0; bx: float = 0.0      # real_x = ax * px + bx
    ay: float = 1.0; by: float = 0.0      # real_y = ay * py + by

    def px_to_real_x(self, px):
        return self.ax * np.asarray(px, dtype=float) + self.bx

    def px_to_real_y(self, py):
        return self.ay * np.asarray(py, dtype=float) + self.by


def _linear_fit(xs: list[float], ys: list[float]) -> Optional[tuple[float, float]]:
    """Least-squares fit y = a*x + b. Returns (a, b) or None if degenerate."""
    if len(xs) < 2:
        return None
    x = np.asarray(xs, dtype=float); y = np.asarray(ys, dtype=float)
    if x.max() - x.min() < 1e-6:
        return None
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def load_axis_mapping(image_id: str) -> Optional[AxisMapping]:
    """Build pixel<->real mapping from GT polyline points (the ground-truth
    pixel<->real correspondence stored in `points[i].pixel`/`real`).

    We fit two linear maps:
        real_x = ax * pix_x + bx
        real_y = ay * pix_y + by
    using all GT polyline points. This is more accurate than relying on
    axis_box + sidecar real range (which is a *data* range, not an axis range).

    Returns None if anything is missing/unreliable (needs_review, too few pts).
    """
    ann_path = LABEL_DIR / f"{image_id}_annotation.json"
    side_path = SIDECAR_DIR / f"{image_id}_axis.json"
    if not ann_path.exists() or not side_path.exists():
        return None
    try:
        ann = json.loads(ann_path.read_text())
        side = json.loads(side_path.read_text())
    except Exception:
        return None

    if side.get("needs_review"):
        return None

    # gather (pixel, real) pairs across all lines
    px_xs: list[float] = []; px_ys: list[float] = []
    rl_xs: list[float] = []; rl_ys: list[float] = []
    for _name, ln in (ann.get("lines") or {}).items():
        for p in (ln or {}).get("points") or []:
            pix = p.get("pixel"); real = p.get("real")
            if not pix or not real or len(pix) < 2 or len(real) < 2:
                continue
            try:
                px_xs.append(float(pix[0])); rl_xs.append(float(real[0]))
                px_ys.append(float(pix[1])); rl_ys.append(float(real[1]))
            except (TypeError, ValueError):
                continue

    if len(px_xs) < 4:
        return None

    fit_x = _linear_fit(px_xs, rl_xs)
    fit_y = _linear_fit(px_ys, rl_ys)
    if fit_x is None or fit_y is None:
        return None
    ax, bx = fit_x
    ay, by = fit_y

    # We store the mapping as (ax, bx, ay, by) inside AxisMapping; the
    # AxisMapping's px_to_real_* methods use these slopes/offsets directly.
    # We piggyback on the existing dataclass fields by encoding the linear map.
    #
    # To preserve the AxisMapping API (px_to_real_x/y), store synthetic anchor
    # points: pix_min->real_min uses the fit at min/max actual pixel range.
    p_xmin = min(px_xs); p_xmax = max(px_xs)
    p_ymin_top = min(px_ys); p_ymax_bot = max(px_ys)
    r_xmin = ax * p_xmin + bx; r_xmax = ax * p_xmax + bx
    r_ytop = ay * p_ymin_top + by  # at small px y -> "top"
    r_ybot = ay * p_ymax_bot + by  # at large px y -> "bottom"
    # ensure y_real_min/max are min/max regardless of slope sign
    y_real_min = min(r_ytop, r_ybot)
    y_real_max = max(r_ytop, r_ybot)

    return AxisMapping(
        x_pix_min=p_xmin, x_pix_max=p_xmax,
        y_pix_top=p_ymin_top, y_pix_bot=p_ymax_bot,
        x_real_min=min(r_xmin, r_xmax), x_real_max=max(r_xmin, r_xmax),
        y_real_min=y_real_min, y_real_max=y_real_max,
        ax=ax, bx=bx, ay=ay, by=by,
    )


def mask_to_xy_curve(mask: np.ndarray, n_bins_per_pixel: float = 1.0) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Convert a 2D bool mask to a single-valued (x_px, y_px) polyline.

    Median-y per x column. Returns None if mask too small.
    """
    if mask is None or mask.size == 0:
        return None
    ys, xs = np.where(mask)
    if xs.size < 5:
        return None
    # bin by integer x columns
    order = np.argsort(xs)
    xs_s = xs[order]; ys_s = ys[order]
    # group: split when x increments
    unique_x, idx_start = np.unique(xs_s, return_index=True)
    medy = np.empty_like(unique_x, dtype=float)
    for i in range(len(unique_x)):
        a = idx_start[i]
        b = idx_start[i + 1] if i + 1 < len(unique_x) else len(xs_s)
        medy[i] = float(np.median(ys_s[a:b]))
    return unique_x.astype(float), medy


def resample_curve(x_px: np.ndarray, y_px: np.ndarray,
                    n_points: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Resample to n_points evenly-spaced x positions over [x.min, x.max]."""
    x_min = float(x_px.min()); x_max = float(x_px.max())
    if x_max - x_min < 1e-6:
        return x_px[:1], y_px[:1]
    sample_x = np.linspace(x_min, x_max, n_points)
    # x_px is sorted (from np.unique); interpolate y_px
    sample_y = np.interp(sample_x, x_px, y_px)
    return sample_x, sample_y


def round_sig(x: float, sig: int = 4) -> float:
    if not math.isfinite(x) or x == 0:
        return float(x)
    d = sig - int(math.floor(math.log10(abs(x)))) - 1
    return round(x, d)


def serialize_curves(image_id: str, cluster_masks: list[np.ndarray],
                      n_points: int = 50, max_curves: int = 8) -> Optional[dict]:
    """Build the JSON block. Returns None if axis mapping unavailable."""
    mp = load_axis_mapping(image_id)
    if mp is None:
        return None

    curves_out = []
    # cluster_masks already sorted by descending area (pipeline.py:135)
    for i, m in enumerate(cluster_masks[:max_curves]):
        out = mask_to_xy_curve(m)
        if out is None:
            continue
        x_px, y_px = out
        rx, ry = resample_curve(x_px, y_px, n_points=n_points)
        rx_real = mp.px_to_real_x(rx)
        ry_real = mp.px_to_real_y(ry)
        pts = []
        for xv, yv in zip(rx_real, ry_real):
            if not (math.isfinite(float(xv)) and math.isfinite(float(yv))):
                continue
            pts.append([round_sig(float(xv), 4), round_sig(float(yv), 4)])
        if len(pts) < 2:
            continue
        # sort by x ascending (already should be) for clarity
        pts.sort(key=lambda p: p[0])
        curves_out.append({
            "id": f"curve_{i+1}",
            "label": None,
            "n_points": len(pts),
            "points": pts,
        })

    if not curves_out:
        return None

    return {
        "axis_units": {"x": None, "y": None},
        "x_range": [round_sig(mp.x_real_min, 4), round_sig(mp.x_real_max, 4)],
        "y_range": [round_sig(mp.y_real_min, 4), round_sig(mp.y_real_max, 4)],
        "curves": curves_out,
    }


def find_longest_cluster_idx(cluster_masks: list[np.ndarray]) -> int:
    """Return index of cluster with widest x-extent (real-world longest curve).

    Falls back to largest area (idx 0) if all extents tied.
    """
    best_idx = 0
    best_extent = -1
    for i, m in enumerate(cluster_masks):
        if m is None or m.sum() < 5:
            continue
        ys, xs = np.where(m)
        if xs.size == 0:
            continue
        ext = int(xs.max() - xs.min())
        if ext > best_extent:
            best_extent = ext
            best_idx = i
    return best_idx


__all__ = [
    "AxisMapping",
    "load_axis_mapping",
    "mask_to_xy_curve",
    "resample_curve",
    "serialize_curves",
    "find_longest_cluster_idx",
]
