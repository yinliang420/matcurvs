"""Pilot v2: same as v1 but with three knob fixes.

Fixes vs v1:
  1. mask_to_pixel_curve uses a *ridge trace* (per-column argmax of mask
     intensity along chart rows) instead of column-median. Falls back to median
     when the column is near-uniform (no clear ridge).
  2. Smoothing window dropped from 9 -> 3.
  3. find_peaks prominence dropped from 0.05*range -> 0.02*range.

Also extends the uniform sample from 30 to 60 images: every (441//60)=7-th
folder, take the first image. The 30 v1 picks are a subset of the v2 picks
(every 14-th folder = every 2nd of the 7-step sequence).
"""
from __future__ import annotations

import json
import os
import sys
import traceback
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import pearsonr

warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, str(Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[3]))))
from pipeline import run_pipeline  # noqa: E402

DATA_ROOT = Path(os.environ.get("RAMAN_POOL", "data/raman"))
OUT_DIR = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[3])) / "experiments" / "results" / "raman_peak_pilot"
DIAG_DIR = OUT_DIR / "diag_v2"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

PEAK_TOL_CM1 = 20.0
SMOOTH_WIN = 3                 # FIX 2: was 9
PROMINENCE_FRAC = 0.02         # FIX 3: was 0.05
N_IMAGES = 60


def _smooth(y: np.ndarray, win: int = SMOOTH_WIN) -> np.ndarray:
    """Moving average smoothing, NaN-aware."""
    if len(y) < win or win <= 1:
        return y.copy()
    kernel = np.ones(win) / win
    valid = ~np.isnan(y)
    y_filled = np.where(valid, y, 0.0)
    smoothed = np.convolve(y_filled, kernel, mode="same")
    weights = np.convolve(valid.astype(float), kernel, mode="same")
    out = np.where(weights > 0, smoothed / np.maximum(weights, 1e-6), np.nan)
    return out


def mask_to_pixel_curve(mask: np.ndarray, x1: int, x2: int, y1: int, y2: int):
    """v2: ridge trace per column.

    For each column x in [x1, x2): collect the foreground rows in [y1, y2).
    If the rows form a clear single cluster (max-mode argmax distinguishable
    from median by > ~10% of mask height), use the argmax (ridge). Otherwise
    fall back to the median (handles thick / noisy mass).
    """
    H, W = mask.shape
    x1 = max(0, x1)
    x2 = min(W, x2)
    y1 = max(0, y1)
    y2 = min(H, y2)
    cols = np.arange(x1, x2)
    height = max(1, y2 - y1)
    ys_pixel = np.full(len(cols), np.nan, dtype=np.float64)

    for i, x in enumerate(cols):
        col = mask[y1:y2, x]
        rows = np.where(col)[0]
        if rows.size == 0:
            continue
        # density-aware ridge: smooth a 1-D occupancy along the column and
        # take its argmax. For a thin curve this is the curve row; for a
        # thick blob it's the blob centroid (similar to median).
        # Use a small Gaussian-ish window proportional to chart height.
        win = max(3, int(height * 0.02))
        if win % 2 == 0:
            win += 1
        kernel = np.ones(win, dtype=np.float64) / win
        density = np.convolve(col.astype(np.float64), kernel, mode="same")
        if density.max() <= 0:
            continue
        ridge_row = int(np.argmax(density))
        median_row = float(np.median(rows))
        # Fall back to median when the density profile is ~uniform: i.e. the
        # ridge row sits far from the median (suggests mode is at edge of
        # support, not a real peak).
        # Heuristic: if the density at ridge is < 1.5x mean density, treat
        # the column as near-uniform and use the median.
        mean_density = density[col > 0].mean() if (col > 0).any() else 0.0
        if mean_density > 0 and density[ridge_row] < 1.5 * mean_density:
            ys_pixel[i] = median_row + y1
        else:
            ys_pixel[i] = float(ridge_row) + y1

    ys_pixel = _smooth(ys_pixel, win=SMOOTH_WIN)
    return cols, ys_pixel


def pixel_curve_to_real(cols, ys_pixel, x1, x2, y1, y2,
                        x_min, x_max, y_min, y_max):
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    xs_real = x_min + (cols - x1) / width * (x_max - x_min)
    ys_norm = 1.0 - (ys_pixel - y1) / height
    ys_real = y_min + ys_norm * (y_max - y_min)
    return xs_real, ys_real


def find_peaks_real(xs_real, ys_real):
    """Run scipy find_peaks on extracted intensity series. v2: prominence=0.02*rng."""
    valid = ~np.isnan(ys_real)
    if valid.sum() < 20:
        return np.array([]), np.array([])
    y_clean = ys_real[valid]
    x_clean = xs_real[valid]
    rng = float(np.nanmax(y_clean) - np.nanmin(y_clean))
    if rng <= 0:
        return np.array([]), np.array([])
    prom = PROMINENCE_FRAC * rng
    idx, _ = find_peaks(y_clean, prominence=prom, distance=10)
    peak_x = x_clean[idx]
    peak_y = y_clean[idx]
    return peak_x, peak_y


def greedy_match(pred_x: np.ndarray, gt_x: np.ndarray, tol: float = PEAK_TOL_CM1):
    if len(pred_x) == 0 or len(gt_x) == 0:
        return []
    diffs = np.abs(pred_x[:, None] - gt_x[None, :])
    matches = []
    used_p, used_g = set(), set()
    flat_idx = np.argsort(diffs, axis=None)
    for fi in flat_idx:
        p, g = np.unravel_index(fi, diffs.shape)
        if diffs[p, g] > tol:
            break
        if p in used_p or g in used_g:
            continue
        used_p.add(p)
        used_g.add(g)
        matches.append((int(p), int(g), float(diffs[p, g])))
        if len(used_p) == len(pred_x) or len(used_g) == len(gt_x):
            break
    return matches


def axis_box_from_lines(axis_lines):
    if axis_lines is None or axis_lines[0] is None or axis_lines[1] is None:
        return None
    x_axis, y_axis = axis_lines
    xa_x1, xa_y1, xa_x2, xa_y2 = x_axis
    ya_x1, ya_y1, ya_x2, ya_y2 = y_axis
    x1 = float(min(xa_x1, xa_x2, ya_x1, ya_x2))
    x2 = float(max(xa_x1, xa_x2, ya_x1, ya_x2))
    y1 = float(min(xa_y1, xa_y2, ya_y1, ya_y2))
    y2 = float(max(xa_y1, xa_y2, ya_y1, ya_y2))
    return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))


def resample(xs_src, ys_src, xs_target):
    valid = ~np.isnan(ys_src)
    if valid.sum() < 4:
        return np.full_like(xs_target, np.nan, dtype=np.float64)
    xs_v = xs_src[valid]
    ys_v = ys_src[valid]
    order = np.argsort(xs_v)
    xs_v = xs_v[order]
    ys_v = ys_v[order]
    return np.interp(xs_target, xs_v, ys_v, left=np.nan, right=np.nan)


def normalized_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    valid = ~np.isnan(pred) & ~np.isnan(gt)
    if valid.sum() < 4:
        return np.nan
    p = pred[valid]
    g = gt[valid]
    rng = float(np.max(g) - np.min(g))
    if rng <= 0:
        return np.nan
    return float(np.sqrt(np.mean((p - g) ** 2)) / rng)


def safe_pearson(p, g) -> float:
    valid = ~np.isnan(p) & ~np.isnan(g)
    if valid.sum() < 4:
        return np.nan
    if np.std(p[valid]) == 0 or np.std(g[valid]) == 0:
        return np.nan
    try:
        return float(pearsonr(p[valid], g[valid])[0])
    except Exception:
        return np.nan


# ----------------------------------------------------------------------------- #
def select_pilot_images(n: int = N_IMAGES):
    folders = sorted([f for f in os.listdir(DATA_ROOT)
                      if (DATA_ROOT / f).is_dir()])
    step = max(1, len(folders) // n)
    picks = []
    for i in range(n):
        idx = i * step
        if idx >= len(folders):
            break
        folder = folders[idx]
        json_path = DATA_ROOT / folder / "chemical_formulas_raman.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            meta = json.load(f)
        if not meta:
            continue
        key = sorted(meta.keys())[0]
        picks.append((folder, key, meta[key]))
    return picks


# ----------------------------------------------------------------------------- #
def save_diag(image_path, axis_box, xs_real, ys_real, gt_x_curve, gt_y_curve,
              pred_peaks_x, pred_peaks_y, gt_peaks_x, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    if img_rgb is not None:
        ax.imshow(img_rgb)
        if axis_box is not None:
            x1, y1, x2, y2 = axis_box
            ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "g-", lw=2)
    ax.set_title(f"input + axis_box\n{Path(image_path).name[:40]}...")
    ax.axis("off")

    ax = axes[0, 1]
    ax.plot(xs_real, ys_real, "b-", lw=1, label="extracted (v2)")
    if len(pred_peaks_x):
        ax.plot(pred_peaks_x, pred_peaks_y, "rv", ms=8, label="pred peaks")
    ax.set_title("Extracted curve (cm^-1 vs intensity)")
    ax.set_xlabel("Raman shift (cm^-1)")
    ax.set_ylabel("Intensity")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(gt_x_curve, gt_y_curve, "k-", lw=1, label="GT (xlsx)")
    for gx in gt_peaks_x:
        ax.axvline(gx, color="orange", alpha=0.5, lw=0.8)
    ax.set_title("Published curve + peaks")
    ax.set_xlabel("Raman shift (cm^-1)")
    ax.set_ylabel("Intensity")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.scatter(gt_peaks_x, np.zeros_like(gt_peaks_x) + 1.0,
               marker="|", s=200, c="orange", label=f"GT (n={len(gt_peaks_x)})")
    if len(pred_peaks_x):
        ax.scatter(pred_peaks_x, np.zeros_like(pred_peaks_x) + 0.6,
                   marker="|", s=200, c="red", label=f"Pred (n={len(pred_peaks_x)})")
    ax.set_ylim(0.0, 1.4)
    ax.set_yticks([])
    ax.set_xlabel("Raman shift (cm^-1)")
    ax.set_title("Peak positions: GT (top) vs Pred (bottom)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(save_path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- #
def process_image(folder: str, image_key: str, meta: dict):
    folder_dir = DATA_ROOT / folder
    img_path = folder_dir / "images" / image_key
    chem = meta.get("chemical_formulas_by_line", {}).get("Line_1", {})
    formula = chem.get("chemical_formula", "")
    xlsx_name = chem.get("excel_file_path", "")
    peaks_meta = chem.get("characterization_data", {}).get("peaks", []) or []
    gt_peaks_x = np.array([p.get("raman_shift_cm1") for p in peaks_meta
                           if p.get("raman_shift_cm1") is not None],
                          dtype=np.float64)

    row = {
        "image_id": f"{folder}/{image_key}",
        "chemical_formula": formula,
        "n_gt_peaks": int(len(gt_peaks_x)),
        "n_pred_peaks": 0,
        "n_matched": 0,
        "peak_mae": np.nan,
        "peak_recall": np.nan,
        "peak_precision": np.nan,
        "curve_pearson": np.nan,
        "curve_nrmse": np.nan,
        "ok": False,
        "reason": "",
    }
    diag_payload = None

    if not img_path.exists():
        row["reason"] = "no_image"
        return row, diag_payload

    xlsx_path = folder_dir / "images" / xlsx_name if xlsx_name else None
    if xlsx_path is None or not xlsx_path.exists():
        row["reason"] = "no_xlsx"
        return row, diag_payload

    try:
        df = pd.read_excel(xlsx_path)
        cols = list(df.columns)
        x_col = cols[0]
        y_col = cols[1]
        gt_x_curve = df[x_col].to_numpy(dtype=np.float64)
        gt_y_curve = df[y_col].to_numpy(dtype=np.float64)
        valid = np.isfinite(gt_x_curve) & np.isfinite(gt_y_curve)
        gt_x_curve = gt_x_curve[valid]
        gt_y_curve = gt_y_curve[valid]
        if len(gt_x_curve) < 10:
            row["reason"] = "xlsx_too_short"
            return row, diag_payload
        order = np.argsort(gt_x_curve)
        gt_x_curve = gt_x_curve[order]
        gt_y_curve = gt_y_curve[order]
    except Exception as e:
        row["reason"] = f"xlsx_read_err:{type(e).__name__}"
        return row, diag_payload

    x_min, x_max = float(gt_x_curve.min()), float(gt_x_curve.max())
    y_min, y_max = float(gt_y_curve.min()), float(gt_y_curve.max())

    try:
        pr = run_pipeline(str(img_path), n_lines=1, separator_mode="v4",
                          morph_close_kernel=5, min_cluster_area=200)
    except Exception as e:
        row["reason"] = f"pipeline_exc:{type(e).__name__}"
        return row, diag_payload

    if not pr.ok:
        row["reason"] = f"pipeline_err:{(pr.error or '')[:60]}"
        return row, diag_payload

    box = axis_box_from_lines(pr.axis_lines)
    if box is None:
        row["reason"] = "no_axis_box"
        return row, diag_payload
    x1, y1, x2, y2 = box
    if x2 - x1 < 30 or y2 - y1 < 30:
        row["reason"] = "axis_box_too_small"
        return row, diag_payload

    if not pr.cluster_masks:
        row["reason"] = "no_cluster_mask"
        return row, diag_payload
    mask = pr.cluster_masks[0]

    cols_px, ys_px = mask_to_pixel_curve(mask, x1, x2, y1, y2)
    if np.sum(~np.isnan(ys_px)) < 20:
        row["reason"] = "curve_empty"
        return row, diag_payload

    xs_real, ys_real = pixel_curve_to_real(cols_px, ys_px, x1, x2, y1, y2,
                                           x_min, x_max, y_min, y_max)

    pred_peaks_x, pred_peaks_y = find_peaks_real(xs_real, ys_real)
    matches = greedy_match(pred_peaks_x, gt_peaks_x, tol=PEAK_TOL_CM1)

    n_pred = int(len(pred_peaks_x))
    n_gt = int(len(gt_peaks_x))
    n_matched = len(matches)
    mae = float(np.mean([m[2] for m in matches])) if matches else np.nan
    recall = n_matched / n_gt if n_gt > 0 else np.nan
    precision = n_matched / n_pred if n_pred > 0 else np.nan

    common_min = max(x_min, float(np.nanmin(xs_real)))
    common_max = min(x_max, float(np.nanmax(xs_real)))
    if common_max - common_min > 50:
        grid = np.linspace(common_min, common_max, 400)
        pred_on_grid = resample(xs_real, ys_real, grid)
        gt_on_grid = resample(gt_x_curve, gt_y_curve, grid)
        pcc = safe_pearson(pred_on_grid, gt_on_grid)
        nrmse = normalized_rmse(pred_on_grid, gt_on_grid)
    else:
        pcc, nrmse = np.nan, np.nan

    row.update({
        "n_pred_peaks": n_pred,
        "n_matched": n_matched,
        "peak_mae": mae,
        "peak_recall": recall,
        "peak_precision": precision,
        "curve_pearson": pcc,
        "curve_nrmse": nrmse,
        "ok": True,
        "reason": "",
    })

    diag_payload = dict(
        image_path=str(img_path), axis_box=box,
        xs_real=xs_real, ys_real=ys_real,
        gt_x_curve=gt_x_curve, gt_y_curve=gt_y_curve,
        pred_peaks_x=pred_peaks_x, pred_peaks_y=pred_peaks_y,
        gt_peaks_x=gt_peaks_x,
    )
    return row, diag_payload


def main():
    picks = select_pilot_images(N_IMAGES)
    print(f"selected {len(picks)} images (target {N_IMAGES})")

    rows = []
    diag_cache = {}
    for i, (folder, key, meta) in enumerate(picks):
        try:
            row, diag = process_image(folder, key, meta)
        except Exception:
            row = {
                "image_id": f"{folder}/{key}",
                "chemical_formula": "", "n_gt_peaks": 0, "n_pred_peaks": 0,
                "n_matched": 0, "peak_mae": np.nan, "peak_recall": np.nan,
                "peak_precision": np.nan, "curve_pearson": np.nan,
                "curve_nrmse": np.nan, "ok": False,
                "reason": "uncaught:" + traceback.format_exc().splitlines()[-1][:60],
            }
            diag = None
        rows.append(row)
        if diag is not None:
            diag_cache[row["image_id"]] = diag
        print(f"[{i+1:>2}/{len(picks)}] {row['image_id'][:50]:50s}  "
              f"ok={row['ok']!s:5s} gt={row['n_gt_peaks']:>2} "
              f"pred={row['n_pred_peaks']:>2} match={row['n_matched']:>2} "
              f"mae={row['peak_mae']:>6.2f} reason={row['reason'][:30]}")

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "peak_pilot_v2_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    # Diagnostic plots: 3 best + 3 worst by recall, restricted to the NEW 30
    # (odd-indexed picks). v1 selection was every 14th = even-indexed of v2's
    # 7-step list. So "new" = odd index in v2 picks order.
    new_ids = {f"{folder}/{key}" for j, (folder, key, _meta)
               in enumerate(picks) if j % 2 == 1}
    df_new = df[df["image_id"].isin(new_ids) & df["ok"]].copy()
    df_new = df_new.dropna(subset=["peak_recall"])
    if len(df_new) >= 1:
        df_new = df_new.sort_values(["peak_recall", "peak_mae"],
                                    ascending=[False, True])
        best = df_new.head(3)
        worst = df_new.tail(3)
        for tag, sub in [("best", best), ("worst", worst)]:
            for _, r in sub.iterrows():
                payload = diag_cache.get(r["image_id"])
                if payload is None:
                    continue
                safe_name = r["image_id"].replace("/", "_")[:80]
                save = DIAG_DIR / f"{tag}_{safe_name}.png"
                try:
                    save_diag(save_path=save, **payload)
                except Exception as e:
                    print(f"diag failed for {r['image_id']}: {e}")
    return df


if __name__ == "__main__":
    main()
