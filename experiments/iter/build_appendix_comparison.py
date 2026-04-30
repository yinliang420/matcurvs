"""Generate the appendix qualitative-comparison figure.

For each of the 4 showcase image_ids (one per stratum), draw a row with
columns: original image / GT polylines / Ours / GPT-5.4 / Gemini /
Qwen / DePlot. Polylines are rendered in pixel space using the cached
JSONs in experiments/results/vlm_showcase/<backend>/<image_id>.json.

Outputs: paper/neurips2026_intro/figures/fig_appendix_comparison.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[2]))
LABEL_DIR = KMEANS_ROOT / "label"
SHOWCASE_DIR = KMEANS_ROOT / "experiments" / "results" / "vlm_showcase"
PAPER_FIG_DIR = KMEANS_ROOT / "paper" / "neurips2026_intro" / "figures"

sys.path.insert(0, str(KMEANS_ROOT / "experiments" / "iter"))


SHOWCASE_PREFIXES = [
    ("L1_single",     "217a5eb3c9f7c97b5738cb13"),
    ("L2_two_line",   "a002b8da910d1e3c2a6ddc7f"),
    ("L3_few_line",   "510aa6c30e12fcc5830a9a0d"),
    ("L4_many_line",  "00e558db0245b40006acf7a1"),
]

# Backend column order + display labels.
BACKEND_COLS = [
    ("vlm_gpt-5.4",                "GPT-5.4"),
    ("vlm_gemini-3.1-pro-preview", "Gemini-3.1-pro"),
    ("vlm_qwen3.5-plus",           "Qwen-3.5-plus"),
    ("deplot",                     "DePlot"),
]

PALETTE = [(230, 30, 30), (30, 180, 30), (30, 100, 230), (240, 180, 30),
           (170, 60, 220), (30, 190, 200), (255, 120, 0), (120, 80, 40)]


def _resolve(prefix: str) -> str | None:
    cands = list(LABEL_DIR.glob(f"{prefix}*_annotation.json"))
    return cands[0].stem.replace("_annotation", "") if cands else None


def _draw_polylines(img_bgr: np.ndarray, trails: list[list[list[float]]],
                    thickness: int = 2) -> np.ndarray:
    out = img_bgr.copy()
    for i, t in enumerate(trails):
        if not t or len(t) < 2:
            continue
        col = PALETTE[i % len(PALETTE)]
        pts = np.array(t, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], False, col,
                      thickness=thickness, lineType=cv2.LINE_AA)
    return out


def _gt_overlay(img_bgr: np.ndarray, ann_path: Path,
                color: tuple = (0, 200, 0)) -> np.ndarray:
    d = json.loads(ann_path.read_text())
    out = img_bgr.copy()
    for _, ln in d["lines"].items():
        pts = np.array([p["pixel"] for p in ln["points"]], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, color,
                          thickness=2, lineType=cv2.LINE_AA)
    return out


def _ours_overlay_from_pipeline(img_bgr, image_id, m2f_ckpt, run_kwargs, k_csv):
    """Re-runs the pipeline once to get cluster_masks, draws as filled bool masks."""
    from pipeline import run_pipeline
    img_path = LABEL_DIR / f"{image_id}.jpg"
    n_lines = None
    if k_csv is not None and image_id in k_csv.index:
        col = "gemini-3.1-pro-preview_k"
        if col in k_csv.columns and k_csv.at[image_id, col] == k_csv.at[image_id, col]:
            n_lines = int(k_csv.at[image_id, col])
    pr = run_pipeline(str(img_path), m2f_ckpt=m2f_ckpt, n_lines=n_lines, **run_kwargs)
    out = img_bgr.copy()
    for i, m in enumerate(pr.cluster_masks or []):
        col = PALETTE[i % len(PALETTE)]
        out[m] = col
    return out


def _vlm_overlay(img_bgr: np.ndarray, image_id: str, backend: str) -> tuple[np.ndarray, str]:
    p = SHOWCASE_DIR / backend / f"{image_id}.json"
    if not p.exists():
        return img_bgr, "no cache"
    d = json.loads(p.read_text())
    trails = d.get("pred_pixel", [])
    n_pred = d.get("n_predicted_trails", 0)
    n_gt = d.get("n_gt_trails", 0)
    success = d.get("success", False)
    if not success:
        return img_bgr, f"failed (k_pred={n_pred}/{n_gt})"
    return _draw_polylines(img_bgr, trails, thickness=2), f"k_pred={n_pred} / k_gt={n_gt}"


def main() -> None:
    sys.path.insert(0, str(KMEANS_ROOT))
    import pandas as pd
    k_csv_path = KMEANS_ROOT / "experiments" / "results" / "iter" / "k_full_label_gemini.csv"
    k_csv = pd.read_csv(k_csv_path).set_index("image_id") if k_csv_path.exists() else None

    m2f_ckpt = str(KMEANS_ROOT / "models" / "mask2foremer_q12_d3_h256_best.pt")
    run_kwargs = dict(separator_mode="v4", morph_close_kernel=5,
                     min_cluster_area=200, m2f_threshold=0.25,
                     m2f_scales=(512, 768, 1024), flip_tta=True,
                     thicken=False, query_nms_mode="off",
                     scatter_branch="off", refine_mode="off")

    cols = [("orig", "Original"), ("gt", "GT polylines"), ("ours", "Ours")] + BACKEND_COLS
    n_cols = len(cols)
    n_rows = len(SHOWCASE_PREFIXES)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.0 * n_cols, 2.0 * n_rows),
                             constrained_layout=True)

    for r, (stratum, prefix) in enumerate(SHOWCASE_PREFIXES):
        image_id = _resolve(prefix)
        if image_id is None:
            continue
        img_path = LABEL_DIR / f"{image_id}.jpg"
        ann_path = LABEL_DIR / f"{image_id}_annotation.json"
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for c, (key, label) in enumerate(cols):
            ax = axes[r, c]
            if key == "orig":
                panel, sub = img, ""
            elif key == "gt":
                panel = _gt_overlay(img, ann_path)
                d = json.loads(ann_path.read_text())
                sub = f"K_gt={len(d['lines'])}"
            elif key == "ours":
                panel = _ours_overlay_from_pipeline(img, image_id, m2f_ckpt,
                                                     run_kwargs, k_csv)
                sub = ""
            else:
                panel, sub = _vlm_overlay(img, image_id, key)
            ax.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
            title = label
            if r == 0:
                ax.set_title(title, fontsize=8)
            if sub:
                ax.set_xlabel(sub, fontsize=6)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(stratum.replace("_", "\n"), fontsize=8)

    out_png = PAPER_FIG_DIR / "fig_appendix_comparison.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"saved → {out_png}")


if __name__ == "__main__":
    main()
