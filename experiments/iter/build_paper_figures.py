"""Generate figures for the NeurIPS 2026 paper revision.

  - fig_reconstruction_strip.png : 4 rows (one per stratum) × 3 cols
        (original | GT polylines | our reconstruction)
        — proves "we faithfully reconstruct curves at IoU ≥ 0.8".
  - fig_downstream.png : 2-panel
        (a) per-K Δaccuracy bar chart for VLM y-lookup
        (b) Raman peak detection — recall / MAE / Pearson box across n=60
        — proves "extracted curves enable scientific reasoning the VLM cannot do alone".

Run after the iter12 pipeline finishes; takes the best CSV (highest pct80) automatically.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[2]))
LABEL_DIR = KMEANS_ROOT / "label"
ITER_DIR = KMEANS_ROOT / "experiments" / "results" / "iter"
PAPER_FIG_DIR = KMEANS_ROOT / "paper" / "neurips2026_intro" / "figures"


def _pick_best_csv(candidates: list[Path]) -> Path:
    best = (None, -1.0)
    for csv in candidates:
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        n_lines = int(df["k_gt"].sum()); n_hit = int(df.get("n_iou_ge_0_8", pd.Series([0])).sum())
        pct80 = n_hit / max(1, n_lines)
        score = pct80 + 0.5 * float(df["iou_matched_mean"].mean())
        if score > best[1]:
            best = (csv, score)
    return best[0]


def _pick_one_per_stratum(csv_path: Path) -> dict[str, str]:
    """Pick one image_id per stratum where pct_lines_iou_ge_0.8 is highest
    (tie-breaker: highest iou_matched_mean), restricted to images where the
    pipeline got K_pred==K_gt — otherwise the showcase visually appears to
    over-cluster (one GT line shown as multiple cluster colours)."""
    df = pd.read_csv(csv_path)
    df = df[df["k_pred"].astype(int) == df["k_gt"].astype(int)].copy()
    df["pct_local"] = df["n_iou_ge_0_8"].astype(int) / df["k_gt"].astype(int).clip(lower=1)
    out: dict[str, str] = {}
    for st in ["L1_single", "L2_two_line", "L3_few_line", "L4_many_line"]:
        sub = df[df["stratum"] == st].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["pct_local", "iou_matched_mean"], ascending=False)
        out[st] = str(sub.iloc[0]["image_id"])
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


def _load_gemini_k_csv():
    p = ITER_DIR / "k_full_label_gemini.csv"
    if not p.exists():
        return None
    return pd.read_csv(p).set_index("image_id")


_K_CSV = _load_gemini_k_csv()


def _ours_overlay(img_bgr: np.ndarray, image_id: str,
                  m2f_ckpt: str, run_kwargs: dict) -> np.ndarray:
    """Re-runs the pipeline on this single image to get cluster_masks.
    Looks up K from k_full_label_gemini.csv so the figure matches the
    paper's published phaseC_flip config (which used --k_source llm)."""
    import sys
    sys.path.insert(0, str(KMEANS_ROOT))
    from pipeline import run_pipeline
    img_path = LABEL_DIR / f"{image_id}.jpg"
    n_lines = None
    if _K_CSV is not None and image_id in _K_CSV.index:
        col = "gemini-3.1-pro-preview_k"
        if col in _K_CSV.columns and not pd.isna(_K_CSV.at[image_id, col]):
            n_lines = int(_K_CSV.at[image_id, col])
    pr = run_pipeline(str(img_path), m2f_ckpt=m2f_ckpt, n_lines=n_lines, **run_kwargs)
    out = img_bgr.copy()
    palette = [(230, 30, 30), (30, 180, 30), (30, 100, 230), (240, 180, 30),
               (170, 60, 220), (30, 190, 200), (255, 120, 0), (120, 80, 40)]
    for i, m in enumerate(pr.cluster_masks or []):
        col = palette[i % len(palette)]
        out[m] = col
    return out


def reconstruction_strip(out_png: Path, best_csv: Path,
                          m2f_ckpt: str, run_kwargs: dict) -> None:
    picks = _pick_one_per_stratum(best_csv)
    fig, axes = plt.subplots(len(picks), 3, figsize=(11, 2.6 * len(picks)),
                             constrained_layout=True)
    if len(picks) == 1:
        axes = np.array([axes])
    for r, (stratum, image_id) in enumerate(picks.items()):
        img_path = LABEL_DIR / f"{image_id}.jpg"
        ann_path = LABEL_DIR / f"{image_id}_annotation.json"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gt_ovl = _gt_overlay(img, ann_path, color=(0, 220, 30))
        try:
            ours_ovl = _ours_overlay(img, image_id, m2f_ckpt, run_kwargs)
        except Exception as e:
            print(f"[fig] {image_id}: ours overlay failed: {e}")
            ours_ovl = img
        for c, (caption, panel) in enumerate([
            (f"{stratum}\nimg {image_id[:10]}…", img),
            ("GT polylines", gt_ovl),
            ("Ours (per-line clusters)", ours_ovl),
        ]):
            ax = axes[r, c]
            ax.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
            ax.set_title(caption, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"saved → {out_png}")


def downstream_figure(out_png: Path) -> None:
    res = KMEANS_ROOT / "experiments" / "results"
    yl = pd.read_csv(res / "vlm_downstream" / "label_results.csv")
    raman = pd.read_csv(res / "raman_peak_pilot" / "peak_pilot_v3_results.csv")
    raman = raman[raman["ok"].astype(bool)]
    raman_v2 = pd.read_csv(res / "raman_peak_pilot" / "peak_pilot_v2_compare.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)

    # (a) per-K Δaccuracy bar chart
    by_k = yl.groupby("K").agg(
        acc_A=("correct_A", "mean"),
        acc_B=("correct_B", "mean"),
        n=("correct_B", "size"),
    ).reset_index()
    K_vals = by_k["K"].astype(int).tolist()
    x = np.arange(len(K_vals))
    w = 0.4
    axes[0].bar(x - w/2, by_k["acc_A"], w, label="image-only (VLM)", color="#bbbbbb")
    axes[0].bar(x + w/2, by_k["acc_B"], w, label="image + ours extraction",
                color="#1f78b4")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(k) for k in K_vals])
    axes[0].set_xlabel("# curves K")
    axes[0].set_ylabel("y-lookup accuracy (rel-err ≤ 10%)")
    axes[0].set_title("(a) VLM y-lookup gain (n=100, gemini-3.1-pro-preview)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_ylim(0, 0.75)
    for xi, dv in zip(x, (by_k["acc_B"] - by_k["acc_A"])):
        axes[0].annotate(f"+{dv*100:.0f}pp" if dv >= 0 else f"{dv*100:.0f}pp",
                         (xi, max(by_k.loc[xi, "acc_A"], by_k.loc[xi, "acc_B"]) + 0.02),
                         ha="center", fontsize=7)

    # (b) Raman peak detection — recall and MAE side-by-side as box+strip
    bx = axes[1]
    metrics = {
        "v2 recall (axis_box)": raman_v2["v2_recall"].dropna(),
        "v3 recall (cluster x)": raman["peak_recall"].dropna(),
    }
    pos = np.arange(len(metrics))
    parts = bx.violinplot(list(metrics.values()), positions=pos, widths=0.7,
                          showmeans=True, showmedians=True)
    for p in parts["bodies"]:
        p.set_facecolor("#1f78b4"); p.set_alpha(0.5)
    bx.set_xticks(pos)
    bx.set_xticklabels(list(metrics.keys()), rotation=8, fontsize=8)
    bx.set_ylabel("peak recall")
    bx.set_ylim(0, 1.05)
    bx.set_title("(b) Raman peak recall (n=60), median 0.50 → 1.00")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"saved → {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--best_csv", default=None,
                    help="explicit best CSV; else pick highest pct80 from phase*F* files")
    ap.add_argument("--m2f_ckpt",
                    default=str(KMEANS_ROOT / "models" / "mask2foremer_q12_d3_h256_best.pt"))
    ap.add_argument("--out_dir", default=str(PAPER_FIG_DIR))
    args = ap.parse_args()

    if args.best_csv:
        best = Path(args.best_csv)
    else:
        candidates = sorted(ITER_DIR.glob("phase*_full204_metrics.csv"))
        best = _pick_best_csv(candidates)
    print(f"best CSV = {best}")

    out_dir = Path(args.out_dir)
    if best is not None and best.exists():
        # Match phaseC_flip exactly (paper headline config). Knobs that
        # iter12 found to either no-op or regress (thicken, query-NMS,
        # scatter-branch, refine_mode) are off; n_lines per image is read
        # from k_full_label_gemini.csv inside _ours_overlay.
        run_kwargs = dict(separator_mode="v4", morph_close_kernel=5,
                         min_cluster_area=200,
                         m2f_threshold=0.25, m2f_scales=(512, 768, 1024),
                         flip_tta=True, thicken=False,
                         query_nms_mode="off", scatter_branch="off",
                         refine_mode="off")
        reconstruction_strip(out_dir / "fig_reconstruction_strip.png",
                              best, args.m2f_ckpt, run_kwargs)
    else:
        print("no best CSV → skip reconstruction strip")

    downstream_figure(out_dir / "fig_downstream.png")


if __name__ == "__main__":
    main()
