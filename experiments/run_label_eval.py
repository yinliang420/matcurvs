"""Evaluate a PipelineResult JSON against ``label/`` annotations.

``label/*_annotation.json`` has schema::

    {"image_size": [w, h],
     "lines": {"Line_i": {"points": [{"pixel": [x,y], "real": [x,y]}, ...]}}}

Here we run the pipeline on every label/ image and score against each
annotation's pixel-polyline using the MatDeplot unified evaluator.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import os

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[1]))
MATDEPLOT_ROOT = Path(os.environ.get("MATDEPLOT_ROOT", "/path/to/MatDeplot/mdp_v2"))
sys.path.insert(0, str(KMEANS_ROOT))
sys.path.insert(0, str(MATDEPLOT_ROOT))

from matdeplot.eval.unified_metrics import evaluate_cell  # noqa: E402
from pipeline import run_pipeline, PipelineResult  # noqa: E402
from experiments.kmeans_matdeplot_adapter import (  # noqa: E402
    pipeline_result_to_json,
)

LABEL_DIR = KMEANS_ROOT / "label"


def stratum_for(n_lines: int, n_points_total: int) -> str:
    if n_lines == 1:
        return "L1_single"
    if n_lines == 2:
        return "L2_two_line"
    if n_lines <= 4:
        return "L3_few_line"
    return "L4_many_line"


def load_gt_from_label(ann_path: Path):
    d = json.loads(ann_path.read_text())
    w, h = d["image_size"]
    trails = []
    total_points = 0
    for _, ln in d["lines"].items():
        pts = ln["points"]
        arr = np.array([[p["pixel"][0], p["pixel"][1]] for p in pts], dtype=float)
        trails.append(arr)
        total_points += len(pts)
    return trails, (w, h), total_points


def score_image(image_id: str, img_path: Path, ann_path: Path,
                m2f_min_fg: float, cell_id: str,
                m2f_ckpt: str | None = None,
                separator_mode: str = "v1",
                morph_close_kernel: int = 0,
                min_cluster_area: int = 0,
                m2f_threshold: float = 0.35,
                m2f_scales: tuple = (640, 1024),
                flip_tta: bool = False,
                tau_abs: float = 0.30,
                n_lines: int | None = None) -> dict | None:
    try:
        gt_trails, (w, h), n_pts = load_gt_from_label(ann_path)
    except Exception as e:
        return {"image_id": image_id, "error": f"gt_load:{e}", "success": False}
    t0 = time.time()
    try:
        r = run_pipeline(str(img_path), m2f_min_fg=m2f_min_fg,
                         m2f_ckpt=m2f_ckpt,
                         n_lines=n_lines,
                         separator_mode=separator_mode,
                         morph_close_kernel=morph_close_kernel,
                         min_cluster_area=min_cluster_area,
                         m2f_threshold=m2f_threshold,
                         m2f_scales=tuple(m2f_scales),
                         flip_tta=flip_tta,
                         tau_abs=tau_abs)
    except Exception as e:
        r = PipelineResult(image_path=str(img_path), ok=False,
                           used_fallback=False, error=str(e))
    elapsed = time.time() - t0
    payload = pipeline_result_to_json(r, cell_id=cell_id, image_id=image_id)
    pred_trails = [np.array(t, dtype=float)
                   for panel in payload["panels"]
                   for t in panel["trails"]]
    m = evaluate_cell(
        pred_trails, gt_trails,
        image_meta={"img_w": w, "img_h": h},
        image_id=image_id, cell_id=cell_id,
        success=bool(r.ok), error=r.error,
    )
    row = asdict(m)
    row["used_fallback"] = bool(r.used_fallback)
    row["m2f_fg_frac"] = float(r.m2f_fg_frac)
    row["n_clusters"] = int(r.n_clusters)
    row["n_gt_lines"] = len(gt_trails)
    row["elapsed_s"] = elapsed
    row["stratum"] = stratum_for(len(gt_trails), n_pts)
    # Cell-5 abstention bookkeeping (one image, summary stats).
    meta = payload.get("meta", {})
    row["n_curves_emitted"] = int(meta.get("n_curves_emitted", 0))
    row["n_curves_abstain"] = int(meta.get("n_curves_abstain", 0))
    row["tau_abs"] = float(meta.get("tau_abs", tau_abs))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="ours_kmeans_m2f_label")
    ap.add_argument("--out", default="experiments/results")
    ap.add_argument("--m2f_min_fg", type=float, default=0.003)
    ap.add_argument("--m2f_ckpt", default=None,
                    help="path to alternative Mask2Former checkpoint")
    ap.add_argument("--separator_mode", default="v1")
    ap.add_argument("--morph_close_kernel", type=int, default=0)
    ap.add_argument("--min_cluster_area", type=int, default=0)
    ap.add_argument("--m2f_threshold", type=float, default=0.35)
    ap.add_argument("--m2f_scales", default="640,1024")
    ap.add_argument("--flip_tta", action="store_true")
    ap.add_argument("--tau_abs", type=float, default=0.30)
    ap.add_argument("--k_csv", default=None,
                    help="path to a cached K predictions CSV; column "
                         "schema: image_id, <backend>_k. Falls back to "
                         "silhouette heuristic when missing.")
    ap.add_argument("--k_backend", default="gemini-3.1-pro-preview",
                    help="column prefix to read K from in --k_csv")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    k_lookup: dict[str, int] = {}
    if args.k_csv and Path(args.k_csv).exists():
        kdf = pd.read_csv(args.k_csv)
        col = f"{args.k_backend}_k"
        if col not in kdf.columns:
            raise ValueError(f"column {col} not in {args.k_csv}; "
                              f"available: {list(kdf.columns)}")
        for _, row in kdf.iterrows():
            try:
                v = int(row[col])
                if v > 0:
                    k_lookup[str(row["image_id"])] = v
            except Exception:
                pass
        print(f"loaded K cache for {len(k_lookup)} images "
              f"from {args.k_csv} ({col})")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / f"{args.cell}.csv"

    anns = sorted(LABEL_DIR.glob("*_annotation.json"))
    if args.limit:
        anns = anns[:args.limit]

    rows = []
    t0 = time.time()
    for i, ann in enumerate(anns):
        image_id = ann.stem.replace("_annotation", "")
        img = LABEL_DIR / f"{image_id}.jpg"
        if not img.exists():
            continue
        r = score_image(image_id, img, ann,
                        m2f_min_fg=args.m2f_min_fg, cell_id=args.cell,
                        m2f_ckpt=args.m2f_ckpt,
                        separator_mode=args.separator_mode,
                        morph_close_kernel=args.morph_close_kernel,
                        min_cluster_area=args.min_cluster_area,
                        m2f_threshold=args.m2f_threshold,
                        m2f_scales=tuple(int(s) for s in args.m2f_scales.split(",")),
                        flip_tta=args.flip_tta,
                        tau_abs=args.tau_abs,
                        n_lines=k_lookup.get(image_id))
        if r is None:
            continue
        rows.append(r)
        print(f"  [{i+1:>3}/{len(anns)}] {image_id[:24]}... "
              f"dtw={r.get('dtw_delta', float('nan')):.2f} "
              f"ccr={r.get('ccr_at_tau', float('nan')):.3f} "
              f"f1={r.get('instance_f1', float('nan')):.3f} "
              f"fb={r.get('used_fallback', '?')} "
              f"K={r.get('n_clusters', '?')}")
    dt = time.time() - t0
    print(f"  total: {dt:.1f}s for {len(rows)} images")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    print("\n=== per-stratum ===")
    grp = df.groupby("stratum").agg(
        n=("image_id", "count"),
        dtw=("dtw_delta", "mean"),
        ccr=("ccr_at_tau", "mean"),
        f1=("instance_f1", "mean"),
        pq=("curve_pq", "mean"),
    ).round(3)
    print(grp.to_string())
    print(f"\nALL: n={len(df)} "
          f"dtw={df['dtw_delta'].mean():.3f} "
          f"ccr={df['ccr_at_tau'].mean():.3f} "
          f"f1={df['instance_f1'].mean():.3f} "
          f"pq={df['curve_pq'].mean():.3f}")
    print(f"\nsaved: {csv_path}")


if __name__ == "__main__":
    main()
