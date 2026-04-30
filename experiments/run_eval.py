"""批量跑 kmeans 管道 → MatDeplot 评测器, 输出每 cell 一个 CSV.

用法::
    python experiments/run_eval.py --cell ours_kmeans_m2f --out experiments/results
    python experiments/run_eval.py --cell ours_kmeans_m2f_fallback --m2f_min_fg 0.99
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import os

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[1]))
MATDEPLOT_ROOT = Path(os.environ.get("MATDEPLOT_ROOT", "/path/to/MatDeplot/mdp_v2"))
sys.path.insert(0, str(KMEANS_ROOT))
sys.path.insert(0, str(MATDEPLOT_ROOT))

from matdeplot.eval.unified_metrics import (  # noqa: E402
    _trails_from_pipeline_json, evaluate_cell, load_gt_trails,
)
from experiments.kmeans_matdeplot_adapter import (  # noqa: E402
    RunSpec, run_pipeline_on_test,
)


def score_run(pred_dir: Path, cell_id: str) -> pd.DataFrame:
    gt = pd.read_parquet(MATDEPLOT_ROOT / "data/gt_unified.parquet")
    rows = []
    for jp in sorted(pred_dir.glob("*.json")):
        payload = json.loads(jp.read_text())
        image_id = payload["image_id"]
        pred_trails = _trails_from_pipeline_json(payload)
        gt_trails, (w, h) = load_gt_trails(image_id, gt, coord="pixel")
        meta = {"img_w": w, "img_h": h}
        m = evaluate_cell(
            pred_trails, gt_trails, image_meta=meta,
            image_id=image_id, cell_id=cell_id,
            success=bool(payload.get("success", True)),
            error=payload.get("error"),
        )
        row = asdict(m)
        row["used_fallback"] = payload.get("meta", {}).get("used_fallback", False)
        row["m2f_fg_frac"] = payload.get("meta", {}).get("m2f_fg_frac", None)
        row["elapsed_s"] = payload.get("meta", {}).get("elapsed_s", None)
        rows.append(row)
    return pd.DataFrame(rows)


def add_strata(df: pd.DataFrame) -> pd.DataFrame:
    strata = json.loads((MATDEPLOT_ROOT / "results/image_strata.json").read_text())
    images = strata.get("images", [])
    if isinstance(images, dict):
        img2str = {i: stratum for stratum, ids in images.items() for i in ids}
    else:
        img2str = {e["image_id"]: e["stratum"] for e in images}
    df["stratum"] = df["image_id"].map(img2str).fillna("UNKNOWN")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", default="experiments/results")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--m2f_min_fg", type=float, default=0.003)
    ap.add_argument("--separator_mode", default="v1",
                    help="v1 legacy / v4 = morph-close + min-area + ΔE dilation")
    ap.add_argument("--morph_close_kernel", type=int, default=0)
    ap.add_argument("--min_cluster_area", type=int, default=0)
    ap.add_argument("--m2f_threshold", type=float, default=0.35)
    ap.add_argument("--m2f_scales", default="640,1024",
                    help="comma-separated scales; e.g. '512,768,1024'")
    ap.add_argument("--flip_tta", action="store_true")
    ap.add_argument("--m2f_ckpt", default=None)
    ap.add_argument("--skip_infer", action="store_true",
                    help="reuse cached pred JSONs")
    args = ap.parse_args()

    out_root = Path(args.out)
    pred_dir = out_root / "pred" / args.cell
    csv_path = out_root / f"{args.cell}.csv"
    pred_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_infer:
        spec = RunSpec(
            cell_id=args.cell,
            m2f_min_fg=args.m2f_min_fg,
            separator_mode=args.separator_mode,
            morph_close_kernel=args.morph_close_kernel,
            min_cluster_area=args.min_cluster_area,
            m2f_threshold=args.m2f_threshold,
            m2f_scales=tuple(int(s) for s in args.m2f_scales.split(",")),
            flip_tta=args.flip_tta,
            m2f_ckpt=args.m2f_ckpt,
        )
        run_pipeline_on_test(pred_dir, spec, limit=args.limit)

    df = score_run(pred_dir, cell_id=args.cell)
    df = add_strata(df)
    df.to_csv(csv_path, index=False)

    # per-stratum summary
    print("\n=== per-stratum summary ===")
    grp = df.groupby("stratum").agg(
        n=("image_id", "count"),
        dtw_delta=("dtw_delta", "mean"),
        ccr_at_tau=("ccr_at_tau", "mean"),
        instance_f1=("instance_f1", "mean"),
        curve_pq=("curve_pq", "mean"),
    ).round(3)
    print(grp.to_string())
    print(f"\nALL: n={len(df)} "
          f"dtw={df['dtw_delta'].mean():.3f} "
          f"ccr={df['ccr_at_tau'].mean():.3f} "
          f"inst_f1={df['instance_f1'].mean():.3f} "
          f"pq={df['curve_pq'].mean():.3f}")
    print(f"\nsaved: {csv_path}")


if __name__ == "__main__":
    main()
