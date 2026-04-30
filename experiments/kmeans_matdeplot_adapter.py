"""把 kmeans + Mask2Former 管道的输出翻译成 MatDeplot 统一评测器能消化的 JSON。

评测器 schema (from ``matdeplot.eval.unified_metrics._trails_from_pipeline_json``):

    {
      "panels": [
        {"trails": [[[x, y], ...], [[x, y], ...]]}
      ]
    }

trails 为像素坐标 (与 gt_unified 的 ``x_pixel``/``y_pixel`` 对齐). 由于我们的
``PipelineResult.cluster_masks`` 只给出每条线的 bool mask, 需要把 mask
skeletonize → 有序多段线.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[1]))
MATDEPLOT_ROOT = Path(os.environ.get("MATDEPLOT_ROOT", "/path/to/MatDeplot/mdp_v2"))
sys.path.insert(0, str(KMEANS_ROOT))
sys.path.insert(0, str(MATDEPLOT_ROOT))

from pipeline import run_pipeline, PipelineResult  # noqa: E402


def mask_to_polyline(mask: np.ndarray, min_len: int = 10,
                     sample_every: int = 2) -> np.ndarray:
    """把 2D bool mask 变成按 x 排序的像素多段线 (N, 2).

    策略:
      1. 骨架化 (skeletonize) 得到单像素宽线.
      2. 取所有骨架像素, 按 x 排序, 对重复 x 合并 y 为均值.
      3. 下采样 (sample_every).
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if mask.sum() < min_len:
        return np.zeros((0, 2), dtype=float)
    skel = skeletonize(mask)
    ys, xs = np.where(skel)
    if len(xs) < min_len:
        # 退化: 直接用所有前景像素
        ys, xs = np.where(mask)
        if len(xs) < min_len:
            return np.zeros((0, 2), dtype=float)
    # 按 x 分组取 y 均值
    df = pd.DataFrame({"x": xs, "y": ys}).groupby("x")["y"].mean().reset_index()
    df = df.sort_values("x").reset_index(drop=True)
    if sample_every > 1 and len(df) > 4 * sample_every:
        df = df.iloc[::sample_every].reset_index(drop=True)
    return df[["x", "y"]].to_numpy(dtype=float)


def pipeline_result_to_json(result: PipelineResult,
                            cell_id: str,
                            image_id: str) -> dict:
    panels = []
    trails = []
    curve_meta = []  # one record per emitted (kept) trail
    confs = list(result.curve_confidences) if result.curve_confidences else []
    abst = list(result.curve_abstain) if result.curve_abstain else []
    for idx, m in enumerate(result.cluster_masks):
        arr = mask_to_polyline(m)
        if arr.shape[0] >= 10:
            trails.append(arr.tolist())
            curve_meta.append({
                "trail_index": len(trails) - 1,
                "cluster_index": idx,
                "confidence": float(confs[idx]) if idx < len(confs) else None,
                "abstain": bool(abst[idx]) if idx < len(abst) else False,
            })
    panels.append({"trails": trails, "curve_meta": curve_meta})
    n_emitted = len(trails)
    n_abstain = sum(1 for c in curve_meta if c.get("abstain"))
    return {
        "image_id": image_id,
        "cell_id": cell_id,
        "panels": panels,
        "success": bool(result.ok),
        "error": result.error,
        "meta": {
            "used_fallback": bool(result.used_fallback),
            "m2f_fg_frac": float(result.m2f_fg_frac),
            "n_clusters": int(result.n_clusters),
            "n_legend_boxes": len(result.legend_boxes),
            "axis_detected": result.axis_lines is not None,
            "tau_abs": float(result.tau_abs),
            "n_curves_emitted": int(n_emitted),
            "n_curves_abstain": int(n_abstain),
        },
    }


@dataclass
class RunSpec:
    cell_id: str
    n_lines: Optional[int] = None
    m2f_min_fg: float = 0.003
    # iter12+ knobs that the headline pipeline (Algorithm 1) actually uses.
    # Defaults retained as the legacy v1 baseline so existing scripts that
    # construct RunSpec(cell_id=...) keep their semantics; the v4+TTA recipe
    # is opt-in via these flags.
    separator_mode: str = "v1"
    morph_close_kernel: int = 0
    min_cluster_area: int = 0
    m2f_threshold: float = 0.35
    m2f_scales: tuple = (640, 1024)
    flip_tta: bool = False
    m2f_ckpt: Optional[str] = None


def get_test_image_paths() -> list[tuple[str, str]]:
    """返回 [(image_id, image_path), ...] for MatDeplot-36 test."""
    splits = json.loads((MATDEPLOT_ROOT / "data/splits.json").read_text())
    gt = pd.read_parquet(MATDEPLOT_ROOT / "data/gt_unified.parquet")
    mf = pd.read_csv(MATDEPLOT_ROOT / "data/raw_manifest.csv")
    test_papers = set(splits["test_paper_ids"])
    gt_ids = set(gt["image_id"].unique())
    out = []
    for _, row in mf.iterrows():
        if row["paper_id"] in test_papers and row["image_id"] in gt_ids:
            out.append((row["image_id"], row["image_path"]))
    return sorted(out)


def run_pipeline_on_test(out_dir: Path, spec: RunSpec,
                         limit: int = 0) -> list[dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = get_test_image_paths()
    if limit > 0:
        paths = paths[:limit]
    results = []
    t0 = time.time()
    for i, (image_id, image_path) in enumerate(paths):
        t_img = time.time()
        try:
            r = run_pipeline(
                image_path,
                n_lines=spec.n_lines,
                m2f_min_fg=spec.m2f_min_fg,
                m2f_ckpt=spec.m2f_ckpt,
                separator_mode=spec.separator_mode,
                morph_close_kernel=spec.morph_close_kernel,
                min_cluster_area=spec.min_cluster_area,
                m2f_threshold=spec.m2f_threshold,
                m2f_scales=tuple(spec.m2f_scales),
                flip_tta=spec.flip_tta,
            )
        except Exception as e:
            r = PipelineResult(image_path=image_path, ok=False,
                               used_fallback=False, error=str(e))
        elapsed = time.time() - t_img
        payload = pipeline_result_to_json(r, cell_id=spec.cell_id,
                                          image_id=image_id)
        payload["meta"]["elapsed_s"] = elapsed
        # 文件名只用 image_id basename 以避免 '/' 路径问题
        safe_id = image_id.replace("/", "__")
        (out_dir / f"{safe_id}__{spec.cell_id}.json").write_text(
            json.dumps(payload, indent=2))
        results.append(payload)
        print(f"  [{i+1:>3}/{len(paths)}] {image_id.split('/')[-1][:40]}... "
              f"ok={r.ok} fb={r.used_fallback} K={r.n_clusters} t={elapsed:.2f}s")
    dt = time.time() - t0
    print(f"  total: {dt:.1f}s for {len(paths)} images")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="ours_kmeans_m2f")
    ap.add_argument("--out", default="experiments/results/pred")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--m2f_min_fg", type=float, default=0.003)
    ap.add_argument("--separator_mode", default="v1")
    ap.add_argument("--morph_close_kernel", type=int, default=0)
    ap.add_argument("--min_cluster_area", type=int, default=0)
    ap.add_argument("--m2f_threshold", type=float, default=0.35)
    ap.add_argument("--m2f_scales", default="640,1024")
    ap.add_argument("--flip_tta", action="store_true")
    ap.add_argument("--m2f_ckpt", default=None)
    args = ap.parse_args()
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
    run_pipeline_on_test(Path(args.out) / args.cell, spec, limit=args.limit)
