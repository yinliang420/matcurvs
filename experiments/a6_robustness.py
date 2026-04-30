"""A6 robustness: rotation ±5°, JPEG q=30, Gaussian noise σ=5.

Perturb the 36 MatDeplot-36 test images, run pipeline, evaluate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import os

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[1]))
MATDEPLOT_ROOT = Path(os.environ.get("MATDEPLOT_ROOT", "/path/to/MatDeplot/mdp_v2"))
sys.path.insert(0, str(KMEANS_ROOT))
sys.path.insert(0, str(MATDEPLOT_ROOT))

from pipeline import run_pipeline, PipelineResult  # noqa: E402
from experiments.kmeans_matdeplot_adapter import (  # noqa: E402
    RunSpec, pipeline_result_to_json, get_test_image_paths,
)


def perturb_rotate(img: np.ndarray, degrees: float = 5.0) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def perturb_jpeg(img: np.ndarray, quality: int = 30) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def perturb_noise(img: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


PERTURBS = {
    "rot5": lambda im: perturb_rotate(im, 5.0),
    "jpeg30": lambda im: perturb_jpeg(im, 30),
    "noise": lambda im: perturb_noise(im, 5.0),
}


def run_one(cell_id: str, perturb_fn, out_dir: Path,
            tmp_dir: Path, m2f_min_fg: float = 0.003,
            limit: int = 0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    paths = get_test_image_paths()
    if limit:
        paths = paths[:limit]
    t0 = time.time()
    for i, (image_id, orig_path) in enumerate(paths):
        img = cv2.imread(orig_path)
        if img is None:
            continue
        img_p = perturb_fn(img)
        tmp_path = tmp_dir / f"{image_id.replace('/', '__')}.jpg"
        cv2.imwrite(str(tmp_path), img_p)
        t_img = time.time()
        try:
            r = run_pipeline(str(tmp_path), m2f_min_fg=m2f_min_fg)
        except Exception as e:
            r = PipelineResult(image_path=str(tmp_path), ok=False,
                               used_fallback=False, error=str(e))
        elapsed = time.time() - t_img
        payload = pipeline_result_to_json(r, cell_id=cell_id, image_id=image_id)
        payload["meta"]["elapsed_s"] = elapsed
        safe_id = image_id.replace("/", "__")
        (out_dir / f"{safe_id}__{cell_id}.json").write_text(
            json.dumps(payload, indent=2))
        print(f"  [{i+1:>3}/{len(paths)}] {cell_id} ok={r.ok} fb={r.used_fallback} "
              f"K={r.n_clusters} t={elapsed:.2f}s")
    print(f"  total: {time.time()-t0:.1f}s for {len(paths)} images")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=list(PERTURBS.keys()) + ["all"],
                    default="all")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = KMEANS_ROOT / "experiments" / "results"
    tmp_root = KMEANS_ROOT / "experiments" / "tmp_perturbed"
    keys = list(PERTURBS) if args.which == "all" else [args.which]
    for k in keys:
        cell = f"ours_robust_{k}"
        pred_dir = root / "pred" / cell
        tmp_dir = tmp_root / k
        print(f"\n### {cell}")
        run_one(cell, PERTURBS[k], pred_dir, tmp_dir, limit=args.limit)


if __name__ == "__main__":
    main()
