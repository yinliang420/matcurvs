"""Extract VLM polylines for the 4 figure-strip showcase images.

For each (image_id, backend) we save:
  experiments/results/vlm_showcase/<backend>/<image_id>.json  containing
  {pred_real: [[(x,y), ...], ...], pred_pixel: [[(x,y), ...], ...]}

so build_paper_figures.py can render an appendix-comparison strip without
re-running the VLM.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[2]))
MATDEPLOT_ROOT = Path(os.environ.get("MATDEPLOT_ROOT", "/path/to/MatDeplot/mdp_v2"))
sys.path.insert(0, str(KMEANS_ROOT))
sys.path.insert(0, str(MATDEPLOT_ROOT))


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ[k.strip()] = v.strip()


_load_env(KMEANS_ROOT / ".env")

from experiments.run_vlm_label_eval import (  # noqa: E402
    load_gt, vlm_real_to_pixel,
)
from matdeplot.baselines.vlm_baseline import (  # noqa: E402
    DEFAULT_PROMPT, VLMResult, call_vlm,
)
from experiments.run_vlm_label_eval import working_backends  # noqa: E402

LABEL_DIR = KMEANS_ROOT / "label"
OUT_DIR = KMEANS_ROOT / "experiments" / "results" / "vlm_showcase"


SHOWCASE_IDS = [
    "217a5eb3c9f7c97b5738cb13",                                          # L1 (XPS)
    "a002b8da910d1e3c2a6ddc7f",                                          # L2 (XRD)
    "510aa6c30e12fcc5830a9a0d",                                          # L3 (Raman)
    "00e558db0245b40006acf7a1",                                          # L4 (XRD stack)
]


def _resolve(prefix: str) -> str | None:
    """Resolve a prefix to a full image_id by listing label/ files."""
    cands = list(LABEL_DIR.glob(f"{prefix}*_annotation.json"))
    if cands:
        return cands[0].stem.replace("_annotation", "")
    return None


def _extract_one(image_id: str, backend) -> dict:
    img_path = LABEL_DIR / f"{image_id}.jpg"
    ann_path = LABEL_DIR / f"{image_id}_annotation.json"
    ann, gt_trails, (w, h), n_pts = load_gt(ann_path)

    t0 = time.time()
    r: VLMResult = call_vlm(str(img_path), image_id=image_id,
                             backend=backend, prompt=DEFAULT_PROMPT)
    elapsed = time.time() - t0

    pred_real = [[(float(x), float(y)) for (x, y) in t.points]
                 for t in r.trails]
    if pred_real:
        pred_pixel_arrays = vlm_real_to_pixel(pred_real, ann, (w, h))
        pred_pixel = [arr.tolist() for arr in pred_pixel_arrays]
    else:
        pred_pixel = []

    return {
        "image_id": image_id,
        "backend": backend.name,
        "success": bool(r.success),
        "error": r.error or "",
        "n_predicted_trails": len(pred_real),
        "n_gt_trails": len(gt_trails),
        "image_size": [int(w), int(h)],
        "pred_real": pred_real,
        "pred_pixel": pred_pixel,
        "input_tokens": int(r.input_tokens or 0),
        "output_tokens": int(r.output_tokens or 0),
        "latency_s": round(elapsed, 2),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # resolve showcase IDs (allow short prefixes)
    resolved = []
    for sid in SHOWCASE_IDS:
        full = sid if (LABEL_DIR / f"{sid}.jpg").exists() else _resolve(sid)
        if full is None:
            print(f"!! could not resolve image_id: {sid}")
            continue
        resolved.append(full)
    print(f"resolved {len(resolved)} showcase image_ids")
    for r in resolved:
        print(f"  {r}")

    backends = working_backends()
    print(f"\nbackends: {[b.name for b in backends]}\n")

    for be in backends:
        be_dir = OUT_DIR / be.name
        be_dir.mkdir(parents=True, exist_ok=True)
        for image_id in resolved:
            out_path = be_dir / f"{image_id}.json"
            if out_path.exists():
                print(f"[cached] {be.name} / {image_id[:12]}")
                continue
            try:
                row = _extract_one(image_id, be)
                out_path.write_text(json.dumps(row, indent=2, ensure_ascii=False))
                print(f"[ok    ] {be.name:<28} {image_id[:12]} "
                      f"n_trails={row['n_predicted_trails']} "
                      f"latency={row['latency_s']:.1f}s success={row['success']}")
            except Exception as e:
                print(f"[fail  ] {be.name:<28} {image_id[:12]} {e}")


if __name__ == "__main__":
    main()
