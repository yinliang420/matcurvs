"""Run google/deplot on the 4 figure-strip showcase images.

DePlot is a Pix2Struct-based chart-to-table model from Google. We feed it the
chart image, ask for a table, then parse the rows into per-series (x, y) lists
and re-anchor them in pixel space the same way `vlm_real_to_pixel` does for
the hosted VLMs (so the polylines render comparably in the appendix figure).

Output schema mirrors `experiments/iter/vlm_extract_showcase.py`:
  experiments/results/vlm_showcase/deplot/<image_id>.json
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
import os

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[2]))
MATDEPLOT_ROOT = Path(os.environ.get("MATDEPLOT_ROOT", "/path/to/MatDeplot/mdp_v2"))
sys.path.insert(0, str(KMEANS_ROOT))
sys.path.insert(0, str(MATDEPLOT_ROOT))

from experiments.run_vlm_label_eval import load_gt, vlm_real_to_pixel  # noqa: E402

LABEL_DIR = KMEANS_ROOT / "label"
OUT_DIR = KMEANS_ROOT / "experiments" / "results" / "vlm_showcase" / "deplot"

SHOWCASE_IDS = [
    "217a5eb3c9f7c97b5738cb13",
    "a002b8da910d1e3c2a6ddc7f",
    "510aa6c30e12fcc5830a9a0d",
    "00e558db0245b40006acf7a1",
]


def _resolve(prefix: str) -> str | None:
    cands = list(LABEL_DIR.glob(f"{prefix}*_annotation.json"))
    return cands[0].stem.replace("_annotation", "") if cands else None


_MODEL = None
_PROC = None


def _load_model():
    global _MODEL, _PROC
    if _MODEL is None:
        from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"loading google/deplot on {device}…")
        _PROC = Pix2StructProcessor.from_pretrained("google/deplot")
        _MODEL = Pix2StructForConditionalGeneration.from_pretrained("google/deplot").to(device).eval()
    return _MODEL, _PROC


def _parse_deplot_table(table_text: str) -> List[List[Tuple[float, float]]]:
    """DePlot returns a TSV-like text:
        TITLE | <title>
        <header_x> | <series1> | <series2> | ...
        <x1> | <y11> | <y12> | ...
        <x2> | <y21> | <y22> | ...

    Return one (x, y) list per data series.
    """
    # DePlot encodes newlines as the literal string "<0x0A>" rather than \n
    norm = table_text.replace("<0x0A>", "\n")
    lines = [l.strip() for l in norm.split("\n") if "|" in l]
    if not lines:
        return []
    # drop TITLE row if present
    rows = [l for l in lines if not l.upper().startswith("TITLE")]
    if not rows:
        return []
    header = [c.strip() for c in rows[0].split("|")]
    if len(header) < 2:
        return []
    n_series = len(header) - 1
    series: List[List[Tuple[float, float]]] = [[] for _ in range(n_series)]
    num_rx = re.compile(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?")
    for row in rows[1:]:
        cells = [c.strip() for c in row.split("|")]
        if len(cells) < 2:
            continue
        # x value
        m = num_rx.search(cells[0])
        if not m:
            continue
        try:
            x = float(m.group(0))
        except ValueError:
            continue
        for j, cell in enumerate(cells[1:1 + n_series]):
            m = num_rx.search(cell)
            if not m:
                continue
            try:
                y = float(m.group(0))
            except ValueError:
                continue
            series[j].append((x, y))
    return [s for s in series if len(s) >= 2]


def _extract_one(image_id: str) -> dict:
    img_path = LABEL_DIR / f"{image_id}.jpg"
    ann_path = LABEL_DIR / f"{image_id}_annotation.json"
    ann, gt_trails, (w, h), _ = load_gt(ann_path)

    model, proc = _load_model()
    img = Image.open(img_path).convert("RGB")
    inputs = proc(images=img,
                  text="Generate underlying data table of the figure below:",
                  return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=512)
    table_text = proc.decode(out[0], skip_special_tokens=True)
    elapsed = time.time() - t0
    print(f"  raw deplot output ({len(table_text)} chars): "
          f"{table_text[:120]}{'…' if len(table_text) > 120 else ''}")

    pred_real = _parse_deplot_table(table_text)
    if pred_real:
        try:
            pred_pixel_arrays = vlm_real_to_pixel(pred_real, ann, (w, h))
            pred_pixel = [arr.tolist() for arr in pred_pixel_arrays]
        except Exception as e:
            print(f"  pixel-mapping failed: {e}")
            pred_pixel = []
    else:
        pred_pixel = []

    return {
        "image_id": image_id,
        "backend": "deplot",
        "success": bool(pred_real),
        "n_predicted_trails": len(pred_real),
        "n_gt_trails": len(gt_trails),
        "image_size": [int(w), int(h)],
        "raw_table": table_text,
        "pred_real": pred_real,
        "pred_pixel": pred_pixel,
        "latency_s": round(elapsed, 2),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    resolved = [r for r in (_resolve(s) or s for s in SHOWCASE_IDS)
                if (LABEL_DIR / f"{r}.jpg").exists()]
    print(f"resolved {len(resolved)} showcase images")

    for image_id in resolved:
        out_path = OUT_DIR / f"{image_id}.json"
        if out_path.exists():
            print(f"[cached] {image_id[:12]}")
            continue
        try:
            row = _extract_one(image_id)
            out_path.write_text(json.dumps(row, indent=2, ensure_ascii=False))
            print(f"[ok    ] {image_id[:12]} n_trails={row['n_predicted_trails']} "
                  f"latency={row['latency_s']:.1f}s")
        except Exception as e:
            print(f"[fail  ] {image_id[:12]}: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
