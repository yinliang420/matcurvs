"""Cache Qwen-VL curve metadata (K + colour + anchors + x-range) for every
image in label/.  Output schema::

    image_id, n_gt_lines, qwen_k, qwen_lat, qwen_err, qwen_curves_json

``qwen_curves_json`` stores the full ``CurveDesc`` list (color/style/
x_lo/x_hi/anchors) as a JSON string for downstream merge logic.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import os

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(KMEANS_ROOT))

from experiments.iter.llm_curves_predictor import predict_curves  # noqa: E402

LABEL_DIR = KMEANS_ROOT / "label"


def _gt_n_lines(ann_path: Path) -> int:
    try:
        d = json.loads(ann_path.read_text())
        return len(d.get("lines", {}))
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qwen-vl-max",
                    choices=["qwen-vl-max", "qwen3-vl-plus"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out",
                    default=str(KMEANS_ROOT /
                                "experiments/results/iter/k_full_label_qwen_curves.csv"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="skip image_ids already in output csv")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    anns = sorted(LABEL_DIR.glob("*_annotation.json"))
    if args.limit:
        anns = anns[: args.limit]

    cached_ids: set[str] = set()
    if args.resume and out_path.exists():
        try:
            cached_ids = set(pd.read_csv(out_path).image_id.astype(str).tolist())
            print(f"resume: {len(cached_ids)} ids already cached")
        except Exception:
            pass

    todo = []
    for ann in anns:
        image_id = ann.stem.replace("_annotation", "")
        if image_id in cached_ids:
            continue
        img = LABEL_DIR / f"{image_id}.jpg"
        if not img.exists():
            continue
        todo.append((image_id, ann, img))

    print(f"queuing {len(todo)} images on backend={args.backend} "
          f"workers={args.workers}")

    rows: list[dict] = []
    if cached_ids and out_path.exists():
        rows.extend(pd.read_csv(out_path).to_dict(orient="records"))

    def _one(image_id, ann, img):
        gt_n = _gt_n_lines(ann)
        cp = predict_curves(img, backend=args.backend)
        return {
            "image_id": image_id,
            "n_gt_lines": gt_n,
            f"{args.backend}_k": cp.k_pred,
            f"{args.backend}_lat": cp.latency_s,
            f"{args.backend}_err": cp.error,
            f"{args.backend}_curves_json":
                json.dumps([asdict(c) for c in cp.curves]),
        }

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one, *t): t for t in todo}
        for i, fut in enumerate(as_completed(futs)):
            rec = fut.result()
            rows.append(rec)
            if (i + 1) % 5 == 0 or (i + 1) == len(todo):
                pd.DataFrame(rows).to_csv(out_path, index=False)
                print(f"  [{i+1:>3}/{len(todo)}] cached → {out_path.name}")

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nfinal: {len(rows)} rows → {out_path}")


if __name__ == "__main__":
    main()
