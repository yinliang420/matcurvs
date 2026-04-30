"""Run pipeline.run_pipeline() over each image in label_qa.json and cache the
serialized curves to experiments/results/vlm_downstream/curves_cache/<id>.json.

K_pred is loaded from experiments/results/iter/k_full_label_gemini.csv
(Gemini K-predictor results from existing K-prediction run).
"""
from __future__ import annotations

import csv
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(KMEANS_ROOT))

from pipeline import run_pipeline  # noqa: E402
from experiments.iter.curve_serializer import serialize_curves, find_longest_cluster_idx  # noqa: E402

LABEL_DIR = KMEANS_ROOT / "label"
QA_PATH = KMEANS_ROOT / "experiments" / "results" / "vlm_downstream" / "label_qa.json"
CACHE_DIR = KMEANS_ROOT / "experiments" / "results" / "vlm_downstream" / "curves_cache"
K_CSV = KMEANS_ROOT / "experiments" / "results" / "iter" / "k_full_label_gemini.csv"
ERROR_LOG = KMEANS_ROOT / "experiments" / "results" / "vlm_downstream" / "extract_errors.json"


def load_k_pred() -> dict[str, int]:
    """Load Gemini K-predictions; fall back to ground-truth K if missing."""
    out: dict[str, int] = {}
    if not K_CSV.exists():
        return out
    with K_CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            k_raw = row.get("gemini-3.1-pro-preview_k") or ""
            try:
                k = int(k_raw)
                out[row["image_id"]] = k
            except (ValueError, TypeError):
                continue
    return out


def gt_k(image_id: str) -> int:
    """Fallback to GT K from annotation."""
    p = LABEL_DIR / f"{image_id}_annotation.json"
    if not p.exists():
        return 1
    try:
        ann = json.loads(p.read_text())
        return max(1, len(ann.get("lines") or {}))
    except Exception:
        return 1


def extract_one(image_id: str, k_pred: int) -> tuple[str, dict | None, str | None]:
    """Returns (image_id, curve_json, error_msg)."""
    cache_path = CACHE_DIR / f"{image_id}.json"
    if cache_path.exists():
        try:
            return image_id, json.loads(cache_path.read_text()), None
        except Exception:
            pass

    img_path = LABEL_DIR / f"{image_id}.jpg"
    if not img_path.exists():
        return image_id, None, f"image_missing: {img_path}"
    try:
        pr = run_pipeline(str(img_path), n_lines=k_pred,
                          separator_mode="v4", morph_close_kernel=3,
                          min_cluster_area=200)
    except Exception as e:
        return image_id, None, f"pipeline_exception: {type(e).__name__}: {e}\n{traceback.format_exc()[:400]}"

    if not pr.ok:
        return image_id, None, f"pipeline_not_ok: {pr.error}"
    if not pr.cluster_masks:
        return image_id, None, "no_cluster_masks"

    # find longest by x-extent and put first in the serialized list
    longest_idx = find_longest_cluster_idx(pr.cluster_masks)
    masks = list(pr.cluster_masks)
    if longest_idx != 0:
        masks[0], masks[longest_idx] = masks[longest_idx], masks[0]

    serialized = serialize_curves(image_id, masks, n_points=50)
    if serialized is None:
        return image_id, None, "serialize_failed (axis mapping or no curves)"

    # mark the first as longest
    if serialized.get("curves"):
        serialized["curves"][0]["label"] = "longest"

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(serialized, indent=2))
    return image_id, serialized, None


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not QA_PATH.exists():
        print(f"QA file missing: {QA_PATH}. Run gen_label_qa.py first.")
        return

    qa = json.loads(QA_PATH.read_text())
    image_ids = sorted(qa.keys())
    if args.limit:
        image_ids = image_ids[:args.limit]
    k_map = load_k_pred()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    errors: dict[str, str] = {}
    ok = 0; fail = 0; cached = 0

    def task(iid: str):
        k = k_map.get(iid)
        if k is None or k <= 0:
            k = gt_k(iid)
        return extract_one(iid, k)

    print(f"Extracting curves for {len(image_ids)} images with {args.workers} workers...")
    # Note: pipeline involves heavy GPU/CPU and can be unsafe to multi-thread (YOLO, m2f).
    # Use process or single-thread for safety; here we use threads but workers=2-4 is safe.
    if args.workers <= 1:
        for iid in image_ids:
            _, j, err = task(iid)
            if j is not None:
                if (CACHE_DIR / f"{iid}.json").exists():
                    cached += 1
                ok += 1
            else:
                fail += 1
                errors[iid] = err or "unknown"
                print(f"[FAIL] {iid}: {err}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(task, iid): iid for iid in image_ids}
            for fut in as_completed(futs):
                iid = futs[fut]
                try:
                    _, j, err = fut.result()
                except Exception as e:
                    j, err = None, f"future_exception: {e}"
                if j is not None:
                    ok += 1
                else:
                    fail += 1
                    errors[iid] = err or "unknown"
                    print(f"[FAIL] {iid}: {err}")
                if (ok + fail) % 10 == 0:
                    print(f"  progress: {ok+fail}/{len(image_ids)} (ok={ok} fail={fail})")

    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    ERROR_LOG.write_text(json.dumps(errors, indent=2))
    print(f"\nDone. ok={ok} fail={fail} cached_dir={CACHE_DIR}")
    print(f"Errors logged to {ERROR_LOG}")


if __name__ == "__main__":
    main()
