"""Run the Cell-6 VLM-as-Judge sweep on a full prediction CSV.

Pipeline:
    1. Load ``experiments/results/<cell>.csv`` to get the list of
       image ids that were scored.
    2. For each image, materialise the v4+TTA polylines (re-run
       ``pipeline.run_pipeline`` once and cache to JSON under
       ``experiments/results/pred_polylines/<cell>/<image_id>.json``).
       The pipeline is serial because Mask2Former + multi-scale TTA
       saturates the local GPU; if a cache file exists we skip.
    3. Dispatch ``len(judges) * len(images)`` judge calls to a
       ThreadPoolExecutor (workers configurable; the bottleneck is
       network latency, not CPU).  Each verdict is appended to the
       output CSV ``experiments/results/iter/judge_<cell>.csv`` as
       soon as it returns, so the script is resumable.

Output schema:
    image_id, judge, k_pred, k_seen, image_pass, per_curve_pass_csv,
    comment, latency_s, error, raw

CLI example:
    python experiments/iter/run_vlm_judge_sweep.py \\
       --cell ours_v4tta_label \\
       --judges qwen-vl-max,gpt-5.4,gemini-3.1-pro-preview,claude-opus-4-6 \\
       --workers 4
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import os

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(KMEANS_ROOT))

from pipeline import run_pipeline                                          # noqa: E402
from experiments.kmeans_matdeplot_adapter import mask_to_polyline          # noqa: E402
from experiments.iter.vlm_reviewer import review, BACKENDS, JudgeVerdict   # noqa: E402

LABEL_DIR = KMEANS_ROOT / "label"
RESULTS_DIR = KMEANS_ROOT / "experiments" / "results"
ITER_DIR = RESULTS_DIR / "iter"
PRED_CACHE_ROOT = RESULTS_DIR / "pred_polylines"


def _polyline_cache_dir(cell: str) -> Path:
    d = PRED_CACHE_ROOT / cell
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ensure_polylines(image_id: str, cell: str,
                      pipeline_kwargs: dict,
                      n_lines: int | None = None) -> list:
    """Return list of polylines (each (N, 2) np.ndarray) for one image,
    materialising and caching them if not already on disk."""
    cache_dir = _polyline_cache_dir(cell)
    cache_path = cache_dir / f"{image_id}.json"
    if cache_path.exists():
        try:
            d = json.loads(cache_path.read_text())
            return [np.asarray(p, dtype=np.int32) for p in d["polylines"]
                     if p is not None and len(p) >= 10]
        except Exception:
            pass
    img_path = LABEL_DIR / f"{image_id}.jpg"
    if not img_path.exists():
        return []
    kwargs = dict(pipeline_kwargs)
    if n_lines is not None and n_lines > 0:
        kwargs["n_lines"] = int(n_lines)
    r = run_pipeline(str(img_path), **kwargs)
    polylines = []
    for m in r.cluster_masks:
        arr = mask_to_polyline(m)
        if arr.shape[0] >= 10:
            polylines.append(arr)
    cache_path.write_text(json.dumps({
        "image_id": image_id,
        "polylines": [p.astype(int).tolist() for p in polylines],
        "n_clusters": len(polylines),
        "used_fallback": bool(r.used_fallback),
        "k_lines_used": int(n_lines) if n_lines is not None else None,
    }))
    return polylines


def _verdict_to_row(v: JudgeVerdict) -> dict:
    return {
        "image_id": v.image_id,
        "judge": v.judge,
        "k_pred": int(v.k_pred),
        "k_seen": v.k_seen if v.k_seen is not None else "",
        "image_pass": v.image_pass if v.image_pass is not None else "",
        "per_curve_pass_csv": ",".join("1" if x else "0"
                                         for x in v.per_curve_pass),
        "comment": v.comment,
        "latency_s": round(float(v.latency_s), 2),
        "error": v.error or "",
        "raw_head": v.raw[:200].replace("\n", " "),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="ours_v4tta_label")
    ap.add_argument("--judges",
                    default="qwen-vl-max,gpt-5.4,"
                            "gemini-3.1-pro-preview,claude-opus-4-6",
                    help="comma-separated judge names")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0,
                    help="if >0, only run on the first N images "
                          "(after sort)")
    ap.add_argument("--resume", action="store_true",
                    help="skip (image, judge) pairs already in the "
                          "output CSV")
    ap.add_argument("--save_overlays", type=int, default=5,
                    help="save overlay JPEGs for this many images per "
                          "judge (debugging)")
    # match the v4+TTA pixel-mode recipe
    ap.add_argument("--separator_mode", default="v4")
    ap.add_argument("--morph_close_kernel", type=int, default=5)
    ap.add_argument("--min_cluster_area", type=int, default=200)
    ap.add_argument("--m2f_threshold", type=float, default=0.25)
    ap.add_argument("--m2f_scales", default="512,768,1024")
    ap.add_argument("--flip_tta", action="store_true", default=True)
    ap.add_argument("--k_csv", default=None,
                    help="optional cached K predictions CSV "
                          "(image_id, <backend>_k); pipeline uses K from "
                          "this cache instead of silhouette heuristic.")
    ap.add_argument("--k_backend", default="gemini-3.1-pro-preview",
                    help="column prefix to read K from in --k_csv")
    args = ap.parse_args()
    k_lookup: dict[str, int] = {}
    if args.k_csv and Path(args.k_csv).exists():
        kdf = pd.read_csv(args.k_csv)
        col = f"{args.k_backend}_k"
        if col not in kdf.columns:
            raise SystemExit(f"column {col} not in {args.k_csv}; "
                              f"available: {list(kdf.columns)}")
        for _, row in kdf.iterrows():
            try:
                v = int(row[col])
                if v > 0:
                    k_lookup[str(row["image_id"])] = v
            except Exception:
                pass
        print(f"loaded K cache for {len(k_lookup)} images "
              f"from {Path(args.k_csv).name} ({col})")

    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    for j in judges:
        if j not in BACKENDS:
            raise SystemExit(f"unknown judge: {j} (have {list(BACKENDS)})")

    pipeline_kwargs = dict(
        separator_mode=args.separator_mode,
        morph_close_kernel=args.morph_close_kernel,
        min_cluster_area=args.min_cluster_area,
        m2f_threshold=args.m2f_threshold,
        m2f_scales=tuple(int(s) for s in args.m2f_scales.split(",")),
        flip_tta=bool(args.flip_tta),
    )

    cell_csv = RESULTS_DIR / f"{args.cell}.csv"
    if not cell_csv.exists():
        raise SystemExit(f"missing {cell_csv}")
    df_cell = pd.read_csv(cell_csv)
    image_ids = sorted(df_cell.image_id.astype(str).tolist())
    if args.limit:
        image_ids = image_ids[: args.limit]
    print(f"sweep: {len(image_ids)} images × {len(judges)} judges = "
          f"{len(image_ids) * len(judges)} calls (workers={args.workers})")

    out_csv = ITER_DIR / f"judge_{args.cell}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    written: set[tuple[str, str]] = set()
    rows_existing: list[dict] = []
    if args.resume and out_csv.exists():
        try:
            prev = pd.read_csv(out_csv)
            for _, row in prev.iterrows():
                # only treat as "done" if the row has a non-empty image_pass
                # value; partial / errored rows get re-tried.
                ip = row.get("image_pass")
                err = row.get("error")
                err = "" if (isinstance(err, float) and np.isnan(err)) else str(err)
                if str(ip) in ("True", "False") and not err:
                    written.add((str(row.image_id), str(row.judge)))
                rows_existing.append(row.to_dict())
            print(f"resume: {len(written)} (image, judge) verdicts already cached")
        except Exception as ex:
            print(f"resume: could not read {out_csv}: {ex}")

    # ---- step 1: materialise polylines (serial, deterministic) ----
    print(f"\nphase 1/2: pipeline polyline cache for {len(image_ids)} images")
    polyline_map: dict[str, list] = {}
    for i, image_id in enumerate(image_ids):
        polylines = _ensure_polylines(image_id, args.cell, pipeline_kwargs,
                                        n_lines=k_lookup.get(image_id))
        polyline_map[image_id] = polylines
        if (i + 1) % 20 == 0 or (i + 1) == len(image_ids):
            print(f"  [{i+1:>3}/{len(image_ids)}] polyline cache OK "
                  f"(latest K={len(polylines)})")

    # ---- step 2: dispatch judge calls ----
    print(f"\nphase 2/2: judge sweep, "
          f"{len([(i,j) for i in image_ids for j in judges if (i,j) not in written])} calls")

    overlay_dir = ITER_DIR / "judge_overlays" / args.cell
    overlay_dir.mkdir(parents=True, exist_ok=True)
    saved_overlays_per_judge: dict[str, int] = {j: 0 for j in judges}
    saved_overlays_lock = threading.Lock()

    rows_lock = threading.Lock()
    rows_acc: list[dict] = list(rows_existing)

    def _persist(progress_idx: int):
        with rows_lock:
            df = pd.DataFrame(rows_acc)
            # dedupe: keep the latest row per (image_id, judge)
            if len(df):
                df = df.drop_duplicates(subset=["image_id", "judge"],
                                          keep="last")
            df.to_csv(out_csv, index=False)
        if progress_idx >= 0:
            print(f"  [{progress_idx:>3}] persisted {len(df)} rows → {out_csv.name}")

    def _job(image_id: str, judge: str):
        polylines = polyline_map.get(image_id, [])
        if not polylines:
            return JudgeVerdict(image_id=image_id, judge=judge, k_pred=0,
                                  error="empty_polylines")
        # save first few overlays per judge for visual inspection
        overlay_out = None
        with saved_overlays_lock:
            cnt = saved_overlays_per_judge[judge]
            if cnt < args.save_overlays:
                overlay_out = str(overlay_dir / f"{image_id}__{judge}.jpg")
                saved_overlays_per_judge[judge] = cnt + 1
        img_path = str(LABEL_DIR / f"{image_id}.jpg")
        return review(img_path, polylines, judge=judge,
                       overlay_out=overlay_out)

    todo = [(i, j) for i in image_ids for j in judges
             if (i, j) not in written]
    t_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_job, i, j): (i, j) for (i, j) in todo}
        n_done = 0
        for fut in as_completed(futs):
            (image_id, judge) = futs[fut]
            try:
                v = fut.result()
            except Exception as e:
                v = JudgeVerdict(image_id=image_id, judge=judge, k_pred=0,
                                   error=f"future:{type(e).__name__}:{e}")
            row = _verdict_to_row(v)
            with rows_lock:
                rows_acc.append(row)
            n_done += 1
            if n_done % 10 == 0 or n_done == len(todo):
                _persist(n_done)
            elapsed = time.monotonic() - t_start
            eta = elapsed / max(n_done, 1) * (len(todo) - n_done)
            print(f"  [{n_done:>4}/{len(todo)}] {image_id[:12]}.. {judge:>22s} "
                   f"image_pass={v.image_pass} k={v.k_pred}->{v.k_seen} "
                   f"t={v.latency_s:>5.1f}s err={v.error or '-'} "
                   f"elapsed={elapsed:>5.0f}s eta={eta:>5.0f}s")
    _persist(-1)
    print(f"\nfinal: {len(rows_acc)} rows → {out_csv}")


if __name__ == "__main__":
    main()
