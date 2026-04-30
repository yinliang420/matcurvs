"""Generate label/ y-lookup Q&A from GT polylines.

For each label/ image with a non-`needs_review` axis sidecar:
  - Find the longest curve (by point count) -> Line_*.points
  - Use real-coords (rx, ry) from points[i].real
  - Sample N=8 query x positions uniformly in interior 5% margin
  - Linearly interpolate y at each x*
  - Skip x* near multivalued regions (vertical gap > 30% y-range)

Stratified sampling: ~100 imgs across K=1..6 buckets (proportional to availability,
floor 8 per non-empty bucket if possible).

Output: experiments/results/vlm_downstream/label_qa.json
        {image_id: [{qid, x_star, y_gt, n_lines, longest_line, x_min, x_max,
                     y_min, y_max}]}
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
import os

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(KMEANS_ROOT))

LABEL_DIR = KMEANS_ROOT / "label"
SIDECAR_DIR = LABEL_DIR / "axis_range_fixed"
OUT_PATH = KMEANS_ROOT / "experiments" / "results" / "vlm_downstream" / "label_qa.json"

SEED = 42
N_QUERIES_PER_IMG = 8
N_TARGET_IMGS = 100


def find_longest_line(lines: dict) -> tuple[str, list[list[float]]]:
    """Return (line_name, points_real) for line with most points."""
    best = ("", [])
    best_n = -1
    for name, ln in (lines or {}).items():
        pts = (ln or {}).get("points") or []
        real = [(p["real"][0], p["real"][1]) for p in pts
                if p.get("real") and len(p["real"]) >= 2]
        if len(real) > best_n:
            best_n = len(real)
            best = (name, real)
    return best


def interpolate_y_at_x(pts: list[tuple[float, float]], x_star: float) -> float | None:
    """Linear interp; pts assumed (or sorted) by x ascending."""
    if not pts:
        return None
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    if x_star < min(xs) or x_star > max(xs):
        return None
    # sort then interp
    order = sorted(range(len(pts)), key=lambda i: xs[i])
    xs_s = [xs[i] for i in order]; ys_s = [ys[i] for i in order]
    for i in range(len(xs_s) - 1):
        if xs_s[i] <= x_star <= xs_s[i + 1]:
            x0, x1 = xs_s[i], xs_s[i + 1]
            if x1 - x0 < 1e-12:
                return ys_s[i]
            t = (x_star - x0) / (x1 - x0)
            return ys_s[i] + t * (ys_s[i + 1] - ys_s[i])
    return ys_s[-1]


def is_multivalued_near(pts: list[tuple[float, float]], x_star: float,
                         x_window: float, y_full_range: float) -> bool:
    """Heuristic: any two adjacent points with |Δy| > 0.3*y_full_range *and*
    both within x_window of x_star -> flagged."""
    if not pts:
        return False
    order = sorted(range(len(pts)), key=lambda i: pts[i][0])
    sorted_pts = [pts[i] for i in order]
    threshold = 0.3 * y_full_range
    for i in range(len(sorted_pts) - 1):
        x0, y0 = sorted_pts[i]
        x1, y1 = sorted_pts[i + 1]
        if abs(x0 - x_star) <= x_window and abs(x1 - x_star) <= x_window:
            if abs(y1 - y0) > threshold:
                return True
    return False


def gen_for_image(image_id: str, ann: dict, side: dict,
                   rng: random.Random) -> list[dict] | None:
    """Generate up to N_QUERIES_PER_IMG questions for one image. Returns None
    if image is unsuitable."""
    if side.get("needs_review"):
        return None
    new_range = side.get("axis_range_new")
    if not new_range:
        return None

    lines = ann.get("lines") or {}
    if not lines:
        return None
    line_name, real_pts = find_longest_line(lines)
    if len(real_pts) < 4:
        return None

    xs = [p[0] for p in real_pts]; ys = [p[1] for p in real_pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max - x_min < 1e-6 or y_max - y_min < 1e-6:
        return None

    margin = 0.05 * (x_max - x_min)
    lo = x_min + margin; hi = x_max - margin
    if hi <= lo:
        return None

    # sample N candidates with rejection for multivalued
    x_window = 0.05 * (x_max - x_min)
    candidates = []
    attempts = 0
    while len(candidates) < N_QUERIES_PER_IMG and attempts < N_QUERIES_PER_IMG * 6:
        attempts += 1
        x_star = rng.uniform(lo, hi)
        if is_multivalued_near(real_pts, x_star, x_window, y_max - y_min):
            continue
        y_gt = interpolate_y_at_x(real_pts, x_star)
        if y_gt is None:
            continue
        # de-dup very close x
        if any(abs(c["x_star"] - x_star) < 0.01 * (x_max - x_min) for c in candidates):
            continue
        qid = f"{image_id}_q{len(candidates):02d}"
        candidates.append({
            "qid": qid,
            "image_id": image_id,
            "x_star": float(x_star),
            "y_gt": float(y_gt),
            "n_lines": len(lines),
            "longest_line": line_name,
            "x_min_curve": float(x_min),
            "x_max_curve": float(x_max),
            "y_min_curve": float(y_min),
            "y_max_curve": float(y_max),
            "x_min_axis": float(new_range["x_min"]),
            "x_max_axis": float(new_range["x_max"]),
            "y_min_axis": float(new_range["y_min"]),
            "y_max_axis": float(new_range["y_max"]),
        })
    if len(candidates) < 3:
        return None
    return candidates


def main() -> None:
    rng = random.Random(SEED)

    # build pool of usable images, grouped by K
    by_k: dict[int, list[str]] = defaultdict(list)
    pool: dict[str, dict] = {}
    for ann_path in sorted(LABEL_DIR.glob("*_annotation.json")):
        image_id = ann_path.stem.replace("_annotation", "")
        side_path = SIDECAR_DIR / f"{image_id}_axis.json"
        if not side_path.exists():
            continue
        try:
            ann = json.loads(ann_path.read_text())
            side = json.loads(side_path.read_text())
        except Exception:
            continue
        if side.get("needs_review"):
            continue
        K = len(ann.get("lines") or {})
        if K < 1 or K > 6:
            continue
        # quick suitability check: longest line >= 4 pts
        _, longest_pts = find_longest_line(ann.get("lines") or {})
        if len(longest_pts) < 4:
            continue
        by_k[K].append(image_id)
        pool[image_id] = {"ann": ann, "side": side}

    print("Usable images by K: " + str({k: len(v) for k, v in sorted(by_k.items())}))

    # stratified sample target N_TARGET_IMGS proportional to availability,
    # floor 8 per bucket (or all if fewer)
    total_avail = sum(len(v) for v in by_k.values())
    if total_avail == 0:
        print("No usable images, abort")
        return
    sample_ids: list[str] = []
    quota: dict[int, int] = {}
    for k, ids in sorted(by_k.items()):
        share = max(8, int(round(N_TARGET_IMGS * len(ids) / total_avail)))
        share = min(share, len(ids))
        quota[k] = share
    # rescale to N_TARGET_IMGS
    cur = sum(quota.values())
    if cur > N_TARGET_IMGS:
        # trim from largest bucket
        for k in sorted(quota, key=lambda kk: -quota[kk]):
            while cur > N_TARGET_IMGS and quota[k] > 8:
                quota[k] -= 1
                cur -= 1
            if cur <= N_TARGET_IMGS:
                break
    elif cur < N_TARGET_IMGS:
        for k in sorted(quota, key=lambda kk: -len(by_k[kk])):
            while cur < N_TARGET_IMGS and quota[k] < len(by_k[k]):
                quota[k] += 1
                cur += 1
            if cur >= N_TARGET_IMGS:
                break

    print(f"Quota: {quota} (total {sum(quota.values())})")
    for k, n in quota.items():
        ids_sorted = sorted(by_k[k])  # deterministic
        rng.shuffle(ids_sorted)
        sample_ids.extend(ids_sorted[:n])

    print(f"Sampled {len(sample_ids)} images")

    # generate questions
    out: dict[str, list[dict]] = {}
    skipped = 0
    for image_id in sample_ids:
        d = pool[image_id]
        qs = gen_for_image(image_id, d["ann"], d["side"], rng)
        if not qs:
            skipped += 1
            continue
        out[image_id] = qs

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    n_q = sum(len(v) for v in out.values())
    print(f"Wrote {len(out)} images with {n_q} questions to {OUT_PATH} (skipped {skipped})")


if __name__ == "__main__":
    main()
