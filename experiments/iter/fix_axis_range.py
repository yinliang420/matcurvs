#!/usr/bin/env python
"""Audit and fix axis_range placeholders in label/*_annotation.json.

For each annotation:
  - Derive correct axis_range from the union of `lines[*].points[*].real`
  - Compare to existing axis_range
  - Write a sidecar JSON to label/axis_range_fixed/<id>_axis.json
  - Append summary row to experiments/results/axis_range_audit.csv

The original annotation files are NEVER modified.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median


REPO_ROOT = Path(__file__).resolve().parents[2]
LABEL_DIR = REPO_ROOT / "label"
SIDECAR_DIR = LABEL_DIR / "axis_range_fixed"
CSV_OUT = REPO_ROOT / "experiments" / "results" / "axis_range_audit.csv"

# A range is treated as a placeholder if it exactly matches [0, 100] x [0, 100]
PLACEHOLDER = {"x_min": 0.0, "x_max": 100.0, "y_min": 0.0, "y_max": 100.0}
SUSPICIOUS_SPAN_THRESHOLD = 1.0


def is_placeholder(rng: dict) -> bool:
    if not rng:
        return False
    try:
        return all(float(rng.get(k, math.nan)) == v for k, v in PLACEHOLDER.items())
    except (TypeError, ValueError):
        return False


def derive_range(lines: dict) -> tuple[dict | None, int, int]:
    xs: list[float] = []
    ys: list[float] = []
    n_lines = 0
    n_points = 0
    for _name, line in (lines or {}).items():
        pts = (line or {}).get("points") or []
        if not pts:
            continue
        n_lines += 1
        for p in pts:
            real = p.get("real")
            if real is None or len(real) < 2:
                continue
            try:
                rx = float(real[0]); ry = float(real[1])
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(rx) and math.isfinite(ry)):
                continue
            xs.append(rx); ys.append(ry)
            n_points += 1
    if not xs or not ys:
        return None, n_lines, n_points
    return (
        {"x_min": min(xs), "x_max": max(xs), "y_min": min(ys), "y_max": max(ys)},
        n_lines,
        n_points,
    )


def quantile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def main():
    SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(LABEL_DIR.glob("*_annotation.json"))
    rows = []
    placeholder_count = 0
    real_count = 0
    no_data_count = 0
    mapping_not_ready = 0
    suspicious = []

    for path in ann_files:
        image_id = path.name.replace("_annotation.json", "")
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            print(f"[skip] {image_id}: {e}")
            continue

        old_range = data.get("axis_range") or {}
        mapping_ready = bool(
            (data.get("axis_detection") or {}).get("mapping_ready", False)
        )
        new_range, n_lines, n_points = derive_range(data.get("lines") or {})

        placeholder = is_placeholder(old_range)
        if placeholder:
            placeholder_count += 1
        else:
            real_count += 1
        if not mapping_ready:
            mapping_not_ready += 1

        x_span_new = (
            new_range["x_max"] - new_range["x_min"] if new_range else float("nan")
        )
        y_span_new = (
            new_range["y_max"] - new_range["y_min"] if new_range else float("nan")
        )

        if new_range is None:
            no_data_count += 1
            needs_review = True
        else:
            needs_review = (
                (not mapping_ready)
                or (x_span_new < SUSPICIOUS_SPAN_THRESHOLD)
                or (y_span_new < SUSPICIOUS_SPAN_THRESHOLD)
                or (n_points < 3)
            )
            if x_span_new < SUSPICIOUS_SPAN_THRESHOLD or y_span_new < SUSPICIOUS_SPAN_THRESHOLD:
                suspicious.append((image_id, x_span_new, y_span_new, n_points))

        sidecar = {
            "image_id": image_id,
            "axis_range_old": old_range,
            "axis_range_new": new_range,
            "mapping_ready": mapping_ready,
            "n_lines": n_lines,
            "n_points": n_points,
            "is_placeholder": placeholder,
            "needs_review": needs_review,
        }
        (SIDECAR_DIR / f"{image_id}_axis.json").write_text(
            json.dumps(sidecar, indent=2)
        )

        rows.append({
            "image_id": image_id,
            "is_placeholder": int(placeholder),
            "mapping_ready": int(mapping_ready),
            "n_lines": n_lines,
            "n_points": n_points,
            "x_min_old": old_range.get("x_min", ""),
            "x_max_old": old_range.get("x_max", ""),
            "y_min_old": old_range.get("y_min", ""),
            "y_max_old": old_range.get("y_max", ""),
            "x_min_new": new_range["x_min"] if new_range else "",
            "x_max_new": new_range["x_max"] if new_range else "",
            "y_min_new": new_range["y_min"] if new_range else "",
            "y_max_new": new_range["y_max"] if new_range else "",
            "x_span_new": x_span_new if new_range else "",
            "y_span_new": y_span_new if new_range else "",
            "needs_review": int(needs_review),
        })

    # write csv
    if rows:
        fieldnames = list(rows[0].keys())
        with CSV_OUT.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    # distributional summary
    x_spans = sorted(
        r["x_span_new"] for r in rows if isinstance(r["x_span_new"], (int, float))
    )
    y_spans = sorted(
        r["y_span_new"] for r in rows if isinstance(r["y_span_new"], (int, float))
    )

    def stats(name, vals):
        if not vals:
            print(f"  {name}: no data")
            return
        print(
            f"  {name}: median={median(vals):.4g} p10={quantile(vals, 0.10):.4g} "
            f"p90={quantile(vals, 0.90):.4g} max={max(vals):.4g} min={min(vals):.4g}"
        )

    print("=" * 60)
    print(f"Total annotations:        {len(rows)}")
    print(f"Placeholder [0,100]x[0,100]: {placeholder_count}")
    print(f"Real / non-placeholder:   {real_count}")
    print(f"No real-coord data:       {no_data_count}")
    print(f"mapping_ready=False:      {mapping_not_ready}")
    print()
    print("Derived span distribution:")
    stats("x_span", x_spans)
    stats("y_span", y_spans)
    print()
    print(f"Suspicious (span < {SUSPICIOUS_SPAN_THRESHOLD}): {len(suspicious)}")
    for sid, xs, ys, npts in suspicious[:20]:
        print(f"   {sid[:24]}.. x_span={xs:.4g} y_span={ys:.4g} n_pts={npts}")
    if len(suspicious) > 20:
        print(f"   ... and {len(suspicious) - 20} more")
    print()
    print(f"Sidecars: {SIDECAR_DIR}")
    print(f"CSV:      {CSV_OUT}")


if __name__ == "__main__":
    main()
