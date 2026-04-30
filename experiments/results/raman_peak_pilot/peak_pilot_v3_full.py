"""Scale v3 (axis-fix) to ALL raman images (~1068 total).

Reuses every helper from peak_pilot_v3 unchanged; only swaps the sampler so we
iterate (folder, image_key) over all `chemical_formulas_raman.json` entries.

Output:
    peak_pilot_v3_full_results.csv  (one row per image)
No diag plots (too many).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

# Reuse all helpers from peak_pilot_v3
sys.path.insert(0, str(Path(__file__).parent))
from peak_pilot_v3 import (  # noqa: E402
    DATA_ROOT, OUT_DIR, process_image,  # process_image is the per-image driver
)


def select_all_images():
    folders = sorted([f for f in os.listdir(DATA_ROOT)
                      if (DATA_ROOT / f).is_dir()])
    picks = []
    for folder in folders:
        json_path = DATA_ROOT / folder / "chemical_formulas_raman.json"
        if not json_path.exists():
            continue
        try:
            meta = json.loads(json_path.read_text())
        except Exception:
            continue
        if not meta:
            continue
        for key in sorted(meta.keys()):
            picks.append((folder, key, meta[key]))
    return picks


def main():
    picks = select_all_images()
    print(f"selected {len(picks)} images (target ~1068)")

    rows = []
    t0 = time.time()
    for i, (folder, key, meta) in enumerate(picks):
        try:
            row, _diag = process_image(folder, key, meta)
        except Exception as e:
            row = {
                "image_id": f"{folder}/{Path(key).stem}",
                "chemical_formula": "", "n_gt_peaks": 0, "n_pred_peaks": 0,
                "n_matched": 0, "peak_mae": float("nan"),
                "peak_recall": float("nan"), "peak_precision": float("nan"),
                "curve_pearson": float("nan"), "curve_nrmse": float("nan"),
                "ok": False, "reason": f"crash:{type(e).__name__}",
                "axis_x1": float("nan"), "axis_x2": float("nan"),
                "curve_x_left": float("nan"), "curve_x_right": float("nan"),
                "left_shift_px": float("nan"), "right_shift_px": float("nan"),
            }
        rows.append(row)
        if (i + 1) % 50 == 0:
            done = pd.DataFrame(rows)
            ok = done[done["ok"]]
            print(f"  [{i+1}/{len(picks)}] elapsed={time.time()-t0:.0f}s  "
                  f"ok={len(ok)}  median recall={ok['peak_recall'].median():.3f}  "
                  f"median MAE={ok['peak_mae'].median():.2f}")

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "peak_pilot_v3_full_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}  ({len(df)} rows, {df['ok'].sum()} ok)")
    if df["ok"].any():
        ok = df[df["ok"]]
        print(f"  median recall   = {ok['peak_recall'].median():.3f}")
        print(f"  agg recall      = {ok['n_matched'].sum()}/{ok['n_gt_peaks'].sum()} "
              f"= {ok['n_matched'].sum()/ok['n_gt_peaks'].sum():.3f}")
        print(f"  median MAE      = {ok['peak_mae'].median():.3f} cm^-1")
        print(f"  median pearson  = {ok['curve_pearson'].median():.3f}")
        print(f"  median nrmse    = {ok['curve_nrmse'].median():.3f}")


if __name__ == "__main__":
    main()
