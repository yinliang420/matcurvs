"""Run 3 hosted VLMs on every label/ image and score against GT.

Backends:
  - gpt-5.4              via viviai.cc (OpenAI-compat)
  - gemini-3.1-pro-preview via viviai.cc (OpenAI-compat, not /v1beta)
  - qwen3.5-plus         via DashScope (OpenAI-compat)

Claude-opus-4-6 is NOT available on the viviai gateway, so we skip it
and flag it in the paper as a gateway limitation.

For each (image, backend) pair:
  1. POST the image + DEFAULT_PROMPT to the backend.
  2. Parse the reply into VLMTrail[] using MatDeplot's tolerant parser.
  3. Convert real-coord trails → pixel trails via the label/ annotation's
     min/max pixel and real ranges. (Linear map.)
  4. Run MatDeplot's unified ``evaluate_cell`` against the GT pixel
     polylines.
  5. Append a row to the per-backend CSV.

Uses ThreadPoolExecutor (concurrency=6) to overlap network calls.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[1]))
MATDEPLOT_ROOT = Path(os.environ.get("MATDEPLOT_ROOT", "/path/to/MatDeplot/mdp_v2"))
sys.path.insert(0, str(KMEANS_ROOT))
sys.path.insert(0, str(MATDEPLOT_ROOT))


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ[k.strip()] = v.strip()


_load_env(KMEANS_ROOT / ".env")

from matdeplot.baselines.vlm_baseline import (  # noqa: E402
    DEFAULT_PROMPT,
    VLMBackendConfig,
    VLMResult,
    call_vlm,
)
from matdeplot.eval.unified_metrics import evaluate_cell  # noqa: E402

LABEL_DIR = KMEANS_ROOT / "label"


def working_backends() -> List[VLMBackendConfig]:
    """Backends using the user-confirmed viviai.cc routes.

    Per user's routing spec:
      - Gemini: https://api.viviai.cc/v1beta (native Gemini endpoint)
      - OpenAI-compat: https://api.viviai.cc/v1 (for GPT-5.4)
      - Anthropic-compat: https://api.viviai.cc/v1/messages

    The user's key currently lacks Claude access (every
    ``claude-*`` model returns ``model_not_found`` on /v1/messages).
    We keep Claude in the list so a later key upgrade re-enables it
    automatically; per-call failures are recorded and don't crash
    the sweep.
    """
    vivi = os.environ.get("VIVI_API_BASE", "https://api.viviai.cc").rstrip("/")
    ds = os.environ.get(
        "DASHSCOPE_API_BASE", "https://dashscope.aliyuncs.com"
    ).rstrip("/")
    return [
        VLMBackendConfig(
            name="vlm_gpt-5.4",
            backend="openai",
            endpoint=f"{vivi}/v1/chat/completions",
            model_id="gpt-5.4",
            api_key_env="OPENAI_API_KEY",
            timeout_s=300,
        ),
        VLMBackendConfig(
            name="vlm_claude-opus-4-6",
            backend="anthropic",
            endpoint=f"{vivi}/v1/messages",
            model_id="claude-opus-4-6",
            api_key_env="ANTHROPIC_API_KEY",
            timeout_s=300,
        ),
        VLMBackendConfig(
            name="vlm_gemini-3.1-pro-preview",
            backend="google",
            endpoint=(
                f"{vivi}/v1beta/models/gemini-3.1-pro-preview:generateContent"
            ),
            model_id="gemini-3.1-pro-preview",
            api_key_env="GOOGLE_API_KEY",
            timeout_s=360,
        ),
        VLMBackendConfig(
            name="vlm_qwen3.5-plus",
            backend="openai",
            endpoint=f"{ds}/compatible-mode/v1/chat/completions",
            model_id="qwen3.5-plus",
            api_key_env="DASHSCOPE_API_KEY",
            timeout_s=360,
        ),
    ]


def _axis_transform_from_ticks(ticks, coord_idx: int):
    """Fit an affine real -> pixel transform from a list of tick pairs.

    ``ticks`` is a list of [real_value, [px_x, px_y]]. ``coord_idx`` is
    0 for x (we read px_x) or 1 for y (px_y). Returns ``(slope, intercept)``
    or ``None`` if fewer than 2 distinct ticks.
    """
    pts = []
    for t in ticks or []:
        if not isinstance(t, (list, tuple)) or len(t) < 2:
            continue
        try:
            real_v = float(t[0])
            px = t[1]
            if not isinstance(px, (list, tuple)) or len(px) < 2:
                continue
            px_v = float(px[coord_idx])
        except (TypeError, ValueError):
            continue
        pts.append((real_v, px_v))
    if len(pts) < 2:
        return None
    # pick the two most-separated ticks (real value)
    pts.sort(key=lambda p: p[0])
    (r1, p1), (r2, p2) = pts[0], pts[-1]
    if r1 == r2:
        return None
    slope = (p2 - p1) / (r2 - r1)
    intercept = p1 - slope * r1
    return (slope, intercept)


def load_gt(ann_path: Path):
    """Return (ann_dict, gt_pixel_trails, (w,h), total_pts).

    GT pixel coords are canonical (they place points on the image).
    The annotation's ``real`` coords are unreliable (y_ticks in 91%
    of label/ annotations are populated with pixel-Y as fake real
    values), so we never use them for VLM comparison.
    """
    d = json.loads(ann_path.read_text())
    w, h = d["image_size"]
    trails: List[np.ndarray] = []
    total = 0
    for _, ln in d["lines"].items():
        pts = ln["points"]
        px = np.array([[p["pixel"][0], p["pixel"][1]] for p in pts],
                       dtype=float)
        trails.append(px)
        total += len(pts)
    return d, trails, (w, h), total


def vlm_real_to_pixel(
    real_trails: List[List[Tuple[float, float]]],
    ann: Dict,
    px_size: Tuple[int, int],
) -> List[np.ndarray]:
    """Map VLM real → pixel using chart-axis geometry.

    Strategy:
      - X: fit real→pixel affine from ``x_ticks`` (reliable in 99% of
        label/ annotations). Fallback: stretch VLM's x-bbox to
        ``x_axis_line`` extent.
      - Y: ``y_ticks`` are buggy (pixel-Y stored as real), so we *don't*
        trust them. Instead, stretch VLM's y-bbox to the chart's plot
        region (between ``y_axis_line`` endpoints, Y-flipped).

    Caveat: this means DTW measures **shape** and **inter-line spacing**
    in Y, not absolute Y placement — we have no reliable way to know
    the chart's Y axis-label range from the annotation.
    """
    w, h = px_size
    det = ann.get("axis_detection") or {}
    rng = ann.get("axis_range") or {}

    # X transform from x_ticks.
    tx = _axis_transform_from_ticks(det.get("x_ticks"), coord_idx=0)
    if tx is None:
        xl = det.get("x_axis_line")
        xmin = rng.get("x_min"); xmax = rng.get("x_max")
        if (isinstance(xl, list) and len(xl) >= 4
                and xmin is not None and xmax is not None and xmax != xmin):
            sx = (xl[2] - xl[0]) / (xmax - xmin)
            bx = xl[0] - sx * xmin
            tx = (sx, bx)

    # Plot-region pixel Y extent.
    yl = det.get("y_axis_line")
    if isinstance(yl, list) and len(yl) >= 4:
        py_top = float(min(yl[1], yl[3]))
        py_bot = float(max(yl[1], yl[3]))
    else:
        py_top, py_bot = 0.0, float(h)

    # Plot-region pixel X extent (used if tx is None).
    xl_ = det.get("x_axis_line")
    if isinstance(xl_, list) and len(xl_) >= 4:
        px_left = float(min(xl_[0], xl_[2]))
        px_right = float(max(xl_[0], xl_[2]))
    else:
        px_left, px_right = 0.0, float(w)

    # Collect all VLM real points across lines.
    all_pts = [p for t in real_trails for p in (t or [])]
    if not all_pts:
        return []
    all_arr = np.array(all_pts, dtype=float)

    # X fallback: stretch VLM x-bbox to plot region.
    if tx is None:
        x0, x1 = float(all_arr[:, 0].min()), float(all_arr[:, 0].max())
        if x1 == x0:
            x1 = x0 + 1.0
        sx = (px_right - px_left) / (x1 - x0)
        bx = px_left - sx * x0
        tx = (sx, bx)
    sx, bx = tx

    # Y: stretch VLM y-bbox to plot region (flip: high real Y → top pixel).
    y0, y1 = float(all_arr[:, 1].min()), float(all_arr[:, 1].max())
    if y1 == y0:
        y1 = y0 + 1.0
    sy = (py_top - py_bot) / (y1 - y0)
    by = py_bot - sy * y0

    out: List[np.ndarray] = []
    for rt in real_trails:
        if not rt:
            continue
        arr = np.array(rt, dtype=float)
        px = sx * arr[:, 0] + bx
        py = sy * arr[:, 1] + by
        out.append(np.column_stack([px, py]))
    return out


def score_one(
    image_id: str,
    ann_path: Path,
    img_path: Path,
    backend: VLMBackendConfig,
    cell_id: str,
) -> Optional[Dict]:
    try:
        ann, gt_trails, (w, h), n_pts = load_gt(ann_path)
    except Exception as e:
        return {"image_id": image_id, "cell_id": cell_id,
                "error": f"gt_load:{e}", "success": False}

    t0 = time.time()
    r: VLMResult = call_vlm(str(img_path), image_id=image_id, backend=backend,
                             prompt=DEFAULT_PROMPT)
    elapsed = time.time() - t0

    pred_real = [[(x, y) for (x, y) in t.points] for t in r.trails]
    pred_px = vlm_real_to_pixel(pred_real, ann, (w, h))

    m = evaluate_cell(
        pred_px, gt_trails,
        image_meta={"img_w": w, "img_h": h},
        image_id=image_id, cell_id=cell_id,
        success=bool(r.success), error=r.error,
    )
    row = asdict(m)
    row["latency_s"] = elapsed
    row["input_tokens"] = r.input_tokens
    row["output_tokens"] = r.output_tokens
    row["n_gt_lines"] = len(gt_trails)
    row["n_pred_lines"] = len(r.trails)
    row["vlm_error"] = r.error or ""
    # stratum
    if len(gt_trails) == 1:
        row["stratum"] = "L1_single"
    elif len(gt_trails) == 2:
        row["stratum"] = "L2_two_line"
    elif len(gt_trails) <= 4:
        row["stratum"] = "L3_few_line"
    else:
        row["stratum"] = "L4_many_line"
    return row


def run_backend(
    backend: VLMBackendConfig,
    pairs: List[Tuple[str, Path, Path]],
    csv_path: Path,
    raw_dir: Path,
    concurrency: int,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {backend.name}: {len(pairs)} images, conc={concurrency} ===")
    rows: List[Dict] = []
    t_start = time.time()

    def _task(p: Tuple[str, Path, Path]) -> Dict:
        image_id, ann, img = p
        try:
            row = score_one(image_id, ann, img, backend, cell_id=backend.name)
        except Exception as e:
            row = {"image_id": image_id, "cell_id": backend.name,
                   "error": f"score_exc:{type(e).__name__}:{e}",
                   "success": False}
        return row or {}

    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(_task, p): p for p in pairs}
        done = 0
        for fut in cf.as_completed(futs):
            done += 1
            row = fut.result()
            if row:
                rows.append(row)
            if done % 10 == 0 or done == len(pairs):
                dt = time.time() - t_start
                print(f"  [{done:>3}/{len(pairs)}] {backend.name} "
                      f"elapsed={dt:.0f}s")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    # quick summary
    if "dtw_delta" in df.columns:
        g = df.groupby("stratum", dropna=False).agg(
            n=("image_id", "count"),
            dtw=("dtw_delta", "mean"),
            ccr=("ccr_at_tau", "mean"),
            f1=("instance_f1", "mean"),
        ).round(3)
        print(g.to_string())
        print(f"  ALL: n={len(df)} dtw={df['dtw_delta'].mean():.3f} "
              f"ccr={df['ccr_at_tau'].mean():.3f} "
              f"f1={df['instance_f1'].mean():.3f}")
    print(f"  saved: {csv_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/results")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conc", type=int, default=6)
    ap.add_argument("--only", default="",
                    help="comma-separated backend names to run (substring match)")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    anns = sorted(LABEL_DIR.glob("*_annotation.json"))
    if args.limit:
        anns = anns[:args.limit]
    pairs: List[Tuple[str, Path, Path]] = []
    for ann in anns:
        iid = ann.stem.replace("_annotation", "")
        img = LABEL_DIR / f"{iid}.jpg"
        if img.exists():
            pairs.append((iid, ann, img))
    print(f"label/ pairs: {len(pairs)}")

    backends = working_backends()
    if args.only:
        want = [s.strip() for s in args.only.split(",") if s.strip()]
        backends = [b for b in backends if any(w in b.name for w in want)]
    print(f"running backends: {[b.name for b in backends]}")

    for be in backends:
        csv_path = out / f"{be.name}_label.csv"
        raw_dir = out / "pred" / f"{be.name}_label"
        run_backend(be, pairs, csv_path, raw_dir, args.conc)


if __name__ == "__main__":
    main()
