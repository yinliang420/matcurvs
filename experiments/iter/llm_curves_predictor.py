"""Curve-metadata predictor: ask a VLM for K + per-curve descriptors so we
can use the answer to merge over-segmented k-means clusters in
``pipeline.run_pipeline``.

Returns ``CurvesPred(image_id, backend, curves, latency_s, raw, error)``
where ``curves`` is a list of dicts::

    [{"color": "red"|"blue"|...|"unknown",
      "style": "solid"|"dashed"|"dotted"|"unknown",
      "x_lo":  float in [0, 1],
      "x_hi":  float in [0, 1]},
     ...]

The VLM is instructed to enumerate distinct DATA curves only (excluding
gridlines, axes, legends).  Tested on Qwen-VL through DashScope.
"""
from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from openai import OpenAI

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[2]))


def _load_env(env_path: Path = KMEANS_ROOT / ".env") -> None:
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


_load_env()

PROMPT = """You are analyzing a scientific chart image. Enumerate the DATA
curves (line plots, scatter trajectories) plotted inside the plot area.

Counting rules (apply in order):
1. If a legend exists, the unique entries determine the curves.
2. Otherwise, count distinct trajectories or coloured groups of points.
3. EXCLUDE chart borders, gridlines, axis lines / ticks, baseline-only
   reference lines (zero-line, fit-line if labeled as such), error bars.

The plot area's bottom-left corner is normalised to (0, 0) and the
top-right corner to (1, 1).  For each curve, return:
- "color": one of {red, blue, green, orange, purple, black, brown,
  cyan, gray, pink, yellow, white, unknown}.  Pick the closest single
  word; no modifiers like "light"/"dark"/"navy".
- "style": one of {solid, dashed, dotted, scatter, unknown}.  Use
  "scatter" only if the curve is a marker-only point cloud.
- "x_lo", "x_hi": fractional x-range covered by this curve,
  with 0 ≤ x_lo < x_hi ≤ 1.
- "anchors": THREE (x, y) sample points along this curve as a list
  of [x, y] floats.  Use the curve's value at x = 0.1, 0.5, 0.9 of
  the plot area (or any three well-separated points along the
  curve's x-range), with y normalized so 0 = bottom of plot, 1 = top.
  Example for a curve at constant y=0.7:
      "anchors": [[0.10, 0.70], [0.50, 0.70], [0.90, 0.70]]

Anchors are critical for distinguishing curves that share a colour
(e.g. four black curves at different y-offsets).

Respond with strict JSON, no extra text:

{"total_lines_num": <integer>,
 "curves": [
    {"color": "...", "style": "...",
     "x_lo": 0.0, "x_hi": 1.0,
     "anchors": [[x, y], [x, y], [x, y]]},
    ...
 ]}

The "curves" array must have exactly "total_lines_num" entries, in
the same order as a left-to-right reading of the legend (or top-to-
bottom if no legend).
"""


BACKENDS = {
    "qwen-vl-max": {
        "model_id": "qwen-vl-max",
        "base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "key_env": "DASHSCOPE_API_KEY",
    },
    "qwen3-vl-plus": {
        "model_id": "qwen3-vl-plus",
        "base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "key_env": "DASHSCOPE_API_KEY",
    },
}


@dataclass
class CurveDesc:
    color: str = "unknown"
    style: str = "unknown"
    x_lo: float = 0.0
    x_hi: float = 1.0
    anchors: list = field(default_factory=list)  # list of (x, y) tuples in [0, 1]


@dataclass
class CurvesPred:
    image_id: str
    backend: str
    k_pred: Optional[int]
    curves: list = field(default_factory=list)   # list[CurveDesc]
    latency_s: float = 0.0
    raw: str = ""
    error: Optional[str] = None


def _encode_image(img_path: Path) -> str:
    return base64.b64encode(Path(img_path).read_bytes()).decode("ascii")


_VALID_COLORS = {"red", "blue", "green", "orange", "purple", "black",
                  "brown", "cyan", "gray", "grey", "pink", "yellow",
                  "white", "unknown"}
_VALID_STYLES = {"solid", "dashed", "dotted", "scatter", "unknown"}


def _parse_curves(raw: str) -> tuple[Optional[int], list[CurveDesc], Optional[str]]:
    if not raw:
        return None, [], "empty_raw"
    txt = raw.strip()
    if "```" in txt:
        # extract inner code block
        parts = txt.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                txt = p
                break
    s = txt.find("{")
    e = txt.rfind("}") + 1
    if s == -1 or e <= 0:
        return None, [], "no_json_braces"
    try:
        d = json.loads(txt[s:e])
    except Exception as ex:
        return None, [], f"json_parse:{type(ex).__name__}"
    k = d.get("total_lines_num")
    if isinstance(k, str) and k.isdigit():
        k = int(k)
    if not isinstance(k, int):
        k = None
    raw_curves = d.get("curves", [])
    if not isinstance(raw_curves, list):
        raw_curves = []
    out: list[CurveDesc] = []
    for c in raw_curves:
        if not isinstance(c, dict):
            continue
        color = str(c.get("color", "unknown")).lower().strip()
        if color == "grey":
            color = "gray"
        if color not in _VALID_COLORS:
            color = "unknown"
        style = str(c.get("style", "unknown")).lower().strip()
        if style not in _VALID_STYLES:
            style = "unknown"
        try:
            x_lo = float(c.get("x_lo", 0.0))
            x_hi = float(c.get("x_hi", 1.0))
        except Exception:
            x_lo, x_hi = 0.0, 1.0
        x_lo = max(0.0, min(1.0, x_lo))
        x_hi = max(0.0, min(1.0, x_hi))
        if x_hi <= x_lo:
            x_lo, x_hi = 0.0, 1.0
        anchors = []
        for pt in c.get("anchors", []) or []:
            try:
                if isinstance(pt, dict):
                    px = float(pt.get("x"))
                    py = float(pt.get("y"))
                else:
                    px, py = float(pt[0]), float(pt[1])
                px = max(0.0, min(1.0, px))
                py = max(0.0, min(1.0, py))
                anchors.append((px, py))
            except Exception:
                continue
        out.append(CurveDesc(color=color, style=style,
                              x_lo=x_lo, x_hi=x_hi, anchors=anchors))
    if k is None and out:
        k = len(out)
    return k, out, None


def predict_curves(image_path: str | Path, backend: str = "qwen-vl-max",
                   timeout: float = 90.0, retries: int = 1) -> CurvesPred:
    cfg = BACKENDS[backend]
    api_key = os.environ.get(cfg["key_env"])
    image_id = Path(image_path).stem
    if not api_key:
        return CurvesPred(image_id=image_id, backend=backend, k_pred=None,
                          latency_s=0.0, raw="",
                          error=f"missing env {cfg['key_env']}")
    client = OpenAI(api_key=api_key, base_url=cfg["base"], timeout=timeout)
    img_b64 = _encode_image(Path(image_path))
    last_err = None
    last_t0 = 0.0
    for attempt in range(retries + 1):
        last_t0 = time.monotonic()
        try:
            resp = client.chat.completions.create(
                model=cfg["model_id"],
                messages=[
                    {"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                        {"type": "text", "text": PROMPT},
                    ]},
                ],
            )
            dt = time.monotonic() - last_t0
            raw = resp.choices[0].message.content or ""
            k, curves, perr = _parse_curves(raw)
            if k is None or not curves:
                if attempt < retries:
                    time.sleep(1.0)
                    last_err = perr or "parse_empty"
                    continue
            return CurvesPred(image_id=image_id, backend=backend, k_pred=k,
                              curves=curves, latency_s=dt, raw=raw[:600],
                              error=perr if (k is None or not curves) else None)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(1.5)
    return CurvesPred(image_id=image_id, backend=backend, k_pred=None,
                      latency_s=time.monotonic() - last_t0, raw="",
                      error=last_err)


if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else \
        str(KMEANS_ROOT / "label" /
            "00d117323855716750e5a0c3245c3d5a9e63adbfa8cafc42050e98b6f1b65b78_1.jpg")
    backend = sys.argv[2] if len(sys.argv) > 2 else "qwen-vl-max"
    r = predict_curves(img, backend=backend)
    print(f"image={r.image_id}  backend={r.backend}  k={r.k_pred}  t={r.latency_s:.1f}s")
    print(f"err={r.error}")
    for i, c in enumerate(r.curves):
        anc = ",".join(f"({x:.2f},{y:.2f})" for x, y in c.anchors)
        print(f"  curve[{i}]: {c.color:>7s} {c.style:>7s} x=[{c.x_lo:.2f},{c.x_hi:.2f}] anchors={anc}")
    print("---raw[:600]---")
    print(r.raw)
