"""Cell-6 VLM-as-judge reviewer for MatDeplot extractions.

Renders the pipeline's polylines as a high-contrast overlay on the
original chart, prompts a hosted multimodal LLM for a per-curve
PASS/FAIL verdict and an image-level pass-iff-all-pass aggregate, and
returns a structured ``JudgeVerdict``.

Supported judges: Qwen-VL-Max / Qwen3-VL-Plus (DashScope) and
GPT-5.4 / Gemini-3.1-pro-preview / Claude Opus 4.6 (viviai.cc).
"""
from __future__ import annotations
import base64
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
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

REVIEW_PROMPT = """You are a strict reviewer of an automated chart-data
extraction. The image is the ORIGINAL chart with each EXTRACTED data
line redrawn ON TOP in a high-contrast colour and labelled #0..#K-1
at its left endpoint. We extracted {K} candidate curves.

For EACH labelled overlay, decide PASS iff
(a) it follows a single real data trajectory inside the plot area;
(b) it does NOT cross between distinct curves, axis lines,
gridlines, legend marks, annotations;
(c) its endpoints are inside the plot area.
Otherwise FAIL. Be strict but reasonable.

Image-level verdict: PASS iff EVERY overlay passes.

Respond with strict JSON, no extra text:
{{"k_seen": <integer>,
  "per_curve": [{{"id": 0, "pass": true|false, "why": "<short>"}}],
  "image_pass": true|false,
  "comment": "<one line>"}}
"""

BACKENDS = {
    "qwen-vl-max": dict(
        model_id="qwen-vl-max",
        base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        key_env="DASHSCOPE_API_KEY",
    ),
    "qwen3-vl-plus": dict(
        model_id="qwen3-vl-plus",
        base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        key_env="DASHSCOPE_API_KEY",
    ),
    "gpt-5.4": dict(
        model_id="gpt-5.4",
        base="https://api.viviai.cc/v1",
        key_env="OPENAI_API_KEY",
    ),
    "gemini-3.1-pro-preview": dict(
        model_id="gemini-3.1-pro-preview",
        base="https://api.viviai.cc/v1",
        key_env="OPENAI_API_KEY",
    ),
    "claude-opus-4-6": dict(
        model_id="claude-opus-4-6",
        base="https://api.viviai.cc/v1",
        key_env="OPENAI_API_KEY",
    ),
}

# High-contrast palette (BGR for OpenCV).  Avoid colours common in
# materials charts (red, blue, green) so the overlay stands out.
_PALETTE = [
    (255,   0, 255),   # magenta
    (  0, 255, 255),   # yellow
    (  0, 255,   0),   # bright green
    (  0, 165, 255),   # orange
    (180, 105, 255),   # hot pink
    ( 50, 205,  50),   # lime
    (255, 140,   0),   # deep sky blue
    (147,  20, 255),   # deep pink
]


def render_overlay(image_path, polylines, out_path: Optional[str] = None,
                    thickness: int = 3) -> bytes:
    """Render ``polylines`` on top of ``image_path`` and return JPEG bytes.

    Each polyline ``i`` is drawn in ``_PALETTE[i % 8]`` with a black
    outlined ``#i`` label at its leftmost endpoint.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    canvas = img.copy()
    for i, pts in enumerate(polylines):
        if pts is None or len(pts) < 2:
            continue
        arr = np.asarray(pts, dtype=np.int32).reshape(-1, 2)
        col = _PALETTE[i % len(_PALETTE)]
        cv2.polylines(canvas, [arr], False, col, thickness, cv2.LINE_AA)
        x0, y0 = int(arr[0, 0]), int(arr[0, 1])
        # double-stroke text: thick black outline + thin coloured fill
        cv2.putText(canvas, f"#{i}", (x0 + 6, y0),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, f"#{i}", (x0 + 6, y0),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("imencode failed")
    if out_path is not None:
        Path(out_path).write_bytes(buf.tobytes())
    return buf.tobytes()


@dataclass
class JudgeVerdict:
    image_id: str
    judge: str
    k_pred: int
    k_seen: Optional[int] = None
    per_curve_pass: list = field(default_factory=list)
    image_pass: Optional[bool] = None
    comment: str = ""
    raw: str = ""
    latency_s: float = 0.0
    error: Optional[str] = None


def _parse(raw: str, k_pred: int):
    if not raw:
        return None, [], None, "", "empty"
    txt = raw.strip()
    if "```" in txt:
        for p in txt.split("```"):
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                txt = p
                break
    s = txt.find("{")
    e = txt.rfind("}") + 1
    if s == -1 or e <= 0:
        return None, [], None, "", "no_json"
    try:
        d = json.loads(txt[s:e])
    except Exception as ex:
        return None, [], None, "", f"json:{type(ex).__name__}"
    k_seen = d.get("k_seen")
    if isinstance(k_seen, str) and k_seen.isdigit():
        k_seen = int(k_seen)
    if not isinstance(k_seen, int):
        k_seen = None
    per_pass = [False] * k_pred
    for entry in (d.get("per_curve") or []):
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry.get("id", -1))
        except Exception:
            continue
        if 0 <= idx < k_pred:
            v = entry.get("pass", False)
            if isinstance(v, str):
                v = v.strip().lower() in ("true", "yes", "pass", "1")
            per_pass[idx] = bool(v)
    img_pass = d.get("image_pass")
    if isinstance(img_pass, str):
        img_pass = img_pass.strip().lower() in ("true", "yes", "pass", "1")
    if img_pass is None and per_pass:
        img_pass = all(per_pass)
    return img_pass, per_pass, k_seen, str(d.get("comment", ""))[:200], None


def review(image_path, polylines, judge: str = "qwen-vl-max",
            timeout: float = 90.0, retries: int = 1,
            overlay_out: Optional[str] = None) -> JudgeVerdict:
    """Run a single judge on an image + polylines and return the verdict."""
    cfg = BACKENDS[judge]
    api_key = os.environ.get(cfg["key_env"])
    image_id = Path(image_path).stem
    k_pred = sum(1 for p in polylines if p is not None and len(p) >= 2)
    if not api_key:
        return JudgeVerdict(image_id=image_id, judge=judge, k_pred=k_pred,
                             error=f"missing {cfg['key_env']}")
    try:
        overlay = render_overlay(image_path, polylines, out_path=overlay_out)
    except Exception as e:
        return JudgeVerdict(image_id=image_id, judge=judge, k_pred=k_pred,
                             error=f"overlay:{type(e).__name__}:{e}")
    img_b64 = base64.b64encode(overlay).decode("ascii")
    client = OpenAI(api_key=api_key, base_url=cfg["base"], timeout=timeout)
    prompt = REVIEW_PROMPT.format(K=k_pred)
    last_err = None
    last_t0 = 0.0
    for attempt in range(retries + 1):
        last_t0 = time.monotonic()
        try:
            resp = client.chat.completions.create(
                model=cfg["model_id"],
                messages=[{"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ]}],
            )
            dt = time.monotonic() - last_t0
            raw = resp.choices[0].message.content or ""
            ip, pp, ks, cm, perr = _parse(raw, k_pred)
            if ip is None and attempt < retries:
                last_err = perr or "parse"
                time.sleep(1.0)
                continue
            return JudgeVerdict(
                image_id=image_id, judge=judge, k_pred=k_pred,
                k_seen=ks, per_curve_pass=pp, image_pass=ip,
                comment=cm, raw=raw[:500], latency_s=dt, error=perr,
            )
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(1.5)
    return JudgeVerdict(
        image_id=image_id, judge=judge, k_pred=k_pred,
        latency_s=time.monotonic() - last_t0, error=last_err,
    )


if __name__ == "__main__":
    """Smoke test: one image, one judge.

    Usage:  python vlm_reviewer.py <image.jpg> [judge]

    The polylines used are the ones the v4+TTA pipeline emits on that
    image (we re-run the pipeline locally so the smoke test does not
    depend on any cached prediction file).
    """
    import sys
    sys.path.insert(0, str(KMEANS_ROOT))
    from pipeline import run_pipeline                                    # noqa: E402
    from experiments.kmeans_matdeplot_adapter import (                  # noqa: E402
        mask_to_polyline,
    )

    img_path = sys.argv[1] if len(sys.argv) > 1 else \
        str(KMEANS_ROOT / "label" /
            "00d117323855716750e5a0c3245c3d5a9e63adbfa8cafc42050e98b6f1b65b78_1.jpg")
    judge = sys.argv[2] if len(sys.argv) > 2 else "qwen-vl-max"

    print(f"running pipeline on {Path(img_path).name} ...")
    r = run_pipeline(
        img_path,
        separator_mode="v4", morph_close_kernel=5, min_cluster_area=200,
        m2f_threshold=0.25, m2f_scales=(512, 768, 1024), flip_tta=True,
    )
    polylines = []
    for m in r.cluster_masks:
        arr = mask_to_polyline(m)
        if arr.shape[0] >= 10:
            polylines.append(arr.astype(np.int32))
    print(f"  pipeline emitted {len(polylines)} curves")

    overlay_out = str(KMEANS_ROOT / "experiments/results/iter/poc_overlay.jpg")
    print(f"calling judge={judge} (overlay -> {overlay_out}) ...")
    v = review(img_path, polylines, judge=judge, overlay_out=overlay_out)
    print(f"\n=== verdict ===")
    print(f"  judge          = {v.judge}")
    print(f"  k_pred         = {v.k_pred}")
    print(f"  k_seen (judge) = {v.k_seen}")
    print(f"  image_pass     = {v.image_pass}")
    print(f"  per_curve_pass = {v.per_curve_pass}")
    print(f"  comment        = {v.comment}")
    print(f"  latency        = {v.latency_s:.1f}s")
    print(f"  error          = {v.error}")
    print(f"  raw[:300]      = {v.raw[:300]!r}")
