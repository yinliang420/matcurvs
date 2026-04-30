"""多后端 LLM-K 预测器:在一张 chart 上问 LLM 有多少条线。

支持的后端:
    - DashScope (Qwen):     qwen-vl-max, qwen3-vl-plus, qwen3.5-plus
    - Vivi gateway (OAI):   gpt-5.4, claude-opus-4-6, gemini-3.1-pro-preview

统一返回 ``KPredResult(image_id, backend, k_pred, latency_s, raw, error)``。
"""
from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass, asdict
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

PROMPT = """You are analyzing a scientific chart image (line plot, scatter plot, or
mixed). Count how many distinct DATA lines/curves are plotted.

Counting rules (apply in order):
1. If a legend exists, the number of unique entries = number of lines.
2. Otherwise, count distinct trajectories/colors of plotted data lines.
3. EXCLUDE chart borders, gridlines, axis lines, axis ticks, baseline-only
   reference lines (zero-line, fit-line if labeled as such), error bars.
4. Treat overlapping lines as separate only if visually distinguishable.

Respond with strict JSON, no extra text:
{"total_lines_num": <integer>}
"""


BACKENDS = {
    # DashScope direct
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
    "qwen3.5-plus": {
        "model_id": "qwen3.5-plus",
        "base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "key_env": "DASHSCOPE_API_KEY",
    },
    # Vivi gateway (OpenAI compatible)
    "gpt-5.4": {
        "model_id": "gpt-5.4",
        "base": "https://api.viviai.cc/v1",
        "key_env": "OPENAI_API_KEY",
    },
    "claude-opus-4-6": {
        "model_id": "claude-opus-4-6",
        "base": "https://api.viviai.cc/v1",
        "key_env": "OPENAI_API_KEY",
    },
    "gemini-3.1-pro-preview": {
        "model_id": "gemini-3.1-pro-preview",
        "base": "https://api.viviai.cc/v1",
        "key_env": "OPENAI_API_KEY",
    },
}


@dataclass
class KPredResult:
    image_id: str
    backend: str
    k_pred: Optional[int]
    latency_s: float
    raw: str
    error: Optional[str] = None


def _encode_image(img_path: Path) -> str:
    return base64.b64encode(Path(img_path).read_bytes()).decode("ascii")


def predict_k(image_path: str | Path, backend: str,
              timeout: float = 60.0, retries: int = 1) -> KPredResult:
    cfg = BACKENDS[backend]
    api_key = os.environ.get(cfg["key_env"])
    image_id = Path(image_path).stem
    if not api_key:
        return KPredResult(image_id=image_id, backend=backend, k_pred=None,
                           latency_s=0.0, raw="",
                           error=f"missing env {cfg['key_env']}")
    client = OpenAI(api_key=api_key, base_url=cfg["base"], timeout=timeout)
    img_b64 = _encode_image(Path(image_path))
    last_err = None
    for attempt in range(retries + 1):
        t0 = time.monotonic()
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
            dt = time.monotonic() - t0
            raw = resp.choices[0].message.content or ""
            k = _parse_k(raw)
            return KPredResult(image_id=image_id, backend=backend, k_pred=k,
                               latency_s=dt, raw=raw[:300],
                               error=None if k is not None else "parse_failed")
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(1.0)
    return KPredResult(image_id=image_id, backend=backend, k_pred=None,
                       latency_s=time.monotonic() - t0, raw="", error=last_err)


def _parse_k(raw: str) -> Optional[int]:
    if not raw:
        return None
    txt = raw.strip()
    # strip markdown fences
    if "```" in txt:
        # take inner block
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
        return None
    try:
        d = json.loads(txt[s:e])
        if "total_lines_num" in d:
            v = d["total_lines_num"]
            if isinstance(v, int):
                return v
            if isinstance(v, str) and v.isdigit():
                return int(v)
    except Exception:
        pass
    return None


def predict_k_all_backends(image_path: str | Path,
                           backends: Optional[list[str]] = None
                           ) -> list[KPredResult]:
    if backends is None:
        backends = list(BACKENDS.keys())
    out = []
    for be in backends:
        r = predict_k(image_path, be)
        out.append(r)
    return out


if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else \
        str(KMEANS_ROOT / "label" /
            "00d117323855716750e5a0c3245c3d5a9e63adbfa8cafc42050e98b6f1b65b78_1.jpg")
    for r in predict_k_all_backends(img):
        print(f"  {r.backend:30s}  K={r.k_pred}  t={r.latency_s:.1f}s  err={r.error}")
