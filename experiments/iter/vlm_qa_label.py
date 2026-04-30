"""Run the VLM y-lookup eval on label/ images, both Conditions A and B.

Inputs:
  - experiments/results/vlm_downstream/label_qa.json     (questions)
  - experiments/results/vlm_downstream/curves_cache/<id>.json  (curves for B)

Outputs:
  - llm_cache/<qid>__A.json     and  llm_cache/<qid>__B.json (one per call)
  - label_results.csv  (per-question results)
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(KMEANS_ROOT))

VLM_DIR = KMEANS_ROOT / "experiments" / "results" / "vlm_downstream"
QA_PATH = VLM_DIR / "label_qa.json"
CURVES_DIR = VLM_DIR / "curves_cache"
LLM_CACHE_DIR = VLM_DIR / "llm_cache"
RESULTS_CSV = VLM_DIR / "label_results.csv"
LABEL_DIR = KMEANS_ROOT / "label"

MODEL_ID = "gemini-3.1-pro-preview"
BASE_URL = "https://api.viviai.cc/v1"
MAX_TOKENS = 2048
TEMPERATURE = 0.0


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


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def make_prompt_A(q: dict) -> str:
    return (
        f"You are reading a scientific line chart. The image is provided.\n"
        f"QUESTION: What is the value of y on the LONGEST plotted curve at x = {q['x_star']:.4g}?\n"
        f"The visible x range is approximately [{q['x_min_axis']:.4g}, {q['x_max_axis']:.4g}] "
        f"and y range [{q['y_min_axis']:.4g}, {q['y_max_axis']:.4g}].\n"
        "Reply with strict JSON, no extra text:\n"
        "{\"y\": <number>}\n"
    )


def make_prompt_B(q: dict, curves_json: dict) -> str:
    base = (
        f"You are reading a scientific line chart. The image is provided.\n"
        f"QUESTION: What is the value of y on the LONGEST plotted curve at x = {q['x_star']:.4g}?\n"
        f"The visible x range is approximately [{q['x_min_axis']:.4g}, {q['x_max_axis']:.4g}] "
        f"and y range [{q['y_min_axis']:.4g}, {q['y_max_axis']:.4g}].\n"
        "\n"
        "We have also algorithmically extracted the curve(s) from this chart. The\n"
        "extraction is approximate (~65% pixel-IoU vs ground truth) but axis units\n"
        "are correct. Use it as a hint, but trust the image when they disagree.\n"
        "The curve labeled \"longest\" is the one to answer about.\n"
        "EXTRACTED CURVES (JSON):\n"
    )
    body = json.dumps(curves_json, ensure_ascii=False, separators=(",", ":"))
    tail = "\n\nReply with strict JSON, no extra text:\n{\"y\": <number>}\n"
    return base + body + tail


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def parse_y(raw: str) -> tuple[Optional[float], str]:
    """Returns (y_pred, parse_method). y_pred=None if parse fails."""
    if not raw:
        return None, "empty"
    txt = raw.strip()
    # 1. JSON parse
    s = txt.find("{"); e = txt.rfind("}") + 1
    if s != -1 and e > 0:
        try:
            d = json.loads(txt[s:e])
            v = d.get("y")
            if isinstance(v, (int, float)):
                return float(v), "json"
            if isinstance(v, str):
                m = _NUM_RE.search(v)
                if m:
                    return float(m.group()), "json_str"
        except Exception:
            pass
    # 2. regex first numeric
    m = _NUM_RE.search(txt)
    if m:
        return float(m.group()), "regex"
    return None, "fail"


def _client(timeout: float = 60.0):
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key, base_url=BASE_URL, timeout=timeout)


def _call_api(client, image_b64: str, prompt: str,
               retries: int = 3) -> tuple[str, dict]:
    last_err = None
    for attempt in range(retries):
        t0 = time.monotonic()
        try:
            kwargs = dict(
                model=MODEL_ID,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                        {"type": "text", "text": prompt},
                    ]},
                ],
            )
            try:
                kwargs["temperature"] = TEMPERATURE
            except Exception:
                pass
            resp = client.chat.completions.create(**kwargs)
            dt = time.monotonic() - t0
            raw = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            usage_d = {}
            if usage:
                usage_d = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
            return raw, {"latency_s": dt, "usage": usage_d, "attempt": attempt + 1, "error": None}
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < retries - 1:
                wait = 4 ** attempt  # 1, 4, 16
                time.sleep(wait)
    return "", {"latency_s": 0.0, "usage": {}, "attempt": retries, "error": last_err}


def cache_key(qid: str, condition: str) -> Path:
    return LLM_CACHE_DIR / f"{qid}__{condition}.json"


def run_one(client, qid: str, image_id: str, condition: str, q: dict,
             curves_json: Optional[dict]) -> dict:
    cache_path = cache_key(qid, condition)
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass

    img_path = LABEL_DIR / f"{image_id}.jpg"
    if not img_path.exists():
        return {"qid": qid, "image_id": image_id, "condition": condition,
                "error": f"image_missing: {img_path}"}
    image_b64 = _encode_image(img_path)
    if condition == "A":
        prompt = make_prompt_A(q)
    elif condition == "B":
        if curves_json is None:
            return {"qid": qid, "image_id": image_id, "condition": condition,
                    "error": "no_curve_for_B"}
        prompt = make_prompt_B(q, curves_json)
    else:
        return {"qid": qid, "image_id": image_id, "condition": condition,
                "error": f"unknown condition {condition}"}
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

    raw, meta = _call_api(client, image_b64, prompt)
    y_pred, parse_method = parse_y(raw)

    out = {
        "qid": qid, "image_id": image_id, "condition": condition,
        "x_star": q["x_star"], "y_gt": q["y_gt"],
        "raw_response": raw, "y_pred": y_pred, "parse_method": parse_method,
        "prompt_hash": prompt_hash,
        "latency_s": meta.get("latency_s"), "usage": meta.get("usage"),
        "attempts": meta.get("attempt"), "error": meta.get("error"),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--conditions", default="A,B")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap number of images (for pilot/sanity)")
    ap.add_argument("--limit_questions", type=int, default=0)
    args = ap.parse_args()

    if not QA_PATH.exists():
        print(f"QA missing: {QA_PATH}")
        return

    qa = json.loads(QA_PATH.read_text())
    image_ids = sorted(qa.keys())
    if args.limit > 0:
        image_ids = image_ids[:args.limit]

    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    client = _client()

    # build (image_id, qid, condition) tasks (deduped via cache)
    tasks = []
    for image_id in image_ids:
        qs = qa[image_id]
        if args.limit_questions > 0:
            qs = qs[:args.limit_questions]
        for q in qs:
            for cond in conditions:
                tasks.append((image_id, q, cond))

    # preload curves cache
    curves_cache: dict[str, Optional[dict]] = {}
    for image_id in image_ids:
        cp = CURVES_DIR / f"{image_id}.json"
        if cp.exists():
            try:
                curves_cache[image_id] = json.loads(cp.read_text())
            except Exception:
                curves_cache[image_id] = None
        else:
            curves_cache[image_id] = None

    print(f"Total {len(tasks)} (image, q, cond) tasks for {len(image_ids)} images.")
    n_done = 0; n_err = 0; n_cached = 0
    results: list[dict] = []

    def task_fn(payload):
        image_id, q, cond = payload
        qid = q["qid"]
        cj = curves_cache.get(image_id)
        return run_one(client, qid, image_id, cond, q, cj)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(task_fn, t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:
                r = {"error": f"future_exception: {e}\n{traceback.format_exc()[:300]}"}
            results.append(r)
            if r.get("error"):
                n_err += 1
            else:
                n_done += 1
            tot = n_done + n_err
            if tot % 25 == 0:
                print(f"  {tot}/{len(tasks)} done={n_done} err={n_err}")

    print(f"All done. ok={n_done} err={n_err}")
    print(f"Per-call cache at {LLM_CACHE_DIR}")
    print("Now run score_label.py to compute results CSV + summary.")


if __name__ == "__main__":
    main()
