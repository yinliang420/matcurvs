"""
MatCurvs-Reasoning runner: VLM/LLM evaluation across three reasoning modes.

Modes
-----
M1  VLM image-only    : prompt = question;  input = chart image only
M2  VLM image + data  : prompt = question + downsampled (x,y) text; input = image + text
M3  LLM text-only     : prompt = question + downsampled (x,y) text; NO image

For M1/M2 the model must be vision-capable (qwen-vl-max, gpt-5.4, gemini-3.1-pro-preview).
For M3 use a strong text LLM (gpt-5.4, claude-opus-4-6, qwen3-max).

Routing: dashscope models go through DASHSCOPE_API_BASE; gpt/claude/gemini go
through the viviai gateway.

Usage:
  python mr_runner.py --set rigorous --method m1 --model qwen-vl-max --workers 8
  python mr_runner.py --set rigorous --method m2 --model gpt-5.4   --workers 4
  python mr_runner.py --set rigorous --method m3 --model qwen3-max --workers 8
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

# ---------- env / config -----------------------------------------------------

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[1]))
EXP_DIR = KMEANS_ROOT / "experiments"
ANS_DIR = EXP_DIR / "mr_answers"
ANS_DIR.mkdir(parents=True, exist_ok=True)

# Manual .env parser (avoid dotenv dependency)
def _load_env_file(p: Path):
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)
_load_env_file(KMEANS_ROOT / ".env")

DASHSCOPE_KEY = os.environ.get("DASHSCOPE_API_KEY")
DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# OpenAI / Anthropic via the OpenAI-compatible SDK. OPENAI_BASE_URL lets you
# route through a third-party gateway (e.g. dmxapi) by setting it in .env.
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Gemini uses the official google-genai SDK. GEMINI_BASE_URL lets you route
# through a gateway; otherwise the SDK talks to Google directly.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GEMINI_BASE_URL = os.environ.get("GEMINI_BASE_URL")

# ---------- model -> endpoint routing ----------------------------------------

DASHSCOPE_MODELS = {"qwen-vl-max", "qwen3-vl-plus", "qwen3-vl-max", "qwen3-max", "qwen3.5-plus", "qwen-vl-plus"}
OPENAI_MODELS = {"gpt-5.4", "gpt-5.4-mini", "claude-opus-4-6"}
GEMINI_MODELS = {"gemini-3.1-pro-preview", "gemini-3.1-pro"}


def model_protocol(model: str) -> str:
    """Returns 'dashscope' / 'openai' / 'gemini'."""
    if model in DASHSCOPE_MODELS:
        return "dashscope"
    if model in OPENAI_MODELS:
        return "openai"
    if model in GEMINI_MODELS:
        return "gemini"
    raise ValueError(f"unknown model: {model}")


def _require_key(key: str | None, env_name: str):
    if not key:
        raise RuntimeError(f"{env_name} not set; create .env from .env.example")


def make_client(model: str):
    proto = model_protocol(model)
    if proto == "dashscope":
        _require_key(DASHSCOPE_KEY, "DASHSCOPE_API_KEY")
        return ("openai", OpenAI(api_key=DASHSCOPE_KEY, base_url=DASHSCOPE_BASE))
    if proto == "openai":
        # Use ANTHROPIC_API_KEY when calling claude-* models; otherwise OPENAI_API_KEY.
        if model.startswith("claude-"):
            _require_key(ANTHROPIC_KEY, "ANTHROPIC_API_KEY")
            api_key = ANTHROPIC_KEY
        else:
            _require_key(OPENAI_KEY, "OPENAI_API_KEY")
            api_key = OPENAI_KEY
        return ("openai", OpenAI(api_key=api_key, base_url=OPENAI_BASE))
    if proto == "gemini":
        _require_key(GOOGLE_API_KEY, "GOOGLE_API_KEY")
        return ("gemini", None)  # google-genai client built lazily inside the call
    raise ValueError(f"unknown model: {model}")


def vision_capable(model: str) -> bool:
    return model in {"qwen-vl-max", "qwen3-vl-plus", "qwen3-vl-max",
                     "gpt-5.4", "claude-opus-4-6",
                     "gemini-3.1-pro-preview", "gemini-3.1-pro"}


# ---------- curve loading + downsampling -------------------------------------

def parse_curves_from_paths(xlsx_paths: list[str], fmt: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    curves: list[tuple[str, np.ndarray, np.ndarray]] = []
    if fmt == "A":
        for xp in xlsx_paths:
            try:
                df = pd.read_excel(xp)
                if df.shape[1] < 2 or len(df) < 2:
                    continue
                x = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
                y = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
                m = ~(np.isnan(x) | np.isnan(y))
                if m.sum() < 2:
                    continue
                order = np.argsort(x[m])
                label = Path(xp).stem.split("_Line_")[-1]
                curves.append((f"Line_{label}", x[m][order], y[m][order]))
            except Exception:
                continue
    elif fmt == "B":
        for xp in xlsx_paths:
            try:
                df = pd.read_excel(xp)
                cols = [str(c) for c in df.columns]
                for c in cols:
                    if not c.endswith("_X"):
                        continue
                    base = c[:-2]
                    yc = f"{base}_Y"
                    if yc not in cols:
                        continue
                    x = pd.to_numeric(df[c], errors="coerce").to_numpy()
                    y = pd.to_numeric(df[yc], errors="coerce").to_numpy()
                    m = ~(np.isnan(x) | np.isnan(y))
                    if m.sum() < 2:
                        continue
                    order = np.argsort(x[m])
                    curves.append((base, x[m][order], y[m][order]))
            except Exception:
                continue
    return curves


def downsample(x: np.ndarray, y: np.ndarray, n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= n:
        return x, y
    idx = np.linspace(0, len(x) - 1, n).round().astype(int)
    return x[idx], y[idx]


def curves_to_text(curves, max_pts_per_curve: int = 500) -> str:
    lines = []
    for label, x, y in curves:
        xd, yd = downsample(x, y, max_pts_per_curve)
        # compact format: one point per line, 5 sig figs
        pts = ", ".join(f"({xx:.5g}, {yy:.5g})" for xx, yy in zip(xd, yd))
        lines.append(f"Curve '{label}' (n={len(xd)} points):")
        lines.append(f"  {pts}")
    return "\n".join(lines)


# ---------- prompt assembly --------------------------------------------------

SYSTEM_M1 = ("You are reading a scientific line chart from a materials-science article. "
             "The user will ask a quantitative question about a curve in the image. "
             "Return STRICT JSON with a single field 'answer' whose value is a number "
             "(or a list of numbers if the question asks for multiple values). "
             "No prose, no units, no other fields.")

SYSTEM_M2 = ("You are reading a scientific line chart from a materials-science article. "
             "You are given BOTH the original image AND the structured (x,y) data points "
             "extracted from each curve. The user will ask a quantitative question. "
             "Use the structured data to compute precise answers. "
             "Return STRICT JSON with a single field 'answer' whose value is a number "
             "(or a list of numbers). No prose, no units, no other fields.")

SYSTEM_M3 = ("You are an agent answering a quantitative question about a scientific curve. "
             "You receive ONLY the structured (x,y) data of the curves (no image). "
             "Compute a precise numerical answer from the data. "
             "Return STRICT JSON with a single field 'answer' whose value is a number "
             "(or a list of numbers). No prose, no units, no other fields.")


# ---------- API call ---------------------------------------------------------

def _parse_answer(text: str):
    """Extract the 'answer' value from the model's JSON output."""
    if not text:
        return None
    # try direct JSON
    s = text.strip()
    # remove markdown fences
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    start = s.find("{")
    end = s.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        d = json.loads(s[start:end])
    except Exception:
        return None
    if "answer" in d:
        return d["answer"]
    # tolerate variations
    for k in ("value", "result", "y", "ans"):
        if k in d:
            return d[k]
    return None


def _gemini_native_call(model: str, method: str, image_path: str | None,
                        question: str, data_text: str | None) -> dict:
    """Gemini call via the official google-genai SDK.

    Defaults to direct Google endpoints. Set GEMINI_BASE_URL in .env to route
    through a third-party gateway (the SDK forwards it to its HTTP client).
    """
    t0 = time.time()
    try:
        # Lazy import so users who never call gemini don't need the dependency.
        from google import genai
        from google.genai import types as genai_types

        client_kwargs = {"api_key": GOOGLE_API_KEY}
        if GEMINI_BASE_URL:
            # SDK accepts http_options.base_url for routing through a gateway.
            client_kwargs["http_options"] = genai_types.HttpOptions(base_url=GEMINI_BASE_URL)
        client = genai.Client(**client_kwargs)

        contents: list = []
        if method in ("m1", "m2"):
            assert image_path is not None
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            mime = "image/png" if str(image_path).lower().endswith(".png") else "image/jpeg"
            contents.append(genai_types.Part.from_bytes(data=image_bytes, mime_type=mime))

        if method == "m1":
            contents.append(
                f"{SYSTEM_M1}\n\nQuestion: {question}\n\n"
                f'Return JSON {{"answer": <number-or-list>}} only.'
            )
        elif method == "m2":
            contents.append(
                f"{SYSTEM_M2}\n\nBelow is the structured (x, y) data:\n\n{data_text}\n\n"
                f"Question: {question}\n\n"
                f'Return JSON {{"answer": <number-or-list>}} only.'
            )
        elif method == "m3":
            contents.append(
                f"{SYSTEM_M3}\n\nBelow is the structured (x, y) data:\n\n{data_text}\n\n"
                f"Question: {question}\n\n"
                f'Return JSON {{"answer": <number-or-list>}} only.'
            )

        resp = client.models.generate_content(model=model, contents=contents)
        text = getattr(resp, "text", None) or ""
        ans = _parse_answer(text)
        return {"answer": ans, "raw_text": text[:500],
                "error": None if ans is not None else "no_answer_parsed",
                "elapsed_s": round(time.time() - t0, 2)}
    except Exception as e:
        return {"answer": None, "raw_text": None,
                "error": f"{type(e).__name__}: {str(e)[:200]}",
                "elapsed_s": round(time.time() - t0, 2)}


def call_api(
    client_tuple,
    model: str,
    method: str,
    image_path: str | None,
    question: str,
    data_text: str | None,
) -> dict:
    """One API call. Returns dict {answer, raw_text, error, elapsed}."""
    proto, client = client_tuple
    if proto == "gemini":
        return _gemini_native_call(model, method, image_path, question, data_text)

    t0 = time.time()
    try:
        if method == "m1":
            assert image_path is not None
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            mime = "image/png" if str(image_path).lower().endswith(".png") else "image/jpeg"
            messages = [
                {"role": "system", "content": SYSTEM_M1},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": f"Question: {question}\n\nReturn JSON {{\"answer\": <number-or-list>}} only."},
                ]},
            ]
        elif method == "m2":
            assert image_path is not None and data_text is not None
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            mime = "image/png" if str(image_path).lower().endswith(".png") else "image/jpeg"
            user_text = (f"Below is the structured (x, y) data extracted from each curve in the image:\n\n"
                         f"{data_text}\n\nQuestion: {question}\n\n"
                         f"Use the data to compute the answer precisely. "
                         f"Return JSON {{\"answer\": <number-or-list>}} only.")
            messages = [
                {"role": "system", "content": SYSTEM_M2},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": user_text},
                ]},
            ]
        elif method == "m3":
            assert data_text is not None
            user_text = (f"Below is the structured (x, y) data extracted from each curve:\n\n"
                         f"{data_text}\n\nQuestion: {question}\n\n"
                         f"Compute the answer precisely from the data. "
                         f"Return JSON {{\"answer\": <number-or-list>}} only.")
            messages = [
                {"role": "system", "content": SYSTEM_M3},
                {"role": "user", "content": user_text},
            ]
        else:
            return {"answer": None, "error": f"unknown method {method}", "elapsed_s": 0}

        resp = client.chat.completions.create(model=model, messages=messages)
        text = resp.choices[0].message.content or ""
        ans = _parse_answer(text)
        return {
            "answer": ans,
            "raw_text": text[:500],
            "error": None if ans is not None else "no_answer_parsed",
            "elapsed_s": round(time.time() - t0, 2),
        }
    except Exception as e:
        return {"answer": None, "raw_text": None,
                "error": f"{type(e).__name__}: {str(e)[:200]}",
                "elapsed_s": round(time.time() - t0, 2)}


# ---------- main loop --------------------------------------------------------

async def run_set(args: argparse.Namespace) -> None:
    set_path = EXP_DIR / f"mr_set_{args.set}.jsonl"
    if not set_path.exists():
        raise FileNotFoundError(set_path)
    items = [json.loads(l) for l in open(set_path)]
    if args.limit:
        items = items[:args.limit]

    out_path = args.output or (ANS_DIR / f"{args.set}_{args.method}_{args.model}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: read which item_ids are done
    done_ids: set[str] = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                d = json.loads(line)
                if "item_id" in d:
                    done_ids.add(d["item_id"])
            except Exception:
                pass
    todo = [it for it in items if it["item_id"] not in done_ids]

    print(f"[init] set={args.set} method={args.method} model={args.model}")
    print(f"  total items: {len(items)} | already done: {len(done_ids)} | todo: {len(todo)}")
    print(f"  output -> {out_path}")
    if not todo:
        return

    if args.method != "m3" and not vision_capable(args.model):
        raise SystemExit(f"model {args.model} is not vision-capable; M1/M2 require a VLM")

    client = make_client(args.model)
    sem = asyncio.Semaphore(args.workers)
    log_lock = asyncio.Lock()

    async def runner(it: dict) -> dict:
        async with sem:
            data_text = None
            if args.method in ("m2", "m3"):
                # load curves and downsample
                xlsx = it.get("l2_xlsx_paths") or []
                fmt = it.get("fmt", "A")
                curves = await asyncio.to_thread(parse_curves_from_paths, xlsx, fmt)
                if not curves:
                    return {"item_id": it["item_id"], "answer": None,
                            "error": "no_curves_loaded", "elapsed_s": 0,
                            "model": args.model, "method": args.method}
                data_text = await asyncio.to_thread(curves_to_text, curves, args.points)
            res = await asyncio.to_thread(
                call_api, client, args.model, args.method,
                it.get("image_path"), it["question"], data_text,
            )
            res["item_id"] = it["item_id"]
            res["model"] = args.model
            res["method"] = args.method
            res["gt_answer"] = it["gt_answer"]
            res["task"] = it["task"]
            res["family"] = it.get("family", "")
            res["level"] = it["level"]
        async with log_lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
        return res

    tasks = [asyncio.create_task(runner(it)) for it in todo]
    n_ok = n_err = 0
    t_start = time.time()
    for i, t in enumerate(asyncio.as_completed(tasks), 1):
        r = await t
        if r.get("answer") is None:
            n_err += 1
        else:
            n_ok += 1
        if i == 1 or i % 20 == 0 or i == len(tasks):
            el = time.time() - t_start
            rate = i / max(el, 1e-9)
            eta = (len(tasks) - i) / max(rate, 1e-9) / 60
            print(f"[{i}/{len(tasks)}] ok={n_ok} err={n_err} | rate={rate:.2f}/s eta={eta:.1f}min")
    print(f"[done] ok={n_ok} err={n_err} | output -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["rigorous", "scaled", "full"], required=True)
    ap.add_argument("--method", choices=["m1", "m2", "m3"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--points", type=int, default=500, help="max points per curve sent in M2/M3")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    asyncio.run(run_set(args))


if __name__ == "__main__":
    main()
