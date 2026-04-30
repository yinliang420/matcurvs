"""
L2 curation: judge MatDeplot extractions on /Volumes/yinliang/数据库项目/data_base/battery_*/
with Qwen thinking mode, copy accepted ones into data_base/L2/.

Layout assumed:
  data_base/battery_NN/<sha>/images/<panel>.jpg
  data_base/battery_NN/<sha>/images/<panel>_Line_<i>.xlsx   (one per extracted curve)

Output:
  data_base/L2/<sha>/<panel>.jpg
  data_base/L2/<sha>/<panel>_Line_<i>.xlsx
  data_base/L2/judgment_log.jsonl    (one record per panel; resumable)

Usage:
  python experiments/l2_curate.py --workers 30
  python experiments/l2_curate.py --limit 20 --workers 8        # smoke test
  python experiments/l2_curate.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from openai import OpenAI

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[1]))
if str(KMEANS_ROOT) not in sys.path:
    sys.path.insert(0, str(KMEANS_ROOT))
from ModalLLMResponse import ModalLLMResponse, ThinkingModalLLMResponse  # noqa: F401  (kept for --thinking branch)

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise RuntimeError("DASHSCOPE_API_KEY not set; create .env from .env.example")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# DATA_BASE points at the source dataset of curated chart images. Override via env.
DATA_BASE = Path(os.environ.get("DATA_BASE", str(KMEANS_ROOT.parent / "data_base")))
DEFAULT_L2_OUT = Path(os.environ.get("L2_OUT", str(KMEANS_ROOT / "data_base_L2")))
DEFAULT_MODEL = "qwen-vl-max"  # multimodal, stable, fast; --thinking switches to qwen3-vl-plus thinking-mode

JUDGE_PROMPT = """You are auditing an automatic curve-extraction system on a scientific line chart from a materials-science article.

You receive TWO images:
- IMAGE 1 (the ORIGINAL chart): a published scientific line chart.
- IMAGE 2 (the EXTRACTION):     the system's reconstruction. Each curve is plotted as a colored polyline labelled Line_1, Line_2, ...

Follow these steps strictly:

STEP 1. Count distinct DATA curves on IMAGE 1 (ignore axis lines, gridlines, legend boxes, error-bar caps, isolated scatter points without a clear connecting line). Call this n_left.
STEP 2. Count distinct colored polylines on IMAGE 2. Call this n_right.
STEP 3. For each curve on IMAGE 2, find its best match on IMAGE 1 and judge whether the overall morphology (rises/falls, peak positions, plateaus, slopes) matches.
STEP 4. Judge whether IMAGE 2's axis range is plausible for the chart family suggested by IMAGE 1 (XRD: 2θ in 5°-90°; Raman: shift in 100-4000 cm⁻¹; XAFS: energy in eV; GCD: voltage 0-5 V vs capacity in mAh/g; CV: current vs voltage 0-5 V; EIS: Z' vs Z''; XPS: binding energy 0-1300 eV; if a non-standard chart, judge plausibility against IMAGE 1's axes).
STEP 5. Judge whether IMAGE 2 is free of spurious extras, broken segments, dramatic kinks, or wildly off-shape artifacts.

HARD RULES:
- If IMAGE 2 is empty / blank / contains zero polylines, set decision="bad".
- If n_right != n_left, set curve_count_match=false (decision usually "bad").
- decision="good" ONLY IF curve_count_match AND shape_match AND range_plausible AND noise_free are ALL true. Otherwise "bad".
- When in doubt, prefer "bad". A false-good costs more than a false-bad.

Return STRICT JSON only, no extra prose, no markdown fences:
{
  "n_left": <integer>,
  "n_right": <integer>,
  "decision": "good" | "bad",
  "curve_count_match": true | false,
  "shape_match": true | false,
  "range_plausible": true | false,
  "noise_free": true | false,
  "reason": "<one short sentence>"
}
"""


def find_panel_groups(data_base: Path) -> list[dict]:
    """Discover (image, extraction-files) groups across two formats and two locations.

    Format A (battery_53/54/55):  <panel>.jpg + <panel>_Line_<i>.xlsx (one xlsx per curve, 2 cols).
    Format B (data_base/t/battery_*): <panel>.jpg + <panel>.xlsx (one xlsx, paired <label>_X/_Y columns).
    """
    groups: list[dict] = []
    battery_roots: list[tuple[Path, str]] = []
    for b in sorted(data_base.glob("battery_*")):
        if b.is_dir():
            battery_roots.append((b, b.name))
    t_dir = data_base / "t"
    if t_dir.is_dir():
        for b in sorted(t_dir.glob("battery_*")):
            if b.is_dir():
                battery_roots.append((b, f"t/{b.name}"))

    for battery_dir, battery_label in battery_roots:
        for sha_dir in sorted(battery_dir.iterdir()):
            if not sha_dir.is_dir():
                continue
            img_dir = sha_dir / "images"
            if not img_dir.exists():
                continue
            for jpg in sorted(img_dir.glob("*.jpg")):
                lines_a = sorted(img_dir.glob(f"{jpg.stem}_Line_*.xlsx"))
                xlsx_b = img_dir / f"{jpg.stem}.xlsx"
                if lines_a:
                    fmt = "A"
                    xlsx_paths = lines_a
                elif xlsx_b.exists():
                    fmt = "B"
                    xlsx_paths = [xlsx_b]
                else:
                    continue
                groups.append({
                    "battery": battery_label,
                    "sha": sha_dir.name,
                    "panel": jpg.stem,
                    "img_path": jpg,
                    "xlsx_paths": xlsx_paths,
                    "fmt": fmt,
                })
    return groups


def parse_curves(xlsx_paths: list[Path], fmt: str) -> list[tuple[str, "np.ndarray", "np.ndarray"]]:
    """Return list of (label, x_array, y_array) regardless of file layout."""
    curves: list[tuple[str, "np.ndarray", "np.ndarray"]] = []
    if fmt == "A":
        for xp in xlsx_paths:
            try:
                df = pd.read_excel(xp)
                if df.shape[1] < 2 or len(df) < 2:
                    continue
                x = pd.to_numeric(df.iloc[:, 0], errors="coerce").values
                y = pd.to_numeric(df.iloc[:, 1], errors="coerce").values
                mask = ~(pd.isna(x) | pd.isna(y))
                if mask.sum() < 2:
                    continue
                label = xp.stem.split("_Line_")[-1]
                curves.append((f"Line_{label}", x[mask], y[mask]))
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
                    x = pd.to_numeric(df[c], errors="coerce").values
                    y = pd.to_numeric(df[yc], errors="coerce").values
                    mask = ~(pd.isna(x) | pd.isna(y))
                    if mask.sum() < 2:
                        continue
                    curves.append((base, x[mask], y[mask]))
            except Exception:
                continue
    return curves


def make_key(g: dict) -> str:
    return f"{g['battery']}/{g['sha']}/{g['panel']}"


def load_done_keys(log_file: Path) -> set[str]:
    done: set[str] = set()
    if not log_file.exists():
        return done
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if "key" in rec:
                    done.add(rec["key"])
            except Exception:
                continue
    return done


def render_extraction_only(curves: list[tuple[str, "np.ndarray", "np.ndarray"]]) -> bytes:
    """Render just the extracted curves as a clean matplotlib plot. Returns PNG bytes."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    if not curves:
        ax.text(0.5, 0.5, "no curves parsed from xlsx", ha="center", va="center",
                transform=ax.transAxes, color="red", fontsize=14)
    for i, (label, x, y) in enumerate(curves):
        ax.plot(x, y, color=cmap(i % 10), linewidth=1.8, label=label)
    ax.set_title(f"Extracted: {len(curves)} curve{'s' if len(curves) != 1 else ''}")
    if curves:
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def call_qwen_two_image(
    client: OpenAI,
    model_name: str,
    img1_b64: str,
    img1_mime: str,
    img2_b64: str,
    img2_mime: str,
    prompt: str,
) -> dict:
    """Send TWO images + prompt to a Qwen multimodal model. Returns parsed JSON dict."""
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{img1_mime};base64,{img1_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:{img2_mime};base64,{img2_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    msg = completion.choices[0].message.content
    start = msg.find("{")
    end = msg.rfind("}") + 1
    if start == -1 or end == 0:
        return {"error": "no JSON in response", "raw": msg[:200]}
    try:
        return json.loads(msg[start:end].strip())
    except json.JSONDecodeError:
        return {"error": "JSON parse fail", "raw": msg[start:end][:200]}


def copy_to_l2(g: dict, l2_root: Path, render_png: bytes | None = None) -> None:
    # Namespace by battery to avoid sha collisions across t/ vs top-level batteries.
    safe_battery = g["battery"].replace("/", "__")
    out_dir = l2_root / f"{safe_battery}__{g['sha']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(g["img_path"], out_dir / g["img_path"].name)
    for xp in g["xlsx_paths"]:
        shutil.copy2(xp, out_dir / xp.name)
    if render_png is not None:
        (out_dir / f"{g['panel']}_extraction.png").write_bytes(render_png)


async def judge_one(
    g: dict,
    client: OpenAI,
    model_name: str,
    sem: asyncio.Semaphore,
) -> dict:
    key = make_key(g)
    async with sem:
        t0 = time.time()
        png_bytes: bytes | None = None
        n_curves = 0
        try:
            curves = await asyncio.to_thread(parse_curves, g["xlsx_paths"], g["fmt"])
            n_curves = len(curves)
            if n_curves == 0:
                return {"key": key, "battery": g["battery"], "sha": g["sha"], "panel": g["panel"],
                        "fmt": g["fmt"], "n_curves": 0, "decision": "bad",
                        "reason": "no curves parsed from xlsx",
                        "elapsed_s": round(time.time() - t0, 2)}
            with open(g["img_path"], "rb") as f:
                img1_b64 = base64.b64encode(f.read()).decode("utf-8")
            png_bytes = await asyncio.to_thread(render_extraction_only, curves)
            img2_b64 = base64.b64encode(png_bytes).decode("utf-8")
            result = await asyncio.to_thread(
                call_qwen_two_image,
                client, model_name, img1_b64, "image/jpeg", img2_b64, "image/png", JUDGE_PROMPT,
            )
        except Exception as e:
            return {"key": key, "battery": g["battery"], "sha": g["sha"], "panel": g["panel"],
                    "fmt": g["fmt"], "n_curves": n_curves, "decision": "bad",
                    "_render": png_bytes,
                    "error": f"runner: {type(e).__name__}: {e}",
                    "elapsed_s": round(time.time() - t0, 2)}

        decision = (result.get("decision") or "bad").strip().lower()
        if decision not in {"good", "bad"}:
            decision = "bad"
        return {
            "key": key,
            "battery": g["battery"],
            "sha": g["sha"],
            "panel": g["panel"],
            "fmt": g["fmt"],
            "n_curves": n_curves,
            "n_left": result.get("n_left"),
            "n_right": result.get("n_right"),
            "decision": decision,
            "curve_count_match": result.get("curve_count_match"),
            "shape_match": result.get("shape_match"),
            "range_plausible": result.get("range_plausible"),
            "noise_free": result.get("noise_free"),
            "reason": result.get("reason", ""),
            "error": result.get("error"),
            "elapsed_s": round(time.time() - t0, 2),
            "_render": png_bytes,
        }


async def main_async(args: argparse.Namespace) -> None:
    l2_out: Path = args.out
    log_file = l2_out / "judgment_log.jsonl"
    l2_out.mkdir(parents=True, exist_ok=True)
    groups = find_panel_groups(DATA_BASE)
    done = load_done_keys(log_file)
    todo = [g for g in groups if make_key(g) not in done]
    if args.limit:
        todo = todo[: args.limit]

    print(f"[scan] total groups: {len(groups):,} | already done: {len(done):,} | todo: {len(todo):,}")
    print(f"[scan] L2 output dir: {l2_out}")
    if args.dry_run or not todo:
        return

    client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
    print(f"[init] DashScope multi-image API model={args.model}, workers={args.workers}")
    sem = asyncio.Semaphore(args.workers)
    log_lock = asyncio.Lock()

    counters = {"good": 0, "bad": 0, "started": 0}

    async def runner(g: dict) -> dict:
        rec = await judge_one(g, client, args.model, sem)
        png = rec.pop("_render", None)
        async with log_lock:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if rec.get("decision") == "good":
            try:
                await asyncio.to_thread(copy_to_l2, g, l2_out, png)
            except Exception as e:
                print(f"[copy err] {rec['key']}: {e}")
        return rec

    tasks = [asyncio.create_task(runner(g)) for g in todo]
    t_start = time.time()
    for i, t in enumerate(asyncio.as_completed(tasks), 1):
        rec = await t
        d = rec.get("decision", "bad")
        counters[d] = counters.get(d, 0) + 1
        if i == 1 or i % 25 == 0 or i == len(tasks):
            elapsed = time.time() - t_start
            rate = i / max(elapsed, 1e-6)
            eta = (len(tasks) - i) / max(rate, 1e-6)
            print(f"[{i}/{len(tasks)}] good={counters['good']} bad={counters['bad']} | "
                  f"rate={rate:.2f}/s eta={eta/60:.1f}min | latest={d} ({rec.get('reason', '')[:60]})")

    print(f"\n[done] accepted to L2: {counters['good']}/{len(tasks)} "
          f"({counters['good']/max(len(tasks),1)*100:.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16, help="concurrent in-flight requests")
    ap.add_argument("--limit", type=int, default=None, help="cap todo for smoke test")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="DashScope model name")
    ap.add_argument("--out", type=Path, default=DEFAULT_L2_OUT, help="L2 output dir (defaults to local SSD because data_base volume is read-only)")
    ap.add_argument("--dry-run", action="store_true", help="just scan, don't call API")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
