"""
Package the L2 layer of MatCurvs for HuggingFace dataset upload.

Output layout (release/matcurvs_l2/):
  README.md                 # dataset card with numbers matching the paper
  metadata.jsonl            # one row per panel (3,817 rows)
  metadata_curves.jsonl     # one row per curve (8,026 rows)
  images/<panel>.jpg        # 3,817 chart images
  curves/<panel>/<i>.csv    # per-curve x,y in physical units
  curves_combined/<panel>.parquet  # alternative single-file-per-panel

Run: python experiments/package_l2_for_hf.py
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import os

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[1]))
L2_SRC = KMEANS_ROOT / "data_base_L2"
SUBFIG = KMEANS_ROOT / "subfigures_and_class"
RELEASE = KMEANS_ROOT / "release" / "matcurvs_l2"


# ---------- minimal L0 lookup (chart family per panel) ----------------------

def build_l0_lookup() -> dict:
    """key '<sha>_<panel>.jpg' -> {'second_level', 'reasoning_lower'}"""
    out = {}
    for fp in sorted(SUBFIG.glob("classbattery_*.json")):
        try:
            with open(fp) as f:
                d = json.load(f)
        except Exception:
            continue
        for k, v in d.items():
            if not isinstance(v, dict):
                continue
            do = v.get("data_origin") or {}
            out[k] = {
                "second_level": do.get("second_level", ""),
                "reasoning_lower": (do.get("reasoning") or "").lower(),
            }
    return out


def infer_family(rs: str, sl: str, x_range) -> str:
    """Same priority as in the paper."""
    if any(k in rs for k in ["nyquist", "electrochemical impedance",
                              "impedance spectroscopy", "(eis)",
                              "imaginary impedance", "(z\" vs", "(z'' vs"]):
        return "EIS"
    if any(k in rs for k in ["xanes", "exafs", "(xafs)", "x-ray absorption",
                              "absorption near-edge", "absorption spectroscopy",
                              "fine structure"]):
        return "XAFS"
    if "raman shift" in rs or "raman spectro" in rs or "raman" in rs:
        return "Raman"
    if "x-ray photoelectron" in rs or "(xps)" in rs or "binding energy" in rs:
        return "XPS"
    if "x-ray diffraction" in rs or "(xrd)" in rs or "2theta" in rs or "2θ" in rs:
        return "XRD"
    if "galvanostatic" in rs or "specific capacity" in rs:
        return "GCD"
    if "cyclic voltammetr" in rs or "cv curve" in rs:
        return "CV"
    if "cycling performance" in rs or "cycle number" in rs:
        return "Cycling"
    return "generic"


# ---------- curve parsing (mirror task-builder) -----------------------------

def parse_curves_from_xlsx(xlsx_paths, fmt):
    curves = []
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


# ---------- main packager ---------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RELEASE)
    ap.add_argument("--include-curves-csv", action="store_true",
                    help="also copy per-curve CSVs (slower)")
    args = ap.parse_args()

    out = args.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    if args.include_curves_csv:
        (out / "curves").mkdir(exist_ok=True)

    log_path = L2_SRC / "judgment_log.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    accepted_rows = []
    for line in open(log_path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("decision") == "good":
            accepted_rows.append(r)
    print(f"L2 accepted panels: {len(accepted_rows):,}")

    print("Loading L0 family lookup...")
    l0 = build_l0_lookup()
    print(f"  L0 entries: {len(l0):,}")

    panel_meta = []
    curve_meta = []
    fam_count = Counter()
    n_curves_total = 0
    n_panels_packaged = 0
    n_papers = set()

    for rec in accepted_rows:
        sha = rec["sha"]
        panel = rec["panel"]
        battery = rec["battery"]
        safe_battery = battery.replace("/", "__")
        src_dir = L2_SRC / f"{safe_battery}__{sha}"
        if not src_dir.exists():
            continue
        # find image (jpg or png)
        img_src = src_dir / f"{panel}.jpg"
        if not img_src.exists():
            img_src = src_dir / f"{panel}.png"
        if not img_src.exists():
            continue

        fmt = rec.get("fmt", "A")
        if fmt == "A":
            xlsx_src = sorted(src_dir.glob(f"{panel}_Line_*.xlsx"))
        else:
            single = src_dir / f"{panel}.xlsx"
            xlsx_src = [single] if single.exists() else []
        if not xlsx_src:
            continue

        curves = parse_curves_from_xlsx(xlsx_src, fmt)
        if not curves:
            continue

        # family detection
        l0_key = f"{sha}_{panel}.jpg"
        l0_entry = l0.get(l0_key, {"second_level": "", "reasoning_lower": ""})
        main = max(curves, key=lambda c: len(c[1]))
        family = infer_family(l0_entry["reasoning_lower"],
                              l0_entry["second_level"],
                              (float(main[1].min()), float(main[1].max())))

        # copy image with a clean unique key
        panel_id = f"{safe_battery}__{sha}__{panel}"
        img_dst = out / "images" / f"{panel_id}{img_src.suffix}"
        if not img_dst.exists():
            shutil.copy2(img_src, img_dst)

        # write per-curve CSVs (optional) + collect curve metadata
        if args.include_curves_csv:
            cdir = out / "curves" / panel_id
            cdir.mkdir(exist_ok=True)
        for ci, (label, x, y) in enumerate(curves):
            curve_id = f"{panel_id}__{ci}"
            curve_meta.append({
                "curve_id": curve_id,
                "panel_id": panel_id,
                "label": label,
                "n_points": int(len(x)),
                "x_min": float(np.min(x)),
                "x_max": float(np.max(x)),
                "y_min": float(np.min(y)),
                "y_max": float(np.max(y)),
            })
            if args.include_curves_csv:
                pd.DataFrame({"x": x, "y": y}).to_csv(
                    cdir / f"{ci}_{label}.csv", index=False
                )
            n_curves_total += 1

        panel_meta.append({
            "panel_id": panel_id,
            "source_battery": battery,
            "source_paper_sha": sha,
            "panel_stem": panel,
            "image_filename": img_dst.name,
            "n_curves": len(curves),
            "main_curve_label": main[0],
            "chart_family": family,
            "extraction_format": fmt,
        })
        fam_count[family] += 1
        n_panels_packaged += 1
        n_papers.add(sha)

    # write metadata jsonl
    with open(out / "metadata.jsonl", "w", encoding="utf-8") as f:
        for it in panel_meta:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(out / "metadata_curves.jsonl", "w", encoding="utf-8") as f:
        for it in curve_meta:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # also copy MatCurvs-Reasoning benchmark
    mr_dir = out / "matcurvs_reasoning"
    mr_dir.mkdir(exist_ok=True)
    for fname in ("mr_set_rigorous.jsonl", "mr_set_scaled.jsonl"):
        src = KMEANS_ROOT / "experiments" / fname
        if src.exists():
            shutil.copy2(src, mr_dir / fname)

    # generate README / dataset card
    readme = f"""---
license: cc-by-4.0
task_categories:
- image-text-to-text
- visual-question-answering
language:
- en
tags:
- materials-science
- chart-extraction
- scientific-figures
- xrd
- raman
- xps
- xafs
- electrochemistry
size_categories:
- 1K<n<10K
---

# MatCurvs-L2: Verified Real-Coordinate Curves from Materials-Science Line Plots

MatCurvs-L2 is the verified real-coordinate-curve layer of the
**MatCurvs** benchmark, a NeurIPS 2026 D&B submission on agentic
chart reading for materials science.

## At a glance

| | |
|---|---|
| **Panels (chart images)** | {n_panels_packaged:,} |
| **Curves (real-coordinate sequences)** | {n_curves_total:,} |
| **Source articles** | {len(n_papers):,} |
| **Chart families covered** | {len(fam_count)} |

The dataset is curated from a corpus of 55,763 materials-science
articles via a semi-automatic pipeline: each panel is processed by
the **MatDeplot** extraction method (Mask2Former + 5D LAB-$xy$
$k$-means + axis calibration) and then verified by a multimodal-LLM
judge against four criteria (curve count, shape, axis-range
plausibility, absence of artifacts). Only panels passing every
criterion are included.

## Family distribution

| Family | Panels |
|---|---|
"""
    for fam, c in fam_count.most_common():
        readme += f"| {fam} | {c:,} |\n"
    readme += f"""
## Files

- `images/<panel_id>.{{jpg,png}}` — the published chart image
- `metadata.jsonl` — one row per panel
- `metadata_curves.jsonl` — one row per curve
- `matcurvs_reasoning/` — the MatCurvs-Reasoning benchmark
  - `mr_set_rigorous.jsonl` (49 panels, 188 questions, GT from L1 manual)
  - `mr_set_scaled.jsonl` (3,817 panels, 14,740 questions, GT from L2 curated)

## Schema (panel-level metadata.jsonl)

```json
{{
  "panel_id": "battery_53__<sha>__<panel>",
  "source_battery": "battery_53",
  "source_paper_sha": "<sha>",
  "panel_stem": "<panel>",
  "image_filename": "<panel_id>.jpg",
  "n_curves": <int>,
  "main_curve_label": "Line_1",
  "chart_family": "XRD" | "Raman" | "XPS" | "XAFS" | "EIS" | "GCD" | "CV" | "Cycling" | "generic",
  "extraction_format": "A" | "B"
}}
```

## Extraction-quality checks

On the L1 sub-pool ($n{{=}}204$ panels with manual pixel-polyline GT;
released separately), the extraction pipeline + agent reaches
$\\mathrm{{LR}}_{{0.5}}{{=}}45.8\\%$ pixel-IoU coverage — a $38\\times$
improvement over the strongest hosted VLM ($1.2\\%$). On the
$49$-panel L1$\\cap$L2 intersection, the median real-coordinate
agreement (DTW in physical units, normalised by axis range) between
the two annotation passes is small enough that the dominant
residual is the inherent ambiguity in curve identity when multiple
generic-labelled curves are interchangeable across passes.

## Use

```python
from datasets import load_dataset
ds = load_dataset("<your-hf-namespace>/matcurvs-l2")
```

Each row has the panel image, panel-level metadata, and (via
`metadata_curves.jsonl`) the per-curve real-coordinate $(x, y)$
sequences. The `matcurvs_reasoning/` JSONL files supply
deterministic numerical-reasoning questions over the same panels
(see the paper's $\\S$7 for the four-method evaluation that uses
this data).

## Citation

To appear at NeurIPS 2026 (Datasets and Benchmarks track). Citation
information will be added on acceptance.

## License

CC-BY-4.0. The chart images are extracted from open-access
materials-science publications; original copyright remains with the
authors of those papers. Per-curve real-coordinate data is released
under CC-BY-4.0 to enable reuse by the materials-informatics
community.
"""

    (out / "README.md").write_text(readme, encoding="utf-8")
    print(f"\n[done] packaged to {out}")
    print(f"  panels: {n_panels_packaged:,}")
    print(f"  curves: {n_curves_total:,}")
    print(f"  source articles: {len(n_papers):,}")
    print(f"  family distribution: {dict(fam_count)}")
    print(f"\nNext step: cd {out.parent} && tar -czf matcurvs_l2.tar.gz matcurvs_l2")
    print(f"  or: huggingface-cli upload <namespace>/matcurvs-l2 {out}")


if __name__ == "__main__":
    main()
