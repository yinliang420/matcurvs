"""
MatCurvs-Reasoning task-set builder.

Builds two test sets for §7:
  1) RIGOROUS  : 49 dual-annotated panels (L1 manual GT exists).
                 GT for the questions comes from L1 manual real-coord curves.
                 Used as headline (n=49 small but rigorous).
  2) SCALED    : 3,817 L2-accepted panels (VLM-judge accepted).
                 GT for the questions comes from L2 MatDeplot real-coord curves.
                 Used as scale-out evaluation.

Each panel gets a uniform set of 3 generic Cat-A questions; Raman panels
get +2 family questions; EIS panels get +1 family question.

Outputs:
  experiments/mr_set_rigorous.jsonl
  experiments/mr_set_scaled.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths
import os

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[1]))
DATA_BASE = Path(os.environ.get("DATA_BASE", str(KMEANS_ROOT.parent / "data_base")))
L2_OUT = Path(os.environ.get("L2_OUT", str(KMEANS_ROOT / "data_base_L2")))
LABEL_DIR = Path(os.environ.get("MATCURVS_L1_DIR", str(KMEANS_ROOT / "label")))
SUBFIG_DIR = Path(os.environ.get("MATCURVS_L0_DIR", str(KMEANS_ROOT / "subfigures_and_class")))

OUT_RIGOROUS = KMEANS_ROOT / "experiments" / "mr_set_rigorous.jsonl"
OUT_SCALED = KMEANS_ROOT / "experiments" / "mr_set_scaled.jsonl"


# --------------------------------------------------------------------- curves

def parse_l1_json(p: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """L1 manual annotation -> [(label, x_real, y_real), ...]"""
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    out = []
    for label, info in d.get("lines", {}).items():
        pts = info.get("points") or []
        xs, ys = [], []
        for pt in pts:
            real = pt.get("real")
            if real is None or len(real) < 2:
                continue
            xs.append(float(real[0]))
            ys.append(float(real[1]))
        if len(xs) >= 2:
            order = np.argsort(xs)
            x = np.asarray(xs)[order]
            y = np.asarray(ys)[order]
            out.append((label, x, y))
    return out


def parse_l2_curves(xlsx_paths: list[Path], fmt: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """L2 MatDeplot Excel(s) -> [(label, x_real, y_real), ...]"""
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
                curves.append((f"Line_{xp.stem.split('_Line_')[-1]}", x[m][order], y[m][order]))
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


def pick_main_curve(curves: list[tuple[str, np.ndarray, np.ndarray]]):
    """Longest curve by point count."""
    if not curves:
        return None
    return max(curves, key=lambda c: len(c[1]))


# ----------------------------------------------------------- family detection

L0_LOOKUP_CACHE: dict | None = None
def build_l0_lookup() -> dict:
    """key '<sha>_<panel>.jpg' -> (chart_type, second_level, reasoning_lower)"""
    global L0_LOOKUP_CACHE
    if L0_LOOKUP_CACHE is not None:
        return L0_LOOKUP_CACHE
    out = {}
    for fp in sorted(glob.glob(str(SUBFIG_DIR / "classbattery_*.json"))):
        try:
            with open(fp, "r") as f:
                d = json.load(f)
        except Exception:
            continue
        for k, v in d.items():
            if not isinstance(v, dict):
                continue
            chart = v.get("chart_type")
            do = v.get("data_origin") or {}
            sl = do.get("second_level", "")
            rs = (do.get("reasoning") or "").lower()
            out[k] = (chart, sl, rs)
    L0_LOOKUP_CACHE = out
    return out


def read_axis_text(panel_jpg: Path) -> tuple[str, str]:
    """Find <panel>.txt next to the .jpg and parse {x, y}. Returns ('','') on miss."""
    txt = panel_jpg.with_suffix(".txt")
    if not txt.exists():
        return "", ""
    try:
        with open(txt, "r", encoding="utf-8") as f:
            content = f.read()
        # the .txt is JSON-like (per figure_class.py)
        d = json.loads(content)
        return str(d.get("x", "")), str(d.get("y", ""))
    except Exception:
        return "", ""


def infer_family(x_label: str, y_label: str, x_range: tuple[float, float],
                 l0_second_level: str = "", l0_reasoning: str = "") -> str:
    """Infer materials chart family from axis labels + x range + L0 prior.

    Returns one of: XRD, Raman, XAFS, XPS, EIS, GCD, CV, Cycling, generic.
    """
    xl = x_label.lower()
    yl = y_label.lower()
    xmin, xmax = x_range
    l0 = l0_second_level.lower()
    rs = l0_reasoning  # already lower

    # ---- 0) Reasoning-text patterns (most reliable; check ALL families).
    # Order: most-specific keywords first so multi-mention reasoning resolves correctly.
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
    if "x-ray diffraction" in rs or "(xrd)" in rs or "2theta" in rs or "2θ" in rs \
       or "two theta" in rs or "diffraction pattern" in rs:
        return "XRD"
    if "galvanostatic" in rs or "specific capacity" in rs or "charge-discharge" in rs:
        return "GCD"
    if "cyclic voltammetr" in rs or "cv curve" in rs:
        return "CV"
    if "cycling performance" in rs or "cycle number" in rs:
        return "Cycling"

    # ---- 1) text-label rules (highest priority when present) ----
    if any(k in xl for k in ["2θ", "2theta", "two theta", "two-theta", "diffraction angle"]):
        return "XRD"
    if "θ" in x_label and ("deg" in xl or "°" in x_label) and 0 <= xmin and xmax <= 180:
        return "XRD"
    if "raman" in xl:
        return "Raman"
    if any(k in xl for k in ["wavenumber", "cm-1", "cm⁻¹"]) and 80 <= xmin and xmax <= 4500:
        return "Raman"
    if "photon energy" in xl or ("energy" in xl and "ev" in xl and xmax > 1000):
        return "XAFS"
    if "binding energy" in xl or ("be" in xl and "ev" in xl):
        return "XPS"
    if any(k in xl for k in ["z'", "re(z)", "real(z)", "z_re"]) and \
       any(k in yl for k in ["z''", "z\"", "im(z)", "imag(z)", "-z"]):
        return "EIS"
    if ("voltage" in xl or "potential" in xl) and ("capacity" in yl):
        return "GCD"
    if ("capacity" in xl) and ("voltage" in yl or "potential" in yl):
        return "GCD"
    if ("voltage" in xl or "potential" in xl) and "current" in yl:
        return "CV"
    if "cycle" in xl:
        return "Cycling"

    # ---- 2) L0 prior + x-range heuristic (used when text labels missing) ----
    # L0 second_level groups Raman with XRD under "Structural Characterization";
    # use x-range to disambiguate within the group.
    if "structural characterization" in l0:
        if 4 <= xmin <= 12 and 30 <= xmax <= 100:        # XRD 2θ degrees
            return "XRD"
        if 80 <= xmin <= 800 and 1000 <= xmax <= 4500:    # Raman cm⁻¹
            return "Raman"
        if xmin >= 400 and xmax <= 4500 and xmax - xmin > 1500:  # FTIR-like wavenumber
            return "Raman"  # collapse FTIR into Raman bucket
    if "compositional analysis" in l0:
        if 0 <= xmin and xmax <= 1300:                     # XPS binding energy
            return "XPS"
    if "performance" in l0:
        # GCD: voltage on one axis (0–5 V) and capacity (0–6000 mAh/g) on the other
        if 0 <= xmin <= 200 and xmax > 30 and xmax < 8000:    # x ~ capacity
            return "GCD"
        if -3 <= xmin <= 0.5 and 0.5 <= xmax <= 6:            # x ~ voltage
            # CV vs GCD differentiated by y range; default CV when x looks like voltage
            return "CV"
        if xmin >= 0 and xmax <= 200 and (xmax - xmin) >= 5:   # cycle number
            return "Cycling"

    # ---- 3) pure x-range fallback (no text, no L0 hint) ----
    if 4 <= xmin <= 12 and 30 <= xmax <= 100:
        return "XRD"
    if 80 <= xmin <= 800 and 1000 <= xmax <= 4500:
        return "Raman"
    if 0 <= xmin <= 200 and 200 <= xmax <= 1300:
        return "XPS"
    if xmin >= 0 and xmin <= 50 and 0 <= xmax <= 1000 and (xmax - xmin) >= 50:
        # ambiguous EIS Z' vs other, leave generic
        pass
    return "generic"


# ----------------------------------------------------------- L2 panel index

def index_l2_accepted() -> list[dict]:
    """Read judgment_log.jsonl and locate the copied accepted artifacts in L2_OUT."""
    log_path = L2_OUT / "judgment_log.jsonl"
    rows = []
    if not log_path.exists():
        return rows
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("decision") != "good":
                continue
            rows.append(r)
    return rows


def resolve_l2_xlsx(rec: dict) -> tuple[Path | None, list[Path], str | None]:
    """For an accepted log row, return (image_path, [xlsx_paths], fmt)."""
    safe_battery = rec["battery"].replace("/", "__")
    out_dir = L2_OUT / f"{safe_battery}__{rec['sha']}"
    if not out_dir.exists():
        return None, [], None
    panel = rec["panel"]
    fmt = rec.get("fmt")
    img = out_dir / f"{panel}.jpg"
    if fmt == "A":
        xlsx = sorted(out_dir.glob(f"{panel}_Line_*.xlsx"))
    else:
        single = out_dir / f"{panel}.xlsx"
        xlsx = [single] if single.exists() else []
    return (img if img.exists() else None), xlsx, fmt


def resolve_l2_jpg_in_source(rec: dict) -> Path | None:
    """Locate the original .jpg in DATA_BASE for axis-label .txt sibling."""
    battery = rec["battery"]
    if battery.startswith("t/"):
        bdir = DATA_BASE / "t" / battery.split("/", 1)[1]
    else:
        bdir = DATA_BASE / battery
    return bdir / rec["sha"] / "images" / f"{rec['panel']}.jpg"


def index_extra_pool(root: Path, family_fixed: str, source_tag: str) -> list[dict]:
    """Scan a SHA-flat directory tree (raman/data, xafs/data style) for panel records.

    Layout assumed: root/<sha>/images/<stem>.{jpg,png} + <stem>_Line_<i>.xlsx
    Returns records analogous to L2 accepted rows.
    """
    out: list[dict] = []
    if not root.exists():
        return out
    for sha_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        imdir = sha_dir / "images"
        if not imdir.exists():
            continue
        for ext in (".jpg", ".png"):
            for img in imdir.glob(f"*{ext}"):
                stem = img.stem
                lines = sorted(imdir.glob(f"{stem}_Line_*.xlsx"))
                if not lines:
                    continue
                out.append({
                    "battery": source_tag,            # e.g. "raman_pool" / "xafs_pool"
                    "sha": sha_dir.name,
                    "panel": stem,
                    "fmt": "A",
                    "img_path": img,
                    "xlsx_paths": lines,
                    "family_fixed": family_fixed,     # bypass family detection
                    "source": source_tag,
                })
    return out


# ----------------------------------------------------------- question makers

RNG = np.random.default_rng(20260429)


def q_y_at_x(curve: tuple[str, np.ndarray, np.ndarray], rng: np.random.Generator) -> dict | None:
    """A1: y(x0) at a deterministic-random x0 inside the inner 60% of the x range."""
    label, x, y = curve
    if len(x) < 5:
        return None
    lo, hi = np.quantile(x, 0.2), np.quantile(x, 0.8)
    x0 = float(rng.uniform(lo, hi))
    y0 = float(np.interp(x0, x, y))
    return {
        "level": "A1",
        "task": "y_at_x",
        "question": f"What is the value of y at x = {x0:.4g} for the curve labelled '{label}'? Return a single numeric value.",
        "gt_answer": y0,
        "x_query": x0,
        "curve_label": label,
    }


def q_argmax_in_range(curve, rng):
    """A2: argmax y in a deterministic-random sub-range of the data."""
    label, x, y = curve
    if len(x) < 8:
        return None
    a = float(rng.uniform(np.quantile(x, 0.05), np.quantile(x, 0.45)))
    b = float(rng.uniform(np.quantile(x, 0.55), np.quantile(x, 0.95)))
    if b <= a:
        return None
    mask = (x >= a) & (x <= b)
    if mask.sum() < 3:
        return None
    x_in = x[mask]; y_in = y[mask]
    x_at_max = float(x_in[int(np.argmax(y_in))])
    return {
        "level": "A2",
        "task": "argmax_in_range",
        "question": f"On the curve labelled '{label}', within the range x ∈ [{a:.4g}, {b:.4g}], at what value of x does y reach its maximum? Return the x value as a single number.",
        "gt_answer": x_at_max,
        "range_a": a,
        "range_b": b,
        "curve_label": label,
    }


def q_fwhm_main_peak(curve, rng):
    """A3 (preferred): FWHM (in x-units) of the most prominent peak.
    Returns None if no detectable peak; caller is expected to use the
    `q_a3_fallback` Q in that case.
    """
    label, x, y = curve
    if len(x) < 10:
        return None
    if np.nanmax(y) - np.nanmin(y) <= 0:
        return None
    peaks, props = find_peaks(y, prominence=(np.nanmax(y) - np.nanmin(y)) * 0.05)
    if len(peaks) == 0:
        return None
    main = peaks[int(np.argmax(props["prominences"]))]
    half_height = y[main] - props["prominences"][int(np.argmax(props["prominences"]))] / 2.0
    li = main
    while li > 0 and y[li] > half_height:
        li -= 1
    ri = main
    while ri < len(y) - 1 and y[ri] > half_height:
        ri += 1
    if li == 0 or ri == len(y) - 1:
        return None
    x_left = np.interp(half_height, [y[li], y[li + 1]], [x[li], x[li + 1]])
    x_right = np.interp(half_height, [y[ri], y[ri - 1]], [x[ri], x[ri - 1]])
    fwhm = float(x_right - x_left)
    if not np.isfinite(fwhm) or fwhm <= 0:
        return None
    return {
        "level": "A3",
        "task": "fwhm_main_peak",
        "question": f"On the curve labelled '{label}', estimate the full-width at half-maximum (FWHM) of the most prominent peak, in the same units as the x-axis. Return a single numeric value.",
        "gt_answer": fwhm,
        "peak_x": float(x[main]),
        "peak_y": float(y[main]),
        "curve_label": label,
    }


def q_a3_fallback(curve, rng):
    """A3 fallback when no peak is detectable — use peak-to-peak amplitude
    (max y − min y) of the full curve. Always defined for any non-trivial curve.
    """
    label, x, y = curve
    if len(x) < 2:
        return None
    rng_y = float(np.nanmax(y) - np.nanmin(y))
    if not np.isfinite(rng_y) or rng_y <= 0:
        return None
    return {
        "level": "A3",
        "task": "y_amplitude",
        "question": f"For the curve labelled '{label}', what is the peak-to-peak amplitude, defined as max(y) − min(y), over the entire curve? Return a single numeric value.",
        "gt_answer": rng_y,
        "y_max": float(np.nanmax(y)),
        "y_min": float(np.nanmin(y)),
        "curve_label": label,
    }


def q_raman_top_peaks(curve, rng):
    """B-Raman 1: top-3 peak positions (cm^-1)."""
    label, x, y = curve
    if len(x) < 20:
        return None
    peaks, props = find_peaks(y, prominence=(np.nanmax(y) - np.nanmin(y)) * 0.05)
    if len(peaks) == 0:
        return None
    order = np.argsort(-props["prominences"])
    topk = peaks[order][:3]
    top_x = sorted([float(x[i]) for i in topk])
    return {
        "level": "B-Raman",
        "task": "raman_top3_peaks",
        "question": f"On the Raman spectrum (curve '{label}'), list the x-positions (in cm⁻¹) of the top-3 most prominent peaks, in ascending order. Return three numeric values comma-separated.",
        "gt_answer": top_x,
        "curve_label": label,
    }


def q_raman_id_ig(curve, rng):
    """B-Raman 2: ID/IG ratio (intensity at ~1350 / ~1580 cm-1)."""
    label, x, y = curve
    if x.min() > 1300 or x.max() < 1600:
        return None
    yD = float(np.interp(1350.0, x, y))
    yG = float(np.interp(1580.0, x, y))
    if yG <= 0:
        return None
    ratio = yD / yG
    return {
        "level": "B-Raman",
        "task": "raman_id_ig_ratio",
        "question": f"On the Raman spectrum (curve '{label}'), compute the I_D/I_G ratio = y(1350 cm⁻¹) / y(1580 cm⁻¹). Return a single numeric value.",
        "gt_answer": ratio,
        "curve_label": label,
    }


def q_stacked_intensity_offset(curves: list, rng) -> dict | None:
    """C-Stack: intensity offset between adjacent stacked curves.

    Convention in materials-science spectra (XRD/Raman/XPS/XAFS): authors
    add a constant intensity to each successive curve so they don't overlap
    visually. The offset is invisible-but-measurable in the extracted data.
    A VLM can guess "looks like ~5 units" but agent + scipy computes it
    precisely from baseline regions.

    GT: difference of robust baseline estimates (median of bottom 25% of y).
    Only fires on panels with >=2 curves.
    """
    if len(curves) < 2:
        return None
    # deterministic: take the two curves with the smallest median-y and second-smallest median-y
    cs = sorted(curves, key=lambda c: float(np.nanmedian(c[2])))
    (l_lo, x_lo, y_lo), (l_hi, x_hi, y_hi) = cs[0], cs[1]
    # robust baseline = median of bottom-25% of y values
    def baseline(y):
        thr = np.nanpercentile(y, 25)
        bot = y[y <= thr]
        return float(np.nanmedian(bot)) if len(bot) > 0 else float(np.nanmin(y))
    bl_lo = baseline(y_lo)
    bl_hi = baseline(y_hi)
    offset = bl_hi - bl_lo
    if not np.isfinite(offset):
        return None
    return {
        "level": "C-Stack",
        "task": "stacked_intensity_offset",
        "question": (f"In this panel, multiple curves are vertically stacked with constant intensity "
                     f"offsets for visual clarity. Estimate the intensity offset of curve '{l_hi}' "
                     f"relative to curve '{l_lo}', defined as the difference of their baseline y-values "
                     f"(median of the lowest 25% of y values for each curve). Return a single numeric value."),
        "gt_answer": offset,
        "curve_label_lower": l_lo,
        "curve_label_upper": l_hi,
        "baseline_lower": bl_lo,
        "baseline_upper": bl_hi,
    }


def q_eis_high_freq_intercept(curve, rng):
    """B-EIS: high-frequency intercept on the real-impedance axis.

    Definition: the smallest Z' value in the dataset (leftmost point on
    Nyquist plot, which corresponds to the highest measurement frequency).
    Always defined whenever the curve has ≥ 2 points.
    """
    label, x, y = curve  # x = Z', y = Z'' (or -Z'')
    if len(x) < 2:
        return None
    z_intercept = float(np.nanmin(x))
    if not np.isfinite(z_intercept):
        return None
    return {
        "level": "B-EIS",
        "task": "eis_high_freq_intercept",
        "question": f"On the Nyquist EIS plot (curve '{label}'), what is the high-frequency intercept on the real-impedance (Z') axis, defined as the smallest Z' value in the dataset (leftmost point of the curve)? Return a single numeric value in the same units as Z'.",
        "gt_answer": z_intercept,
        "curve_label": label,
    }


# ---------------------------------------------------------------------- main

def make_questions_for_panel(curves, family, rng) -> list[dict]:
    main = pick_main_curve(curves)
    if main is None:
        return []
    q1 = q_y_at_x(main, rng)
    q2 = q_argmax_in_range(main, rng)
    # A3: try FWHM first; if no detectable peak, fall back to amplitude
    q3 = q_fwhm_main_peak(main, rng) or q_a3_fallback(main, rng)
    out = [q for q in (q1, q2, q3) if q is not None]
    if family == "Raman":
        for fn in (q_raman_top_peaks, q_raman_id_ig):
            q = fn(main, rng)
            if q is not None:
                out.append(q)
    elif family == "EIS":
        q = q_eis_high_freq_intercept(main, rng)
        if q is not None:
            out.append(q)
    # C-Stack: stacked intensity offset, fires on multi-curve spectra-style panels.
    if family in {"XRD", "Raman", "XPS", "XAFS"} and len(curves) >= 2:
        q = q_stacked_intensity_offset(curves, rng)
        if q is not None:
            out.append(q)
    return out


def build_rigorous_set(l0_lookup) -> list[dict]:
    """49 dual-annotated panels: L1 stems ∩ L2-accepted panels."""
    l1_stems = {p.stem.replace("_annotation", ""): p for p in LABEL_DIR.glob("*_annotation.json")}
    accepted = index_l2_accepted()
    by_panel = {r["panel"]: r for r in accepted}
    items = []
    rng_local = np.random.default_rng(42)
    for stem, json_path in sorted(l1_stems.items()):
        if stem not in by_panel:
            continue
        rec = by_panel[stem]
        # GT comes from L1
        l1_curves = parse_l1_json(json_path)
        if not l1_curves:
            continue
        # Family detection: try .txt axis labels (if source volume mounted),
        # else fall back to L0 second_level + x-range heuristic.
        src_jpg = resolve_l2_jpg_in_source(rec)
        x_label = y_label = ""
        if src_jpg and src_jpg.exists():
            x_label, y_label = read_axis_text(src_jpg)
        l0_key = f"{rec['sha']}_{rec['panel']}.jpg"
        l0_entry = l0_lookup.get(l0_key, ("", "", ""))
        l0_second, l0_reason = l0_entry[1], l0_entry[2] if len(l0_entry) > 2 else ""
        main = pick_main_curve(l1_curves)
        if main is None:
            continue
        family = infer_family(x_label, y_label,
                              (float(main[1].min()), float(main[1].max())),
                              l0_second_level=l0_second,
                              l0_reasoning=l0_reason)
        # also resolve image path under label/ for the rigorous set
        img_path_label = LABEL_DIR / f"{stem}.jpg"
        if not img_path_label.exists():
            continue
        # resolve L2 xlsx paths (so M2/M3 runners can load extracted curves)
        l2_dir = L2_OUT / f"{rec['battery'].replace('/', '__')}__{rec['sha']}"
        fmt = rec.get("fmt")
        if fmt == "A":
            l2_xlsx_paths = sorted(l2_dir.glob(f"{stem}_Line_*.xlsx"))
        else:
            single = l2_dir / f"{stem}.xlsx"
            l2_xlsx_paths = [single] if single.exists() else []
        questions = make_questions_for_panel(l1_curves, family, rng_local)
        for qi, q in enumerate(questions):
            items.append({
                "set": "rigorous",
                "item_id": f"rig_{stem}_q{qi+1}",
                "panel": stem,
                "battery": rec["battery"],
                "sha": rec["sha"],
                "image_path": str(img_path_label),
                "l1_json_path": str(json_path),
                "l2_xlsx_paths": [str(p) for p in l2_xlsx_paths],
                "l2_accepted_dir": str(l2_dir),
                "fmt": fmt,
                "n_l1_curves": len(l1_curves),
                "main_curve_label": main[0],
                "family": family,
                "x_label": x_label,
                "y_label": y_label,
                **q,
                "gt_source": "L1_manual",
            })
    return items


def build_scaled_set(l0_lookup) -> list[dict]:
    """All 3,817 L2-accepted panels: GT from L2 MatDeplot output."""
    accepted = index_l2_accepted()
    items = []
    rng_local = np.random.default_rng(7)
    for rec in accepted:
        img, xlsx_paths, fmt = resolve_l2_xlsx(rec)
        if img is None or not xlsx_paths:
            continue
        l2_curves = parse_l2_curves(xlsx_paths, fmt)
        if not l2_curves:
            continue
        # Family detection: try .txt (if mounted) → L0 second_level + range
        src_jpg = resolve_l2_jpg_in_source(rec)
        x_label = y_label = ""
        if src_jpg and src_jpg.exists():
            x_label, y_label = read_axis_text(src_jpg)
        l0_key = f"{rec['sha']}_{rec['panel']}.jpg"
        l0_entry = l0_lookup.get(l0_key, ("", "", ""))
        l0_second = l0_entry[1] if len(l0_entry) > 1 else ""
        l0_reason = l0_entry[2] if len(l0_entry) > 2 else ""
        main = pick_main_curve(l2_curves)
        if main is None:
            continue
        family = infer_family(x_label, y_label,
                              (float(main[1].min()), float(main[1].max())),
                              l0_second_level=l0_second,
                              l0_reasoning=l0_reason)
        questions = make_questions_for_panel(l2_curves, family, rng_local)
        for qi, q in enumerate(questions):
            items.append({
                "set": "scaled",
                "item_id": f"scl_{rec['battery'].replace('/','__')}_{rec['sha']}_{rec['panel']}_q{qi+1}",
                "panel": rec["panel"],
                "battery": rec["battery"],
                "sha": rec["sha"],
                "image_path": str(img),
                "l2_xlsx_paths": [str(p) for p in xlsx_paths],
                "fmt": fmt,
                "n_l2_curves": len(l2_curves),
                "main_curve_label": main[0],
                "family": family,
                "x_label": x_label,
                "y_label": y_label,
                **q,
                "gt_source": "L2_curated",
            })
    return items


def build_full_set(l0_lookup) -> list[dict]:
    """Merged MatCurvs-Reasoning pool: L2-accepted + raman/data + xafs/data.

    All three sources treated as ONE benchmark in §7. raman/xafs panels get
    family hardcoded (Raman/XAFS) since they are pre-stratified by source.
    """
    items = []
    rng_local = np.random.default_rng(11)

    # ---- 1) L2-accepted (3,817 panels) ----
    accepted = index_l2_accepted()
    for rec in accepted:
        img, xlsx_paths, fmt = resolve_l2_xlsx(rec)
        if img is None or not xlsx_paths:
            continue
        l2_curves = parse_l2_curves(xlsx_paths, fmt)
        if not l2_curves:
            continue
        l0_key = f"{rec['sha']}_{rec['panel']}.jpg"
        l0_entry = l0_lookup.get(l0_key, ("", "", ""))
        l0_second = l0_entry[1] if len(l0_entry) > 1 else ""
        l0_reason = l0_entry[2] if len(l0_entry) > 2 else ""
        main = pick_main_curve(l2_curves)
        if main is None:
            continue
        family = infer_family("", "",
                              (float(main[1].min()), float(main[1].max())),
                              l0_second_level=l0_second,
                              l0_reasoning=l0_reason)
        questions = make_questions_for_panel(l2_curves, family, rng_local)
        for qi, q in enumerate(questions):
            items.append({
                "set": "full",
                "source": "L2",
                "item_id": f"l2_{rec['battery'].replace('/','__')}_{rec['sha']}_{rec['panel']}_q{qi+1}",
                "panel": rec["panel"],
                "battery": rec["battery"],
                "sha": rec["sha"],
                "image_path": str(img),
                "l2_xlsx_paths": [str(p) for p in xlsx_paths],
                "fmt": fmt,
                "n_l2_curves": len(l2_curves),
                "main_curve_label": main[0],
                "family": family,
                "level": q["level"],
                "task": q["task"],
                "question": q["question"],
                "gt_answer": q["gt_answer"],
                **{k: v for k, v in q.items() if k not in ("level", "task", "question", "gt_answer")},
                "gt_source": "L2_curated",
            })

    # ---- 2) Raman pool (1,069 panels, family hardcoded) ----
    raman_panels = index_extra_pool(RAMAN_POOL, family_fixed="Raman", source_tag="raman_pool")
    for rec in raman_panels:
        curves = parse_l2_curves(rec["xlsx_paths"], rec["fmt"])
        if not curves:
            continue
        main = pick_main_curve(curves)
        if main is None:
            continue
        questions = make_questions_for_panel(curves, "Raman", rng_local)
        for qi, q in enumerate(questions):
            items.append({
                "set": "full",
                "source": "raman_pool",
                "item_id": f"raman_{rec['sha']}_{rec['panel']}_q{qi+1}",
                "panel": rec["panel"],
                "battery": rec["battery"],
                "sha": rec["sha"],
                "image_path": str(rec["img_path"]),
                "l2_xlsx_paths": [str(p) for p in rec["xlsx_paths"]],
                "fmt": rec["fmt"],
                "n_l2_curves": len(curves),
                "main_curve_label": main[0],
                "family": "Raman",
                "level": q["level"],
                "task": q["task"],
                "question": q["question"],
                "gt_answer": q["gt_answer"],
                **{k: v for k, v in q.items() if k not in ("level", "task", "question", "gt_answer")},
                "gt_source": "raman_pool_extraction",
            })

    # ---- 3) XAFS pool (1,955 panels, family hardcoded) ----
    xafs_panels = index_extra_pool(XAFS_POOL, family_fixed="XAFS", source_tag="xafs_pool")
    for rec in xafs_panels:
        curves = parse_l2_curves(rec["xlsx_paths"], rec["fmt"])
        if not curves:
            continue
        main = pick_main_curve(curves)
        if main is None:
            continue
        questions = make_questions_for_panel(curves, "XAFS", rng_local)
        for qi, q in enumerate(questions):
            items.append({
                "set": "full",
                "source": "xafs_pool",
                "item_id": f"xafs_{rec['sha']}_{rec['panel']}_q{qi+1}",
                "panel": rec["panel"],
                "battery": rec["battery"],
                "sha": rec["sha"],
                "image_path": str(rec["img_path"]),
                "l2_xlsx_paths": [str(p) for p in rec["xlsx_paths"]],
                "fmt": rec["fmt"],
                "n_l2_curves": len(curves),
                "main_curve_label": main[0],
                "family": "XAFS",
                "level": q["level"],
                "task": q["task"],
                "question": q["question"],
                "gt_answer": q["gt_answer"],
                **{k: v for k, v in q.items() if k not in ("level", "task", "question", "gt_answer")},
                "gt_source": "xafs_pool_extraction",
            })
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rigorous-only", action="store_true")
    ap.add_argument("--scaled-only", action="store_true")
    ap.add_argument("--full", action="store_true",
                    help="Build the merged MatCurvs-Reasoning set "
                         "(L2 + raman + xafs).")
    args = ap.parse_args()
    l0 = build_l0_lookup()
    print(f"L0 lookup: {len(l0):,} entries")

    if args.full:
        full = build_full_set(l0)
        OUT_FULL.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_FULL, "w", encoding="utf-8") as f:
            for it in full:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        from collections import Counter
        src = Counter(i["source"] for i in full)
        fam = Counter(i["family"] for i in full)
        lvl = Counter(i["level"] for i in full)
        print(f"\nFULL SET → {OUT_FULL}")
        print(f"  panels covered: {len({(i['source'], i['sha'], i['panel']) for i in full})}")
        print(f"  total questions: {len(full)}")
        print(f"  by source: {dict(src)}")
        print(f"  by family: {dict(fam)}")
        print(f"  by level : {dict(lvl)}")
        return

    if not args.scaled_only:
        rig = build_rigorous_set(l0)
        OUT_RIGOROUS.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_RIGOROUS, "w", encoding="utf-8") as f:
            for it in rig:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"\nRIGOROUS SET → {OUT_RIGOROUS}")
        print(f"  panels covered: {len({i['panel'] for i in rig})}")
        print(f"  total questions: {len(rig)}")
        from collections import Counter
        fam = Counter(i["family"] for i in rig)
        lvl = Counter(i["level"] for i in rig)
        print(f"  by family: {dict(fam)}")
        print(f"  by level : {dict(lvl)}")

    if not args.rigorous_only:
        scl = build_scaled_set(l0)
        with open(OUT_SCALED, "w", encoding="utf-8") as f:
            for it in scl:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"\nSCALED SET → {OUT_SCALED}")
        print(f"  panels covered: {len({i['panel'] for i in scl})}")
        print(f"  total questions: {len(scl)}")
        from collections import Counter
        fam = Counter(i["family"] for i in scl)
        lvl = Counter(i["level"] for i in scl)
        print(f"  by family: {dict(fam)}")
        print(f"  by level : {dict(lvl)}")


if __name__ == "__main__":
    main()
