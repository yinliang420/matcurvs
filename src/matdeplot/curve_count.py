"""Curve-count providers for the paper setting and local alternatives."""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .config import CurveCountConfig, SeparationConfig
from .separation import pixel_features

_IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def _image_mime_type(image_path: Path) -> str:
    """Return the data-URL MIME type for a supported chart image."""

    suffix = Path(image_path).suffix.lower()
    try:
        return _IMAGE_MIME_TYPES[suffix]
    except KeyError as exc:
        supported = ", ".join(sorted(_IMAGE_MIME_TYPES))
        raise ValueError(
            f"Unsupported image format '{suffix or '<none>'}'; "
            f"expected one of: {supported}"
        ) from exc


CURVE_COUNT_PROMPT = """You are analyzing a scientific chart image (line plot, scatter plot, or
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


@dataclass(frozen=True)
class CurveCountResult:
    value: int
    source: str
    model: Optional[str] = None
    latency_seconds: float = 0.0
    raw_response: Optional[str] = None
    fallback_reason: Optional[str] = None


def parse_curve_count(raw: str) -> Optional[int]:
    """Parse the strict response while tolerating a Markdown JSON fence."""

    if not raw:
        return None
    text = raw.strip()
    if "```" in text:
        for part in text.split("```"):
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{"):
                text = candidate
                break
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= 0:
        return None
    try:
        payload = json.loads(text[start:end])
    except json.JSONDecodeError:
        return None
    value = payload.get("total_lines_num")
    if isinstance(value, int) and value >= 1:
        return value
    if isinstance(value, str) and value.isdigit() and int(value) >= 1:
        return int(value)
    return None


class GeminiCurveCounter:
    """Paper curve-count provider through Gemini's compatible endpoint."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini counting"
            )
        endpoint = (
            base_url
            or os.environ.get("GEMINI_BASE_URL")
            or "https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.model = model
        self.client = OpenAI(api_key=key, base_url=endpoint, timeout=timeout)

    def count(self, image_path: Path) -> CurveCountResult:
        image_path = Path(image_path)
        mime_type = _image_mime_type(image_path)
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        started = time.monotonic()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{encoded}"
                            },
                        },
                        {"type": "text", "text": CURVE_COUNT_PROMPT},
                    ],
                }
            ],
        )
        raw = response.choices[0].message.content or ""
        value = parse_curve_count(raw)
        if value is None:
            raise RuntimeError("Gemini curve-count response was not valid JSON")
        return CurveCountResult(
            value=value,
            source="gemini",
            model=self.model,
            latency_seconds=time.monotonic() - started,
            raw_response=raw[:300],
        )


def load_cached_counts(path: Path) -> Dict[str, int]:
    """Load image-to-count mappings from JSON or a paper-format CSV."""

    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {str(key): int(value) for key, value in payload.items()}

    import pandas as pd

    frame = pd.read_csv(path)
    if "image_id" not in frame.columns:
        raise ValueError("curve-count CSV must contain an image_id column")
    count_columns = [
        column
        for column in frame.columns
        if column.endswith("_k") or column in {"k", "curve_count"}
    ]
    if len(count_columns) != 1:
        raise ValueError(
            "curve-count CSV must contain exactly one count column ending in '_k'"
        )
    column = count_columns[0]
    return {
        str(row["image_id"]): int(row[column])
        for _, row in frame.dropna(subset=[column]).iterrows()
    }


def silhouette_curve_count(
    image_bgr: np.ndarray,
    pixel_mask: np.ndarray,
    count_config: CurveCountConfig,
    separation_config: SeparationConfig,
    maximum_samples: int = 5000,
) -> CurveCountResult:
    """Select K deterministically by silhouette score over K=2,...,6."""

    features, _, _ = pixel_features(
        image_bgr,
        pixel_mask,
        separation_config.spatial_weight,
    )
    if len(features) < 20:
        return CurveCountResult(
            value=1,
            source="silhouette",
            fallback_reason="too_few_foreground_pixels",
        )

    rng = np.random.default_rng(separation_config.random_state)
    if len(features) > maximum_samples:
        indices = np.sort(
            rng.choice(len(features), size=maximum_samples, replace=False)
        )
        sample = features[indices]
    else:
        sample = features

    maximum_k = min(count_config.candidate_max, len(sample) - 1)
    candidates = range(count_config.candidate_min, maximum_k + 1)
    best_k: Optional[int] = None
    best_score = -np.inf
    for candidate in candidates:
        labels = KMeans(
            n_clusters=candidate,
            n_init=separation_config.kmeans_n_init,
            random_state=separation_config.random_state,
        ).fit_predict(sample)
        score = float(silhouette_score(sample, labels))
        if score > best_score:
            best_score = score
            best_k = candidate
    if best_k is None:
        return CurveCountResult(
            value=1,
            source="silhouette",
            fallback_reason="no_valid_candidate",
        )
    return CurveCountResult(value=best_k, source="silhouette")
