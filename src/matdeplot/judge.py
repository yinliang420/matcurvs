"""Optional final extraction-quality judge for public MatDeplot outputs."""

from __future__ import annotations

import base64
import io
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openai import OpenAI

from .schema import CurveRecord

_IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

CRITERION_NAMES: Tuple[str, str, str] = (
    "curve_count_match",
    "shape_match",
    "range_plausible",
)

JUDGE_PROMPT = """You are auditing an automatic curve-extraction system on a scientific line chart from a materials-science article.

You receive TWO images:
- IMAGE 1 (the ORIGINAL chart): a published scientific line chart.
- IMAGE 2 (the EXTRACTION):     the system's reconstruction. Each curve is plotted as a colored polyline labelled Line_1, Line_2, ...

Follow these steps strictly:

STEP 1. Count distinct DATA curves on IMAGE 1 (ignore axis lines, gridlines, legend boxes, error-bar caps, isolated scatter points without a clear connecting line). Call this n_left.
STEP 2. Count distinct colored polylines on IMAGE 2. Call this n_right.
STEP 3. For each curve on IMAGE 2, find its best match on IMAGE 1 and judge whether the overall morphology (rises/falls, peak positions, plateaus, slopes) matches without substantial spurious spikes, bridging lines, broken segments, dramatic kinks, or excessive staircase artifacts.
STEP 4. Judge whether IMAGE 2's axis range is plausible for the chart family suggested by IMAGE 1 (XRD: 2θ in 5°-90°; Raman: shift in 100-4000 cm⁻¹; XAFS: energy in eV; GCD: voltage 0-5 V vs capacity in mAh/g; CV: current vs voltage 0-5 V; EIS: Z' vs Z''; XPS: binding energy 0-1300 eV; if a non-standard chart, judge plausibility against IMAGE 1's axes).

HARD RULES:
- If IMAGE 2 is empty / blank / contains zero polylines, set decision="bad".
- If n_right != n_left, set curve_count_match=false (decision usually "bad").
- decision="good" ONLY IF curve_count_match AND shape_match AND range_plausible are ALL true. Otherwise "bad".
- When in doubt, prefer "bad". A false-good costs more than a false-bad.

Return STRICT JSON only, no extra prose, no markdown fences:
{
  "n_left": <integer>,
  "n_right": <integer>,
  "decision": "good" | "bad",
  "curve_count_match": true | false,
  "shape_match": true | false,
  "range_plausible": true | false,
  "reason": "<one short sentence>"
}
"""


@dataclass(frozen=True)
class JudgeVerdict:
    model: str
    admitted: bool = False
    n_left: Optional[int] = None
    n_right: Optional[int] = None
    curve_count_match: Optional[bool] = None
    shape_match: Optional[bool] = None
    range_plausible: Optional[bool] = None
    reason: str = ""
    latency_seconds: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_json_text(raw: str) -> Tuple[Optional[str], Optional[str]]:
    if raw is None:
        return None, "empty_response"
    text = raw.strip()
    if not text:
        return None, "empty_response"
    if "```" in text:
        for part in text.split("```"):
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                return candidate, None
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0:
        return None, "no_json"
    if end <= start:
        return text[start:], None
    return text[start:end], None


def parse_judge_response(
    raw: str,
    model: str,
    latency_seconds: float = 0.0,
) -> JudgeVerdict:
    """Parse a judge response and recompute admission locally."""

    json_text, extraction_error = _extract_json_text(raw)
    if extraction_error is not None:
        return JudgeVerdict(
            model=model,
            admitted=False,
            reason=extraction_error,
            latency_seconds=latency_seconds,
            error=extraction_error,
        )
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError:
        return JudgeVerdict(
            model=model,
            admitted=False,
            reason="invalid_json",
            latency_seconds=latency_seconds,
            error="invalid_json",
        )

    counts = []
    for key in ("n_left", "n_right"):
        value = payload.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return JudgeVerdict(
                model=model,
                admitted=False,
                reason="missing_or_invalid_fields",
                latency_seconds=latency_seconds,
                error="missing_or_invalid_fields",
            )
        counts.append(value)

    criteria = {}
    for name in CRITERION_NAMES:
        value = payload.get(name)
        if not isinstance(value, bool):
            return JudgeVerdict(
                model=model,
                admitted=False,
                reason="missing_or_invalid_fields",
                latency_seconds=latency_seconds,
                error="missing_or_invalid_fields",
            )
        criteria[name] = value

    reason = payload.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)
    local_curve_count_match = criteria["curve_count_match"] is True and counts[0] == counts[1]
    admitted = (
        local_curve_count_match
        and criteria["shape_match"] is True
        and criteria["range_plausible"] is True
    )
    return JudgeVerdict(
        model=model,
        admitted=admitted,
        n_left=counts[0],
        n_right=counts[1],
        curve_count_match=local_curve_count_match,
        shape_match=criteria["shape_match"],
        range_plausible=criteria["range_plausible"],
        reason=reason,
        latency_seconds=latency_seconds,
        error=None,
    )


def _coerce_points(points: Sequence[Sequence[float]], field_name: str) -> List[Tuple[float, float]]:
    if not isinstance(points, list):
        raise ValueError(f"extraction schema validation failed: {field_name} must be a list")
    converted: List[Tuple[float, float]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(
                f"extraction schema validation failed: {field_name} must contain finite numeric [x, y] pairs"
            )
        x, y = point
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
            raise ValueError(
                f"extraction schema validation failed: {field_name} must contain finite numeric [x, y] pairs"
            )
        if isinstance(y, bool) or not isinstance(y, (int, float)) or not math.isfinite(float(y)):
            raise ValueError(
                f"extraction schema validation failed: {field_name} must contain finite numeric [x, y] pairs"
            )
        converted.append((float(x), float(y)))
    return converted


def render_extracted_curves(curves: Sequence[CurveRecord]) -> bytes:
    """Render real-valued curves as a clean scientific plot and return PNG bytes."""

    if not curves:
        raise ValueError("render_extracted_curves requires at least one curve")

    figure, axis = plt.subplots(figsize=(8, 6))
    palette = plt.get_cmap("tab10")
    try:
        for index, curve in enumerate(curves):
            if len(curve.real_points) < 2:
                raise ValueError("each curve must contain at least two real points")
            if len(curve.pixel_points) != len(curve.real_points):
                raise ValueError("pixel_points and real_points must have the same length")
            xs = []
            ys = []
            for x_value, y_value in curve.real_points:
                x_number = float(x_value)
                y_number = float(y_value)
                if not (float("-inf") < x_number < float("inf")) or not (
                    float("-inf") < y_number < float("inf")
                ):
                    raise ValueError("real_points must contain finite numeric values")
                xs.append(x_number)
                ys.append(y_number)
            axis.plot(
                xs,
                ys,
                color=palette(index % 10),
                linewidth=1.8,
                label=f"Line_{curve.curve_index + 1}",
            )
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_title(f"Extracted curves ({len(curves)})")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best", fontsize=9)
        figure.tight_layout()
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(figure)


def load_curves_from_extraction(path: Path) -> List[CurveRecord]:
    """Load public extraction JSON into CurveRecord objects."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("extraction schema validation failed: top-level payload must be an object")
    curves_payload = payload.get("curves")
    if not isinstance(curves_payload, list):
        raise ValueError("extraction schema validation failed: curves must be a list")

    curves: List[CurveRecord] = []
    required_keys = {
        "curve_index",
        "pixel_points",
        "real_points",
        "confidence",
        "abstain",
    }
    for index, entry in enumerate(curves_payload):
        if not isinstance(entry, dict):
            raise ValueError("extraction schema validation failed: each curve entry must be an object")
        missing = sorted(required_keys - set(entry))
        if missing:
            raise ValueError(
                "extraction schema validation failed: curve entry {0} is missing required key(s): {1}".format(
                    index, ", ".join(missing)
                )
            )
        curve_index = entry["curve_index"]
        if not isinstance(curve_index, int) or isinstance(curve_index, bool):
            raise ValueError("extraction schema validation failed: curve_index must be an int")
        confidence = entry["confidence"]
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
        ):
            raise ValueError(
                "extraction schema validation failed: confidence must be a finite number"
            )
        abstain = entry["abstain"]
        if not isinstance(abstain, bool):
            raise ValueError("extraction schema validation failed: abstain must be a bool")
        curves.append(
            CurveRecord(
                curve_index=curve_index,
                pixel_points=_coerce_points(entry["pixel_points"], "pixel_points"),
                real_points=_coerce_points(entry["real_points"], "real_points"),
                confidence=float(confidence),
                abstain=abstain,
            )
        )
    return curves


def _image_mime_type(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    try:
        return _IMAGE_MIME_TYPES[suffix]
    except KeyError as exc:
        supported = ", ".join(sorted(_IMAGE_MIME_TYPES))
        raise ValueError(
            f"Unsupported image format '{suffix or '<none>'}'; expected one of: {supported}"
        ) from exc


class QwenExtractionJudge:
    """DashScope-compatible two-image extraction judge."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-vl-max",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout: float = 90.0,
        retries: int = 1,
        client: Optional[object] = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.retries = max(0, int(retries))
        self._client = client

    def _resolve_api_key(self) -> Optional[str]:
        return self.api_key or os.environ.get("DASHSCOPE_API_KEY")

    def _client_or_create(self, api_key: str):
        if self._client is None:
            self._client = OpenAI(
                api_key=api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    def review(
        self,
        image_path: Path,
        curves: Sequence[CurveRecord],
    ) -> JudgeVerdict:
        api_key = self._resolve_api_key()
        if not api_key:
            return JudgeVerdict(
                model=self.model,
                admitted=False,
                reason="DASHSCOPE_API_KEY is required",
                error="missing_DASHSCOPE_API_KEY",
            )

        image_path = Path(image_path)
        try:
            original_bytes = image_path.read_bytes()
            original_mime = _image_mime_type(image_path)
            original_b64 = base64.b64encode(original_bytes).decode("ascii")
        except Exception as exc:
            return JudgeVerdict(
                model=self.model,
                admitted=False,
                reason=str(exc),
                error=f"input:{type(exc).__name__}",
            )
        try:
            rendered_png = render_extracted_curves(curves)
            rendered_b64 = base64.b64encode(rendered_png).decode("ascii")
        except Exception as exc:
            return JudgeVerdict(
                model=self.model,
                admitted=False,
                reason=str(exc),
                error=f"render:{type(exc).__name__}",
            )

        started = time.monotonic()
        client = self._client_or_create(api_key)
        last_error: Optional[BaseException] = None
        last_parse_verdict: Optional[JudgeVerdict] = None
        for _ in range(self.retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{original_mime};base64,{original_b64}"
                                    },
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{rendered_b64}"
                                    },
                                },
                                {"type": "text", "text": JUDGE_PROMPT},
                            ],
                        }
                    ],
                )
                raw = response.choices[0].message.content or ""
                verdict = parse_judge_response(
                    raw,
                    model=self.model,
                    latency_seconds=time.monotonic() - started,
                )
                if verdict.error is None:
                    return verdict
                last_parse_verdict = verdict
            except Exception as exc:  # pragma: no cover - exact provider errors vary
                last_error = exc

        if last_parse_verdict is not None:
            return last_parse_verdict
        return JudgeVerdict(
            model=self.model,
            admitted=False,
            reason=str(last_error) if last_error is not None else "provider_error",
            latency_seconds=time.monotonic() - started,
            error="provider_error",
        )


__all__ = [
    "CRITERION_NAMES",
    "JUDGE_PROMPT",
    "JudgeVerdict",
    "QwenExtractionJudge",
    "load_curves_from_extraction",
    "parse_judge_response",
    "render_extracted_curves",
]
