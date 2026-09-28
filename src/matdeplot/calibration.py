"""Axis-tick extraction and pixel-to-data calibration.

Calibration never uses ground-truth curve annotations. When normal OCR output
contains too few numeric ticks, the released configuration can recover a
pixel-to-value mapping from evenly spaced synthetic ticks on the detected axis.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import CalibrationConfig
from .schema import AxisDetection, AxisMapping, AxisTick, Point

_NUMBER_PATTERN = re.compile(
    r"^[+\-−]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?$"
)


@dataclass(frozen=True)
class CalibrationResult:
    x_mapping: AxisMapping
    y_mapping: AxisMapping
    diagnostics: Dict[str, object] = field(default_factory=dict)

    @property
    def calibrated(self) -> bool:
        return self.x_mapping.calibrated and self.y_mapping.calibrated


def _parse_number(text: str) -> Optional[float]:
    normalized = text.strip().replace("−", "-").replace(",", "")
    if not _NUMBER_PATTERN.match(normalized):
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def _create_ocr():
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(
            "Axis calibration requires the optional 'paddleocr' package"
        ) from exc
    try:
        return PaddleOCR(use_textline_orientation=True, lang="en")
    except TypeError:
        return PaddleOCR(use_angle_cls=True, lang="en")


def _normalize_ocr_result(result) -> List[Tuple[np.ndarray, str, float]]:
    if not result:
        return []
    first = result[0]
    if first is None:
        return []
    if isinstance(first, list):
        normalized = []
        for box, recognition in first:
            text, score = recognition
            normalized.append((np.asarray(box, dtype=float), str(text), float(score)))
        return normalized

    polygons = first.get("dt_polys", first.get("rec_polys", []))
    texts = first.get("rec_texts", [])
    scores = first.get("rec_scores", [])
    normalized = []
    for index, text in enumerate(texts):
        if index >= len(polygons):
            continue
        score = scores[index] if index < len(scores) else 0.0
        normalized.append(
            (np.asarray(polygons[index], dtype=float), str(text), float(score))
        )
    return normalized


def extract_numeric_ticks(
    image_bgr: np.ndarray,
    axis_line: Sequence[float],
    axis_type: str,
    ocr=None,
) -> List[AxisTick]:
    """Extract numeric ticks near one detected axis in original coordinates."""

    if axis_type not in {"x", "y"}:
        raise ValueError("axis_type must be 'x' or 'y'")
    if ocr is None:
        ocr = _create_ocr()

    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in axis_line]
    if axis_type == "x":
        margin = max(4, round(height / 10))
        start = max(0, y1 - margin)
        end = min(height, y1 + margin)
        roi = image_bgr[start:end, :]
        offset_x, offset_y = 0, start
        maximum_distance = max(3, round(height / 15))
    else:
        margin = max(4, round(width / 10))
        start = max(0, x1 - margin)
        end = min(width, x1 + margin)
        roi = image_bgr[:, start:end]
        offset_x, offset_y = start, 0
        maximum_distance = max(3, round(width / 15))

    raw = ocr.ocr(roi)
    observations = _normalize_ocr_result(raw)
    candidates: List[Tuple[AxisTick, float]] = []
    for polygon, text, confidence in observations:
        value = _parse_number(text)
        if value is None or polygon.size == 0:
            continue
        center_x = float(polygon[:, 0].mean() + offset_x)
        center_y = float(polygon[:, 1].mean() + offset_y)
        distance = abs(center_y - y1) if axis_type == "x" else abs(center_x - x1)
        if distance > maximum_distance:
            continue
        pixel = center_x if axis_type == "x" else center_y
        candidates.append((AxisTick(pixel=pixel, value=value), confidence))

    # Retain the highest-confidence observation for each numeric value.
    by_value = {}
    for tick, confidence in candidates:
        current = by_value.get(tick.value)
        if current is None or confidence > current[1]:
            by_value[tick.value] = (tick, confidence)
    return sorted(
        (item[0] for item in by_value.values()),
        key=lambda tick: tick.pixel,
    )


def generate_equidistant_ticks(
    axis_line: Sequence[float],
    axis_type: str,
    count: int = 6,
) -> List[AxisTick]:
    """Generate evenly spaced pixel/value pairs along one detected axis."""

    if axis_type not in {"x", "y"}:
        raise ValueError("axis_type must be 'x' or 'y'")
    if int(count) < 2:
        raise ValueError("count must be at least two")

    x1, y1, x2, y2 = [int(round(value)) for value in axis_line]
    if axis_type == "x":
        start, end = min(x1, x2), max(x1, x2)
        pixels = np.linspace(start, end, int(count)).astype(int)
        values = np.linspace(start, end, int(count))
    else:
        start, end = min(y1, y2), max(y1, y2)
        pixels = np.linspace(start, end, int(count)).astype(int)
        values = np.linspace(end, start, int(count))
    return [
        AxisTick(pixel=float(pixel), value=float(value))
        for pixel, value in zip(pixels, values)
    ]


def fit_axis_mapping(
    ticks: Iterable[AxisTick],
    config: CalibrationConfig,
) -> AxisMapping:
    """Fit the best linear or log-value mapping from pixels to data units."""

    tick_list = list(ticks)
    if len(tick_list) < config.minimum_ticks:
        return AxisMapping(
            kind="unavailable",
            coefficients=(1.0, 0.0),
            ticks=tick_list,
            calibrated=False,
        )

    pixels = np.asarray([tick.pixel for tick in tick_list], dtype=float)
    values = np.asarray([tick.value for tick in tick_list], dtype=float)
    if np.unique(pixels).size < 2:
        return AxisMapping(
            kind="unavailable",
            coefficients=(1.0, 0.0),
            ticks=tick_list,
            calibrated=False,
        )

    linear = np.polyfit(pixels, values, 1)
    linear_prediction = np.polyval(linear, pixels)
    scale = max(float(np.ptp(values)), 1e-12)
    linear_error = float(np.sqrt(np.mean((linear_prediction - values) ** 2)) / scale)

    best_kind = "linear"
    best_coefficients = (float(linear[0]), float(linear[1]))
    best_error = linear_error
    if config.allow_log_scale and np.all(values > 0):
        log_values = np.log10(values)
        log_fit = np.polyfit(pixels, log_values, 1)
        log_prediction = np.power(10.0, np.polyval(log_fit, pixels))
        log_error = float(
            np.sqrt(np.mean((log_prediction - values) ** 2)) / scale
        )
        if log_error < best_error:
            best_kind = "log10"
            best_coefficients = (float(log_fit[0]), float(log_fit[1]))

    return AxisMapping(
        kind=best_kind,
        coefficients=best_coefficients,
        ticks=tick_list,
        calibrated=True,
    )


def _calibrate_one_axis(
    image_bgr: np.ndarray,
    axis_line: Sequence[float],
    axis_type: str,
    config: CalibrationConfig,
    ocr,
) -> Tuple[AxisMapping, Dict[str, object]]:
    primary_ticks = extract_numeric_ticks(
        image_bgr,
        axis_line,
        axis_type,
        ocr=ocr,
    )
    use_synthetic = (
        config.synthetic_fallback
        and len(primary_ticks) < config.minimum_ticks
    )
    selected_ticks = (
        generate_equidistant_ticks(
            axis_line,
            axis_type,
            count=config.synthetic_tick_count,
        )
        if use_synthetic
        else primary_ticks
    )
    mapping = fit_axis_mapping(selected_ticks, config)
    diagnostics = {
        "primary_method": "ocr_ticks",
        "primary_tick_count": len(primary_ticks),
        "primary_status": "failed" if use_synthetic else "success",
        "fallback_triggered": use_synthetic,
        "fallback_method": "equidistant_ticks" if use_synthetic else None,
        "fallback_status": "rescued" if use_synthetic else "not_triggered",
        "selected_method": "equidistant_ticks" if use_synthetic else "ocr_ticks",
        "synthetic_ticks_used": use_synthetic,
        "selected_tick_count": len(selected_ticks),
        "scale_type": mapping.kind,
    }
    return mapping, diagnostics


def calibrate_axes(
    image_bgr: np.ndarray,
    axes: AxisDetection,
    config: CalibrationConfig,
    ocr=None,
) -> CalibrationResult:
    if axes.x_axis is None or axes.y_axis is None:
        unavailable = AxisMapping(
            kind="unavailable",
            coefficients=(1.0, 0.0),
            ticks=[],
            calibrated=False,
        )
        empty = {
            "primary_method": "ocr_ticks",
            "primary_tick_count": 0,
            "primary_status": "failed",
            "fallback_triggered": False,
            "fallback_method": None,
            "fallback_status": "not_triggered",
            "selected_method": None,
            "synthetic_ticks_used": False,
            "selected_tick_count": 0,
            "scale_type": "unavailable",
        }
        return CalibrationResult(
            unavailable,
            unavailable,
            diagnostics={
                "x": dict(empty),
                "y": dict(empty),
                "calibrated": False,
                "used_synthetic_ticks": False,
            },
        )

    shared_ocr = ocr if ocr is not None else _create_ocr()
    x_mapping, x_diagnostics = _calibrate_one_axis(
        image_bgr,
        axes.x_axis,
        "x",
        config,
        shared_ocr,
    )
    y_mapping, y_diagnostics = _calibrate_one_axis(
        image_bgr,
        axes.y_axis,
        "y",
        config,
        shared_ocr,
    )
    calibrated = x_mapping.calibrated and y_mapping.calibrated
    used_synthetic = bool(
        x_diagnostics["synthetic_ticks_used"]
        or y_diagnostics["synthetic_ticks_used"]
    )
    return CalibrationResult(
        x_mapping=x_mapping,
        y_mapping=y_mapping,
        diagnostics={
            "x": x_diagnostics,
            "y": y_diagnostics,
            "calibrated": calibrated,
            "used_synthetic_ticks": used_synthetic,
        },
    )


def map_value(mapping: AxisMapping, pixel: float) -> float:
    slope, intercept = mapping.coefficients
    transformed = slope * float(pixel) + intercept
    if mapping.kind == "log10":
        return float(math.pow(10.0, transformed))
    return float(transformed)


def map_polyline(
    pixel_points: Iterable[Point],
    calibration: CalibrationResult,
) -> List[Point]:
    if not calibration.calibrated:
        return []
    return [
        (
            map_value(calibration.x_mapping, x),
            map_value(calibration.y_mapping, y),
        )
        for x, y in pixel_points
    ]
