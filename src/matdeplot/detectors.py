"""YOLO detectors for subfigures, plot bounds, and legends."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .schema import AxisDetection, Box

_MODEL_CACHE: Dict[str, object] = {}


def _load_yolo(checkpoint: Path):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"YOLO checkpoint not found: {checkpoint}")
    key = str(checkpoint.resolve())
    if key not in _MODEL_CACHE:
        from ultralytics import YOLO

        _MODEL_CACHE[key] = YOLO(key)
    return _MODEL_CACHE[key]


def _predict_boxes(
    image_path: Path,
    checkpoint: Path,
    confidence: float,
) -> List[Box]:
    model = _load_yolo(checkpoint)
    prediction = model.predict(
        source=str(image_path),
        conf=float(confidence),
        verbose=False,
    )[0]
    boxes: List[Box] = []
    for detected in prediction.boxes:
        x1, y1, x2, y2 = map(float, detected.xyxy[0].tolist())
        score = float(detected.conf[0])
        boxes.append(Box(x1=x1, y1=y1, x2=x2, y2=y2, confidence=score))
    return boxes


def detect_axes(
    image_path: Path,
    checkpoint: Path,
    confidence: float = 0.5,
) -> AxisDetection:
    """Detect the plot box and convert it to horizontal and vertical axes."""

    boxes = _predict_boxes(image_path, checkpoint, confidence)
    if not boxes:
        return AxisDetection(x_axis=None, y_axis=None, plot_box=None)
    best = max(
        boxes,
        key=lambda box: box.confidence if box.confidence is not None else 0.0,
    )
    x_axis = (best.x1, best.y2, best.x2, best.y2)
    y_axis = (best.x1, best.y1, best.x1, best.y2)
    return AxisDetection(x_axis=x_axis, y_axis=y_axis, plot_box=best)


def axes_are_valid(axes: AxisDetection) -> bool:
    """Return whether an axis result contains two finite, nonzero lines."""

    if not isinstance(axes, AxisDetection):
        return False
    for line in (axes.x_axis, axes.y_axis):
        if line is None or len(line) != 4:
            return False
        try:
            values = [float(value) for value in line]
        except (TypeError, ValueError):
            return False
        if not all(np.isfinite(value) for value in values):
            return False
        if np.hypot(values[2] - values[0], values[3] - values[1]) <= 1.0:
            return False
    return axes.plot_box is not None


def detect_axes_with_fallback(
    image_path: Path,
    checkpoint: Path,
    confidence: float = 0.5,
    fallback=None,
):
    """Run Axis YOLO first and recover invalid results with CV+VLM."""

    diagnostics = {
        "primary_method": "axis_yolo",
        "primary_status": "failed",
        "primary_error": None,
        "fallback_triggered": False,
        "fallback_method": "cv_vlm_axes",
        "fallback_status": "not_triggered",
        "fallback_error": None,
        "selected_method": None,
    }
    try:
        primary = detect_axes(image_path, checkpoint, confidence=confidence)
        if axes_are_valid(primary):
            diagnostics.update(
                primary_status="success",
                selected_method="axis_yolo",
            )
            return primary, diagnostics
        diagnostics["primary_error"] = "invalid_or_missing_axes"
    except Exception as exc:
        diagnostics["primary_error"] = f"{type(exc).__name__}: {exc}"

    diagnostics["fallback_triggered"] = True
    detector = fallback
    if detector is None:
        from .axis_fallback import detect_axes_cv_vlm

        detector = detect_axes_cv_vlm
    try:
        recovered = detector(image_path)
        if not axes_are_valid(recovered):
            raise RuntimeError("invalid_or_missing_axes")
    except Exception as exc:
        diagnostics["fallback_status"] = "failed"
        diagnostics["fallback_error"] = f"{type(exc).__name__}: {exc}"
        return None, diagnostics

    diagnostics["fallback_status"] = "rescued"
    diagnostics["selected_method"] = "cv_vlm_axes"
    return recovered, diagnostics


def detect_legends(
    image_path: Path,
    checkpoint: Path,
    confidence: float = 0.3,
) -> List[Box]:
    """Return every legend or clutter box above the confidence threshold."""

    return _predict_boxes(image_path, checkpoint, confidence)


def detect_subfigures(
    image_path: Path,
    checkpoint: Path,
    confidence: float = 0.25,
) -> List[Box]:
    """Return subfigure boxes in deterministic reading order."""

    boxes = _predict_boxes(image_path, checkpoint, confidence)
    return sorted(boxes, key=lambda box: (round(box.y1 / 20.0), box.x1))


def build_valid_region(
    image_shape: Tuple[int, ...],
    axes: AxisDetection,
    legend_boxes: List[Box],
    plot_padding: int = 4,
) -> np.ndarray:
    """Build a boolean mask for pixels eligible for curve extraction."""

    height, width = image_shape[:2]
    valid = np.ones((height, width), dtype=bool)
    if axes.plot_box is not None:
        valid[:] = False
        # Keep the production implementation's truncation semantics. A
        # round-to-nearest conversion changes the eligible region by one pixel
        # and consequently changes the paper configuration's clustering.
        x1 = max(0, int(axes.plot_box.x1) + plot_padding)
        y1 = max(0, int(axes.plot_box.y1) + plot_padding)
        x2 = min(width, int(axes.plot_box.x2) - plot_padding)
        y2 = min(height, int(axes.plot_box.y2) - plot_padding)
        if x2 > x1 and y2 > y1:
            valid[y1:y2, x1:x2] = True

    for box in legend_boxes:
        x1 = max(0, int(box.x1))
        y1 = max(0, int(box.y1))
        x2 = min(width, int(box.x2))
        y2 = min(height, int(box.y2))
        if x2 > x1 and y2 > y1:
            valid[y1:y2, x1:x2] = False
    return valid


def read_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return image
