"""Minimal one-pass CV+VLM axis recovery."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from openai import OpenAI

from .schema import AxisDetection, Box

AxisLine = Tuple[int, int, int, int]


def _line_from_payload(value) -> AxisLine:
    if not isinstance(value, list) or len(value) != 4:
        raise RuntimeError("Axis fallback returned an invalid line")
    try:
        floats = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Axis fallback returned an invalid line") from exc
    if not all(np.isfinite(item) for item in floats):
        raise RuntimeError("Axis fallback returned an invalid line")
    return tuple(int(round(item)) for item in floats)


class AxisSelectionClient:
    """Select Hough candidates with one OpenAI-compatible VLM provider."""

    def __init__(self, api_key=None, base_url=None, model=None, timeout=60.0):
        key = (
            api_key
            or os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not key:
            raise RuntimeError("Axis fallback requires a configured API key")
        self.model = model or os.environ.get(
            "MATDEPLOT_AXIS_FALLBACK_MODEL",
            "qwen-vl-max",
        )
        endpoint = base_url or os.environ.get(
            "MATDEPLOT_AXIS_FALLBACK_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.client = OpenAI(api_key=key, base_url=endpoint, timeout=timeout)

    def _complete(self, image_bgr: np.ndarray, prompt: str) -> Dict[str, object]:
        ok, encoded = cv2.imencode(".png", image_bgr)
        if not ok:
            raise RuntimeError("Could not encode axis fallback image")
        payload = base64.b64encode(encoded.tobytes()).decode("ascii")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{payload}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            text = response.choices[0].message.content or ""
        except Exception as exc:
            raise RuntimeError("Axis fallback provider request failed") from exc
        start, end = text.find("{"), text.rfind("}") + 1
        if start < 0 or end <= 0:
            raise RuntimeError("Axis fallback response was not valid JSON")
        try:
            parsed = json.loads(text[start:end])
        except json.JSONDecodeError as exc:
            raise RuntimeError("Axis fallback response was not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("Axis fallback response was not valid JSON")
        return parsed

    def select_candidates(
        self,
        image_bgr: np.ndarray,
        candidates: Sequence[AxisLine],
        orientation: str,
    ) -> List[AxisLine]:
        prompt = (
            "Select the two most likely scientific-chart axis lines from the "
            f"{orientation} candidates {list(candidates)}. Prioritize long lines "
            "near the plot boundary with nearby tick labels. Return strict JSON: "
            '{"primary_axis":[x1,y1,x2,y2],'
            '"secondary_axis":[x1,y1,x2,y2]}'
        )
        payload = self._complete(image_bgr, prompt)
        selected = [_line_from_payload(payload.get("primary_axis"))]
        secondary = payload.get("secondary_axis")
        if secondary is not None:
            selected.append(_line_from_payload(secondary))
        return selected[:2]

    def select_axes(self, marked_bgr: np.ndarray) -> Dict[str, str]:
        prompt = (
            "Choose the main x and y axes in this scientific chart. Horizontal "
            "candidates are red/orange and vertical candidates are green/blue. "
            'Return strict JSON: {"x_axis":"red|orange",'
            '"y_axis":"green|blue"}'
        )
        payload = self._complete(marked_bgr, prompt)
        x_color = payload.get("x_axis")
        y_color = payload.get("y_axis")
        if x_color not in {"red", "orange"} or y_color not in {"green", "blue"}:
            raise RuntimeError("Axis fallback selected an unknown candidate")
        return {"x_axis": x_color, "y_axis": y_color}


def _preprocess(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    enhanced = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(16, 16),
    ).apply(blurred)
    return cv2.GaussianBlur(enhanced, (3, 3), 0)


def _hough_axis_candidates(image_bgr: np.ndarray) -> List[AxisLine]:
    edges = cv2.Canny(_preprocess(image_bgr), 10, 150)
    _, width = edges.shape
    parameters = [
        (70, max(100, width // 4), 5),
        (100, max(150, width // 3), 2),
    ]
    candidates: List[AxisLine] = []
    for threshold, minimum_length, maximum_gap in parameters:
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=threshold,
            minLineLength=minimum_length,
            maxLineGap=maximum_gap,
        )
        if lines is not None:
            candidates.extend(tuple(map(int, line[0])) for line in lines)

    unique: List[AxisLine] = []
    seen = set()
    for line in candidates:
        x1, y1, x2, y2 = line
        if np.hypot(x2 - x1, y2 - y1) < 100:
            continue
        normalized = (
            (x1, y1, x2, y2)
            if (x1, y1) <= (x2, y2)
            else (x2, y2, x1, y1)
        )
        if normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def _orientation(line: AxisLine) -> Optional[str]:
    x1, y1, x2, y2 = line
    angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
    if min(angle, 180 - angle) <= 2:
        return "horizontal"
    if abs(angle - 90) <= 2:
        return "vertical"
    return None


def _draw_selected(
    image_bgr: np.ndarray,
    horizontal: Sequence[AxisLine],
    vertical: Sequence[AxisLine],
) -> Tuple[np.ndarray, Dict[str, AxisLine]]:
    if not horizontal or not vertical:
        raise RuntimeError("Axis fallback did not retain both orientations")
    marked = image_bgr.copy()
    labels = {}
    palette = [
        ("red", (0, 0, 255), horizontal[0]),
        ("orange", (0, 165, 255), horizontal[min(1, len(horizontal) - 1)]),
        ("green", (0, 255, 0), vertical[0]),
        ("blue", (255, 0, 0), vertical[min(1, len(vertical) - 1)]),
    ]
    for name, color, line in palette:
        cv2.line(marked, line[:2], line[2:], color, 5)
        labels[name] = line
    return marked, labels


def _to_detection(x_line: AxisLine, y_line: AxisLine) -> AxisDetection:
    _, x_right = sorted((float(x_line[0]), float(x_line[2])))
    x_y = (float(x_line[1]) + float(x_line[3])) / 2.0
    y_top, _ = sorted((float(y_line[1]), float(y_line[3])))
    y_x = (float(y_line[0]) + float(y_line[2])) / 2.0
    x_axis = (y_x, x_y, x_right, x_y)
    y_axis = (y_x, y_top, y_x, x_y)
    plot_box = Box(x1=y_x, y1=y_top, x2=x_right, y2=x_y)
    return AxisDetection(x_axis=x_axis, y_axis=y_axis, plot_box=plot_box)


def detect_axes_cv_vlm(
    image_path: Path,
    client: Optional[AxisSelectionClient] = None,
) -> AxisDetection:
    """Recover plot axes from Hough candidates and one VLM selection pass."""

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    candidates = _hough_axis_candidates(image)
    horizontal = [line for line in candidates if _orientation(line) == "horizontal"]
    vertical = [line for line in candidates if _orientation(line) == "vertical"]
    if not horizontal or not vertical:
        raise RuntimeError("Axis fallback found insufficient candidates")

    selector = client or AxisSelectionClient()
    selected_vertical = selector.select_candidates(image, vertical, "vertical")
    selected_horizontal = selector.select_candidates(image, horizontal, "horizontal")
    marked, labels = _draw_selected(image, selected_horizontal, selected_vertical)
    selected = selector.select_axes(marked)
    try:
        x_line = labels[selected["x_axis"]]
        y_line = labels[selected["y_axis"]]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Axis fallback selected an unknown candidate") from exc
    return _to_detection(x_line, y_line)
