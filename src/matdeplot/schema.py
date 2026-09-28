"""Public, JSON-serializable contracts for MatDeplot outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Point = Tuple[float, float]


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: Optional[float] = None


@dataclass
class AxisDetection:
    x_axis: Optional[Tuple[float, float, float, float]]
    y_axis: Optional[Tuple[float, float, float, float]]
    plot_box: Optional[Box]


@dataclass
class AxisTick:
    pixel: float
    value: float


@dataclass
class AxisMapping:
    kind: str
    coefficients: Tuple[float, float]
    ticks: List[AxisTick]
    calibrated: bool


@dataclass
class CurveRecord:
    curve_index: int
    pixel_points: List[Point]
    real_points: List[Point]
    confidence: float
    abstain: bool


@dataclass
class FailureRecord:
    stage: str
    message: str
    exception_type: Optional[str] = None


@dataclass
class ExtractionResult:
    image_id: str
    image_size: Tuple[int, int]
    success: bool
    used_fallback: bool = False
    curve_count_source: str = ""
    requested_curve_count: Optional[int] = None
    foreground_fraction: float = 0.0
    axes: Optional[AxisDetection] = None
    legend_boxes: List[Box] = field(default_factory=list)
    curves: List[CurveRecord] = field(default_factory=list)
    failures: List[FailureRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
