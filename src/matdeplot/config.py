"""Typed configuration for the released MatDeplot pipeline.

All paper-facing entry points load the same JSON configuration. Keeping the
published recipe in one place prevents document and chart inference from
silently drifting to legacy experiment defaults.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class ModelFiles:
    subfigure: str
    axis: str
    legend: str
    foreground: str


@dataclass(frozen=True)
class ForegroundConfig:
    encoder: str
    hidden_dim: int
    n_queries: int
    decoder_layers: int
    threshold: float
    objectness_threshold: float
    fallback_fraction: float
    tta_scales: Tuple[int, ...]
    flip_tta: bool


@dataclass(frozen=True)
class SeparationConfig:
    mode: str
    spatial_weight: float
    kmeans_n_init: int
    random_state: int
    morph_close_kernel: int
    dilation_kernel: int
    min_cluster_area: int
    delta_e_threshold: float


@dataclass(frozen=True)
class CurveCountConfig:
    paper_provider: str
    paper_model: str
    open_fallback: str
    candidate_min: int
    candidate_max: int


@dataclass(frozen=True)
class CalibrationConfig:
    minimum_ticks: int
    allow_log_scale: bool
    synthetic_fallback: bool
    synthetic_tick_count: int


@dataclass(frozen=True)
class AbstentionConfig:
    threshold: float


@dataclass(frozen=True)
class MatDeplotConfig:
    schema_version: str
    models: ModelFiles
    foreground: ForegroundConfig
    separation: SeparationConfig
    curve_count: CurveCountConfig
    calibration: CalibrationConfig
    abstention: AbstentionConfig

    def model_paths(self, model_directory: Path) -> Dict[str, Path]:
        root = Path(model_directory)
        return {
            "subfigure": root / self.models.subfigure,
            "axis": root / self.models.axis,
            "legend": root / self.models.legend,
            "foreground": root / self.models.foreground,
        }


_TOP_LEVEL_KEYS = {
    "schema_version",
    "models",
    "foreground",
    "separation",
    "curve_count",
    "calibration",
    "abstention",
}


def _require_keys(section: str, payload: Dict[str, Any], expected: set) -> None:
    missing = expected - set(payload)
    extra = set(payload) - expected
    if missing or extra:
        raise ValueError(
            f"Invalid {section} configuration: "
            f"missing={sorted(missing)}, unknown={sorted(extra)}"
        )


def load_config(path: Path) -> MatDeplotConfig:
    """Load and validate a MatDeplot JSON configuration."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require_keys("top-level", payload, _TOP_LEVEL_KEYS)

    model_keys = {"subfigure", "axis", "legend", "foreground"}
    foreground_keys = {
        "encoder",
        "hidden_dim",
        "n_queries",
        "decoder_layers",
        "threshold",
        "objectness_threshold",
        "fallback_fraction",
        "tta_scales",
        "flip_tta",
    }
    separation_keys = {
        "mode",
        "spatial_weight",
        "kmeans_n_init",
        "random_state",
        "morph_close_kernel",
        "dilation_kernel",
        "min_cluster_area",
        "delta_e_threshold",
    }
    count_keys = {
        "paper_provider",
        "paper_model",
        "open_fallback",
        "candidate_min",
        "candidate_max",
    }
    calibration_keys = {
        "minimum_ticks",
        "allow_log_scale",
        "synthetic_fallback",
        "synthetic_tick_count",
    }
    abstention_keys = {"threshold"}

    _require_keys("models", payload["models"], model_keys)
    _require_keys("foreground", payload["foreground"], foreground_keys)
    _require_keys("separation", payload["separation"], separation_keys)
    _require_keys("curve_count", payload["curve_count"], count_keys)
    _require_keys("calibration", payload["calibration"], calibration_keys)
    _require_keys("abstention", payload["abstention"], abstention_keys)

    fg = dict(payload["foreground"])
    fg["tta_scales"] = tuple(int(value) for value in fg["tta_scales"])
    config = MatDeplotConfig(
        schema_version=str(payload["schema_version"]),
        models=ModelFiles(**payload["models"]),
        foreground=ForegroundConfig(**fg),
        separation=SeparationConfig(**payload["separation"]),
        curve_count=CurveCountConfig(**payload["curve_count"]),
        calibration=CalibrationConfig(**payload["calibration"]),
        abstention=AbstentionConfig(**payload["abstention"]),
    )
    _validate_values(config)
    return config


def _validate_values(config: MatDeplotConfig) -> None:
    if not 0.0 < config.foreground.threshold < 1.0:
        raise ValueError("foreground.threshold must lie in (0, 1)")
    if not 0.0 <= config.foreground.fallback_fraction < 1.0:
        raise ValueError("foreground.fallback_fraction must lie in [0, 1)")
    if not config.foreground.tta_scales:
        raise ValueError("foreground.tta_scales must not be empty")
    if config.separation.mode != "v4":
        raise ValueError("The released paper configuration requires mode='v4'")
    if config.curve_count.candidate_min < 1:
        raise ValueError("curve_count.candidate_min must be at least one")
    if config.curve_count.candidate_max < config.curve_count.candidate_min:
        raise ValueError("curve-count candidate range is invalid")
    if config.calibration.minimum_ticks < 2:
        raise ValueError("calibration.minimum_ticks must be at least two")
    if config.calibration.synthetic_tick_count < 2:
        raise ValueError("calibration.synthetic_tick_count must be at least two")
    if not 0.0 <= config.abstention.threshold <= 1.0:
        raise ValueError("abstention.threshold must lie in [0, 1]")


def load_paper_config() -> MatDeplotConfig:
    """Load the immutable configuration used by the paper headline."""

    repository_root = Path(__file__).resolve().parents[2]
    return load_config(repository_root / "configs" / "paper.json")
