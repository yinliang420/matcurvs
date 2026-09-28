"""End-to-end chart-level MatDeplot extraction."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from .calibration import CalibrationResult, calibrate_axes, map_polyline
from .config import MatDeplotConfig, load_paper_config
from .curve_count import (
    CurveCountResult,
    GeminiCurveCounter,
    silhouette_curve_count,
)
from .detectors import (
    build_valid_region,
    detect_axes_with_fallback,
    detect_legends,
    read_image,
)
from .foreground import get_line_prob_multiscale
from .polyline import mask_to_polyline
from .schema import (
    CurveRecord,
    ExtractionResult,
    FailureRecord,
)
from .separation import prepare_pixel_mask, separate_curves


class MatDeplotPipeline:
    """Run the paper's Stage-3 cells on a single chart image."""

    def __init__(
        self,
        model_directory: Path,
        config: Optional[MatDeplotConfig] = None,
    ) -> None:
        self.config = config or load_paper_config()
        self.model_paths = self.config.model_paths(Path(model_directory))

    def extract(
        self,
        image_path: Path,
        curve_count_source: str = "silhouette",
        explicit_curve_count: Optional[int] = None,
        cached_counts: Optional[Dict[str, int]] = None,
        calibrate: bool = True,
    ) -> ExtractionResult:
        image_path = Path(image_path)
        started = time.monotonic()
        failures = []
        try:
            image = read_image(image_path)
        except Exception as exc:
            return ExtractionResult(
                image_id=image_path.stem,
                image_size=(0, 0),
                success=False,
                failures=[
                    FailureRecord(
                        stage="input",
                        message=str(exc),
                        exception_type=type(exc).__name__,
                    )
                ],
            )

        height, width = image.shape[:2]
        axes, axis_diagnostics = detect_axes_with_fallback(
            image_path,
            self.model_paths["axis"],
            confidence=0.5,
        )
        if axes is None:
            error = RuntimeError(
                axis_diagnostics.get("fallback_error")
                or "axis detection failed"
            )
            return self._failed_result(
                image_path,
                width,
                height,
                "axis_detection",
                error,
                started,
                metadata={"axis_detection": axis_diagnostics},
            )

        try:
            legend_boxes = detect_legends(
                image_path,
                self.model_paths["legend"],
                confidence=0.3,
            )
        except Exception as exc:
            legend_boxes = []
            failures.append(
                FailureRecord(
                    stage="legend_detection",
                    message=str(exc),
                    exception_type=type(exc).__name__,
                )
            )

        valid_region = build_valid_region(
            image.shape,
            axes,
            legend_boxes,
        )
        try:
            foreground_probability = get_line_prob_multiscale(
                image_path,
                ckpt_path=self.model_paths["foreground"],
                scales=self.config.foreground.tta_scales,
                obj_threshold=self.config.foreground.objectness_threshold,
                flip_tta=self.config.foreground.flip_tta,
            )
        except Exception as exc:
            return self._failed_result(
                image_path,
                width,
                height,
                "foreground",
                exc,
                started,
                failures=failures,
                axes=axes,
                legend_boxes=legend_boxes,
                metadata={"axis_detection": axis_diagnostics},
            )

        foreground_mask = (
            foreground_probability >= self.config.foreground.threshold
        ) & valid_region
        foreground_fraction = float(foreground_mask.mean())
        used_fallback = (
            foreground_fraction < self.config.foreground.fallback_fraction
        )
        pixel_mask = prepare_pixel_mask(
            image,
            foreground_mask,
            valid_region,
            used_fallback,
            self.config.separation,
        )

        try:
            count_result = self._resolve_curve_count(
                image_path=image_path,
                image=image,
                pixel_mask=pixel_mask,
                source=curve_count_source,
                explicit=explicit_curve_count,
                cached_counts=cached_counts,
            )
        except Exception as exc:
            failures.append(
                FailureRecord(
                    stage="curve_count",
                    message=str(exc),
                    exception_type=type(exc).__name__,
                )
            )
            count_result = silhouette_curve_count(
                image,
                pixel_mask,
                self.config.curve_count,
                self.config.separation,
            )
            count_result = CurveCountResult(
                value=count_result.value,
                source=count_result.source,
                fallback_reason=f"{type(exc).__name__}: {exc}",
            )

        cluster_masks = separate_curves(
            image,
            pixel_mask,
            count_result.value,
            valid_region,
            self.config.separation,
        )

        calibration: Optional[CalibrationResult] = None
        calibration_diagnostics = {
            "calibrated": False,
            "used_synthetic_ticks": False,
            "skipped": not calibrate,
        }
        if calibrate:
            try:
                calibration = calibrate_axes(
                    image,
                    axes,
                    self.config.calibration,
                )
                calibration_diagnostics.update(calibration.diagnostics)
                calibration_diagnostics["calibrated"] = calibration.calibrated
                calibration_diagnostics["skipped"] = False
                if not calibration.calibrated:
                    failures.append(
                        FailureRecord(
                            stage="calibration",
                            message="Fewer than two reliable ticks were found on an axis",
                        )
                    )
            except Exception as exc:
                failures.append(
                    FailureRecord(
                        stage="calibration",
                        message=str(exc),
                        exception_type=type(exc).__name__,
                    )
                )

        curves = []
        for cluster_index, mask in enumerate(cluster_masks):
            pixel_points = mask_to_polyline(mask)
            if not pixel_points:
                continue
            confidence = self._curve_confidence(
                mask,
                foreground_probability,
                used_fallback,
            )
            real_points = (
                map_polyline(pixel_points, calibration)
                if calibration is not None
                else []
            )
            curves.append(
                CurveRecord(
                    curve_index=cluster_index,
                    pixel_points=pixel_points,
                    real_points=real_points,
                    confidence=confidence,
                    abstain=confidence < self.config.abstention.threshold,
                )
            )

        return ExtractionResult(
            image_id=image_path.stem,
            image_size=(width, height),
            success=True,
            used_fallback=used_fallback,
            curve_count_source=count_result.source,
            requested_curve_count=count_result.value,
            foreground_fraction=foreground_fraction,
            axes=axes,
            legend_boxes=legend_boxes,
            curves=curves,
            failures=failures,
            metadata={
                "schema_version": self.config.schema_version,
                "curve_count_model": count_result.model,
                "curve_count_fallback_reason": count_result.fallback_reason,
                "axis_detection": axis_diagnostics,
                "calibration": calibration_diagnostics,
                "elapsed_seconds": time.monotonic() - started,
                "paper_configuration": True,
            },
        )

    def _resolve_curve_count(
        self,
        image_path: Path,
        image: np.ndarray,
        pixel_mask: np.ndarray,
        source: str,
        explicit: Optional[int],
        cached_counts: Optional[Dict[str, int]],
    ) -> CurveCountResult:
        if source == "explicit":
            if explicit is None or int(explicit) < 1:
                raise ValueError("explicit curve count must be a positive integer")
            return CurveCountResult(value=int(explicit), source="explicit")
        if source == "cached":
            if cached_counts is None or image_path.stem not in cached_counts:
                raise KeyError(f"no cached curve count for {image_path.stem}")
            return CurveCountResult(
                value=int(cached_counts[image_path.stem]),
                source="cached",
            )
        if source == "gemini":
            counter = GeminiCurveCounter(self.config.curve_count.paper_model)
            return counter.count(image_path)
        if source == "silhouette":
            return silhouette_curve_count(
                image,
                pixel_mask,
                self.config.curve_count,
                self.config.separation,
            )
        raise ValueError(
            "curve_count_source must be explicit, cached, gemini, or silhouette"
        )

    @staticmethod
    def _curve_confidence(
        mask: np.ndarray,
        foreground_probability: np.ndarray,
        used_fallback: bool,
    ) -> float:
        if not mask.any():
            return 0.0
        if not used_fallback:
            return float(foreground_probability[mask].mean())
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ring = (cv2.dilate(mask.astype(np.uint8), kernel) > 0) & ~mask
        inner_area = float(mask.sum())
        outer_area = float(ring.sum())
        return inner_area / (inner_area + max(outer_area, 1.0))

    @staticmethod
    def _failed_result(
        image_path: Path,
        width: int,
        height: int,
        stage: str,
        error: Exception,
        started: float,
        failures=None,
        axes=None,
        legend_boxes=None,
        metadata=None,
    ) -> ExtractionResult:
        all_failures = list(failures or [])
        all_failures.append(
            FailureRecord(
                stage=stage,
                message=str(error),
                exception_type=type(error).__name__,
            )
        )
        result_metadata = dict(metadata or {})
        result_metadata["elapsed_seconds"] = time.monotonic() - started
        return ExtractionResult(
            image_id=image_path.stem,
            image_size=(width, height),
            success=False,
            axes=axes,
            legend_boxes=list(legend_boxes or []),
            failures=all_failures,
            metadata=result_metadata,
        )
