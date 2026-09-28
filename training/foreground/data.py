"""Selected V11 foreground-training data utilities."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
VALID_SOURCES = {"synthetic", "real"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class ChartRecord:
    image_path: Path
    annotation_path: Path
    source: str


def _normalize_root(root) -> Path:
    return Path(root)


def _require_valid_source(source: str) -> None:
    if source not in VALID_SOURCES:
        raise ValueError("source must be one of {0}".format(sorted(VALID_SOURCES)))


def _series_iterable(annotation: Mapping) -> List[Mapping]:
    if "lines" in annotation:
        container = annotation["lines"]
    elif "data_series" in annotation:
        container = annotation["data_series"]
    else:
        raise ValueError("Annotation must contain lines or data_series")

    if isinstance(container, dict):
        items = list(container.values())
    elif isinstance(container, list):
        items = list(container)
    else:
        raise ValueError("Annotation series container must be a dict or list")

    if not items:
        raise ValueError("Annotation series container is empty")
    return items


def _normalize_records(records: Sequence[ChartRecord], expected_source: Optional[str] = None) -> List[ChartRecord]:
    normalized = list(records)
    if not normalized:
        raise ValueError("At least one chart record is required")
    for record in normalized:
        if not isinstance(record, ChartRecord):
            raise TypeError("records must contain ChartRecord instances")
        _require_valid_source(record.source)
        if expected_source is not None and record.source != expected_source:
            raise ValueError("Expected all records to have source {0}".format(expected_source))
    return normalized


def _canonical_path_key(path) -> str:
    return str(Path(path).resolve())


def _validate_nonnegative_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("{0} must be an integer".format(name))
    if int(value) < 0:
        raise ValueError("{0} must be >= 0".format(name))
    return int(value)


def _validate_positive_int(value, name: str) -> int:
    normalized = _validate_nonnegative_int(value, name)
    if normalized < 1:
        raise ValueError("{0} must be >= 1".format(name))
    return normalized


def _collect_stem_map(paths: Sequence[Path], stem_suffix: str = "") -> Dict[str, Path]:
    by_stem: Dict[str, Path] = {}
    for path in sorted(paths):
        stem = path.name[: -len(stem_suffix)] if stem_suffix else path.stem
        previous = by_stem.get(stem)
        if previous is not None:
            raise ValueError(
                "Duplicate chart stem detected for {0}: {1} and {2}".format(
                    stem,
                    previous,
                    path,
                )
            )
        by_stem[stem] = path
    return by_stem


def collect_pairs(root, source: str, recursive: bool = False) -> List[ChartRecord]:
    _require_valid_source(source)
    root_path = _normalize_root(root)
    if not root_path.exists():
        raise FileNotFoundError("Data root not found: {0}".format(root_path))
    if not root_path.is_dir():
        raise NotADirectoryError("Data root is not a directory: {0}".format(root_path))

    finder = root_path.rglob if recursive else root_path.glob
    annotation_paths = list(finder("*_annotation.json"))
    image_paths = []
    for suffix in IMAGE_EXTENSIONS:
        image_paths.extend(finder("*{0}".format(suffix)))

    annotation_by_stem = _collect_stem_map(annotation_paths, stem_suffix="_annotation.json")
    image_by_stem = _collect_stem_map(image_paths)

    missing_image_stems = sorted(set(annotation_by_stem) - set(image_by_stem))
    if missing_image_stems:
        missing_stem = missing_image_stems[0]
        raise ValueError(
            "Missing paired image for annotation: {0}".format(annotation_by_stem[missing_stem])
        )

    missing_annotation_stems = sorted(set(image_by_stem) - set(annotation_by_stem))
    if missing_annotation_stems:
        missing_stem = missing_annotation_stems[0]
        raise ValueError(
            "Missing paired annotation for image: {0}".format(image_by_stem[missing_stem])
        )

    records = []
    for chart_stem, annotation_path in sorted(annotation_by_stem.items()):
        image_path = image_by_stem[chart_stem]
        records.append(
            ChartRecord(
                image_path=image_path,
                annotation_path=annotation_path,
                source=source,
            )
        )

    if not records:
        raise ValueError("No annotation/image pairs found under {0}".format(root_path))
    return sorted(records, key=lambda record: (record.image_path.stem, str(record.image_path)))


def _record_identity(record: ChartRecord) -> Tuple[str, str]:
    return (str(record.image_path.resolve()), str(record.annotation_path.resolve()))


def assert_non_overlapping_records(**split_records: Sequence[ChartRecord]) -> None:
    seen = {}
    for split_name, records in split_records.items():
        for record in records:
            identity = _record_identity(record)
            previous = seen.get(identity)
            if previous is not None:
                raise ValueError("Record overlap detected between splits {0} and {1}".format(previous, split_name))
            seen[identity] = split_name


def split_synthetic_pairs(records: Sequence[ChartRecord], val_ratio: float = 0.05, seed: int = 42):
    normalized = _normalize_records(records, expected_source="synthetic")
    if not 0.0 < float(val_ratio) < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    if len(normalized) < 2:
        raise ValueError("Synthetic split requires at least 2 records")

    indices = np.random.RandomState(seed).permutation(len(normalized))
    n_val = max(1, int(round(len(indices) * float(val_ratio))))
    if n_val >= len(indices):
        raise ValueError("Synthetic split would leave an empty training split")

    train = [normalized[index] for index in indices[n_val:]]
    val = [normalized[index] for index in indices[:n_val]]
    assert_non_overlapping_records(train=train, val=val)
    return train, val


def split_real_pairs(records: Sequence[ChartRecord], n_test: int = 40, n_val: int = 20, seed: int = 42):
    normalized = _normalize_records(records, expected_source="real")
    n_test = _validate_nonnegative_int(n_test, "n_test")
    n_val = _validate_nonnegative_int(n_val, "n_val")
    indices = np.random.RandomState(seed).permutation(len(normalized))
    test_end = min(n_test, len(indices))
    val_end = min(test_end + n_val, len(indices))
    train = [normalized[index] for index in indices[val_end:]]
    val = [normalized[index] for index in indices[test_end:val_end]]
    test = [normalized[index] for index in indices[:test_end]]
    assert_non_overlapping_records(train=train, val=val, test=test)
    return train, val, test


def build_train_records(
    real_train: Sequence[ChartRecord],
    synthetic_train: Sequence[ChartRecord],
    real_repeat: int,
    seed: int,
) -> List[ChartRecord]:
    real_train = list(real_train)
    synthetic_train = list(synthetic_train)
    if not real_train and not synthetic_train:
        raise ValueError("At least one training record is required")
    assert_non_overlapping_records(real=real_train, synthetic=synthetic_train)
    real_repeat = _validate_positive_int(real_repeat, "real_repeat")

    records = list(synthetic_train) + list(real_train) * max(1, real_repeat)
    random.Random(seed).shuffle(records)
    return records


def deterministic_split(records: Sequence[ChartRecord], val_fraction: float = 0.05, seed: int = 42):
    normalized = _normalize_records(records)
    if not 0.0 < float(val_fraction) < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if len(normalized) < 2:
        raise ValueError("deterministic_split would leave an empty training split")

    indices = np.random.RandomState(seed).permutation(len(normalized))
    n_val = max(1, int(round(len(indices) * float(val_fraction))))
    if n_val >= len(indices):
        raise ValueError("deterministic_split would leave an empty training split")

    train = [normalized[index] for index in indices[n_val:]]
    val = [normalized[index] for index in indices[:n_val]]
    assert_non_overlapping_records(train=train, val=val)
    return train, val


def explicit_split(train_records: Sequence[ChartRecord], val_records: Sequence[ChartRecord]):
    train = _normalize_records(train_records)
    val = _normalize_records(val_records)
    assert_non_overlapping_records(train=train, val=val)
    return train, val


def load_annotation(annotation_path) -> Mapping:
    path = Path(annotation_path)
    if not path.is_file():
        raise FileNotFoundError("Annotation file not found: {0}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        annotation = json.load(handle)
    if not isinstance(annotation, dict):
        raise ValueError("Annotation root must be a JSON object")
    return annotation


def _normalize_image_shape(image_shape) -> Tuple[int, int]:
    if isinstance(image_shape, np.ndarray):
        height, width = image_shape.shape[:2]
    elif len(image_shape) == 2:
        height, width = int(image_shape[0]), int(image_shape[1])
    elif len(image_shape) >= 3:
        height, width = int(image_shape[0]), int(image_shape[1])
    else:
        raise ValueError("image_shape must describe at least height and width")
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must be positive")
    return height, width


def _normalize_point(point, width: int, height: int) -> Tuple[int, int]:
    if isinstance(point, dict):
        if "pixel" in point:
            coords = point["pixel"]
        elif "pixel_coords" in point:
            coords = point["pixel_coords"]
        else:
            raise ValueError("Annotation point is missing pixel or pixel_coords")
    else:
        coords = point

    if not isinstance(coords, (list, tuple)) or len(coords) != 2:
        raise ValueError("Annotation point must be a two-item coordinate")

    x = float(coords[0])
    y = float(coords[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError("Annotation point coordinates must be finite")
    if not (0.0 <= x < width and 0.0 <= y < height):
        raise ValueError("Annotation point is out of bounds for the image")
    return int(round(x)), int(round(y))


def _scatter_radius(series: Mapping) -> int:
    marker_size = int(series.get("marker_size", 6))
    return max(3, marker_size // 2)


def _is_scatter_series(series: Mapping, points: Sequence[Tuple[int, int]]) -> bool:
    display_mode = str(series.get("display_mode", "")).strip().lower()
    render_mode = str(series.get("render_mode", "")).strip().lower()
    return display_mode == "scatter" or render_mode == "scatter_only" or len(points) <= 1


def _has_markers(series: Mapping) -> bool:
    display_mode = str(series.get("display_mode", "")).strip().lower()
    render_mode = str(series.get("render_mode", "")).strip().lower()
    return display_mode in {"line_marker", "line-marker"} or render_mode in {"line_marker", "line-marker"}


def rasterize_annotation(annotation, image_shape, line_width: int = 3, max_instances: int = 14):
    if isinstance(annotation, (str, Path)):
        annotation = load_annotation(annotation)
    if not isinstance(annotation, dict):
        raise ValueError("annotation must be a mapping or annotation path")

    height, width = _normalize_image_shape(image_shape)
    series_items = _series_iterable(annotation)

    binary_mask = np.zeros((height, width), dtype=np.uint8)
    instance_masks = np.zeros((max_instances, height, width), dtype=np.uint8)
    valid_flags = np.zeros((max_instances,), dtype=bool)

    stored_instances = 0
    valid_series = 0
    for series in series_items:
        if not isinstance(series, dict):
            raise ValueError("Annotation series entries must be objects")
        points_raw = series.get("points")
        if not isinstance(points_raw, list) or not points_raw:
            raise ValueError("Annotation series points may not be empty")

        points = [_normalize_point(point, width=width, height=height) for point in points_raw]
        mask = np.zeros((height, width), dtype=np.uint8)
        radius = _scatter_radius(series)
        if _is_scatter_series(series, points):
            for point in points:
                cv2.circle(mask, point, radius, color=1, thickness=-1, lineType=cv2.LINE_AA)
        else:
            polyline = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(mask, [polyline], isClosed=False, color=1, thickness=int(line_width), lineType=cv2.LINE_AA)
            if _has_markers(series):
                marker_step = max(1, int(series.get("marker_step", 1)))
                for point in points[::marker_step]:
                    cv2.circle(mask, point, radius, color=1, thickness=-1, lineType=cv2.LINE_AA)

        if mask.max() == 0:
            raise ValueError("Annotation series produced an empty rasterized mask")

        valid_series += 1
        binary_mask = np.maximum(binary_mask, mask)
        if stored_instances < max_instances:
            instance_masks[stored_instances] = mask
            valid_flags[stored_instances] = True
            stored_instances += 1

    if valid_series == 0:
        raise ValueError("Annotation does not contain any valid series")
    return binary_mask, instance_masks, valid_flags


def clamp_box(box, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    if box is None or len(box) != 4:
        return None
    x1 = max(0, min(width, int(math.floor(float(box[0])))))
    y1 = max(0, min(height, int(math.floor(float(box[1])))))
    x2 = max(0, min(width, int(math.ceil(float(box[2])))))
    y2 = max(0, min(height, int(math.ceil(float(box[3])))))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _boxes_to_numpy(result) -> np.ndarray:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return np.zeros((0, 4), dtype=np.float32)
    xyxy = getattr(boxes, "xyxy", None)
    if xyxy is None:
        return np.zeros((0, 4), dtype=np.float32)
    if hasattr(xyxy, "cpu"):
        xyxy = xyxy.cpu().numpy()
    else:
        xyxy = np.asarray(xyxy)
    if xyxy.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return xyxy.astype(np.float32)


def _normalize_region_box(box, name: str) -> Optional[List[int]]:
    if box is None:
        return None
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("{0} must be a 4-item box or null".format(name))
    normalized = []
    for value in box:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("{0} coordinates must be numeric".format(name))
        if not math.isfinite(float(value)):
            raise ValueError("{0} coordinates must be finite".format(name))
        normalized.append(int(value))
    return normalized


def _normalize_region_entry(entry, key: str) -> Dict[str, Optional[List[List[int]]]]:
    if not isinstance(entry, Mapping):
        raise ValueError("Region entry for {0} must be a mapping".format(key))
    axis_box = _normalize_region_box(entry.get("axis_box"), "axis_box")
    legend_boxes_raw = entry.get("legend_boxes", [])
    if legend_boxes_raw is None:
        legend_boxes_raw = []
    if not isinstance(legend_boxes_raw, list):
        raise ValueError("legend_boxes for {0} must be a list".format(key))
    legend_boxes = [_normalize_region_box(box, "legend_boxes") for box in legend_boxes_raw]
    return {
        "axis_box": axis_box,
        "legend_boxes": legend_boxes,
    }


def _normalize_region_map(
    region_map: Mapping,
    required_real_records: Sequence[ChartRecord],
) -> Dict[str, Dict[str, Optional[List[List[int]]]]]:
    if not isinstance(region_map, Mapping):
        raise ValueError("region_map must be a mapping")

    normalized_map: Dict[str, Dict[str, Optional[List[List[int]]]]] = {}
    for raw_key, raw_entry in region_map.items():
        canonical_key = _canonical_path_key(raw_key)
        normalized_map[canonical_key] = _normalize_region_entry(raw_entry, str(raw_key))

    for record in required_real_records:
        canonical_key = _canonical_path_key(record.image_path)
        if canonical_key not in normalized_map:
            raise ValueError(
                "region_map is missing required real-record entry for {0}".format(record.image_path)
            )
    return normalized_map


def detect_yolo_regions_for_image(
    image_path,
    axis_model,
    legend_model,
    axis_confidence: float = 0.5,
    legend_confidence: float = 0.3,
) -> Dict[str, Optional[List[List[int]]]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unreadable image for YOLO region detection: {0}".format(image_path))
    height, width = image.shape[:2]

    axis_results = axis_model.predict(source=str(image_path), conf=float(axis_confidence), verbose=False)
    legend_results = legend_model.predict(source=str(image_path), conf=float(legend_confidence), verbose=False)

    axis_boxes = []
    for result in axis_results:
        for raw_box in _boxes_to_numpy(result):
            clamped = clamp_box(raw_box.tolist(), width=width, height=height)
            if clamped is not None:
                axis_boxes.append(clamped)
    largest_axis = None
    if axis_boxes:
        largest_axis = max(axis_boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))

    legend_boxes = []
    for result in legend_results:
        for raw_box in _boxes_to_numpy(result):
            clamped = clamp_box(raw_box.tolist(), width=width, height=height)
            if clamped is not None:
                legend_boxes.append(list(clamped))

    return {
        "axis_box": list(largest_axis) if largest_axis is not None else None,
        "legend_boxes": legend_boxes,
    }


def build_yolo_region_map(
    records: Sequence[ChartRecord],
    axis_weights,
    legend_weights,
    axis_confidence: float = 0.5,
    legend_confidence: float = 0.3,
    cache_path=None,
) -> Dict[str, Dict[str, Optional[List[List[int]]]]]:
    real_records = [record for record in _normalize_records(records) if record.source == "real"]
    cache_file = Path(cache_path) if cache_path is not None else None
    if cache_file is not None and cache_file.is_file():
        with cache_file.open("r", encoding="utf-8") as handle:
            cached_map = json.load(handle)
        return _normalize_region_map(cached_map, required_real_records=real_records)

    if not real_records:
        return {}

    axis_weights = Path(axis_weights)
    legend_weights = Path(legend_weights)
    if not axis_weights.is_file():
        raise FileNotFoundError("Axis YOLO weights not found: {0}".format(axis_weights))
    if not legend_weights.is_file():
        raise FileNotFoundError("Legend YOLO weights not found: {0}".format(legend_weights))

    from ultralytics import YOLO

    axis_model = YOLO(str(axis_weights))
    legend_model = YOLO(str(legend_weights))
    region_map = {}
    for record in real_records:
        region_map[_canonical_path_key(record.image_path)] = detect_yolo_regions_for_image(
            record.image_path,
            axis_model=axis_model,
            legend_model=legend_model,
            axis_confidence=axis_confidence,
            legend_confidence=legend_confidence,
        )

    region_map = _normalize_region_map(region_map, required_real_records=real_records)

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("w", encoding="utf-8") as handle:
            json.dump(region_map, handle, indent=2, sort_keys=True)
    return region_map


def detect_yolo_regions(
    image_path,
    axis_weights,
    legend_weights,
    axis_confidence: float = 0.5,
    legend_confidence: float = 0.3,
):
    axis_weights = Path(axis_weights)
    legend_weights = Path(legend_weights)
    if not axis_weights.is_file():
        raise FileNotFoundError("Axis YOLO weights not found: {0}".format(axis_weights))
    if not legend_weights.is_file():
        raise FileNotFoundError("Legend YOLO weights not found: {0}".format(legend_weights))

    from ultralytics import YOLO

    axis_model = YOLO(str(axis_weights))
    legend_model = YOLO(str(legend_weights))
    return detect_yolo_regions_for_image(
        image_path,
        axis_model=axis_model,
        legend_model=legend_model,
        axis_confidence=axis_confidence,
        legend_confidence=legend_confidence,
    )


def build_region_map(
    records: Sequence[ChartRecord],
    axis_weights,
    legend_weights,
    axis_confidence: float = 0.5,
    legend_confidence: float = 0.3,
    cache_path=None,
):
    return build_yolo_region_map(
        records=records,
        axis_weights=axis_weights,
        legend_weights=legend_weights,
        axis_confidence=axis_confidence,
        legend_confidence=legend_confidence,
        cache_path=cache_path,
    )


def build_keep_mask_from_regions(image_shape, axis_box=None, legend_boxes=None) -> np.ndarray:
    height, width = _normalize_image_shape(image_shape)
    keep_mask = np.ones((height, width), dtype=bool)
    if axis_box is not None:
        keep_mask[:] = False
        clamped_axis = clamp_box(axis_box, width=width, height=height)
        if clamped_axis is not None:
            x1, y1, x2, y2 = clamped_axis
            keep_mask[y1:y2, x1:x2] = True
    for legend_box in legend_boxes or ():
        clamped_legend = clamp_box(legend_box, width=width, height=height)
        if clamped_legend is None:
            continue
        x1, y1, x2, y2 = clamped_legend
        keep_mask[y1:y2, x1:x2] = False
    return keep_mask


def keep_mask_from_regions(image_shape, axis_box=None, legend_boxes=None) -> np.ndarray:
    return build_keep_mask_from_regions(
        image_shape=image_shape,
        axis_box=axis_box,
        legend_boxes=legend_boxes,
    )


def apply_regions_to_sample(image, binary_mask, instance_masks, region):
    if not region:
        return image, binary_mask, instance_masks

    keep_mask = build_keep_mask_from_regions(
        image.shape[:2],
        axis_box=region.get("axis_box"),
        legend_boxes=region.get("legend_boxes"),
    )
    masked_image = np.array(image, copy=True)
    masked_binary = np.array(binary_mask, copy=True)
    masked_instances = np.array(instance_masks, copy=True)

    masked_image[~keep_mask] = 255
    masked_binary = (masked_binary > 0).astype(np.uint8)
    masked_binary[~keep_mask] = 0
    masked_instances = (masked_instances > 0).astype(np.uint8)
    masked_instances[:, ~keep_mask] = 0
    return masked_image, masked_binary, masked_instances


def apply_region_masking(image, binary_mask, region):
    dummy_instances = np.zeros((1, binary_mask.shape[0], binary_mask.shape[1]), dtype=np.uint8)
    masked_image, masked_binary, _ = apply_regions_to_sample(
        image=image,
        binary_mask=binary_mask,
        instance_masks=dummy_instances,
        region=region,
    )
    return masked_image, masked_binary


def _resize_image(image: np.ndarray, image_size: int) -> np.ndarray:
    return cv2.resize(image, (int(image_size), int(image_size)), interpolation=cv2.INTER_LINEAR)


def _resize_mask(mask: np.ndarray, image_size: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (int(image_size), int(image_size)), interpolation=cv2.INTER_NEAREST)


def _normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return image.transpose(2, 0, 1)


class UnifiedChartDataset(Dataset):
    def __init__(
        self,
        records: Sequence[ChartRecord],
        image_size: int,
        max_instances: int,
        is_train: bool,
        line_width: int,
        real_weight: float,
        region_map: Optional[Mapping[str, Mapping]] = None,
    ):
        self.records = _normalize_records(records)
        if not self.records:
            raise ValueError("UnifiedChartDataset requires at least one record")
        self.image_size = int(image_size)
        self.max_instances = int(max_instances)
        self.is_train = bool(is_train)
        self.line_width = int(line_width)
        self.real_weight = float(real_weight)
        real_records = [record for record in self.records if record.source == "real"]
        if region_map:
            self.region_map = _normalize_region_map(region_map, required_real_records=real_records)
        else:
            self.region_map = {}

    def __len__(self):
        return len(self.records)

    def _load_image(self, image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Unreadable image: {0}".format(image_path))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _apply_train_augmentations(
        self,
        image: np.ndarray,
        binary_mask: np.ndarray,
        instance_masks: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            image = np.ascontiguousarray(np.fliplr(image))
            binary_mask = np.ascontiguousarray(np.fliplr(binary_mask))
            instance_masks = np.ascontiguousarray(np.flip(instance_masks, axis=2))

        if random.random() < 0.3:
            alpha = 1.0 + random.uniform(-0.2, 0.2)
            beta = random.uniform(-16.0, 16.0)
            image = np.clip(image.astype(np.float32) * alpha + beta, 0.0, 255.0).astype(np.uint8)

        return image, binary_mask, instance_masks

    def __getitem__(self, index):
        record = self.records[index]
        image = self._load_image(record.image_path)
        annotation = load_annotation(record.annotation_path)
        binary_mask, instance_masks, valid_flags = rasterize_annotation(
            annotation,
            image_shape=image.shape[:2],
            line_width=self.line_width,
            max_instances=self.max_instances,
        )

        if record.source == "real":
            region = self.region_map.get(_canonical_path_key(record.image_path))
            image, binary_mask, instance_masks = apply_regions_to_sample(image, binary_mask, instance_masks, region)

        image = _resize_image(image, self.image_size)
        binary_mask = _resize_mask(binary_mask, self.image_size)
        resized_instances = np.zeros((self.max_instances, self.image_size, self.image_size), dtype=np.uint8)
        for mask_index in range(self.max_instances):
            resized_instances[mask_index] = _resize_mask(instance_masks[mask_index], self.image_size)
        instance_masks = resized_instances

        if self.is_train:
            image, binary_mask, instance_masks = self._apply_train_augmentations(image, binary_mask, instance_masks)

        sample_weight = self.real_weight if record.source == "real" else 1.0
        return {
            "image": torch.from_numpy(_normalize_image(image)).to(dtype=torch.float32),
            "binary_mask": torch.from_numpy(binary_mask.astype(np.float32)),
            "instance_masks": torch.from_numpy(instance_masks.astype(np.float32)),
            "valid_flags": torch.from_numpy(valid_flags.astype(bool)),
            "sample_weight": torch.tensor(sample_weight, dtype=torch.float32),
            "is_real": torch.tensor(1.0 if record.source == "real" else 0.0, dtype=torch.float32),
        }
