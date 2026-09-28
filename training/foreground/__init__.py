"""Selected V11 foreground data exports."""

from importlib import import_module
from typing import TYPE_CHECKING

from training.foreground.engine import (
    create_optimizer,
    evaluate_loader,
    get_autocast_context,
    make_scheduler,
    save_checkpoint,
    train_one_epoch,
)
from training.foreground.data import (
    ChartRecord,
    UnifiedChartDataset,
    apply_region_masking,
    apply_regions_to_sample,
    assert_non_overlapping_records,
    build_keep_mask_from_regions,
    build_region_map,
    build_train_records,
    build_yolo_region_map,
    clamp_box,
    collect_pairs,
    detect_yolo_regions,
    detect_yolo_regions_for_image,
    deterministic_split,
    explicit_split,
    keep_mask_from_regions,
    load_annotation,
    rasterize_annotation,
    split_real_pairs,
    split_synthetic_pairs,
)
from training.foreground.losses import BinaryMaskIoUCriterion

if TYPE_CHECKING:
    from training.foreground.model import (
        Mask2FormerStyleModel,
        build_paper_model,
        normalize_state_dict_keys,
    )


_LAZY_MODEL_EXPORTS = {
    "Mask2FormerStyleModel",
    "build_paper_model",
    "normalize_state_dict_keys",
}


def __getattr__(name):
    if name in _LAZY_MODEL_EXPORTS:
        module = import_module("training.foreground.model")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError("module {0!r} has no attribute {1!r}".format(__name__, name))

__all__ = [
    "BinaryMaskIoUCriterion",
    "ChartRecord",
    "Mask2FormerStyleModel",
    "UnifiedChartDataset",
    "apply_region_masking",
    "apply_regions_to_sample",
    "assert_non_overlapping_records",
    "build_keep_mask_from_regions",
    "build_region_map",
    "build_train_records",
    "build_yolo_region_map",
    "clamp_box",
    "collect_pairs",
    "detect_yolo_regions",
    "detect_yolo_regions_for_image",
    "deterministic_split",
    "explicit_split",
    "build_paper_model",
    "create_optimizer",
    "evaluate_loader",
    "get_autocast_context",
    "keep_mask_from_regions",
    "load_annotation",
    "make_scheduler",
    "normalize_state_dict_keys",
    "rasterize_annotation",
    "save_checkpoint",
    "split_real_pairs",
    "split_synthetic_pairs",
    "train_one_epoch",
]
