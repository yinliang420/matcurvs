"""Path-explicit CLI and orchestration for the selected foreground recipe."""

from __future__ import annotations

import argparse
import csv
import copy
import json
import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from training.foreground.data import (
    ChartRecord,
    UnifiedChartDataset,
    build_region_map,
    build_train_records,
    collect_pairs,
    split_real_pairs,
    split_synthetic_pairs,
)
from training.foreground.engine import (
    create_optimizer,
    evaluate_loader,
    make_scheduler,
    save_checkpoint,
    train_one_epoch,
)
from training.foreground.losses import BinaryMaskIoUCriterion


TOP_LEVEL_KEYS = {
    "schema_version",
    "selected_experiment",
    "model",
    "training",
    "splits",
    "preprocessing",
    "selection",
}
MODEL_KEYS = {"encoder", "hidden_dim", "n_queries", "decoder_layers"}
TRAINING_KEYS = {
    "epochs",
    "batch_size",
    "image_size",
    "line_width",
    "max_instances",
    "real_weight",
    "real_repeat",
    "learning_rate",
    "warmup_epochs",
    "binary_threshold",
    "weight_decay",
    "num_workers",
    "seed",
}
SPLIT_KEYS = {"synthetic_val_ratio", "real_test_size", "real_val_size"}
PREPROCESSING_KEYS = {"use_yolo_real_mask", "yolo_axis_confidence", "yolo_legend_confidence"}
SELECTION_KEYS = {"best_on", "metric"}
EXPECTED_MODEL = {
    "encoder": "mit_b3",
    "hidden_dim": 256,
    "n_queries": 12,
    "decoder_layers": 3,
}
REQUIRED_CHECKPOINT_KEYS = {
    "model",
    "optimizer",
    "scheduler",
    "epoch",
    "config",
    "best_score",
    "metrics",
}
EXPECTED_TRAINING = {
    "epochs": 100,
    "batch_size": 2,
    "image_size": 640,
    "line_width": 3,
    "max_instances": 14,
    "real_weight": 3.0,
    "real_repeat": 64,
    "learning_rate": 0.0001,
    "warmup_epochs": 5,
    "binary_threshold": 0.5,
    "weight_decay": 0.05,
    "num_workers": 8,
    "seed": 42,
}
EXPECTED_SPLITS = {
    "synthetic_val_ratio": 0.05,
    "real_test_size": 40,
    "real_val_size": 20,
}
EXPECTED_PREPROCESSING = {
    "use_yolo_real_mask": True,
    "yolo_axis_confidence": 0.5,
    "yolo_legend_confidence": 0.3,
}
EXPECTED_SELECTION = {
    "best_on": "real",
    "metric": "real_binary_iou",
}
EXPECTED_CONFIG = {
    "schema_version": "1.0",
    "selected_experiment": "v11_raw40k_q12_d3_h256",
    "model": EXPECTED_MODEL,
    "training": EXPECTED_TRAINING,
    "splits": EXPECTED_SPLITS,
    "preprocessing": EXPECTED_PREPROCESSING,
    "selection": EXPECTED_SELECTION,
}
METRICS_FIELDNAMES = [
    "epoch",
    "train_loss",
    "synthetic_val_loss",
    "synthetic_binary_iou",
    "synthetic_binary_precision",
    "synthetic_binary_recall",
    "synthetic_binary_f1",
    "synthetic_pred_fg_ratio",
    "synthetic_gt_fg_ratio",
    "synthetic_score",
    "real_val_loss",
    "real_binary_iou",
    "real_binary_precision",
    "real_binary_recall",
    "real_binary_f1",
    "real_pred_fg_ratio",
    "real_gt_fg_ratio",
    "real_score",
    "best_score",
]
RESERVED_ARTIFACT_NAMES = ("best.pt", "last.pt", "metrics.csv", "final_test_metrics.json")
CHECKPOINT_METRICS_KEYS = {"history", "current_row", "best_epoch", "rng_state"}


@dataclass(frozen=True)
class TrainingPaths:
    synthetic_dir: Path
    real_dir: Path
    output_dir: Path
    axis_model: Optional[Path]
    legend_model: Optional[Path]


def _load_model_helpers():
    from training.foreground.model import build_paper_model, normalize_state_dict_keys

    return build_paper_model, normalize_state_dict_keys


def _lexists(path: Path) -> bool:
    return os.path.lexists(str(path))


def _assert_exact_keys(payload: Mapping, expected: Sequence[str], location: str) -> None:
    actual = set(payload)
    expected_set = set(expected)
    if actual != expected_set:
        unknown = sorted(actual - expected_set)
        missing = sorted(expected_set - actual)
        details = []
        if unknown:
            details.append("Unknown config keys at {0}: {1}".format(location, ", ".join(unknown)))
        if missing:
            details.append("Missing config keys at {0}: {1}".format(location, ", ".join(missing)))
        raise ValueError("; ".join(details))


def _looks_like_absolute_path(value: str) -> bool:
    if value.startswith(("/", "~")):
        return True
    if len(value) >= 3 and value[1] == ":" and value[2] in ("\\", "/"):
        return True
    return False


def _reject_embedded_absolute_paths(node, location: str = "root") -> None:
    if isinstance(node, Mapping):
        for key, value in node.items():
            _reject_embedded_absolute_paths(value, location="{0}.{1}".format(location, key))
        return
    if isinstance(node, list):
        for index, value in enumerate(node):
            _reject_embedded_absolute_paths(value, location="{0}[{1}]".format(location, index))
        return
    if isinstance(node, str) and _looks_like_absolute_path(node):
        raise ValueError("Config contains embedded absolute path at {0}".format(location))


def _require_positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("{0} must be a positive integer".format(name))
    return value


def _require_positive_float(value, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError("{0} must be > 0".format(name))
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{0} must be > 0".format(name)) from exc
    if normalized <= 0.0:
        raise ValueError("{0} must be > 0".format(name))
    return normalized


def _assert_exact_value(actual, expected, location: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise ValueError("{0} must exactly match the selected recipe".format(location))
        _assert_exact_keys(actual, expected.keys(), location)
        for key, expected_value in expected.items():
            _assert_exact_value(actual[key], expected_value, "{0}.{1}".format(location, key))
        return
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError("{0} must exactly match the selected recipe".format(location))


def _resolved_output_dir(output_dir: Path) -> Path:
    if _lexists(output_dir) and output_dir.is_symlink():
        raise ValueError("output_dir must not be a symlink")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("output_dir must be a directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir.resolve()


def _reserved_artifact_paths(output_dir: Path) -> Dict[str, Path]:
    return {name: output_dir / name for name in RESERVED_ARTIFACT_NAMES}


def _validate_reserved_artifact_paths(output_dir: Path, allow_existing: bool) -> Dict[str, Path]:
    resolved_output_dir = _resolved_output_dir(output_dir)
    artifact_paths = _reserved_artifact_paths(output_dir)
    for name, artifact_path in artifact_paths.items():
        if artifact_path.parent.resolve() != resolved_output_dir:
            raise ValueError("{0} must stay inside the user output_dir".format(name))
        if _lexists(artifact_path) and artifact_path.is_symlink():
            raise ValueError("{0} must not be a symlink".format(name))
        if not allow_existing and _lexists(artifact_path):
            raise ValueError("Fresh runs require an empty reserved artifact path: {0}".format(name))
    return artifact_paths


def _make_temp_path(target_path: Path) -> Path:
    handle = tempfile.NamedTemporaryFile(
        prefix=".{0}.".format(target_path.name),
        suffix=".tmp",
        dir=str(target_path.parent),
        delete=False,
    )
    temp_path = Path(handle.name)
    handle.close()
    return temp_path


def _atomic_write_bytes(target_path: Path, writer_fn) -> None:
    if target_path.is_symlink():
        raise ValueError("{0} must not be a symlink".format(target_path.name))
    temp_path = _make_temp_path(target_path)
    try:
        writer_fn(temp_path)
        os.replace(str(temp_path), str(target_path))
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    def _writer(temp_path: Path) -> None:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    _atomic_write_bytes(path, _writer)


def _atomic_write_metrics_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)

    def _writer(temp_path: Path) -> None:
        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=METRICS_FIELDNAMES)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    _atomic_write_bytes(path, _writer)


def _atomic_save_checkpoint(path: Path, **kwargs) -> None:
    def _writer(temp_path: Path) -> None:
        save_checkpoint(temp_path, **kwargs)

    _atomic_write_bytes(path, _writer)


def _atomic_copy_file(source_path: Path, target_path: Path) -> None:
    def _writer(temp_path: Path) -> None:
        shutil.copyfile(str(source_path), str(temp_path))

    _atomic_write_bytes(target_path, _writer)


def _row_to_csv_strings(row: Mapping[str, object]) -> Dict[str, str]:
    normalized = {}
    for field in METRICS_FIELDNAMES:
        if field not in row:
            raise ValueError("Metric row must include {0}".format(field))
        normalized[field] = str(row[field])
    return normalized


def _serialize_python_random_state(state) -> Mapping[str, object]:
    version, internal_state, gaussian = state
    return {
        "version": int(version),
        "internal_state": [int(value) for value in internal_state],
        "gaussian": gaussian,
    }


def _deserialize_python_random_state(payload: Mapping[str, object]):
    return (
        int(payload["version"]),
        tuple(int(value) for value in payload["internal_state"]),
        payload["gaussian"],
    )


def _serialize_numpy_random_state(state) -> Mapping[str, object]:
    return {
        "bit_generator": state[0],
        "state": state[1].tolist(),
        "pos": int(state[2]),
        "has_gauss": int(state[3]),
        "cached_gaussian": float(state[4]),
    }


def _deserialize_numpy_random_state(payload: Mapping[str, object]):
    return (
        payload["bit_generator"],
        np.array(payload["state"], dtype=np.uint32),
        int(payload["pos"]),
        int(payload["has_gauss"]),
        float(payload["cached_gaussian"]),
    )


def _require_mapping_keys(payload: Mapping, expected_keys, location: str) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("{0} must be a mapping".format(location))
    actual = set(payload)
    expected = set(expected_keys)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        details = []
        if missing:
            details.append("missing {0}".format(", ".join(missing)))
        if unknown:
            details.append("unknown {0}".format(", ".join(unknown)))
        raise ValueError("{0} has invalid keys: {1}".format(location, "; ".join(details)))


def _validate_rng_state_payload(payload: Mapping[str, object]):
    _require_mapping_keys(payload, {"python", "numpy", "torch", "train_loader_generator"}, "rng_state")

    python_state = payload["python"]
    _require_mapping_keys(python_state, {"version", "internal_state", "gaussian"}, "rng_state.python")
    if isinstance(python_state["version"], bool) or not isinstance(python_state["version"], int):
        raise ValueError("rng_state.python.version must be an integer")
    if not isinstance(python_state["internal_state"], list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in python_state["internal_state"]
    ):
        raise ValueError("rng_state.python.internal_state must be a list of integers")
    if python_state["gaussian"] is not None and not isinstance(python_state["gaussian"], (int, float)):
        raise ValueError("rng_state.python.gaussian must be numeric or null")

    numpy_state = payload["numpy"]
    _require_mapping_keys(
        numpy_state,
        {"bit_generator", "state", "pos", "has_gauss", "cached_gaussian"},
        "rng_state.numpy",
    )
    if not isinstance(numpy_state["bit_generator"], str):
        raise ValueError("rng_state.numpy.bit_generator must be a string")
    if not isinstance(numpy_state["state"], list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in numpy_state["state"]
    ):
        raise ValueError("rng_state.numpy.state must be a list of integers")
    for key in ("pos", "has_gauss"):
        if isinstance(numpy_state[key], bool) or not isinstance(numpy_state[key], int):
            raise ValueError("rng_state.numpy.{0} must be an integer".format(key))
    if not isinstance(numpy_state["cached_gaussian"], (int, float)):
        raise ValueError("rng_state.numpy.cached_gaussian must be numeric")

    torch_state = payload["torch"]
    _require_mapping_keys(torch_state, {"cpu", "cuda"}, "rng_state.torch")
    if not torch.is_tensor(torch_state["cpu"]) or torch_state["cpu"].ndim != 1:
        raise ValueError("rng_state.torch.cpu must be a 1D tensor")
    if not isinstance(torch_state["cuda"], list):
        raise ValueError("rng_state.torch.cuda must be a list")
    for index, state in enumerate(torch_state["cuda"]):
        if not torch.is_tensor(state) or state.ndim != 1:
            raise ValueError("rng_state.torch.cuda[{0}] must be a 1D tensor".format(index))

    train_loader_generator = payload["train_loader_generator"]
    if train_loader_generator is not None and (
        not torch.is_tensor(train_loader_generator) or train_loader_generator.ndim != 1
    ):
        raise ValueError("rng_state.train_loader_generator must be a 1D tensor or null")

    return copy.deepcopy(payload)


def _capture_rng_state(train_generator: Optional[torch.Generator]) -> Mapping[str, object]:
    payload = {
        "python": _serialize_python_random_state(random.getstate()),
        "numpy": _serialize_numpy_random_state(np.random.get_state()),
        "torch": {
            "cpu": torch.random.get_rng_state().clone(),
            "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
        },
        "train_loader_generator": train_generator.get_state().clone() if train_generator is not None else None,
    }
    return payload


def _restore_rng_state(payload: Mapping[str, object], train_generator: Optional[torch.Generator]) -> None:
    normalized_payload = _validate_rng_state_payload(payload)
    random.setstate(_deserialize_python_random_state(normalized_payload["python"]))
    np.random.set_state(_deserialize_numpy_random_state(normalized_payload["numpy"]))
    torch.random.set_rng_state(normalized_payload["torch"]["cpu"])
    if train_generator is not None and normalized_payload["train_loader_generator"] is not None:
        train_generator.set_state(normalized_payload["train_loader_generator"])
    if torch.cuda.is_available() and normalized_payload["torch"]["cuda"]:
        torch.cuda.set_rng_state_all(normalized_payload["torch"]["cuda"])


def _seed_data_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _build_checkpoint_metrics(
    history: Sequence[Mapping[str, object]],
    current_row: Mapping[str, object],
    best_epoch: int,
    rng_state: Mapping[str, object],
) -> Mapping[str, object]:
    return {
        "history": [dict(row) for row in history],
        "current_row": dict(current_row),
        "best_epoch": int(best_epoch),
        "rng_state": copy.deepcopy(rng_state),
    }


def _validate_checkpoint_metrics_payload(payload, expected_epoch: int):
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint metrics payload must be a mapping")
    if set(payload) != CHECKPOINT_METRICS_KEYS:
        raise ValueError("Checkpoint metrics payload keys do not match the released training schema")
    history = payload["history"]
    if not isinstance(history, list) or not history:
        raise ValueError("Checkpoint metrics history must be a non-empty list")
    normalized_history = []
    for index, row in enumerate(history):
        if not isinstance(row, Mapping):
            raise ValueError("Checkpoint metric rows must be mappings")
        normalized_row = dict(row)
        _row_to_csv_strings(normalized_row)
        if int(normalized_row["epoch"]) != index:
            raise ValueError("Checkpoint metrics history epoch continuity is invalid")
        normalized_history.append(normalized_row)
    current_row = dict(payload["current_row"])
    _row_to_csv_strings(current_row)
    if current_row != normalized_history[-1]:
        raise ValueError("Checkpoint current_row must match the final history row exactly")
    if int(current_row["epoch"]) != expected_epoch or len(normalized_history) != expected_epoch + 1:
        raise ValueError("Checkpoint metrics history length does not match the checkpoint epoch")
    best_epoch = int(payload["best_epoch"])
    if best_epoch < 0 or best_epoch > expected_epoch:
        raise ValueError("Checkpoint best_epoch is out of range")
    rng_state = payload["rng_state"]
    if not isinstance(rng_state, Mapping):
        raise ValueError("Checkpoint rng_state must be a mapping")
    return normalized_history, current_row, best_epoch, copy.deepcopy(rng_state)


def validate_training_config(config) -> None:
    if not isinstance(config, Mapping):
        raise ValueError("Training config must be a mapping")
    _assert_exact_keys(config, TOP_LEVEL_KEYS, "root")
    _reject_embedded_absolute_paths(config)
    _assert_exact_value(config["schema_version"], EXPECTED_CONFIG["schema_version"], "schema_version")
    _assert_exact_value(config["selected_experiment"], EXPECTED_CONFIG["selected_experiment"], "selected_experiment")

    model = config["model"]
    if not isinstance(model, Mapping):
        raise ValueError("model must be a mapping")
    _assert_exact_keys(model, MODEL_KEYS, "model")
    if dict(model) != EXPECTED_MODEL:
        raise ValueError("Selected model shape must match the released paper recipe")
    _assert_exact_value(model, EXPECTED_MODEL, "model")

    training = config["training"]
    if not isinstance(training, Mapping):
        raise ValueError("training must be a mapping")
    _assert_exact_keys(training, TRAINING_KEYS, "training")
    _assert_exact_value(training, EXPECTED_TRAINING, "training")

    splits = config["splits"]
    if not isinstance(splits, Mapping):
        raise ValueError("splits must be a mapping")
    _assert_exact_keys(splits, SPLIT_KEYS, "splits")
    _assert_exact_value(splits, EXPECTED_SPLITS, "splits")

    preprocessing = config["preprocessing"]
    if not isinstance(preprocessing, Mapping):
        raise ValueError("preprocessing must be a mapping")
    _assert_exact_keys(preprocessing, PREPROCESSING_KEYS, "preprocessing")
    _assert_exact_value(preprocessing, EXPECTED_PREPROCESSING, "preprocessing")

    selection = config["selection"]
    if not isinstance(selection, Mapping):
        raise ValueError("selection must be a mapping")
    _assert_exact_keys(selection, SELECTION_KEYS, "selection")
    _assert_exact_value(selection, EXPECTED_SELECTION, "selection")


def load_training_config(path) -> dict:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    validate_training_config(payload)
    return copy.deepcopy(payload)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the released paper-aligned foreground model.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the selected YAML recipe.")
    parser.add_argument("--synthetic-dir", type=Path, required=True, help="Synthetic chart root.")
    parser.add_argument("--real-dir", type=Path, required=True, help="Real chart root.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Runtime output directory.")
    parser.add_argument("--axis-model", type=Path, default=None, help="Axis YOLO weights for real masking.")
    parser.add_argument("--legend-model", type=Path, default=None, help="Legend YOLO weights for real masking.")
    parser.add_argument("--device", default="cpu", help="Torch device string.")
    parser.add_argument("--resume", type=Path, default=None, help="Path to the last.pt checkpoint to resume from.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and data discovery without training.")
    return parser


def _require_runtime_paths(config: Mapping, paths: TrainingPaths) -> None:
    if not paths.synthetic_dir.is_dir():
        raise FileNotFoundError("Synthetic data root not found: {0}".format(paths.synthetic_dir))
    if not paths.real_dir.is_dir():
        raise FileNotFoundError("Real data root not found: {0}".format(paths.real_dir))
    if config["preprocessing"]["use_yolo_real_mask"]:
        if paths.axis_model is None or paths.legend_model is None:
            raise ValueError("axis_model and legend_model are required when use_yolo_real_mask is true")
        if not paths.axis_model.is_file():
            raise FileNotFoundError("Axis YOLO weights not found: {0}".format(paths.axis_model))
        if not paths.legend_model.is_file():
            raise FileNotFoundError("Legend YOLO weights not found: {0}".format(paths.legend_model))


def _discover_records(config: Mapping, paths: TrainingPaths) -> Dict[str, Sequence[ChartRecord]]:
    _require_runtime_paths(config, paths)
    synthetic_records = collect_pairs(paths.synthetic_dir, source="synthetic")
    real_records = collect_pairs(paths.real_dir, source="real")
    synthetic_train, synthetic_val = split_synthetic_pairs(
        synthetic_records,
        val_ratio=float(config["splits"]["synthetic_val_ratio"]),
        seed=int(config["training"]["seed"]),
    )
    real_train, real_val, real_test = split_real_pairs(
        real_records,
        n_test=int(config["splits"]["real_test_size"]),
        n_val=int(config["splits"]["real_val_size"]),
        seed=int(config["training"]["seed"]),
    )
    train_records = build_train_records(
        real_train=real_train,
        synthetic_train=synthetic_train,
        real_repeat=int(config["training"]["real_repeat"]),
        seed=int(config["training"]["seed"]),
    )
    if not train_records:
        raise ValueError("Training split is empty")
    if not synthetic_val:
        raise ValueError("Synthetic validation split is empty")
    if not real_val:
        raise ValueError("Real validation split is empty")
    if not real_test:
        raise ValueError("Real test split is empty")
    return {
        "synthetic_records": synthetic_records,
        "real_records": real_records,
        "synthetic_train": synthetic_train,
        "synthetic_val": synthetic_val,
        "real_train": real_train,
        "real_val": real_val,
        "real_test": real_test,
        "train_records": train_records,
    }


def validate_input_discovery(config: Mapping, paths: TrainingPaths) -> Dict[str, Sequence[ChartRecord]]:
    return _discover_records(config, paths)


def _build_region_map_if_needed(config: Mapping, paths: TrainingPaths, real_records: Sequence[ChartRecord]):
    if not config["preprocessing"]["use_yolo_real_mask"]:
        return {}
    return build_region_map(
        real_records,
        axis_weights=paths.axis_model,
        legend_weights=paths.legend_model,
        axis_confidence=float(config["preprocessing"]["yolo_axis_confidence"]),
        legend_confidence=float(config["preprocessing"]["yolo_legend_confidence"]),
    )


def _make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    generator: Optional[torch.Generator] = None,
    worker_init_fn=None,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_metrics_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    _atomic_write_metrics_csv(path, rows)


def _write_final_metrics(path: Path, metrics: Mapping[str, object]) -> None:
    _atomic_write_json(path, metrics)


def _load_checkpoint_payload(
    checkpoint_path: Path,
    config: Mapping,
):
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Checkpoint not found: {0}".format(checkpoint_path))
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("Checkpoint payload could not be loaded safely") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint payload must be a mapping")
    if set(payload) != REQUIRED_CHECKPOINT_KEYS:
        raise ValueError("Checkpoint keys must match the saved training payload exactly")
    if payload["config"] != config:
        raise ValueError("Checkpoint config mismatch")
    return dict(payload)


def _validate_resume_lineage(paths: TrainingPaths, resume_path: Path) -> Tuple[Path, Path]:
    expected_resume_path = Path(paths.output_dir) / "last.pt"
    if resume_path.resolve() != expected_resume_path.resolve():
        raise ValueError("Resume checkpoint must be output_dir/last.pt")
    best_path = Path(paths.output_dir) / "best.pt"
    metrics_path = Path(paths.output_dir) / "metrics.csv"
    if best_path.is_symlink():
        raise ValueError("best.pt must not be a symlink")
    if metrics_path.is_symlink():
        raise ValueError("metrics.csv must not be a symlink")
    return best_path, metrics_path


def _read_metrics_history(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != METRICS_FIELDNAMES:
            raise ValueError("metrics.csv columns do not match the selected training history schema")
        rows = list(reader)
    return rows


def _validate_csv_history_against_checkpoint(csv_rows, checkpoint_history) -> None:
    normalized_checkpoint_rows = [_row_to_csv_strings(row) for row in checkpoint_history]
    if len(csv_rows) > len(normalized_checkpoint_rows):
        raise ValueError("metrics.csv diverges from the checkpoint history")
    for index, csv_row in enumerate(csv_rows):
        if dict(csv_row) != normalized_checkpoint_rows[index]:
            raise ValueError("metrics.csv diverges from the checkpoint history")


def _repair_or_validate_best_checkpoint(
    best_checkpoint_path: Path,
    last_checkpoint_path: Path,
    config: Mapping,
    last_payload: Mapping,
    last_history: Sequence[Mapping[str, object]],
    best_epoch: int,
) -> Mapping[str, object]:
    latest_is_best = best_epoch == int(last_payload["epoch"])
    if not best_checkpoint_path.is_file():
        if latest_is_best:
            _atomic_copy_file(last_checkpoint_path, best_checkpoint_path)
            return copy.deepcopy(last_payload)
        raise ValueError("best.pt is missing and cannot be reconstructed from last.pt")
    try:
        best_payload = _load_checkpoint_payload(best_checkpoint_path, config)
    except ValueError:
        if latest_is_best:
            _atomic_copy_file(last_checkpoint_path, best_checkpoint_path)
            return copy.deepcopy(last_payload)
        raise
    if float(best_payload["best_score"]) != float(last_payload["best_score"]) or int(best_payload["epoch"]) != int(best_epoch):
        if latest_is_best:
            _atomic_copy_file(last_checkpoint_path, best_checkpoint_path)
            return copy.deepcopy(last_payload)
        raise ValueError("best.pt diverges from the last checkpoint best state")
    best_history, _, _, _ = _validate_checkpoint_metrics_payload(best_payload["metrics"], int(best_payload["epoch"]))
    expected_best_history = list(last_history[: int(best_payload["epoch"]) + 1])
    if best_history != expected_best_history:
        if latest_is_best:
            _atomic_copy_file(last_checkpoint_path, best_checkpoint_path)
            return copy.deepcopy(last_payload)
        raise ValueError("best.pt history diverges from the last checkpoint best state")
    return best_payload


def _load_resume_checkpoint(
    payload: Mapping,
    model,
    optimizer,
    scheduler,
    normalize_state_dict_keys,
) -> Tuple[int, float, Sequence[Mapping[str, object]], int, Mapping[str, object]]:
    history, current_row, best_epoch, rng_state = _validate_checkpoint_metrics_payload(
        payload["metrics"],
        int(payload["epoch"]),
    )
    normalized_state_dict = normalize_state_dict_keys(payload["model"])
    load_result = model.load_state_dict(normalized_state_dict)
    if hasattr(load_result, "missing_keys") and (load_result.missing_keys or load_result.unexpected_keys):
        raise ValueError("Checkpoint model state could not be restored strictly")
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if float(current_row["best_score"]) != float(payload["best_score"]):
        raise ValueError("Checkpoint best_score does not match the stored metrics history")
    return int(payload["epoch"]) + 1, float(payload["best_score"]), history, best_epoch, rng_state


def _load_model_checkpoint(
    payload: Mapping,
    model,
    normalize_state_dict_keys,
) -> None:
    _validate_checkpoint_metrics_payload(payload["metrics"], int(payload["epoch"]))
    normalized_state_dict = normalize_state_dict_keys(payload["model"])
    load_result = model.load_state_dict(normalized_state_dict)
    if hasattr(load_result, "missing_keys") and (load_result.missing_keys or load_result.unexpected_keys):
        raise ValueError("Checkpoint model state could not be restored strictly")


def run_training(config: Mapping, paths: TrainingPaths, device: str = "cpu", resume=None):
    validate_training_config(config)
    artifact_paths = _validate_reserved_artifact_paths(Path(paths.output_dir), allow_existing=resume is not None)
    last_checkpoint_path = artifact_paths["last.pt"]
    best_checkpoint_path = artifact_paths["best.pt"]
    metrics_path = artifact_paths["metrics.csv"]
    final_test_metrics_path = artifact_paths["final_test_metrics.json"]

    resume_payload = None
    checkpoint_history = []
    checkpoint_best_epoch = -1
    resume_rng_state = None
    if resume is not None and Path(resume).name != "last.pt":
        raise ValueError("Resume checkpoint must be last.pt")
    if resume is not None:
        best_checkpoint_path, metrics_path = _validate_resume_lineage(paths, Path(resume))
        resume_payload = _load_checkpoint_payload(last_checkpoint_path, config)
        checkpoint_history, current_row, checkpoint_best_epoch, resume_rng_state = _validate_checkpoint_metrics_payload(
            resume_payload["metrics"],
            int(resume_payload["epoch"]),
        )
        if float(current_row["best_score"]) != float(resume_payload["best_score"]):
            raise ValueError("Checkpoint best_score does not match the stored metrics history")
        if metrics_path.exists():
            csv_rows = _read_metrics_history(metrics_path)
            _validate_csv_history_against_checkpoint(csv_rows, checkpoint_history)
        _repair_or_validate_best_checkpoint(
            best_checkpoint_path=best_checkpoint_path,
            last_checkpoint_path=last_checkpoint_path,
            config=config,
            last_payload=resume_payload,
            last_history=checkpoint_history,
            best_epoch=checkpoint_best_epoch,
        )

    records = _discover_records(config, paths)
    _set_seed(int(config["training"]["seed"]))
    region_map = _build_region_map_if_needed(config, paths, records["real_records"])

    image_size = int(config["training"]["image_size"])
    max_instances = int(config["training"]["max_instances"])
    line_width = int(config["training"]["line_width"])
    real_weight = float(config["training"]["real_weight"])
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"]["num_workers"])

    train_dataset = UnifiedChartDataset(
        records=records["train_records"],
        image_size=image_size,
        max_instances=max_instances,
        is_train=True,
        line_width=line_width,
        real_weight=real_weight,
        region_map=region_map,
    )
    synthetic_val_dataset = UnifiedChartDataset(
        records=records["synthetic_val"],
        image_size=image_size,
        max_instances=max_instances,
        is_train=False,
        line_width=line_width,
        real_weight=real_weight,
        region_map={},
    )
    real_val_dataset = UnifiedChartDataset(
        records=records["real_val"],
        image_size=image_size,
        max_instances=max_instances,
        is_train=False,
        line_width=line_width,
        real_weight=real_weight,
        region_map=region_map,
    )
    real_test_dataset = UnifiedChartDataset(
        records=records["real_test"],
        image_size=image_size,
        max_instances=max_instances,
        is_train=False,
        line_width=line_width,
        real_weight=real_weight,
        region_map=region_map,
    )

    train_loader_generator = torch.Generator()
    train_loader_generator.manual_seed(int(config["training"]["seed"]))
    train_loader = _make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=train_loader_generator,
        worker_init_fn=_seed_data_worker,
    )
    synthetic_val_loader = _make_loader(
        synthetic_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    real_val_loader = _make_loader(
        real_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    real_test_loader = _make_loader(
        real_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    build_paper_model, normalize_state_dict_keys = _load_model_helpers()
    model = build_paper_model()
    runtime_device = torch.device(device)
    model.to(runtime_device)
    criterion = BinaryMaskIoUCriterion()
    optimizer = create_optimizer(
        model,
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = make_scheduler(
        optimizer,
        epochs=int(config["training"]["epochs"]),
        warmup_epochs=int(config["training"]["warmup_epochs"]),
    )

    start_epoch = 0
    best_score = float("-inf")
    if resume_payload is not None:
        start_epoch, best_score, checkpoint_history, checkpoint_best_epoch, _ = _load_resume_checkpoint(
            payload=resume_payload,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            normalize_state_dict_keys=normalize_state_dict_keys,
        )
        _restore_rng_state(resume_rng_state, train_loader_generator)

    metrics_rows = [dict(row) for row in checkpoint_history]
    epochs = int(config["training"]["epochs"])
    binary_threshold = float(config["training"]["binary_threshold"])

    for epoch in range(start_epoch, epochs):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=runtime_device,
            epoch=epoch,
        )
        synthetic_val_metrics = evaluate_loader(
            model=model,
            loader=synthetic_val_loader,
            criterion=criterion,
            device=runtime_device,
            binary_threshold=binary_threshold,
        )
        real_val_metrics = evaluate_loader(
            model=model,
            loader=real_val_loader,
            criterion=criterion,
            device=runtime_device,
            binary_threshold=binary_threshold,
        )
        scheduler.step()
        score = float(real_val_metrics["binary_iou"])
        is_new_best = score > best_score
        if is_new_best:
            best_score = score
            checkpoint_best_epoch = epoch
        metrics_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["train_loss"],
                "synthetic_val_loss": synthetic_val_metrics["val_loss"],
                "synthetic_binary_iou": synthetic_val_metrics["binary_iou"],
                "synthetic_binary_precision": synthetic_val_metrics["binary_precision"],
                "synthetic_binary_recall": synthetic_val_metrics["binary_recall"],
                "synthetic_binary_f1": synthetic_val_metrics["binary_f1"],
                "synthetic_pred_fg_ratio": synthetic_val_metrics["pred_fg_ratio"],
                "synthetic_gt_fg_ratio": synthetic_val_metrics["gt_fg_ratio"],
                "synthetic_score": synthetic_val_metrics["score"],
                "real_val_loss": real_val_metrics["val_loss"],
                "real_binary_iou": real_val_metrics["binary_iou"],
                "real_binary_precision": real_val_metrics["binary_precision"],
                "real_binary_recall": real_val_metrics["binary_recall"],
                "real_binary_f1": real_val_metrics["binary_f1"],
                "real_pred_fg_ratio": real_val_metrics["pred_fg_ratio"],
                "real_gt_fg_ratio": real_val_metrics["gt_fg_ratio"],
                "real_score": real_val_metrics["score"],
                "best_score": best_score,
            }
        )
        current_row = dict(metrics_rows[-1])
        checkpoint_metrics = _build_checkpoint_metrics(
            history=metrics_rows,
            current_row=current_row,
            best_epoch=checkpoint_best_epoch,
            rng_state=_capture_rng_state(train_loader_generator),
        )
        _atomic_save_checkpoint(
            last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            config=config,
            best_score=best_score,
            metrics=checkpoint_metrics,
        )
        if is_new_best:
            _atomic_copy_file(last_checkpoint_path, best_checkpoint_path)
        _write_metrics_csv(metrics_path, metrics_rows)

    if not best_checkpoint_path.is_file():
        raise ValueError("best.pt was not created during training")
    best_payload = _load_checkpoint_payload(best_checkpoint_path, config)
    _load_model_checkpoint(
        payload=best_payload,
        model=model,
        normalize_state_dict_keys=normalize_state_dict_keys,
    )
    final_test_metrics = evaluate_loader(
        model=model,
        loader=real_test_loader,
        criterion=criterion,
        device=runtime_device,
        binary_threshold=binary_threshold,
    )
    _write_final_metrics(final_test_metrics_path, final_test_metrics)
    return {
        "best_score": best_score,
        "start_epoch": start_epoch,
        "metrics_path": metrics_path,
        "final_test_metrics_path": final_test_metrics_path,
    }


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    config = load_training_config(args.config)
    paths = TrainingPaths(
        synthetic_dir=Path(args.synthetic_dir),
        real_dir=Path(args.real_dir),
        output_dir=Path(args.output_dir),
        axis_model=Path(args.axis_model) if args.axis_model is not None else None,
        legend_model=Path(args.legend_model) if args.legend_model is not None else None,
    )
    try:
        _require_runtime_paths(config, paths)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        validate_input_discovery(config, paths)
        print("foreground_training_dry_run_ok")
        return
    run_training(config=config, paths=paths, device=args.device, resume=args.resume)


if __name__ == "__main__":
    main()
