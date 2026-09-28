"""Anonymous YOLO training entrypoints for the selected release recipes."""

import argparse
from pathlib import Path, PurePosixPath, PureWindowsPath

import yaml


RECIPE_ROOT_KEYS = {
    "schema_version",
    "role",
    "base_model",
    "classes",
    "train",
    "hyperparameters",
}
SUPPORTED_SCHEMA_VERSION = "1.0"
EXPECTED_CLASSES_BY_ROLE = {
    "axis": ["axis"],
    "legend_clutter": ["legend", "clutter"],
    "subfigure": ["subfigure"],
}
EXPECTED_TRAIN_KEYS = {
    "workers",
    "seed",
    "deterministic",
    "pretrained",
    "optimizer",
    "amp",
    "save",
    "val",
    "plots",
    "epochs",
    "patience",
    "batch",
    "imgsz",
    "save_period",
}
EXPECTED_HYPERPARAMETER_KEYS = {
    "close_mosaic",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "warmup_momentum",
    "warmup_bias_lr",
    "box",
    "cls",
    "dfl",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "mosaic",
    "mixup",
    "cutmix",
}
RESERVED_TRAINING_KEYS = {"data", "project", "device", "name", "exist_ok"}
SELECTED_BASE_MODEL = "yolo11x.pt"


def _is_forbidden_pathlike_string(value):
    if not isinstance(value, str):
        return False
    if not value:
        return False
    if value.startswith(("~", "/", "\\")):
        return True
    if "/" in value or "\\" in value:
        return True
    if value in {".", ".."}:
        return True
    if value.startswith("./") or value.startswith("../"):
        return True
    if len(value) >= 3 and value[1] == ":" and value[2] in ("\\", "/"):
        return True
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        return True
    return False


def _assert_no_embedded_paths(value, breadcrumb):
    if isinstance(value, dict):
        for key, nested_value in value.items():
            _assert_no_embedded_paths(nested_value, breadcrumb + "." + str(key))
        return
    if isinstance(value, list):
        for index, nested_value in enumerate(value):
            _assert_no_embedded_paths(nested_value, breadcrumb + "[" + str(index) + "]")
        return
    if _is_forbidden_pathlike_string(value):
        raise ValueError("Recipe contains forbidden path-like value at {0}: {1!r}".format(breadcrumb, value))


def _require_mapping(name, value):
    if not isinstance(value, dict):
        raise ValueError("{0} must be a mapping".format(name))


def _validate_exact_keys(name, value, expected_keys):
    _require_mapping(name, value)
    actual_keys = set(value)
    missing_keys = sorted(expected_keys - actual_keys)
    unknown_keys = sorted(actual_keys - expected_keys)
    if missing_keys:
        raise ValueError("{0} is missing keys: {1}".format(name, ", ".join(missing_keys)))
    if unknown_keys:
        raise ValueError("{0} contains unknown keys: {1}".format(name, ", ".join(unknown_keys)))


def _normalize_recipe_path(path):
    return Path(path)


def _normalize_dataset_names(names):
    if isinstance(names, list):
        return list(names)
    if isinstance(names, dict):
        normalized = {}
        for raw_key, value in names.items():
            try:
                key = int(raw_key)
            except (TypeError, ValueError):
                raise ValueError("Dataset names mapping must use integer keys")
            normalized[key] = value
        expected_keys = list(range(len(normalized)))
        if sorted(normalized) != expected_keys:
            raise ValueError("Dataset names mapping must be contiguous from 0")
        return [normalized[index] for index in expected_keys]
    raise ValueError("Dataset names must be a list or integer-keyed mapping")


def load_recipe(path):
    recipe_path = _normalize_recipe_path(path)
    if not recipe_path.is_file():
        raise FileNotFoundError("Recipe file not found: {0}".format(recipe_path))
    with recipe_path.open("r", encoding="utf-8") as handle:
        recipe = yaml.safe_load(handle)

    _require_mapping("recipe", recipe)
    actual_root_keys = set(recipe)
    missing_root_keys = sorted(RECIPE_ROOT_KEYS - actual_root_keys)
    unknown_root_keys = sorted(actual_root_keys - RECIPE_ROOT_KEYS)
    if missing_root_keys:
        raise ValueError("Missing root keys: {0}".format(", ".join(missing_root_keys)))
    if unknown_root_keys:
        raise ValueError("Unknown root keys: {0}".format(", ".join(unknown_root_keys)))

    if recipe["schema_version"] != SUPPORTED_SCHEMA_VERSION:
        raise ValueError("Unsupported schema_version: {0}".format(recipe["schema_version"]))

    role = recipe["role"]
    if role not in EXPECTED_CLASSES_BY_ROLE:
        raise ValueError("Unsupported role: {0}".format(role))

    if recipe["base_model"] != SELECTED_BASE_MODEL:
        raise ValueError("Recipe must declare base_model {0}".format(SELECTED_BASE_MODEL))

    expected_classes = EXPECTED_CLASSES_BY_ROLE[role]
    if recipe["classes"] != expected_classes:
        raise ValueError("Recipe classes for role {0} must be {1}".format(role, expected_classes))

    _validate_exact_keys("train", recipe["train"], EXPECTED_TRAIN_KEYS)
    _validate_exact_keys("hyperparameters", recipe["hyperparameters"], EXPECTED_HYPERPARAMETER_KEYS)

    reserved_train_keys = sorted(set(recipe["train"]) & RESERVED_TRAINING_KEYS)
    if reserved_train_keys:
        raise ValueError("Recipe train section contains reserved keys: {0}".format(", ".join(reserved_train_keys)))
    reserved_hyperparameter_keys = sorted(set(recipe["hyperparameters"]) & RESERVED_TRAINING_KEYS)
    if reserved_hyperparameter_keys:
        raise ValueError(
            "Recipe hyperparameters section contains reserved keys: {0}".format(
                ", ".join(reserved_hyperparameter_keys)
            )
        )

    _assert_no_embedded_paths(recipe, "recipe")
    return recipe


def verify_dataset_classes(dataset_yaml, expected_classes):
    dataset_path = Path(dataset_yaml)
    if not dataset_path.is_file():
        raise FileNotFoundError("Dataset YAML not found: {0}".format(dataset_path))
    with dataset_path.open("r", encoding="utf-8") as handle:
        dataset = yaml.safe_load(handle)

    _require_mapping("dataset", dataset)

    missing_required = [key for key in ("train", "val", "nc", "names") if key not in dataset]
    if missing_required:
        raise ValueError("Dataset YAML is missing keys: {0}".format(", ".join(missing_required)))

    names = _normalize_dataset_names(dataset["names"])
    if dataset["nc"] != len(names):
        raise ValueError("Dataset nc does not match names length")

    expected = list(expected_classes)
    if names != expected:
        raise ValueError("Expected classes {0}, found {1}".format(expected, names))
    if dataset["nc"] != len(expected):
        raise ValueError("Expected nc {0}, found {1}".format(len(expected), dataset["nc"]))
    return names


def build_train_kwargs(recipe, dataset_yaml, output_directory, device):
    _require_mapping("recipe", recipe)
    output_path = Path(output_directory)
    if output_path.exists() and not output_path.is_dir():
        raise ValueError("Output path must be a directory path: {0}".format(output_path))
    if not output_path.parent.exists():
        raise FileNotFoundError("Output parent directory not found: {0}".format(output_path.parent))

    reserved_collisions = sorted(
        (set(recipe.get("train", {})) | set(recipe.get("hyperparameters", {}))) & RESERVED_TRAINING_KEYS
    )
    if reserved_collisions:
        raise ValueError("Recipe collides with reserved training keys: {0}".format(", ".join(reserved_collisions)))

    verify_dataset_classes(dataset_yaml, recipe["classes"])

    kwargs = {
        "data": str(Path(dataset_yaml)),
        "project": str(output_path),
        "device": str(device),
        "name": recipe["role"],
        "exist_ok": False,
    }
    kwargs.update(recipe["train"])
    kwargs.update(recipe["hyperparameters"])
    return kwargs


def _load_yolo_class():
    from ultralytics import YOLO

    return YOLO


def run_training(recipe_path, dataset_yaml, base_model, output_directory, device="0", run_name=None):
    recipe = load_recipe(recipe_path)

    dataset_path = Path(dataset_yaml)
    if not dataset_path.is_file():
        raise FileNotFoundError("Dataset YAML not found: {0}".format(dataset_path))

    base_model_path = Path(base_model)
    if not base_model_path.is_file():
        raise FileNotFoundError("Base model file not found: {0}".format(base_model_path))
    if base_model_path.name != recipe["base_model"]:
        raise ValueError(
            "Selected base model filename must match recipe base model filename: {0}".format(recipe["base_model"])
        )

    kwargs = build_train_kwargs(recipe, dataset_path, output_directory, device)
    if run_name is not None:
        kwargs["name"] = run_name

    model = _load_yolo_class()(str(base_model_path))
    return model.train(**kwargs)


def build_argument_parser():
    parser = argparse.ArgumentParser(description="Train selected anonymous YOLO release recipes.")
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--run-name", default=None)
    return parser


def main():
    args = build_argument_parser().parse_args()
    return run_training(
        args.recipe,
        args.dataset,
        args.base_model,
        args.output,
        device=args.device,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
