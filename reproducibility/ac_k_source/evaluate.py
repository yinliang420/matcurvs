from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import tempfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


CONDITIONS = (
    "gemini_submitted",
    "gpt54",
    "qwen_vl_max",
    "qwen3_vl_plus",
    "local_qwen25",
    "silhouette",
)
EXPECTED_PREDICTION_MEMBERS = ("masks.npy", "selected_k.npy", "emitted_k.npy")
PREDICTION_ARCHIVE_SUFFIX = ".npz.zip"
PER_CURVE_FIELDS = (
    "condition",
    "panel_id",
    "curve_id",
    "matched_prediction_index",
    "iou",
    "pass_0_3",
    "pass_0_4",
    "pass_0_5",
)
SUMMARY_FIELDS = (
    "condition",
    "n_panels",
    "n_gt_curves",
    "lr_0_3_hits",
    "lr_0_3",
    "lr_0_4_hits",
    "lr_0_4",
    "lr_0_5_hits",
    "lr_0_5",
)
PANEL_ID_PATTERN = re.compile(r"^[0-9a-f]{64}_[1-9][0-9]*$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MANIFEST_LINE_PATTERN = re.compile(r"^([0-9a-f]{64})  ([^\n]+)$")
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
FIXED_ZIP_MODE = 0o600
MAX_MANIFEST_FILES = 5000
MAX_ANNOTATION_BYTES = 2 * 1024 * 1024
MAX_PREDICTION_BYTES = 256 * 1024 * 1024
MAX_MEMBER_BYTES = {
    "masks.npy": 128 * 1024 * 1024,
    "selected_k.npy": 1 * 1024 * 1024,
    "emitted_k.npy": 1 * 1024 * 1024,
}
MAX_COMPRESSION_RATIO = 1000
READ_CHUNK_BYTES = 64 * 1024
EXPECTED_GT_CURVES = 203
EXPECTED_PANELS = 90
FLOAT_FIELDS = {"iou", "lr_0_3", "lr_0_4", "lr_0_5"}
_VERIFY_MANIFEST_TEST_HOOK = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular_file(path: Path, *, label: str, max_bytes: Optional[int] = None) -> os.stat_result:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    try:
        st = path.stat()
    except FileNotFoundError as exc:
        raise ValueError(f"missing {label}: {path}") from exc
    if not stat.S_ISREG(st.st_mode):
        raise ValueError(f"{label} must be a regular file: {path}")
    if max_bytes is not None and st.st_size > max_bytes:
        raise ValueError(f"{label} exceeds size ceiling: {path}")
    return st


def _validate_capsule_root(capsule: Path) -> Path:
    root = Path(capsule)
    if root.is_symlink():
        raise ValueError(f"capsule root must not be a symlink: {root}")
    if not root.is_dir():
        raise ValueError(f"capsule root must be a directory: {root}")
    return root


def _stat_identity(st: os.stat_result) -> Tuple[int, int, int, int]:
    return (st.st_dev, st.st_ino, st.st_mode, st.st_size)


def _is_runtime_pyc_path(pure_path: PurePosixPath) -> bool:
    return "__pycache__" in pure_path.parts and pure_path.suffix == ".pyc"


def _walk_regular_files(root: Path) -> Dict[str, Tuple[Path, Tuple[int, int, int, int]]]:
    files: Dict[str, Tuple[Path, Tuple[int, int, int, int]]] = {}
    for current_root, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current_root)
        rel_dir = current_path.relative_to(root)
        for dirname in list(dirnames):
            if dirname.startswith("."):
                raise ValueError(f"hidden path is not allowed: {(rel_dir / dirname).as_posix()}")
            full_dir = current_path / dirname
            if full_dir.is_symlink():
                raise ValueError(f"symlink is not allowed: {(rel_dir / dirname).as_posix()}")
        for filename in filenames:
            rel_path = (rel_dir / filename).as_posix()
            full_path = current_path / filename
            pure_path = PurePosixPath(rel_path)
            if _is_runtime_pyc_path(pure_path):
                continue
            if "__pycache__" in pure_path.parts:
                raise ValueError(f"unexpected runtime cache file: {rel_path}")
            if filename.startswith("."):
                raise ValueError(f"hidden path is not allowed: {rel_path}")
            if rel_path == "MANIFEST.sha256":
                _require_regular_file(full_path, label="manifest")
                continue
            if full_path.is_symlink():
                raise ValueError(f"symlink is not allowed: {rel_path}")
            st = _require_regular_file(full_path, label="capsule file")
            files[rel_path] = (full_path, _stat_identity(st))
    if len(files) > MAX_MANIFEST_FILES:
        raise ValueError(f"too many capsule files: {len(files)}")
    return files


def _parse_manifest_line(line: str) -> tuple[str, str]:
    match = MANIFEST_LINE_PATTERN.fullmatch(line)
    if match is None:
        raise ValueError("invalid manifest line format")
    digest, rel_path = match.groups()
    pure_path = PurePosixPath(rel_path)
    if pure_path.is_absolute() or "\\" in rel_path or ".." in pure_path.parts:
        raise ValueError(f"unsafe manifest path: {rel_path}")
    if _is_runtime_pyc_path(pure_path):
        raise ValueError(f"runtime cache files must not appear in manifest: {rel_path}")
    if any(part.startswith(".") for part in pure_path.parts):
        raise ValueError(f"hidden manifest path: {rel_path}")
    if len(pure_path.parts) == 0:
        raise ValueError(f"unsafe manifest path: {rel_path}")
    return digest, rel_path


def verify_manifest(capsule: Path) -> dict[str, str]:
    root = _validate_capsule_root(Path(capsule))
    manifest_path = root / "MANIFEST.sha256"
    _require_regular_file(manifest_path, label="manifest", max_bytes=2 * 1024 * 1024)
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    if len(lines) > MAX_MANIFEST_FILES:
        raise ValueError(f"manifest exceeds file-count ceiling: {len(lines)}")

    manifest_entries: dict[str, str] = {}
    for line in lines:
        digest, rel_path = _parse_manifest_line(line)
        if rel_path in manifest_entries:
            raise ValueError(f"duplicate manifest entry: {rel_path}")
        manifest_entries[rel_path] = digest

    actual_files = _walk_regular_files(root)
    missing = sorted(set(manifest_entries) - set(actual_files))
    if missing:
        raise ValueError(f"missing manifest files: {missing[0]}")
    extra = sorted(set(actual_files) - set(manifest_entries))
    if extra:
        raise ValueError(f"extra capsule files: {extra[0]}")

    root_fd = os.open(root, os.O_RDONLY)
    try:
        for rel_path, expected_digest in manifest_entries.items():
            if _VERIFY_MANIFEST_TEST_HOOK is not None:
                _VERIFY_MANIFEST_TEST_HOOK(root, rel_path)
            _, expected_identity = actual_files[rel_path]
            file_fd = None
            try:
                file_fd = _open_verified_file_fd(root_fd, rel_path, expected_identity)
                actual_digest = _sha256_open_fd(file_fd, expected_identity)
            finally:
                if file_fd is not None:
                    os.close(file_fd)
            if actual_digest != expected_digest:
                raise ValueError(f"hash mismatch for {rel_path}")
    finally:
        os.close(root_fd)
    return manifest_entries


def _open_verified_file_fd(
    root_fd: int,
    rel_path: str,
    expected_identity: Tuple[int, int, int, int],
) -> int:
    parts = PurePosixPath(rel_path).parts
    nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    current_fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | directory_flag | nofollow_flag, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        file_fd = os.open(parts[-1], os.O_RDONLY | nofollow_flag, dir_fd=current_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(f"swapped manifest file became a symlink: {rel_path}") from exc
        raise ValueError(f"failed to open manifest file safely: {rel_path}") from exc
    finally:
        os.close(current_fd)
    if _stat_identity(os.fstat(file_fd)) != expected_identity:
        os.close(file_fd)
        raise ValueError(f"manifest file was swapped after walk: {rel_path}")
    return file_fd


def _sha256_open_fd(file_fd: int, expected_identity: Tuple[int, int, int, int]) -> str:
    digest = hashlib.sha256()
    os.lseek(file_fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(file_fd, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    if _stat_identity(os.fstat(file_fd)) != expected_identity:
        raise ValueError("manifest file was swapped during hashing")
    return digest.hexdigest()


def _validate_exact_keys(payload: Dict[str, Any], expected_keys: set[str], *, label: str) -> None:
    actual_keys = set(payload)
    if actual_keys != expected_keys:
        extra = sorted(actual_keys - expected_keys)
        missing = sorted(expected_keys - actual_keys)
        pieces: List[str] = []
        if extra:
            pieces.append(f"extra={extra}")
        if missing:
            pieces.append(f"missing={missing}")
        raise ValueError(f"{label} has invalid keys: {' '.join(pieces)}")


def _validate_positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def load_annotation(path: Path) -> dict[str, Any]:
    _require_regular_file(path, label="annotation", max_bytes=MAX_ANNOTATION_BYTES)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("annotation payload must be an object")
    _validate_exact_keys(
        payload,
        {"schema_version", "panel_id", "image_sha256", "image_size", "curves"},
        label="annotation",
    )
    if payload["schema_version"] != "1.0":
        raise ValueError("annotation schema_version must be 1.0")
    panel_id = payload["panel_id"]
    if not isinstance(panel_id, str) or PANEL_ID_PATTERN.fullmatch(panel_id) is None:
        raise ValueError("annotation panel_id is invalid")
    if path.stem != panel_id:
        raise ValueError("annotation filename does not match panel_id")
    image_sha256 = payload["image_sha256"]
    if not isinstance(image_sha256, str) or SHA256_PATTERN.fullmatch(image_sha256) is None:
        raise ValueError("annotation image_sha256 is invalid")
    image_size = payload["image_size"]
    if not (isinstance(image_size, list) and len(image_size) == 2):
        raise ValueError("annotation image_size must have two entries")
    width = _validate_positive_int(image_size[0], label="annotation image width")
    height = _validate_positive_int(image_size[1], label="annotation image height")
    curves = payload["curves"]
    if not isinstance(curves, list) or not curves:
        raise ValueError("annotation curves must be a non-empty list")
    seen_curve_ids: set[str] = set()
    normalized_curves: List[Dict[str, Any]] = []
    for curve in curves:
        if not isinstance(curve, dict):
            raise ValueError("curve entry must be an object")
        _validate_exact_keys(curve, {"curve_id", "pixel_points"}, label="curve")
        curve_id = curve["curve_id"]
        if not isinstance(curve_id, str) or not curve_id:
            raise ValueError("curve_id must be a non-empty string")
        if curve_id in seen_curve_ids:
            raise ValueError(f"duplicate curve_id: {curve_id}")
        seen_curve_ids.add(curve_id)
        pixel_points = curve["pixel_points"]
        if not isinstance(pixel_points, list) or len(pixel_points) < 2:
            raise ValueError(f"curve {curve_id} must have at least two pixel_points")
        normalized_points: List[Tuple[float, float]] = []
        for point in pixel_points:
            if not (isinstance(point, list) and len(point) == 2):
                raise ValueError(f"curve {curve_id} has invalid point shape")
            x, y = point
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                raise ValueError(f"curve {curve_id} has non-numeric point")
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(f"curve {curve_id} has non-finite point")
            if x < 0 or y < 0 or x >= width or y >= height:
                raise ValueError(f"curve {curve_id} point is out of bounds")
            normalized_points.append((float(x), float(y)))
        normalized_curves.append({"curve_id": curve_id, "pixel_points": normalized_points})
    return {
        "schema_version": payload["schema_version"],
        "panel_id": panel_id,
        "image_sha256": image_sha256,
        "image_size": (width, height),
        "curves": normalized_curves,
    }


def _read_zip_member_bytes(archive: zipfile.ZipFile, info: zipfile.ZipInfo, *, max_bytes: int) -> bytes:
    chunks: List[bytes] = []
    total = 0
    with archive.open(info, "r") as handle:
        while True:
            chunk = handle.read(READ_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"zip member exceeds size ceiling: {info.filename}")
            chunks.append(chunk)
    return b"".join(chunks)


def _load_np_array(member_name: str, payload: bytes) -> np.ndarray:
    try:
        array = np.load(io.BytesIO(payload), allow_pickle=False)
    except ValueError as exc:
        raise ValueError(f"pickle is not allowed in {member_name}") from exc
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{member_name} did not decode to an ndarray")
    return array


def _read_scalar_int(array: np.ndarray, *, label: str) -> int:
    if array.shape != ():
        raise ValueError(f"{label} must be a scalar")
    value = array.item()
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def read_prediction_npz(path: Path) -> dict[str, Any]:
    _require_regular_file(path, label="prediction archive", max_bytes=MAX_PREDICTION_BYTES)
    with zipfile.ZipFile(path, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if names != list(EXPECTED_PREDICTION_MEMBERS) or len(set(names)) != len(names):
            raise ValueError("prediction archive members are invalid")
        arrays: dict[str, np.ndarray] = {}
        for info in infos:
            if info.flag_bits & 0x1:
                raise ValueError(f"prediction archive member is encrypted: {info.filename}")
            if info.date_time != FIXED_ZIP_TIMESTAMP or info.create_system != 3:
                raise ValueError(f"prediction archive member metadata is invalid: {info.filename}")
            if ((info.external_attr >> 16) & 0o777) != FIXED_ZIP_MODE:
                raise ValueError(f"prediction archive member permissions are invalid: {info.filename}")
            if info.file_size > MAX_MEMBER_BYTES[info.filename]:
                raise ValueError(f"prediction archive member exceeds size ceiling: {info.filename}")
            compressed_size = max(info.compress_size, 1)
            if info.file_size / compressed_size > MAX_COMPRESSION_RATIO:
                raise ValueError(f"prediction archive member compression ratio is too high: {info.filename}")
            payload = _read_zip_member_bytes(archive, info, max_bytes=MAX_MEMBER_BYTES[info.filename])
            arrays[info.filename] = _load_np_array(info.filename, payload)

    masks = arrays["masks.npy"]
    if masks.ndim != 3 or masks.shape[0] <= 0 or masks.shape[1] <= 0 or masks.shape[2] <= 0:
        raise ValueError("masks.npy must be rank-3 with non-empty dimensions")
    if masks.dtype == np.bool_:
        masks = masks.astype(np.uint8)
    elif masks.dtype == np.uint8:
        unique_values = set(int(value) for value in np.unique(masks))
        if not unique_values.issubset({0, 1}):
            raise ValueError("masks.npy must be binary")
    else:
        raise ValueError("masks.npy must have dtype bool or uint8")
    emitted_k = _read_scalar_int(arrays["emitted_k.npy"], label="emitted_k")
    selected_k = _read_scalar_int(arrays["selected_k.npy"], label="selected_k")
    if emitted_k != masks.shape[0]:
        raise ValueError("emitted_k does not match masks count")
    if selected_k <= 0:
        raise ValueError("selected_k must be positive")
    return {
        "masks": masks.astype(np.uint8, copy=False),
        "selected_k": selected_k,
        "emitted_k": emitted_k,
    }


def rasterize_polyline(pixel_points: list[tuple[float, float]], image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    canvas = np.zeros((height, width), dtype=np.uint8)
    # Mirror the historical evaluator exactly: cast float pixel coordinates to int32
    # without extra rounding before drawing the antialiased 5 px polyline.
    polyline = np.asarray(pixel_points, dtype=np.float64).astype(np.int32).reshape(1, -1, 2)
    cv2.polylines(canvas, polyline, isClosed=False, color=255, thickness=5, lineType=cv2.LINE_AA)
    return canvas > 0


def compute_iou(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_bool = np.asarray(lhs, dtype=bool)
    rhs_bool = np.asarray(rhs, dtype=bool)
    if lhs_bool.shape != rhs_bool.shape:
        raise ValueError("mask shape mismatch")
    union = np.logical_or(lhs_bool, rhs_bool).sum()
    if union == 0:
        return 0.0
    intersection = np.logical_and(lhs_bool, rhs_bool).sum()
    return float(intersection / union)


def score_panel(
    *,
    condition: str,
    panel_id: str,
    curves: List[Dict[str, Any]],
    image_size: Tuple[int, int],
    prediction_masks: np.ndarray,
) -> List[Dict[str, Any]]:
    gt_masks = [rasterize_polyline(curve["pixel_points"], image_size) for curve in curves]
    pred_masks = np.asarray(prediction_masks, dtype=bool)
    if pred_masks.ndim != 3:
        raise ValueError("prediction masks must be rank-3")
    width, height = image_size
    if pred_masks.shape[1:] != (height, width):
        raise ValueError("prediction mask spatial shape does not match annotation")

    assignment: Dict[int, Tuple[int, float]] = {}
    if gt_masks and pred_masks.shape[0] > 0:
        iou_matrix = np.zeros((len(gt_masks), pred_masks.shape[0]), dtype=np.float64)
        for gt_index, gt_mask in enumerate(gt_masks):
            for pred_index in range(pred_masks.shape[0]):
                iou_matrix[gt_index, pred_index] = compute_iou(gt_mask, pred_masks[pred_index])
        gt_indices, pred_indices = linear_sum_assignment(1.0 - iou_matrix)
        for gt_index, pred_index in zip(gt_indices.tolist(), pred_indices.tolist()):
            assignment[gt_index] = (pred_index, float(iou_matrix[gt_index, pred_index]))

    rows: List[Dict[str, Any]] = []
    for gt_index, curve in enumerate(curves):
        if gt_index in assignment:
            pred_index, iou = assignment[gt_index]
            matched_prediction_index = pred_index
        else:
            iou = 0.0
            matched_prediction_index = ""
        rows.append(
            {
                "condition": condition,
                "panel_id": panel_id,
                "curve_id": curve["curve_id"],
                "matched_prediction_index": matched_prediction_index,
                "iou": iou,
                "pass_0_3": int(iou >= 0.3),
                "pass_0_4": int(iou >= 0.4),
                "pass_0_5": int(iou >= 0.5),
            }
        )
    return rows


def _load_clean_split(path: Path) -> list[str]:
    _require_regular_file(path, label="clean90 split", max_bytes=128 * 1024)
    panel_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(panel_ids) != EXPECTED_PANELS:
        raise ValueError(f"clean90 split must contain exactly {EXPECTED_PANELS} ids")
    if len(set(panel_ids)) != EXPECTED_PANELS:
        raise ValueError("clean90 split contains duplicate ids")
    for panel_id in panel_ids:
        if PANEL_ID_PATTERN.fullmatch(panel_id) is None:
            raise ValueError(f"clean90 split contains invalid panel id: {panel_id}")
    return panel_ids


def _load_curve_count_cache(path: Path, panel_ids: list[str]) -> dict[str, int]:
    _require_regular_file(path, label="curve-count cache", max_bytes=2 * 1024 * 1024)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["panel_id", "selected_k"]:
            raise ValueError(f"curve-count cache has invalid columns: {reader.fieldnames}")
        rows = list(reader)
    if len(rows) != EXPECTED_PANELS:
        raise ValueError(f"curve-count cache must contain exactly {EXPECTED_PANELS} rows")
    cache: dict[str, int] = {}
    for row in rows:
        panel_id = row["panel_id"]
        if panel_id in cache:
            raise ValueError(f"duplicate panel_id in curve-count cache: {panel_id}")
        if panel_id not in panel_ids:
            raise ValueError(f"unexpected panel_id in curve-count cache: {panel_id}")
        try:
            selected_k = int(row["selected_k"])
        except ValueError as exc:
            raise ValueError(f"selected_k is not an integer for {panel_id}") from exc
        if selected_k <= 0:
            raise ValueError(f"selected_k must be positive for {panel_id}")
        cache[panel_id] = selected_k
    if set(cache) != set(panel_ids):
        raise ValueError("curve-count cache does not match clean90 split")
    return cache


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for condition in CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        if len(condition_rows) != EXPECTED_GT_CURVES:
            raise ValueError(f"condition {condition} must contain exactly {EXPECTED_GT_CURVES} GT curves")
        lr_03_hits = sum(int(row["pass_0_3"]) for row in condition_rows)
        lr_04_hits = sum(int(row["pass_0_4"]) for row in condition_rows)
        lr_05_hits = sum(int(row["pass_0_5"]) for row in condition_rows)
        summaries.append(
            {
                "condition": condition,
                "n_panels": EXPECTED_PANELS,
                "n_gt_curves": EXPECTED_GT_CURVES,
                "lr_0_3_hits": lr_03_hits,
                "lr_0_3": lr_03_hits / EXPECTED_GT_CURVES,
                "lr_0_4_hits": lr_04_hits,
                "lr_0_4": lr_04_hits / EXPECTED_GT_CURVES,
                "lr_0_5_hits": lr_05_hits,
                "lr_0_5": lr_05_hits / EXPECTED_GT_CURVES,
            }
        )
    return summaries


def _prepare_output_dir(capsule: Path, output_dir: Path) -> Tuple[Path, Path]:
    capsule_resolved = capsule.resolve()
    output_path = Path(output_dir)
    if output_path.is_symlink():
        raise ValueError("output directory must not be a symlink")

    output_resolved_parent = output_path.parent.resolve()
    if output_path.exists():
        if not output_path.is_dir():
            raise ValueError("output path must be a directory")
        if output_path.resolve().is_relative_to(capsule_resolved):
            raise ValueError("output directory must be outside the capsule")
        if any(output_path.iterdir()):
            raise ValueError("output directory must be empty")
        if output_path.parent.resolve().is_relative_to(capsule_resolved):
            raise ValueError("output directory must be outside the capsule")
        temp_path = output_path.parent / f".{output_path.name}.tmp-{uuid.uuid4().hex}"
        temp_path.mkdir()
        return temp_path, output_path

    if output_resolved_parent.is_relative_to(capsule_resolved):
        raise ValueError("output directory must be outside the capsule")
    temp_path = output_path.parent / f".{output_path.name}.tmp-{uuid.uuid4().hex}"
    temp_path.mkdir()
    return temp_path, output_path


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    _require_regular_file(path, label="csv file", max_bytes=32 * 1024 * 1024)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"csv file is missing a header: {path}")
        return list(reader.fieldnames), list(reader)


def _parse_finite_float(value: str, *, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite float in {label}")
    return parsed


def _compare_csv_semantic(expected_path: Path, actual_path: Path, fieldnames: Tuple[str, ...]) -> None:
    expected_header, expected_rows = _read_csv_rows(expected_path)
    actual_header, actual_rows = _read_csv_rows(actual_path)
    if expected_header != list(fieldnames) or actual_header != list(fieldnames):
        raise ValueError(f"expected csv schema mismatch for {expected_path.name}")
    if len(expected_rows) != len(actual_rows):
        raise ValueError(f"expected csv row-count mismatch for {expected_path.name}")
    for row_index, (expected_row, actual_row) in enumerate(zip(expected_rows, actual_rows)):
        for field in fieldnames:
            expected_value = expected_row[field]
            actual_value = actual_row[field]
            if field in FLOAT_FIELDS:
                expected_float = _parse_finite_float(expected_value, label=f"{expected_path.name}:{field}")
                actual_float = _parse_finite_float(actual_value, label=f"{actual_path.name}:{field}")
                if abs(expected_float - actual_float) > 1e-12:
                    raise ValueError(f"expected csv mismatch for {expected_path.name} row {row_index} field {field}")
            elif expected_value != actual_value:
                raise ValueError(f"expected csv mismatch for {expected_path.name} row {row_index} field {field}")


def evaluate_capsule(capsule: Path, output_dir: Path, *, check_expected: bool = False) -> dict[str, Any]:
    root = _validate_capsule_root(Path(capsule))
    verify_manifest(root)

    panel_ids = _load_clean_split(root / "splits" / "clean90.txt")
    annotations: Dict[str, Dict[str, Any]] = {}
    total_gt_curves = 0
    for panel_id in panel_ids:
        annotation = load_annotation(root / "annotations" / f"{panel_id}.json")
        annotations[panel_id] = annotation
        total_gt_curves += len(annotation["curves"])
    if len(annotations) != EXPECTED_PANELS:
        raise ValueError(f"annotation set must contain exactly {EXPECTED_PANELS} panels")
    if total_gt_curves != EXPECTED_GT_CURVES:
        raise ValueError(f"annotation set must contain exactly {EXPECTED_GT_CURVES} GT curves")

    per_curve_rows: List[Dict[str, Any]] = []
    for condition in CONDITIONS:
        cache = _load_curve_count_cache(root / "curve_counts" / f"{condition}.csv", panel_ids)
        prediction_dir = root / "predictions" / condition
        if prediction_dir.is_symlink() or not prediction_dir.is_dir():
            raise ValueError(f"prediction directory is invalid for {condition}")
        prediction_files = sorted(path.name for path in prediction_dir.iterdir())
        expected_files = [f"{panel_id}{PREDICTION_ARCHIVE_SUFFIX}" for panel_id in panel_ids]
        if prediction_files != expected_files:
            raise ValueError(f"prediction directory does not match clean90 split for {condition}")
        for panel_id in panel_ids:
            prediction = read_prediction_npz(prediction_dir / f"{panel_id}{PREDICTION_ARCHIVE_SUFFIX}")
            if prediction["selected_k"] != cache[panel_id]:
                raise ValueError(f"curve-count cache disagrees with prediction metadata for {condition}/{panel_id}")
            annotation = annotations[panel_id]
            per_curve_rows.extend(
                score_panel(
                    condition=condition,
                    panel_id=panel_id,
                    curves=annotation["curves"],
                    image_size=annotation["image_size"],
                    prediction_masks=prediction["masks"],
                )
            )

    if len(per_curve_rows) != len(CONDITIONS) * EXPECTED_GT_CURVES:
        raise ValueError("per-curve row count is invalid")
    summary_rows = summarize_rows(per_curve_rows)

    work_dir, final_dir = _prepare_output_dir(root, Path(output_dir))
    try:
        _write_csv(work_dir / "per_curve.csv", PER_CURVE_FIELDS, per_curve_rows)
        _write_csv(work_dir / "ac_k_source_table.csv", SUMMARY_FIELDS, summary_rows)
        if check_expected:
            _compare_csv_semantic(root / "expected" / "per_curve.csv", work_dir / "per_curve.csv", PER_CURVE_FIELDS)
            _compare_csv_semantic(
                root / "expected" / "ac_k_source_table.csv",
                work_dir / "ac_k_source_table.csv",
                SUMMARY_FIELDS,
            )
        if final_dir.exists():
            if any(final_dir.iterdir()):
                raise ValueError("output directory must remain empty before promotion")
            final_dir.rmdir()
        os.replace(work_dir, final_dir)
        realized_output = final_dir
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise

    return {
        "output_dir": str(realized_output),
        "n_panels": EXPECTED_PANELS,
        "n_gt_curves": EXPECTED_GT_CURVES,
        "n_rows": len(per_curve_rows),
    }


def _default_output_dir(capsule: Path) -> Path:
    temp_dir = tempfile.mkdtemp(prefix="ac_k_source_eval_")
    return Path(temp_dir)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the public AC k-source capsule.")
    parser.add_argument("--capsule", required=True, type=Path, help="Path to the public capsule root.")
    parser.add_argument("--output", type=Path, help="Directory for regenerated CSV outputs.")
    parser.add_argument("--verify-only", action="store_true", help="Verify MANIFEST.sha256 and exit.")
    parser.add_argument("--check-expected", action="store_true", help="Compare regenerated outputs with capsule/expected.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    capsule = _validate_capsule_root(args.capsule)
    if args.verify_only:
        verify_manifest(capsule)
        print("manifest_ok files_verified")
        return 0
    output_dir = args.output if args.output is not None else _default_output_dir(capsule)
    result = evaluate_capsule(capsule, output_dir, check_expected=args.check_expected)
    print(
        f"evaluation_ok panels={result['n_panels']} gt_curves={result['n_gt_curves']} "
        f"rows={result['n_rows']} output_dir={result['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
