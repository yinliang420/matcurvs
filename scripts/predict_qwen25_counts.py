#!/usr/bin/env python3
"""Predict chart curve counts with a local Qwen2.5-VL-7B-Instruct checkpoint.

The resulting CSV can be supplied to ``scripts/run_chart.py`` with
``--curve-count-source cached --curve-count-cache <CSV>``. The script never
uses annotations or a hosted API. It records non-identifying provenance beside
the cache so resumed runs cannot silently mix configurations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from matdeplot.curve_count import CURVE_COUNT_PROMPT, parse_curve_count


BACKEND = "qwen2.5-vl"
COUNT_COLUMN = f"{BACKEND}_k"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_images(paths: list[Path]) -> list[Path]:
    """Validate input images and require unique cache keys (their stems)."""

    if not paths:
        raise ValueError("at least one --image is required")
    images: list[Path] = []
    seen_ids: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"image not found: {path}")
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            supported = ", ".join(sorted(SUPPORTED_IMAGE_SUFFIXES))
            raise ValueError(f"unsupported image format {path.suffix!r}; expected {supported}")
        if path.stem in seen_ids:
            raise ValueError(f"duplicate image stem is not a valid cache key: {path.stem}")
        seen_ids.add(path.stem)
        images.append(path)
    return images


def checkpoint_metadata(model_path: Path) -> dict[str, Any]:
    """Hash the local Qwen checkpoint, including every safetensors shard."""

    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(
            "Qwen2.5-VL local checkpoints must contain config.json and "
            "model.safetensors.index.json"
        )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid safetensors index: {index_path}")
    shard_names = sorted(set(str(name) for name in weight_map.values()))
    shards: dict[str, str] = {}
    for name in shard_names:
        shard_path = model_path / name
        if Path(name).name != name or not shard_path.is_file():
            raise FileNotFoundError(f"checkpoint shard not found: {shard_path}")
        shards[name] = sha256(shard_path)
    return {
        "config_sha256": sha256(config_path),
        "weight_index_sha256": sha256(index_path),
        "weight_shards_sha256": shards,
    }


def environment_metadata() -> dict[str, str | None]:
    """Return dependency versions without embedding host names or file paths."""

    import torch
    import transformers

    try:
        from importlib.metadata import version

        qwen_vl_utils_version = version("qwen-vl-utils")
    except Exception:
        qwen_vl_utils_version = "unknown"
    try:
        from importlib.metadata import version

        flash_attention_version = version("flash-attn")
    except Exception:
        flash_attention_version = "unknown"
    return {
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "qwen_vl_utils": qwen_vl_utils_version,
        "flash_attn": flash_attention_version,
    }


def build_metadata(
    model_path: Path,
    model_id: str,
    images: list[Path],
    max_new_tokens: int,
    attention_implementation: str,
    dtype: str,
) -> dict[str, Any]:
    """Create the stable provenance contract for a new or resumed cache."""

    return {
        "schema_version": "1.0",
        "backend": BACKEND,
        "model_id": model_id,
        "checkpoint": checkpoint_metadata(model_path),
        "prompt_sha256": hashlib.sha256(CURVE_COUNT_PROMPT.encode("utf-8")).hexdigest(),
        "generation": {
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "attention_implementation": attention_implementation,
            "torch_dtype": dtype,
        },
        "environment": environment_metadata(),
        "input_images": [
            {"image_id": image.stem, "sha256": sha256(image)} for image in images
        ],
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist metadata so an interrupted job cannot leave partial JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_completed(output_csv: Path) -> dict[str, dict[str, str]]:
    """Load a well-formed cache; failed rows remain eligible for retry."""

    if not output_csv.exists():
        return {}
    with output_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = {"image_id", COUNT_COLUMN, f"{BACKEND}_latency_s", f"{BACKEND}_error"}
        if not reader.fieldnames or not fields.issubset(reader.fieldnames):
            raise ValueError(f"invalid existing cache: {output_csv}")
        rows = list(reader)
    identifiers = [row["image_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"cache contains duplicate image IDs: {output_csv}")
    return {row["image_id"]: row for row in rows}


def validate_resume(metadata_path: Path, expected: dict[str, Any]) -> None:
    """Reject a resume if it would mix results from different conditions."""

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"cannot resume without matching metadata: {metadata_path}; use a new output path"
        )
    observed = json.loads(metadata_path.read_text(encoding="utf-8"))
    if observed != expected:
        raise ValueError(
            "refusing to resume: existing metadata differs from the requested "
            "model, checkpoint, prompt, settings, environment, or input images"
        )


def write_cache(output_csv: Path, rows: dict[str, dict[str, str]], images: list[Path]) -> None:
    """Atomically write one cache row per requested image, in input order."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image_id", COUNT_COLUMN, f"{BACKEND}_latency_s", f"{BACKEND}_error"]
    temporary = output_csv.with_name(f".{output_csv.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for image in images:
            if image.stem in rows:
                writer.writerow(rows[image.stem])
    temporary.replace(output_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--raw-jsonl", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--attn-implementation",
        choices=["flash_attention_2", "sdpa"],
        default="flash_attention_2",
    )
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"model directory not found: {args.model_path}")
    images = collect_images(args.image)
    metadata_path = args.metadata_json or args.output_csv.with_name(
        f"{args.output_csv.stem}_metadata.json"
    )
    metadata = build_metadata(
        args.model_path,
        args.model_id,
        images,
        args.max_new_tokens,
        args.attn_implementation,
        args.torch_dtype,
    )
    outputs = [args.output_csv, metadata_path]
    if args.raw_jsonl is not None:
        outputs.append(args.raw_jsonl)
    if args.resume:
        if not args.output_csv.is_file():
            raise FileNotFoundError(f"cannot resume without a cache: {args.output_csv}")
        validate_resume(metadata_path, metadata)
        completed = load_completed(args.output_csv)
    else:
        existing = next((path for path in outputs if path.exists()), None)
        if existing is not None:
            raise FileExistsError(f"refusing to overwrite existing output: {existing}; use --resume")
        completed = {}

    try:
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Qwen2.5-VL requires the dependencies in requirements-qwen25.txt"
        ) from exc

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=getattr(torch, args.torch_dtype),
            attn_implementation=args.attn_implementation,
            device_map="auto",
            local_files_only=True,
        )
    except Exception as exc:
        if args.attn_implementation == "flash_attention_2":
            raise RuntimeError(
                "could not load FlashAttention 2; install flash-attn or use "
                "--attn-implementation sdpa (which is not the recorded setting)"
            ) from exc
        raise
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    if not args.resume:
        write_json_atomic(metadata_path, metadata)
    if args.raw_jsonl is not None:
        args.raw_jsonl.parent.mkdir(parents=True, exist_ok=True)
    raw_handle = (
        args.raw_jsonl.open("a", encoding="utf-8") if args.raw_jsonl is not None else None
    )
    try:
        for index, image in enumerate(images, start=1):
            previous = completed.get(image.stem)
            if previous is not None and previous[COUNT_COLUMN]:
                continue
            started = time.monotonic()
            raw, error, value = "", "", None
            try:
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": str(image)},
                    {"type": "text", "text": CURVE_COUNT_PROMPT},
                ]}]
                prompt = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)
                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=args.max_new_tokens,
                    )
                generated_ids = [
                    output_ids[len(input_ids):]
                    for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
                ]
                raw = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                value = parse_curve_count(raw)
                if value is None:
                    error = "parse_failed"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            latency = time.monotonic() - started
            row = {
                "image_id": image.stem,
                COUNT_COLUMN: "" if value is None else str(value),
                f"{BACKEND}_latency_s": f"{latency:.4f}",
                f"{BACKEND}_error": error,
            }
            completed[image.stem] = row
            if raw_handle is not None:
                raw_handle.write(json.dumps({
                    "image_id": image.stem,
                    "raw": raw,
                    "k_pred": value,
                    "error": error,
                    "latency_s": latency,
                }) + "\n")
                raw_handle.flush()
            write_cache(args.output_csv, completed, images)
            print(
                f"[{index:>3}/{len(images)}] {image.name} K={value} error={error or '-'}",
                flush=True,
            )
    finally:
        if raw_handle is not None:
            raw_handle.close()

    valid = sum(bool(completed[image.stem][COUNT_COLUMN]) for image in images)
    print(f"completed {len(images)}/{len(images)}; valid K values: {valid}", flush=True)


if __name__ == "__main__":
    main()
