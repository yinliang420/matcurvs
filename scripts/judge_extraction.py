#!/usr/bin/env python3
"""Replay the public extraction judge on one image/extraction pair."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

from matdeplot.judge import (
    JudgeVerdict,
    QwenExtractionJudge,
    load_curves_from_extraction,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--extraction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=["qwen-vl-max"],
        default="qwen-vl-max",
    )
    return parser


def _temp_output_path(output_path: Path) -> Path:
    return output_path.with_name(
        f".{output_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )


def write_json_atomic(output_path: Path, payload: dict) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temp_output_path(output_path)
    try:
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, output_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _loader_error_verdict(model: str, exc: Exception) -> JudgeVerdict:
    return JudgeVerdict(
        model=model,
        admitted=False,
        reason="Extraction payload could not be loaded.",
        error=f"extraction:{type(exc).__name__}",
    )


def _review_error_verdict(model: str, exc: Exception) -> JudgeVerdict:
    return JudgeVerdict(
        model=model,
        admitted=False,
        reason="Final quality judge failed operationally.",
        error=f"judge:{type(exc).__name__}",
    )


def _exit_code_for_verdict(verdict: JudgeVerdict) -> int:
    if verdict.error is not None:
        return 2
    if verdict.admitted:
        return 0
    return 1


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    judge = QwenExtractionJudge(model=args.model)

    try:
        curves = load_curves_from_extraction(args.extraction)
    except Exception as exc:
        verdict = _loader_error_verdict(args.model, exc)
        write_json_atomic(args.output, verdict.to_dict())
        return 2

    try:
        verdict = judge.review(args.image, curves)
    except Exception as exc:
        verdict = _review_error_verdict(args.model, exc)

    write_json_atomic(args.output, verdict.to_dict())
    return _exit_code_for_verdict(verdict)


if __name__ == "__main__":
    raise SystemExit(main())
