#!/usr/bin/env python3
"""Run the paper-aligned MatDeplot pipeline on one chart image."""

from __future__ import annotations

import argparse
from pathlib import Path

from matdeplot.config import load_config, load_paper_config
from matdeplot.curve_count import load_cached_counts
from matdeplot.output import write_extraction
from matdeplot.pipeline import MatDeplotPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for extraction.json and per-curve XLSX files.",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--curve-count-source",
        choices=["explicit", "cached", "gemini", "silhouette"],
        default="silhouette",
    )
    parser.add_argument("--curve-count", type=int)
    parser.add_argument("--curve-count-cache", type=Path)
    parser.add_argument(
        "--pixel-only",
        action="store_true",
        help="Skip optional OCR axis calibration.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config) if args.config else load_paper_config()
    cached_counts = (
        load_cached_counts(args.curve_count_cache)
        if args.curve_count_cache
        else None
    )
    pipeline = MatDeplotPipeline(args.models, config=config)
    result = pipeline.extract(
        image_path=args.image,
        curve_count_source=args.curve_count_source,
        explicit_curve_count=args.curve_count,
        cached_counts=cached_counts,
        calibrate=not args.pixel_only,
    )
    write_extraction(result, args.output)
    if not result.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
