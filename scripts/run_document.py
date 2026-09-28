#!/usr/bin/env python3
"""Run the paper's PDF/figure-to-curve extraction workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from matdeplot.document_pipeline import DocumentPipeline, QwenPanelClassifier
from matdeplot.judge import QwenExtractionJudge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--curve-count-source",
        choices=["cached", "gemini", "silhouette"],
        default="silhouette",
    )
    parser.add_argument("--curve-count-cache", type=Path)
    parser.add_argument("--assume-line-charts", action="store_true")
    parser.add_argument("--pixel-only", action="store_true")
    parser.add_argument("--judge", choices=["qwen-vl-max"])
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    classifier = None if args.assume_line_charts else QwenPanelClassifier()
    judge = None if args.judge is None else QwenExtractionJudge(model=args.judge)
    workflow = DocumentPipeline(args.models, classifier=classifier, judge=judge)
    workflow.run(
        input_path=args.input,
        output_directory=args.output,
        curve_count_source=args.curve_count_source,
        curve_count_cache=args.curve_count_cache,
        assume_line_charts=args.assume_line_charts,
        calibrate=not args.pixel_only,
    )


if __name__ == "__main__":
    main()
