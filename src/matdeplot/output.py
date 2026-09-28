"""Writers for the public MatDeplot extraction schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from .schema import ExtractionResult


def write_extraction(result: ExtractionResult, output_directory: Path) -> Dict[str, Path]:
    """Write one JSON record and one XLSX workbook per extracted curve."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    record_path = output_directory / "extraction.json"
    record_path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    written = {"record": record_path}

    for curve in result.curves:
        pixel_x = [point[0] for point in curve.pixel_points]
        pixel_y = [point[1] for point in curve.pixel_points]
        real_x = [point[0] for point in curve.real_points]
        real_y = [point[1] for point in curve.real_points]
        calibrated = len(real_x) == len(pixel_x)
        frame = pd.DataFrame(
            {
                "pixel_x": pixel_x,
                "pixel_y": pixel_y,
                "x": real_x if calibrated else [None] * len(pixel_x),
                "y": real_y if calibrated else [None] * len(pixel_y),
            }
        )
        curve_path = output_directory / f"curve_{curve.curve_index + 1:03d}.xlsx"
        frame.to_excel(curve_path, index=False)
        written[f"curve_{curve.curve_index + 1:03d}"] = curve_path
    return written
