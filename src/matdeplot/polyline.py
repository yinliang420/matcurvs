"""Convert per-curve masks into ordered original-image polylines."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from skimage.morphology import skeletonize

from .schema import Point


def mask_to_polyline(
    mask: np.ndarray,
    minimum_length: int = 10,
    sample_every: int = 2,
) -> List[Point]:
    """Skeletonize a curve mask and retain one mean y value for each x."""

    binary = mask.astype(bool)
    if int(binary.sum()) < minimum_length:
        return []
    skeleton = skeletonize(binary)
    ys, xs = np.where(skeleton)
    if len(xs) < minimum_length:
        ys, xs = np.where(binary)
    if len(xs) < minimum_length:
        return []

    points = pd.DataFrame({"x": xs, "y": ys})
    ordered = (
        points.groupby("x", as_index=False)["y"]
        .mean()
        .sort_values("x")
        .reset_index(drop=True)
    )
    if sample_every > 1 and len(ordered) > 4 * sample_every:
        ordered = ordered.iloc[::sample_every]
    return [
        (float(row.x), float(row.y))
        for row in ordered.itertuples(index=False)
    ]
