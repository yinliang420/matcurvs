"""Selected V11 foreground-model helpers."""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch

from matdeplot.foreground import Mask2FormerStyleModel


def normalize_state_dict_keys(
    state_dict: Mapping[str, torch.Tensor],
) -> "OrderedDict[str, torch.Tensor]":
    """Strip the DataParallel prefix used by some private training checkpoints."""
    normalized = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        normalized[key] = value
    return normalized


def build_paper_model() -> Mask2FormerStyleModel:
    """Build the released paper-aligned foreground architecture."""
    return Mask2FormerStyleModel(
        encoder_name="mit_b3",
        hidden_dim=256,
        n_queries=12,
        dec_layers=3,
    )


__all__ = ["Mask2FormerStyleModel", "build_paper_model", "normalize_state_dict_keys"]
