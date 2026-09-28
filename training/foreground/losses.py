"""Selected V11 foreground-training losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryMaskIoUCriterion(nn.Module):
    """Weighted soft binary-IoU over the union of query-mask predictions."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, outputs, batch):
        logits = outputs["mask_logits"]
        target_mask = batch["binary_mask"].float()
        sample_weight = batch["sample_weight"].float().to(logits.device)
        if target_mask.ndim == 3:
            target_mask = target_mask.unsqueeze(1)
        elif target_mask.ndim != 4:
            raise ValueError("binary_mask must have shape [B, H, W] or [B, 1, H, W]")
        if target_mask.shape[1] != 1:
            raise ValueError("binary_mask channel dimension must be 1")
        if sample_weight.ndim != 1:
            raise ValueError("sample_weight must have shape [B]")
        if logits.shape[0] != target_mask.shape[0] or logits.shape[0] != sample_weight.shape[0]:
            raise ValueError("batch dimensions must match across logits, binary_mask, and sample_weight")

        target_mask = F.interpolate(
            target_mask.to(logits.device),
            size=logits.shape[-2:],
            mode="nearest",
        )
        union_probability = logits.sigmoid().amax(dim=1, keepdim=True)
        intersection = (union_probability * target_mask).flatten(1).sum(dim=1)
        union = (
            union_probability
            + target_mask
            - union_probability * target_mask
        ).flatten(1).sum(dim=1)
        per_sample_loss = 1.0 - (intersection + self.eps) / (union + self.eps)
        weight_sum = sample_weight.sum()
        if weight_sum <= 0:
            raise ValueError("sample_weight must sum to a positive value")
        loss = (per_sample_loss * sample_weight).sum() / weight_sum
        stats = {
            "loss": loss.detach(),
            "per_sample_loss": per_sample_loss.detach(),
            "union_probability": union_probability.detach(),
            "target_mask": target_mask.detach(),
        }
        return loss, stats


__all__ = ["BinaryMaskIoUCriterion"]
