"""Selected V11 foreground-training engine."""

from __future__ import annotations

import contextlib
import math
from collections import OrderedDict
from pathlib import Path

import torch


class WarmupCosineScheduler:
    """Five-step warmup followed by cosine decay with checkpointable state."""

    def __init__(self, optimizer, epochs: int, warmup_epochs: int = 5) -> None:
        self.optimizer = optimizer
        self.epochs = int(epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.last_epoch = -1

    @property
    def effective_warmup_epochs(self) -> int:
        return min(self.warmup_epochs, self.epochs)

    def _lr_factor(self, epoch: int) -> float:
        if self.effective_warmup_epochs > 0 and epoch < self.effective_warmup_epochs:
            return float(epoch + 1) / float(self.effective_warmup_epochs)
        if self.epochs <= self.effective_warmup_epochs:
            return 1.0
        decay_steps = self.epochs - self.effective_warmup_epochs
        progress = min(
            max(epoch - self.effective_warmup_epochs + 1, 0),
            decay_steps,
        ) / float(decay_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def step(self) -> None:
        self.last_epoch += 1
        factor = self._lr_factor(self.last_epoch)
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * factor

    def state_dict(self):
        return {
            "epochs": self.epochs,
            "warmup_epochs": self.warmup_epochs,
            "base_lrs": list(self.base_lrs),
            "last_epoch": self.last_epoch,
        }

    def load_state_dict(self, state_dict) -> None:
        self.epochs = int(state_dict["epochs"])
        self.warmup_epochs = int(state_dict["warmup_epochs"])
        self.base_lrs = list(state_dict["base_lrs"])
        self.last_epoch = int(state_dict["last_epoch"])
        if self.last_epoch >= 0:
            factor = self._lr_factor(self.last_epoch)
            for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
                group["lr"] = base_lr * factor


def _move_batch_to_device(batch, device: torch.device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def get_autocast_context(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def create_optimizer(model, learning_rate: float, weight_decay: float):
    encoder_params = [
        parameter
        for parameter in model.encoder.parameters()
        if parameter.requires_grad
    ]
    encoder_ids = {id(parameter) for parameter in encoder_params}
    head_params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in encoder_ids
    ]
    if not encoder_params:
        raise ValueError("model.encoder must expose trainable parameters")
    if not head_params:
        raise ValueError("model must expose non-encoder trainable parameters")
    return torch.optim.AdamW(
        [
            {
                "params": encoder_params,
                "lr": float(learning_rate) * 0.1,
                "weight_decay": float(weight_decay),
            },
            {
                "params": head_params,
                "lr": float(learning_rate),
                "weight_decay": float(weight_decay),
            },
        ]
    )


def make_scheduler(optimizer, epochs: int, warmup_epochs: int = 5):
    epochs = int(epochs)
    warmup_epochs = int(warmup_epochs)
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be >= 0")
    return WarmupCosineScheduler(
        optimizer=optimizer,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
    )


def train_one_epoch(model, loader, criterion, optimizer, device, epoch: int):
    del epoch
    model.train()
    total_weighted_loss = 0.0
    total_sample_weight = 0.0
    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        images = batch["image"]
        optimizer.zero_grad(set_to_none=True)
        with get_autocast_context(device):
            outputs = model(images)
            loss, stats = criterion(outputs, batch)
        loss.backward()
        optimizer.step()
        sample_weight = batch["sample_weight"].detach().double()
        batch_weight = float(sample_weight.sum().item())
        per_sample_loss = stats["per_sample_loss"].detach().double()
        total_weighted_loss += float((per_sample_loss * sample_weight).sum().item())
        total_sample_weight += batch_weight
    if total_sample_weight <= 0.0:
        raise ValueError("loader must yield at least one batch")
    return {"train_loss": total_weighted_loss / total_sample_weight}


@torch.inference_mode()
def evaluate_loader(model, loader, criterion, device, binary_threshold: float):
    model.eval()
    total_weighted_loss = 0.0
    total_sample_weight = 0.0
    total_pixels = 0
    total_intersection = 0
    total_union = 0
    total_pred_fg = 0
    total_gt_fg = 0

    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        images = batch["image"]
        outputs = model(images)
        loss, stats = criterion(outputs, batch)
        union_probability = stats["union_probability"]
        target_mask = stats["target_mask"] >= 0.5
        prediction = union_probability >= float(binary_threshold)

        sample_weight = batch["sample_weight"].detach().double()
        per_sample_loss = stats["per_sample_loss"].detach().double()
        batch_weight = float(sample_weight.sum().item())
        total_weighted_loss += float((per_sample_loss * sample_weight).sum().item())
        total_sample_weight += batch_weight
        total_pixels += prediction.numel()
        total_intersection += int(torch.logical_and(prediction, target_mask).sum().item())
        total_union += int(torch.logical_or(prediction, target_mask).sum().item())
        total_pred_fg += int(prediction.sum().item())
        total_gt_fg += int(target_mask.sum().item())

    if total_sample_weight <= 0.0:
        raise ValueError("loader must yield at least one batch")

    binary_iou = total_intersection / total_union if total_union else 0.0
    binary_precision = total_intersection / total_pred_fg if total_pred_fg else 0.0
    binary_recall = total_intersection / total_gt_fg if total_gt_fg else 0.0
    if binary_precision + binary_recall:
        binary_f1 = 2.0 * binary_precision * binary_recall / (binary_precision + binary_recall)
    else:
        binary_f1 = 0.0
    pred_fg_ratio = total_pred_fg / float(total_pixels) if total_pixels else 0.0
    gt_fg_ratio = total_gt_fg / float(total_pixels) if total_pixels else 0.0

    return {
        "val_loss": total_weighted_loss / total_sample_weight,
        "binary_iou": binary_iou,
        "binary_precision": binary_precision,
        "binary_recall": binary_recall,
        "binary_f1": binary_f1,
        "pred_fg_ratio": pred_fg_ratio,
        "gt_fg_ratio": gt_fg_ratio,
        "score": binary_iou,
    }


def save_checkpoint(path, model, optimizer, scheduler, epoch: int, config, best_score: float, metrics):
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = OrderedDict(
        [
            ("model", model.state_dict()),
            ("optimizer", optimizer.state_dict()),
            ("scheduler", scheduler.state_dict()),
            ("epoch", int(epoch)),
            ("config", config),
            ("best_score", float(best_score)),
            ("metrics", metrics),
        ]
    )
    torch.save(payload, checkpoint_path)


__all__ = [
    "create_optimizer",
    "evaluate_loader",
    "get_autocast_context",
    "make_scheduler",
    "save_checkpoint",
    "train_one_epoch",
]
