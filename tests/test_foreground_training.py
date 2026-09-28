import contextlib
import copy
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Optional
from unittest import mock

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from matdeplot.foreground import Mask2FormerStyleModel
from training.foreground.engine import (
    create_optimizer,
    evaluate_loader,
    get_autocast_context,
    make_scheduler,
    save_checkpoint,
    train_one_epoch,
)
from training.foreground.data import ChartRecord, UnifiedChartDataset
from training.foreground.losses import BinaryMaskIoUCriterion
from training.foreground.model import build_paper_model, normalize_state_dict_keys


class UntrustedCheckpointObject:
    pass


def weighted_soft_iou(union_probability, target, weights):
    resized_target = F.interpolate(
        target.float(),
        size=union_probability.shape[-2:],
        mode="nearest",
    )
    intersection = (union_probability * resized_target).flatten(1).sum(dim=1)
    union = (
        union_probability
        + resized_target
        - union_probability * resized_target
    ).flatten(1).sum(dim=1)
    per_sample_loss = 1.0 - (intersection + 1e-6) / (union + 1e-6)
    normalized_weights = weights / weights.sum()
    return (per_sample_loss * normalized_weights).sum()


class FakeForegroundModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=1, bias=False),
            nn.ReLU(),
        )
        self.head = nn.Conv2d(4, 2, kernel_size=1, bias=False)
        self.aux_head = nn.Conv2d(4, 1, kernel_size=1, bias=False)
        self.objectness_head = nn.Linear(4, 2, bias=False)

    def forward(self, images):
        encoded = self.encoder(images)
        pooled = encoded.mean(dim=(2, 3))
        return {
            "mask_logits": self.head(encoded),
            "objectness_logits": self.objectness_head(pooled),
            "aux_logits": self.aux_head(encoded),
        }


class IndexedLogitModel(nn.Module):
    def __init__(self, logits_by_index):
        super().__init__()
        self.encoder = nn.Identity()
        self._dummy = nn.Parameter(torch.zeros(()))
        self.register_buffer("logits_by_index", logits_by_index)

    def forward(self, images):
        indices = images[:, 0, 0, 0].round().to(dtype=torch.long)
        logits = self.logits_by_index.index_select(0, indices) + self._dummy * 0.0
        batch_size = logits.shape[0]
        height, width = logits.shape[-2:]
        return {
            "mask_logits": logits,
            "objectness_logits": torch.zeros((batch_size, logits.shape[1]), device=logits.device),
            "aux_logits": torch.zeros((batch_size, 1, height, width), device=logits.device),
        }


class ForegroundTrainingTest(unittest.TestCase):
    maxDiff = None

    def make_batch(self):
        images = torch.tensor(
            [
                [
                    [[0.1, 0.2], [0.3, 0.4]],
                    [[0.5, 0.6], [0.7, 0.8]],
                    [[0.9, 1.0], [0.2, 0.1]],
                ],
                [
                    [[0.2, 0.4], [0.6, 0.8]],
                    [[0.1, 0.3], [0.5, 0.7]],
                    [[0.9, 0.7], [0.5, 0.3]],
                ],
            ],
            dtype=torch.float32,
        )
        target = torch.tensor(
            [
                [[[1.0, 0.0], [1.0, 0.0]]],
                [[[0.0, 1.0], [0.0, 1.0]]],
            ],
            dtype=torch.float32,
        )
        sample_weight = torch.tensor([1.0, 3.0], dtype=torch.float32)
        return {
            "image": images,
            "binary_mask": target,
            "sample_weight": sample_weight,
        }

    def make_logits(self):
        return torch.tensor(
            [
                [
                    [[2.0, -2.0], [0.0, -1.0]],
                    [[-3.0, -4.0], [3.0, -2.0]],
                ],
                [
                    [[-2.0, 1.0], [-4.0, 2.0]],
                    [[-1.5, -2.5], [-3.5, 0.5]],
                ],
            ],
            dtype=torch.float32,
        )

    def write_chart_fixture(self, root: Path, stem: str):
        image = np.full((12, 12, 3), 255, dtype=np.uint8)
        image[2:10, 4:8] = np.array([32, 96, 192], dtype=np.uint8)
        image_path = root / "{0}.png".format(stem)
        annotation_path = root / "{0}_annotation.json".format(stem)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        self.assertTrue(cv2.imwrite(str(image_path), image))
        annotation_path.write_text(
            (
                '{"lines":{"Line_1":{"display_mode":"line","points":'
                '[{"pixel":[2,2]},{"pixel":[6,6]},{"pixel":[9,9]}]}}}'
            ),
            encoding="utf-8",
        )
        return ChartRecord(
            image_path=image_path,
            annotation_path=annotation_path,
            source="synthetic",
        )

    def make_aggregation_batches(self, batch_sizes):
        images = torch.zeros((3, 3, 2, 2), dtype=torch.float32)
        images[:, 0, 0, 0] = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
        binary_mask = torch.tensor(
            [
                [[[1.0, 0.0], [1.0, 0.0]]],
                [[[0.0, 1.0], [0.0, 1.0]]],
                [[[1.0, 1.0], [0.0, 0.0]]],
            ],
            dtype=torch.float32,
        )
        sample_weight = torch.tensor([1.0, 4.0, 2.0], dtype=torch.float32)
        logits = torch.tensor(
            [
                [
                    [[1.5, -2.5], [0.2, -0.2]],
                    [[-4.0, -4.0], [-4.0, -4.0]],
                ],
                [
                    [[-2.0, 1.5], [-1.0, 1.0]],
                    [[-4.0, -4.0], [-4.0, -4.0]],
                ],
                [
                    [[1.0, 0.4], [-1.5, -2.0]],
                    [[-4.0, -4.0], [-4.0, -4.0]],
                ],
            ],
            dtype=torch.float32,
        )

        batches = []
        start = 0
        for batch_size in batch_sizes:
            end = start + batch_size
            batches.append(
                {
                    "image": images[start:end].clone(),
                    "binary_mask": binary_mask[start:end].clone(),
                    "sample_weight": sample_weight[start:end].clone(),
                }
            )
            start = end
        return batches, logits, binary_mask, sample_weight

    def test_binary_mask_iou_matches_independent_expected_value(self):
        criterion = BinaryMaskIoUCriterion()
        logits = self.make_logits()
        batch = self.make_batch()
        outputs = {
            "mask_logits": logits,
            "objectness_logits": torch.full((2, 2), -99.0),
            "aux_logits": torch.full((2, 1, 2, 2), 99.0),
        }

        loss, stats = criterion(outputs, batch)

        union_probability = logits.sigmoid().amax(dim=1, keepdim=True)
        expected = weighted_soft_iou(
            union_probability,
            batch["binary_mask"],
            batch["sample_weight"],
        )
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))
        self.assertTrue(torch.allclose(stats["loss"], expected, atol=1e-6))
        self.assertEqual(tuple(stats["union_probability"].shape), (2, 1, 2, 2))

    def test_binary_mask_iou_resizes_target_and_ignores_other_outputs(self):
        criterion = BinaryMaskIoUCriterion()
        outputs = {
            "mask_logits": torch.zeros((1, 2, 4, 4), dtype=torch.float32),
            "objectness_logits": torch.randn(1, 2),
            "aux_logits": torch.randn(1, 1, 4, 4),
        }
        batch = {
            "binary_mask": torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32),
            "sample_weight": torch.tensor([2.0], dtype=torch.float32),
        }

        loss, stats = criterion(outputs, batch)

        expected_target = F.interpolate(batch["binary_mask"], size=(4, 4), mode="nearest")
        expected_prob = torch.full((1, 1, 4, 4), 0.5, dtype=torch.float32)
        expected = weighted_soft_iou(expected_prob, expected_target, batch["sample_weight"])
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))
        self.assertTrue(torch.equal(stats["target_mask"], expected_target))

    def test_dataset_loader_batch_without_channel_dimension_integrates_with_engine(self):
        criterion = BinaryMaskIoUCriterion()
        model = FakeForegroundModel()
        optimizer = create_optimizer(model, learning_rate=1e-2, weight_decay=0.0)

        with tempfile.TemporaryDirectory() as tempdir:
            record = self.write_chart_fixture(Path(tempdir), "chart")
            dataset = UnifiedChartDataset(
                records=[record],
                image_size=8,
                line_width=3,
                max_instances=4,
                is_train=False,
                real_weight=3.0,
            )
            batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
            self.assertEqual(tuple(batch["binary_mask"].shape), (1, 8, 8))

            outputs = model(batch["image"])
            loss, stats = criterion(outputs, batch)
            self.assertEqual(tuple(stats["target_mask"].shape), (1, 1, 8, 8))
            self.assertGreaterEqual(float(loss.item()), 0.0)

            train_metrics = train_one_epoch(
                model=model,
                loader=DataLoader(dataset, batch_size=1, shuffle=False),
                criterion=criterion,
                optimizer=optimizer,
                device=torch.device("cpu"),
                epoch=0,
            )
            eval_metrics = evaluate_loader(
                model=model,
                loader=DataLoader(dataset, batch_size=1, shuffle=False),
                criterion=criterion,
                device=torch.device("cpu"),
                binary_threshold=0.5,
            )

        self.assertIn("train_loss", train_metrics)
        self.assertEqual(
            set(eval_metrics),
            {
                "val_loss",
                "binary_iou",
                "binary_precision",
                "binary_recall",
                "binary_f1",
                "pred_fg_ratio",
                "gt_fg_ratio",
                "score",
            },
        )

    def test_build_paper_model_reuses_released_class(self):
        model = build_paper_model()
        self.assertIsInstance(model, Mask2FormerStyleModel)
        self.assertEqual(model.hidden_dim, 256)
        self.assertEqual(model.query_embed.num_embeddings, 12)
        self.assertEqual(len(model.decoder.layers), 3)

    def test_normalize_state_dict_keys_strips_module_prefix_only(self):
        state_dict = {
            "module.encoder.weight": torch.tensor([1.0]),
            "head.bias": torch.tensor([2.0]),
        }

        normalized = normalize_state_dict_keys(state_dict)

        self.assertEqual(set(normalized), {"encoder.weight", "head.bias"})
        self.assertTrue(torch.equal(normalized["encoder.weight"], torch.tensor([1.0])))
        self.assertTrue(torch.equal(normalized["head.bias"], torch.tensor([2.0])))

    def test_create_optimizer_splits_encoder_and_head_learning_rates(self):
        model = FakeForegroundModel()

        optimizer = create_optimizer(model, learning_rate=1e-3, weight_decay=0.05)

        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-4)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 1e-3)
        self.assertAlmostEqual(optimizer.param_groups[0]["weight_decay"], 0.05)
        self.assertAlmostEqual(optimizer.param_groups[1]["weight_decay"], 0.05)
        encoder_ids = {id(param) for param in model.encoder.parameters()}
        group0_ids = {id(param) for param in optimizer.param_groups[0]["params"]}
        group1_ids = {id(param) for param in optimizer.param_groups[1]["params"]}
        self.assertEqual(group0_ids, encoder_ids)
        self.assertTrue(group1_ids)
        self.assertTrue(group0_ids.isdisjoint(group1_ids))

    def test_scheduler_warms_up_for_five_epochs_then_decays(self):
        model = FakeForegroundModel()
        optimizer = create_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
        scheduler = make_scheduler(optimizer, epochs=10, warmup_epochs=5)
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        observed = []

        for epoch in range(10):
            optimizer.step()
            scheduler.step()
            decay_steps = 10 - 5
            if epoch < 5:
                factor = float(epoch + 1) / 5.0
            else:
                progress = min(max(epoch - 5 + 1, 0), decay_steps) / float(decay_steps)
                factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            expected = [base_lr * factor for base_lr in base_lrs]
            current = [group["lr"] for group in optimizer.param_groups]
            observed.append(current)
            for actual, want in zip(current, expected):
                self.assertAlmostEqual(actual, want, places=12)

        self.assertEqual(len(observed), 10)

    def test_scheduler_short_run_uses_effective_warmup_and_resume(self):
        model = FakeForegroundModel()
        optimizer = create_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
        scheduler = make_scheduler(optimizer, epochs=1, warmup_epochs=5)
        base_lrs = [group["lr"] for group in optimizer.param_groups]

        optimizer.step()
        scheduler.step()
        for actual, want in zip([group["lr"] for group in optimizer.param_groups], base_lrs):
            self.assertAlmostEqual(actual, want, places=12)

        model_three = FakeForegroundModel()
        optimizer_three = create_optimizer(model_three, learning_rate=1e-3, weight_decay=0.0)
        scheduler_three = make_scheduler(optimizer_three, epochs=3, warmup_epochs=5)
        base_lrs_three = [group["lr"] for group in optimizer_three.param_groups]
        expected_factors = [1.0 / 3.0, 2.0 / 3.0, 1.0]

        optimizer_three.step()
        scheduler_three.step()
        for actual, want in zip(
            [group["lr"] for group in optimizer_three.param_groups],
            [base_lr * expected_factors[0] for base_lr in base_lrs_three],
        ):
            self.assertAlmostEqual(actual, want, places=12)

        resumed_model = FakeForegroundModel()
        resumed_optimizer = create_optimizer(resumed_model, learning_rate=1e-3, weight_decay=0.0)
        resumed_scheduler = make_scheduler(resumed_optimizer, epochs=99, warmup_epochs=99)
        resumed_scheduler.load_state_dict(scheduler_three.state_dict())

        self.assertEqual(resumed_scheduler.state_dict()["last_epoch"], 0)
        self.assertEqual(resumed_scheduler.state_dict()["epochs"], 3)
        self.assertEqual(resumed_scheduler.state_dict()["warmup_epochs"], 5)
        for actual, want in zip(
            [group["lr"] for group in resumed_optimizer.param_groups],
            [base_lr * expected_factors[0] for base_lr in base_lrs_three],
        ):
            self.assertAlmostEqual(actual, want, places=12)

        optimizer_three.step()
        scheduler_three.step()
        resumed_optimizer.step()
        resumed_scheduler.step()
        expected_second = [base_lr * expected_factors[1] for base_lr in base_lrs_three]
        for actual, want in zip([group["lr"] for group in optimizer_three.param_groups], expected_second):
            self.assertAlmostEqual(actual, want, places=12)
        for actual, want in zip([group["lr"] for group in resumed_optimizer.param_groups], expected_second):
            self.assertAlmostEqual(actual, want, places=12)

        optimizer_three.step()
        scheduler_three.step()
        resumed_optimizer.step()
        resumed_scheduler.step()
        expected_third = [base_lr * expected_factors[2] for base_lr in base_lrs_three]
        for actual, want in zip([group["lr"] for group in optimizer_three.param_groups], expected_third):
            self.assertAlmostEqual(actual, want, places=12)
        for actual, want in zip([group["lr"] for group in resumed_optimizer.param_groups], expected_third):
            self.assertAlmostEqual(actual, want, places=12)

    def test_train_one_epoch_updates_parameters_on_cpu(self):
        model = FakeForegroundModel()
        criterion = BinaryMaskIoUCriterion()
        optimizer = create_optimizer(model, learning_rate=1e-2, weight_decay=0.0)
        batch = self.make_batch()
        before = model.head.weight.detach().clone()

        metrics = train_one_epoch(
            model=model,
            loader=[batch],
            criterion=criterion,
            optimizer=optimizer,
            device=torch.device("cpu"),
            epoch=0,
        )

        after = model.head.weight.detach()
        self.assertIn("train_loss", metrics)
        self.assertGreaterEqual(metrics["train_loss"], 0.0)
        self.assertFalse(torch.allclose(before, after))

    def test_epoch_loss_aggregation_matches_global_weighted_objective_and_batching(self):
        criterion = BinaryMaskIoUCriterion()
        batched_loader_a, logits, binary_mask, sample_weight = self.make_aggregation_batches([2, 1])
        batched_loader_b, _, _, _ = self.make_aggregation_batches([1, 2])
        model_a = IndexedLogitModel(logits)
        model_b = IndexedLogitModel(logits)
        optimizer_a = torch.optim.AdamW(model_a.parameters(), lr=0.0)
        optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=0.0)

        per_sample_losses = []
        for sample_index in range(logits.shape[0]):
            single_loss = weighted_soft_iou(
                logits[sample_index : sample_index + 1].sigmoid().amax(dim=1, keepdim=True),
                binary_mask[sample_index : sample_index + 1],
                torch.tensor([1.0], dtype=torch.float32),
            )
            per_sample_losses.append(float(single_loss.item()))
        total_weight = float(sample_weight.sum().item())
        expected_global = sum(
            loss * float(weight)
            for loss, weight in zip(per_sample_losses, sample_weight.tolist())
        ) / total_weight

        train_metrics_a = train_one_epoch(
            model=model_a,
            loader=batched_loader_a,
            criterion=criterion,
            optimizer=optimizer_a,
            device=torch.device("cpu"),
            epoch=0,
        )
        train_metrics_b = train_one_epoch(
            model=model_b,
            loader=batched_loader_b,
            criterion=criterion,
            optimizer=optimizer_b,
            device=torch.device("cpu"),
            epoch=0,
        )
        eval_metrics_a = evaluate_loader(
            model=model_a,
            loader=batched_loader_a,
            criterion=criterion,
            device=torch.device("cpu"),
            binary_threshold=0.5,
        )
        eval_metrics_b = evaluate_loader(
            model=model_b,
            loader=batched_loader_b,
            criterion=criterion,
            device=torch.device("cpu"),
            binary_threshold=0.5,
        )

        self.assertAlmostEqual(train_metrics_a["train_loss"], expected_global, places=6)
        self.assertAlmostEqual(train_metrics_b["train_loss"], expected_global, places=6)
        self.assertAlmostEqual(eval_metrics_a["val_loss"], expected_global, places=6)
        self.assertAlmostEqual(eval_metrics_b["val_loss"], expected_global, places=6)
        self.assertAlmostEqual(train_metrics_a["train_loss"], train_metrics_b["train_loss"], places=12)
        self.assertAlmostEqual(eval_metrics_a["val_loss"], eval_metrics_b["val_loss"], places=12)

    def test_evaluate_loader_reports_exact_metrics_and_score(self):
        criterion = BinaryMaskIoUCriterion()
        logits = self.make_logits()
        batch = self.make_batch()

        class ConstantModel(nn.Module):
            def forward(self, images):
                del images
                return {
                    "mask_logits": logits,
                    "objectness_logits": torch.full((2, 2), -10.0),
                    "aux_logits": torch.full((2, 1, 2, 2), 10.0),
                }

        metrics = evaluate_loader(
            model=ConstantModel(),
            loader=[batch],
            criterion=criterion,
            device=torch.device("cpu"),
            binary_threshold=0.7,
        )

        union_probability = logits.sigmoid().amax(dim=1, keepdim=True)
        resized_target = F.interpolate(batch["binary_mask"], size=(2, 2), mode="nearest")
        prediction = union_probability >= 0.7
        target = resized_target >= 0.5
        intersection = torch.logical_and(prediction, target).sum().item()
        pred_total = prediction.sum().item()
        target_total = target.sum().item()
        union = torch.logical_or(prediction, target).sum().item()
        total_pixels = prediction.numel()
        expected_loss = weighted_soft_iou(
            union_probability,
            batch["binary_mask"],
            batch["sample_weight"],
        ).item()
        expected_precision = intersection / pred_total
        expected_recall = intersection / target_total
        expected_f1 = 2.0 * expected_precision * expected_recall / (
            expected_precision + expected_recall
        )

        self.assertEqual(
            set(metrics),
            {
                "val_loss",
                "binary_iou",
                "binary_precision",
                "binary_recall",
                "binary_f1",
                "pred_fg_ratio",
                "gt_fg_ratio",
                "score",
            },
        )
        self.assertAlmostEqual(metrics["val_loss"], expected_loss, places=6)
        self.assertAlmostEqual(metrics["binary_iou"], intersection / union, places=6)
        self.assertAlmostEqual(metrics["binary_precision"], expected_precision, places=6)
        self.assertAlmostEqual(metrics["binary_recall"], expected_recall, places=6)
        self.assertAlmostEqual(metrics["binary_f1"], expected_f1, places=6)
        self.assertAlmostEqual(metrics["pred_fg_ratio"], pred_total / total_pixels, places=6)
        self.assertAlmostEqual(metrics["gt_fg_ratio"], target_total / total_pixels, places=6)
        self.assertAlmostEqual(metrics["score"], metrics["binary_iou"], places=6)

    def test_get_autocast_context_returns_noop_context_on_cpu(self):
        context_manager = get_autocast_context(torch.device("cpu"))
        self.assertIsInstance(context_manager, contextlib.nullcontext)
        with context_manager:
            value = torch.tensor([1.0]) + 1.0
        self.assertTrue(torch.equal(value, torch.tensor([2.0])))

    def test_save_checkpoint_writes_exact_top_level_keys(self):
        model = FakeForegroundModel()
        optimizer = create_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
        scheduler = make_scheduler(optimizer, epochs=6, warmup_epochs=5)
        metrics = {
            "val_loss": 0.1,
            "binary_iou": 0.2,
            "binary_precision": 0.3,
            "binary_recall": 0.4,
            "binary_f1": 0.5,
            "pred_fg_ratio": 0.6,
            "gt_fg_ratio": 0.7,
            "score": 0.2,
        }
        config = {"training": {"learning_rate": 1e-3}}

        with tempfile.TemporaryDirectory() as tempdir:
            checkpoint_path = Path(tempdir) / "checkpoint.pt"
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=3,
                config=config,
                best_score=0.2,
                metrics=metrics,
            )

            payload = torch.load(checkpoint_path, map_location="cpu")

        self.assertEqual(
            list(payload.keys()),
            ["model", "optimizer", "scheduler", "epoch", "config", "best_score", "metrics"],
        )
        self.assertEqual(payload["epoch"], 3)
        self.assertEqual(payload["config"], config)
        self.assertEqual(payload["best_score"], 0.2)
        self.assertEqual(payload["metrics"], metrics)


class ForegroundTrainingCliTest(unittest.TestCase):
    maxDiff = None

    def load_selected_config(self):
        config_path = ROOT / "training" / "foreground" / "configs" / "paper_q12_d3_h256.yaml"
        with config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def write_config(self, root: Path, config):
        config_path = root / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return config_path

    def make_record(self, root: Path, name: str, source: str):
        image_path = root / (name + ".png")
        annotation_path = root / (name + "_annotation.json")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"")
        annotation_path.write_text("{}", encoding="utf-8")
        return ChartRecord(image_path=image_path, annotation_path=annotation_path, source=source)

    def make_record_list(self, root: Path, prefix: str, count: int, source: str):
        return [
            self.make_record(root, "{0}_{1}".format(prefix, index), source)
            for index in range(count)
        ]

    def make_metric_row(self, epoch: int, best_score: float, real_score: Optional[float] = None):
        score = best_score if real_score is None else real_score
        return {
            "epoch": epoch,
            "train_loss": 0.9,
            "synthetic_val_loss": 0.6,
            "synthetic_binary_iou": 0.2,
            "synthetic_binary_precision": 0.2,
            "synthetic_binary_recall": 0.2,
            "synthetic_binary_f1": 0.2,
            "synthetic_pred_fg_ratio": 0.2,
            "synthetic_gt_fg_ratio": 0.2,
            "synthetic_score": 0.2,
            "real_val_loss": 1.0 - score,
            "real_binary_iou": score,
            "real_binary_precision": score,
            "real_binary_recall": score,
            "real_binary_f1": score,
            "real_pred_fg_ratio": 0.1,
            "real_gt_fg_ratio": 0.1,
            "real_score": score,
            "best_score": best_score,
        }

    def make_checkpoint_metrics(self, history_rows, best_epoch: int, seed: int = 42):
        from training.foreground import train as train_module

        generator = torch.Generator()
        generator.manual_seed(seed)
        return train_module._build_checkpoint_metrics(
            history=history_rows,
            current_row=history_rows[-1],
            best_epoch=best_epoch,
            rng_state=train_module._capture_rng_state(generator),
        )

    def test_load_training_config_rejects_unknown_nested_keys_and_embedded_paths(self):
        from training.foreground.train import load_training_config

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)

            unknown_top = self.load_selected_config()
            unknown_top["unexpected"] = 1
            unknown_top_path = self.write_config(root / "unknown_top", unknown_top)
            with self.assertRaisesRegex(ValueError, "Unknown config keys"):
                load_training_config(unknown_top_path)

            unknown_nested = self.load_selected_config()
            unknown_nested["training"]["momentum"] = 0.9
            unknown_nested_path = self.write_config(root / "unknown_nested", unknown_nested)
            with self.assertRaisesRegex(ValueError, "Unknown config keys"):
                load_training_config(unknown_nested_path)

            embedded_path = self.load_selected_config()
            embedded_path["selected_experiment"] = "/private/data/run42"
            embedded_path_path = self.write_config(root / "embedded_path", embedded_path)
            with self.assertRaisesRegex(ValueError, "embedded absolute path"):
                load_training_config(embedded_path_path)

    def test_load_training_config_rejects_invalid_shape_splits_and_hyperparameters(self):
        from training.foreground.train import load_training_config

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)

            invalid_shape = self.load_selected_config()
            invalid_shape["model"]["hidden_dim"] = 128
            invalid_shape_path = self.write_config(root / "invalid_shape", invalid_shape)
            with self.assertRaisesRegex(ValueError, "Selected model shape"):
                load_training_config(invalid_shape_path)

            invalid_split = self.load_selected_config()
            invalid_split["splits"]["synthetic_val_ratio"] = 1.0
            invalid_split_path = self.write_config(root / "invalid_split", invalid_split)
            with self.assertRaisesRegex(ValueError, "synthetic_val_ratio"):
                load_training_config(invalid_split_path)

            invalid_hparams = self.load_selected_config()
            invalid_hparams["training"]["learning_rate"] = 0.0
            invalid_hparams_path = self.write_config(root / "invalid_hparams", invalid_hparams)
            with self.assertRaisesRegex(ValueError, "learning_rate"):
                load_training_config(invalid_hparams_path)

            drifted_selected_experiment = self.load_selected_config()
            drifted_selected_experiment["selected_experiment"] = "v11_other"
            drifted_selected_experiment_path = self.write_config(
                root / "drifted_selected_experiment",
                drifted_selected_experiment,
            )
            with self.assertRaisesRegex(ValueError, "selected_experiment"):
                load_training_config(drifted_selected_experiment_path)

            drifted_epochs = self.load_selected_config()
            drifted_epochs["training"]["epochs"] = 101
            drifted_epochs_path = self.write_config(root / "drifted_epochs", drifted_epochs)
            with self.assertRaisesRegex(ValueError, "training.epochs"):
                load_training_config(drifted_epochs_path)

            drifted_seed = self.load_selected_config()
            drifted_seed["training"]["seed"] = 43
            drifted_seed_path = self.write_config(root / "drifted_seed", drifted_seed)
            with self.assertRaisesRegex(ValueError, "training.seed"):
                load_training_config(drifted_seed_path)

            drifted_repeat = self.load_selected_config()
            drifted_repeat["training"]["real_repeat"] = 63
            drifted_repeat_path = self.write_config(root / "drifted_repeat", drifted_repeat)
            with self.assertRaisesRegex(ValueError, "training.real_repeat"):
                load_training_config(drifted_repeat_path)

            drifted_threshold = self.load_selected_config()
            drifted_threshold["training"]["binary_threshold"] = 0.4
            drifted_threshold_path = self.write_config(root / "drifted_threshold", drifted_threshold)
            with self.assertRaisesRegex(ValueError, "training.binary_threshold"):
                load_training_config(drifted_threshold_path)

            drifted_axis_conf = self.load_selected_config()
            drifted_axis_conf["preprocessing"]["yolo_axis_confidence"] = 0.4
            drifted_axis_conf_path = self.write_config(root / "drifted_axis_conf", drifted_axis_conf)
            with self.assertRaisesRegex(ValueError, "preprocessing.yolo_axis_confidence"):
                load_training_config(drifted_axis_conf_path)

            drifted_legend_conf = self.load_selected_config()
            drifted_legend_conf["preprocessing"]["yolo_legend_confidence"] = 0.2
            drifted_legend_conf_path = self.write_config(root / "drifted_legend_conf", drifted_legend_conf)
            with self.assertRaisesRegex(ValueError, "preprocessing.yolo_legend_confidence"):
                load_training_config(drifted_legend_conf_path)

            fractional_test_size = self.load_selected_config()
            fractional_test_size["splits"]["real_test_size"] = 40.9
            fractional_test_size_path = self.write_config(root / "fractional_test_size", fractional_test_size)
            with self.assertRaisesRegex(ValueError, "splits.real_test_size"):
                load_training_config(fractional_test_size_path)

            fractional_val_size = self.load_selected_config()
            fractional_val_size["splits"]["real_val_size"] = 20.1
            fractional_val_size_path = self.write_config(root / "fractional_val_size", fractional_val_size)
            with self.assertRaisesRegex(ValueError, "splits.real_val_size"):
                load_training_config(fractional_val_size_path)

    def test_main_requires_axis_and_legend_models_when_yolo_masking_is_enabled(self):
        from training.foreground.train import main

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            config_path = self.write_config(root, self.load_selected_config())
            synthetic_dir = root / "synthetic"
            real_dir = root / "real"
            output_dir = root / "output"
            synthetic_dir.mkdir()
            real_dir.mkdir()

            argv = [
                "training.foreground.train",
                "--config",
                str(config_path),
                "--synthetic-dir",
                str(synthetic_dir),
                "--real-dir",
                str(real_dir),
                "--output-dir",
                str(output_dir),
                "--dry-run",
            ]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit) as excinfo:
                    main()

        self.assertEqual(excinfo.exception.code, 2)

    def test_main_dry_run_validates_inputs_without_constructing_runtime_or_writing(self):
        from training.foreground.train import main

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            config_path = self.write_config(root, self.load_selected_config())
            synthetic_dir = root / "synthetic"
            real_dir = root / "real"
            output_dir = root / "output"
            axis_model = root / "axis.pt"
            legend_model = root / "legend.pt"
            synthetic_dir.mkdir()
            real_dir.mkdir()
            axis_model.write_bytes(b"axis")
            legend_model.write_bytes(b"legend")

            synthetic_records = self.make_record_list(root / "fixtures", "synthetic", 5, "synthetic")
            real_records = self.make_record_list(root / "fixtures", "real", 3, "real")
            synthetic_train = synthetic_records[1:]
            synthetic_val = synthetic_records[:1]
            real_train = real_records[2:]
            real_val = real_records[1:2]
            real_test = real_records[:1]
            train_records = synthetic_train + real_train * 64

            stdout = tempfile.SpooledTemporaryFile(mode="w+")
            argv = [
                "training.foreground.train",
                "--config",
                str(config_path),
                "--synthetic-dir",
                str(synthetic_dir),
                "--real-dir",
                str(real_dir),
                "--output-dir",
                str(output_dir),
                "--axis-model",
                str(axis_model),
                "--legend-model",
                str(legend_model),
                "--dry-run",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch(
                "training.foreground.train.collect_pairs",
                side_effect=[synthetic_records, real_records],
            ) as collect_pairs_mock, mock.patch(
                "training.foreground.train.split_synthetic_pairs",
                return_value=(synthetic_train, synthetic_val),
            ) as split_synthetic_mock, mock.patch(
                "training.foreground.train.split_real_pairs",
                return_value=(real_train, real_val, real_test),
            ) as split_real_mock, mock.patch(
                "training.foreground.train.build_train_records",
                return_value=train_records,
            ) as build_train_records_mock, mock.patch(
                "training.foreground.train.build_region_map"
            ) as build_region_map_mock, mock.patch(
                "training.foreground.train.DataLoader"
            ) as dataloader_mock, mock.patch(
                "training.foreground.train.run_training"
            ) as run_training_mock, contextlib.redirect_stdout(stdout):
                main()

            stdout.seek(0)
            self.assertEqual(stdout.read(), "foreground_training_dry_run_ok\n")
            self.assertFalse(output_dir.exists())
            collect_pairs_mock.assert_has_calls(
                [
                    mock.call(synthetic_dir, source="synthetic"),
                    mock.call(real_dir, source="real"),
                ]
            )
            split_synthetic_mock.assert_called_once_with(
                synthetic_records,
                val_ratio=0.05,
                seed=42,
            )
            split_real_mock.assert_called_once_with(
                real_records,
                n_test=40,
                n_val=20,
                seed=42,
            )
            build_train_records_mock.assert_called_once_with(
                real_train=real_train,
                synthetic_train=synthetic_train,
                real_repeat=64,
                seed=42,
            )
            build_region_map_mock.assert_not_called()
            dataloader_mock.assert_not_called()
            run_training_mock.assert_not_called()

    def test_run_training_uses_selected_splits_outputs_and_best_checkpoint_policy(self):
        from training.foreground.train import TrainingPaths, run_training

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            config = self.load_selected_config()
            synthetic_dir = root / "synthetic"
            real_dir = root / "real"
            output_dir = root / "output"
            axis_model = root / "axis.pt"
            legend_model = root / "legend.pt"
            synthetic_dir.mkdir()
            real_dir.mkdir()
            axis_model.write_bytes(b"axis")
            legend_model.write_bytes(b"legend")
            paths = TrainingPaths(
                synthetic_dir=synthetic_dir,
                real_dir=real_dir,
                output_dir=output_dir,
                axis_model=axis_model,
                legend_model=legend_model,
            )

            synthetic_records = self.make_record_list(root / "fixtures", "synthetic", 5, "synthetic")
            real_records = self.make_record_list(root / "fixtures", "real", 4, "real")
            synthetic_train = synthetic_records[1:]
            synthetic_val = synthetic_records[:1]
            real_train = real_records[2:]
            real_val = real_records[1:2]
            real_test = real_records[:1]
            train_records = synthetic_train + real_train * 64
            region_map = {str(real_records[0].image_path.resolve()): {"axis_box": None, "legend_boxes": []}}
            dataset_calls = []
            dataloader_calls = []
            checkpoint_paths = []
            saved_checkpoint_payloads = {}

            class FakeDataset:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs
                    self.records = kwargs["records"]
                    labels = ["train", "synthetic_val", "real_val", "real_test"]
                    self.label = labels[len(dataset_calls)]
                    dataset_calls.append(self)

            def fake_dataloader(dataset, batch_size, shuffle, num_workers, generator=None, worker_init_fn=None):
                loader = types.SimpleNamespace(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    generator=generator,
                    worker_init_fn=worker_init_fn,
                )
                dataloader_calls.append(loader)
                return loader

            class FakeStatefulModel:
                def __init__(self):
                    self.state_tag = "init"
                    self.loaded_states = []

                def to(self, device):
                    del device
                    return self

                def state_dict(self):
                    return {"module.tag": self.state_tag}

                def load_state_dict(self, state_dict):
                    if set(state_dict) != {"tag"}:
                        raise AssertionError("unexpected model state keys: {0}".format(sorted(state_dict)))
                    self.loaded_states.append(dict(state_dict))
                    self.state_tag = state_dict["tag"]

            fake_model = FakeStatefulModel()

            def fake_save_checkpoint(path, **kwargs):
                checkpoint_path = Path(path)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "model": dict(kwargs["model"].state_dict()),
                    "optimizer": kwargs["optimizer"].state_dict(),
                    "scheduler": kwargs["scheduler"].state_dict(),
                    "epoch": kwargs["epoch"],
                    "config": kwargs["config"],
                    "best_score": kwargs["best_score"],
                    "metrics": kwargs["metrics"],
                }
                torch.save(payload, checkpoint_path)
                saved_checkpoint_payloads[checkpoint_path.name] = payload
                checkpoint_paths.append(checkpoint_path)

            val_scores = [0.2] + [0.3] + [0.1] * 98
            val_index = {"value": 0}
            synthetic_eval_count = {"value": 0}

            def fake_evaluate_loader(model, loader, criterion, device, binary_threshold):
                del model, criterion, device
                self.assertEqual(binary_threshold, 0.5)
                if loader.dataset.label == "synthetic_val":
                    synthetic_eval_count["value"] += 1
                    return {
                        "val_loss": 0.6,
                        "binary_iou": 0.15,
                        "binary_precision": 0.2,
                        "binary_recall": 0.3,
                        "binary_f1": 0.24,
                        "pred_fg_ratio": 0.1,
                        "gt_fg_ratio": 0.12,
                        "score": 0.15,
                    }
                if loader.dataset.label == "real_val":
                    score = val_scores[val_index["value"]]
                    val_index["value"] += 1
                    return {
                        "val_loss": 1.0 - score,
                        "binary_iou": score,
                        "binary_precision": 0.8,
                        "binary_recall": 0.7,
                        "binary_f1": 0.75,
                        "pred_fg_ratio": 0.2,
                        "gt_fg_ratio": 0.25,
                        "score": score,
                    }
                self.assertEqual(loader.dataset.label, "real_test")

            def fake_test_evaluate_loader(model, loader, criterion, device, binary_threshold):
                del criterion, device, binary_threshold
                if loader.dataset.label == "synthetic_val":
                    return fake_evaluate_loader(model, loader, None, None, 0.5)
                if loader.dataset.label == "real_val":
                    return fake_evaluate_loader(model, loader, None, None, 0.5)
                self.assertEqual(model.state_tag, "epoch_1")
                return {
                    "val_loss": 0.11,
                    "binary_iou": 0.91,
                    "binary_precision": 0.92,
                    "binary_recall": 0.93,
                    "binary_f1": 0.925,
                    "pred_fg_ratio": 0.17,
                    "gt_fg_ratio": 0.19,
                    "score": 0.91,
                }

            evaluate_loader_mock = mock.Mock(side_effect=fake_test_evaluate_loader)
            build_paper_model_mock = mock.Mock(return_value=fake_model)
            normalize_state_dict_keys_mock = mock.Mock(
                side_effect=lambda state: {key.replace("module.", "", 1): value for key, value in state.items()}
            )

            def fake_train_one_epoch(model, loader, criterion, optimizer, device, epoch):
                del loader, criterion, optimizer, device
                model.state_tag = "epoch_{0}".format(epoch)
                return {"train_loss": 0.9 - epoch * 0.001}

            with mock.patch(
                "training.foreground.train.collect_pairs",
                side_effect=[synthetic_records, real_records],
            ) as collect_pairs_mock, mock.patch(
                "training.foreground.train.split_synthetic_pairs",
                return_value=(synthetic_train, synthetic_val),
            ) as split_synthetic_mock, mock.patch(
                "training.foreground.train.split_real_pairs",
                return_value=(real_train, real_val, real_test),
            ) as split_real_mock, mock.patch(
                "training.foreground.train.build_train_records",
                return_value=train_records,
            ) as build_train_records_mock, mock.patch(
                "training.foreground.train.build_region_map",
                return_value=region_map,
            ) as build_region_map_mock, mock.patch(
                "training.foreground.train.UnifiedChartDataset",
                side_effect=FakeDataset,
            ), mock.patch(
                "training.foreground.train.DataLoader",
                side_effect=fake_dataloader,
            ), mock.patch(
                "training.foreground.train._load_model_helpers",
                return_value=(build_paper_model_mock, normalize_state_dict_keys_mock),
            ), mock.patch(
                "training.foreground.train.BinaryMaskIoUCriterion",
                return_value=mock.Mock(name="criterion"),
            ) as criterion_cls_mock, mock.patch(
                "training.foreground.train.create_optimizer",
                return_value=types.SimpleNamespace(state_dict=lambda: {"optimizer": "state"}),
            ) as create_optimizer_mock, mock.patch(
                "training.foreground.train.make_scheduler",
                return_value=types.SimpleNamespace(
                    state_dict=lambda: {"scheduler": "state"},
                    step=lambda: None,
                ),
            ) as make_scheduler_mock, mock.patch(
                "training.foreground.train.train_one_epoch",
                side_effect=fake_train_one_epoch,
            ) as train_one_epoch_mock, mock.patch(
                "training.foreground.train.torch.load",
                side_effect=torch.load,
            ), mock.patch(
                "training.foreground.train.evaluate_loader",
                evaluate_loader_mock,
            ), mock.patch(
                "training.foreground.train.save_checkpoint",
                side_effect=fake_save_checkpoint,
            ) as save_checkpoint_mock:
                result = run_training(config=config, paths=paths, device="cpu", resume=None)

            self.assertEqual(result["best_score"], 0.3)
            collect_pairs_mock.assert_has_calls(
                [
                    mock.call(synthetic_dir, source="synthetic"),
                    mock.call(real_dir, source="real"),
                ]
            )
            split_synthetic_mock.assert_called_once_with(synthetic_records, val_ratio=0.05, seed=42)
            split_real_mock.assert_called_once_with(real_records, n_test=40, n_val=20, seed=42)
            build_train_records_mock.assert_called_once_with(
                real_train=real_train,
                synthetic_train=synthetic_train,
                real_repeat=64,
                seed=42,
            )
            build_region_map_mock.assert_called_once_with(
                real_records,
                axis_weights=axis_model,
                legend_weights=legend_model,
                axis_confidence=0.5,
                legend_confidence=0.3,
            )
            criterion_cls_mock.assert_called_once_with()
            create_optimizer_mock.assert_called_once()
            make_scheduler_mock.assert_called_once()
            self.assertEqual(train_one_epoch_mock.call_count, 100)
            self.assertEqual(train_one_epoch_mock.call_args_list[0].kwargs["epoch"], 0)
            self.assertEqual(train_one_epoch_mock.call_args_list[-1].kwargs["epoch"], 99)
            self.assertEqual(evaluate_loader_mock.call_count, 201)
            self.assertEqual(synthetic_eval_count["value"], 100)
            self.assertEqual(len(dataset_calls), 4)
            self.assertTrue(dataset_calls[0].kwargs["is_train"])
            self.assertEqual(dataset_calls[0].kwargs["records"], train_records)
            self.assertEqual(dataset_calls[1].kwargs["records"], synthetic_val)
            self.assertEqual(dataset_calls[2].kwargs["records"], real_val)
            self.assertEqual(dataset_calls[2].kwargs["region_map"], region_map)
            self.assertEqual(dataset_calls[3].kwargs["records"], real_test)
            self.assertEqual([loader.batch_size for loader in dataloader_calls], [2, 2, 2, 2])
            self.assertEqual([loader.shuffle for loader in dataloader_calls], [True, False, False, False])
            self.assertEqual([loader.num_workers for loader in dataloader_calls], [8, 8, 8, 8])
            self.assertIsNotNone(dataloader_calls[0].generator)
            self.assertIsNotNone(dataloader_calls[0].worker_init_fn)
            self.assertIsNone(dataloader_calls[1].generator)
            self.assertTrue(all(path.parent == output_dir for path in checkpoint_paths))
            self.assertTrue(all(path.name.startswith(".last.pt.") and path.name.endswith(".tmp") for path in checkpoint_paths))
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {"best.pt", "last.pt", "metrics.csv", "final_test_metrics.json"},
            )
            self.assertTrue(save_checkpoint_mock.call_args_list[-1].args[0].name.startswith(".last.pt."))
            self.assertTrue((output_dir / "best.pt").is_file())
            self.assertEqual(normalize_state_dict_keys_mock.call_count, 1)
            self.assertEqual(fake_model.loaded_states[-1], {"tag": "epoch_1"})
            best_payload = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)
            last_payload = torch.load(output_dir / "last.pt", map_location="cpu", weights_only=True)
            self.assertEqual(best_payload["model"]["module.tag"], "epoch_1")
            self.assertEqual(last_payload["model"]["module.tag"], "epoch_99")

            with (output_dir / "metrics.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 100)
            self.assertEqual(
                set(rows[0]),
                {
                    "epoch",
                    "train_loss",
                    "synthetic_val_loss",
                    "synthetic_binary_iou",
                    "synthetic_binary_precision",
                    "synthetic_binary_recall",
                    "synthetic_binary_f1",
                    "synthetic_pred_fg_ratio",
                    "synthetic_gt_fg_ratio",
                    "synthetic_score",
                    "real_val_loss",
                    "real_binary_iou",
                    "real_binary_precision",
                    "real_binary_recall",
                    "real_binary_f1",
                    "real_pred_fg_ratio",
                    "real_gt_fg_ratio",
                    "real_score",
                    "best_score",
                },
            )
            self.assertEqual(rows[0]["epoch"], "0")
            self.assertEqual(rows[-1]["epoch"], "99")
            self.assertEqual(rows[0]["synthetic_binary_iou"], "0.15")
            self.assertEqual(rows[1]["best_score"], "0.3")

            final_metrics = json.loads((output_dir / "final_test_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(final_metrics["binary_iou"], 0.91)
            self.assertEqual(final_metrics["score"], 0.91)

    def test_run_training_resume_requires_last_checkpoint_and_restores_full_state(self):
        from training.foreground.train import TrainingPaths, run_training

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            config = self.load_selected_config()
            synthetic_dir = root / "synthetic"
            real_dir = root / "real"
            output_dir = root / "output"
            axis_model = root / "axis.pt"
            legend_model = root / "legend.pt"
            synthetic_dir.mkdir()
            real_dir.mkdir()
            output_dir.mkdir()
            axis_model.write_bytes(b"axis")
            legend_model.write_bytes(b"legend")
            resume_path = output_dir / "last.pt"
            wrong_resume_path = output_dir / "best.pt"
            wrong_resume_path.write_bytes(b"x")
            best_path = output_dir / "best.pt"
            metrics_path = output_dir / "metrics.csv"
            paths = TrainingPaths(
                synthetic_dir=synthetic_dir,
                real_dir=real_dir,
                output_dir=output_dir,
                axis_model=axis_model,
                legend_model=legend_model,
            )

            synthetic_records = self.make_record_list(root / "fixtures", "synthetic", 5, "synthetic")
            real_records = self.make_record_list(root / "fixtures", "real", 4, "real")
            synthetic_train = synthetic_records[1:]
            synthetic_val = synthetic_records[:1]
            real_train = real_records[2:]
            real_val = real_records[1:2]
            real_test = real_records[:1]
            train_records = synthetic_train + real_train * 64

            class FakeStatefulModel:
                def __init__(self):
                    self.state_tag = "init"
                    self.loaded_states = []

                def to(self, device):
                    del device
                    return self

                def state_dict(self):
                    return {"module.tag": self.state_tag}

                def load_state_dict(self, state_dict):
                    if set(state_dict) != {"tag"}:
                        raise AssertionError("unexpected model state keys: {0}".format(sorted(state_dict)))
                    self.loaded_states.append(dict(state_dict))
                    self.state_tag = state_dict["tag"]

            fake_model = FakeStatefulModel()
            fake_optimizer = mock.Mock()
            fake_scheduler = mock.Mock()
            fake_optimizer.state_dict.return_value = {"optimizer": "state"}
            fake_scheduler.state_dict.return_value = {"scheduler": "state"}
            history_rows = [self.make_metric_row(epoch, 0.77, 0.77) for epoch in range(96)]
            checkpoint = {
                "model": {"module.tag": "resume_state"},
                "optimizer": {"optimizer_state": 1},
                "scheduler": {"scheduler_state": 2},
                "epoch": 95,
                "config": copy.deepcopy(config),
                "best_score": 0.77,
                "metrics": self.make_checkpoint_metrics(history_rows, best_epoch=95),
            }
            torch.save(checkpoint, resume_path)
            best_checkpoint = {
                "model": {"module.tag": "best_pre_resume"},
                "optimizer": {"optimizer_state": 1},
                "scheduler": {"scheduler_state": 2},
                "epoch": 95,
                "config": copy.deepcopy(config),
                "best_score": 0.77,
                "metrics": self.make_checkpoint_metrics(history_rows, best_epoch=95),
            }
            torch.save(best_checkpoint, best_path)
            with metrics_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=list(history_rows[0].keys()),
                )
                writer.writeheader()
                for row in history_rows:
                    writer.writerow(row)

            with self.assertRaisesRegex(ValueError, "last.pt"):
                run_training(config=config, paths=paths, device="cpu", resume=wrong_resume_path)

            resume_real_scores = iter([0.8, 0.81, 0.82, 0.83])

            def fake_resume_evaluate_loader(model, loader, criterion, device, binary_threshold):
                del criterion, device
                self.assertEqual(binary_threshold, 0.5)
                if loader.dataset.label == "synthetic_val":
                    return {
                        "val_loss": 0.6,
                        "binary_iou": 0.2,
                        "binary_precision": 0.2,
                        "binary_recall": 0.2,
                        "binary_f1": 0.2,
                        "pred_fg_ratio": 0.2,
                        "gt_fg_ratio": 0.2,
                        "score": 0.2,
                    }
                if loader.dataset.label == "real_val":
                    score = next(resume_real_scores)
                    return {
                        "val_loss": 1.0 - score,
                        "binary_iou": score,
                        "binary_precision": score,
                        "binary_recall": score,
                        "binary_f1": score,
                        "pred_fg_ratio": 0.1,
                        "gt_fg_ratio": 0.1,
                        "score": score,
                    }
                self.assertEqual(loader.dataset.label, "real_test")
                self.assertEqual(model.state_tag, "epoch_99")
                return {
                    "val_loss": 0.02,
                    "binary_iou": 0.95,
                    "binary_precision": 0.95,
                    "binary_recall": 0.95,
                    "binary_f1": 0.95,
                    "pred_fg_ratio": 0.1,
                    "gt_fg_ratio": 0.1,
                    "score": 0.95,
                }

            evaluate_loader_mock = mock.Mock(side_effect=fake_resume_evaluate_loader)

            def fake_train_one_epoch(model, loader, criterion, optimizer, device, epoch):
                del loader, criterion, optimizer, device
                model.state_tag = "epoch_{0}".format(epoch)
                return {"train_loss": 0.9}

            def fake_save_checkpoint(path, **kwargs):
                checkpoint_path = Path(path)
                payload = {
                    "model": {"module.tag": kwargs["model"].state_tag},
                    "optimizer": kwargs["optimizer"].state_dict(),
                    "scheduler": kwargs["scheduler"].state_dict(),
                    "epoch": kwargs["epoch"],
                    "config": kwargs["config"],
                    "best_score": kwargs["best_score"],
                    "metrics": kwargs["metrics"],
                }
                torch.save(payload, checkpoint_path)

            with mock.patch(
                "training.foreground.train.collect_pairs",
                side_effect=[synthetic_records, real_records],
            ), mock.patch(
                "training.foreground.train.split_synthetic_pairs",
                return_value=(synthetic_train, synthetic_val),
            ), mock.patch(
                "training.foreground.train.split_real_pairs",
                return_value=(real_train, real_val, real_test),
            ), mock.patch(
                "training.foreground.train.build_train_records",
                return_value=train_records,
            ), mock.patch(
                "training.foreground.train.build_region_map",
                return_value={},
            ), mock.patch(
                "training.foreground.train.UnifiedChartDataset",
                side_effect=lambda **kwargs: types.SimpleNamespace(
                    kwargs=kwargs,
                    label=(
                        "train"
                        if kwargs["is_train"]
                        else "synthetic_val"
                        if kwargs["records"] == synthetic_val
                        else "real_val"
                        if kwargs["records"] == real_val
                        else "real_test"
                    ),
                ),
            ), mock.patch(
                "training.foreground.train.DataLoader",
                side_effect=lambda dataset, batch_size, shuffle, num_workers, generator=None, worker_init_fn=None: types.SimpleNamespace(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    generator=generator,
                    worker_init_fn=worker_init_fn,
                ),
            ), mock.patch(
                "training.foreground.train._load_model_helpers",
                return_value=(
                    mock.Mock(return_value=fake_model),
                    mock.Mock(
                        side_effect=lambda state: {
                            key.replace("module.", "", 1): value for key, value in state.items()
                        }
                    ),
                ),
            ) as load_model_helpers_mock, mock.patch(
                "training.foreground.train.BinaryMaskIoUCriterion",
                return_value=mock.Mock(),
            ), mock.patch(
                "training.foreground.train.create_optimizer",
                return_value=fake_optimizer,
            ), mock.patch(
                "training.foreground.train.make_scheduler",
                return_value=fake_scheduler,
            ), mock.patch(
                "training.foreground.train.train_one_epoch",
                side_effect=fake_train_one_epoch,
            ) as train_one_epoch_mock, mock.patch(
                "training.foreground.train.evaluate_loader",
                evaluate_loader_mock,
            ), mock.patch(
                "training.foreground.train.save_checkpoint",
                side_effect=fake_save_checkpoint,
            ):
                result = run_training(config=config, paths=paths, device="cpu", resume=resume_path)

            self.assertEqual(result["start_epoch"], 96)
            self.assertGreater(result["best_score"], 0.77)
            load_model_helpers_mock.assert_called_once_with()
            self.assertEqual(fake_model.loaded_states[0], {"tag": "resume_state"})
            fake_optimizer.load_state_dict.assert_called_once_with(checkpoint["optimizer"])
            fake_scheduler.load_state_dict.assert_called_once_with(checkpoint["scheduler"])
            self.assertEqual(train_one_epoch_mock.call_count, 4)
            self.assertEqual(train_one_epoch_mock.call_args_list[0].kwargs["epoch"], 96)
            self.assertEqual(train_one_epoch_mock.call_args_list[-1].kwargs["epoch"], 99)
            self.assertEqual(fake_model.loaded_states[-1], {"tag": "epoch_99"})

            with metrics_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["epoch"]) for row in rows], list(range(100)))
            self.assertEqual(rows[0]["best_score"], "0.77")
            self.assertEqual(rows[-1]["best_score"], "0.83")

            final_metrics = json.loads((output_dir / "final_test_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(final_metrics["binary_iou"], 0.95)
            self.assertEqual(final_metrics["score"], 0.95)

            incompatible = copy.deepcopy(checkpoint)
            incompatible["config"]["training"]["batch_size"] = 99
            torch.save(incompatible, resume_path)
            with self.assertRaisesRegex(ValueError, "config mismatch"):
                run_training(config=config, paths=paths, device="cpu", resume=resume_path)

            metrics_path.unlink()
            rerun_real_scores = iter([0.8, 0.81, 0.82, 0.83])

            def fake_resume_after_missing_csv(model, loader, criterion, device, binary_threshold):
                del criterion, device
                self.assertEqual(binary_threshold, 0.5)
                if loader.dataset.label == "synthetic_val":
                    return {
                        "val_loss": 0.6,
                        "binary_iou": 0.2,
                        "binary_precision": 0.2,
                        "binary_recall": 0.2,
                        "binary_f1": 0.2,
                        "pred_fg_ratio": 0.2,
                        "gt_fg_ratio": 0.2,
                        "score": 0.2,
                    }
                if loader.dataset.label == "real_val":
                    score = next(rerun_real_scores)
                    return {
                        "val_loss": 1.0 - score,
                        "binary_iou": score,
                        "binary_precision": score,
                        "binary_recall": score,
                        "binary_f1": score,
                        "pred_fg_ratio": 0.1,
                        "gt_fg_ratio": 0.1,
                        "score": score,
                    }
                return {
                    "val_loss": 0.02,
                    "binary_iou": 0.95,
                    "binary_precision": 0.95,
                    "binary_recall": 0.95,
                    "binary_f1": 0.95,
                    "pred_fg_ratio": 0.1,
                    "gt_fg_ratio": 0.1,
                    "score": 0.95,
                }

            torch.save(checkpoint, resume_path)
            with mock.patch(
                "training.foreground.train.collect_pairs",
                side_effect=[synthetic_records, real_records],
            ), mock.patch(
                "training.foreground.train.split_synthetic_pairs",
                return_value=(synthetic_train, synthetic_val),
            ), mock.patch(
                "training.foreground.train.split_real_pairs",
                return_value=(real_train, real_val, real_test),
            ), mock.patch(
                "training.foreground.train.build_train_records",
                return_value=train_records,
            ), mock.patch(
                "training.foreground.train.build_region_map",
                return_value={},
            ), mock.patch(
                "training.foreground.train.UnifiedChartDataset",
                side_effect=lambda **kwargs: types.SimpleNamespace(
                    kwargs=kwargs,
                    label=(
                        "train"
                        if kwargs["is_train"]
                        else "synthetic_val"
                        if kwargs["records"] == synthetic_val
                        else "real_val"
                        if kwargs["records"] == real_val
                        else "real_test"
                    ),
                ),
            ), mock.patch(
                "training.foreground.train.DataLoader",
                side_effect=lambda dataset, batch_size, shuffle, num_workers, generator=None, worker_init_fn=None: types.SimpleNamespace(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    generator=generator,
                    worker_init_fn=worker_init_fn,
                ),
            ), mock.patch(
                "training.foreground.train._load_model_helpers",
                return_value=(
                    mock.Mock(return_value=FakeStatefulModel()),
                    mock.Mock(
                        side_effect=lambda state: {
                            key.replace("module.", "", 1): value for key, value in state.items()
                        }
                    ),
                ),
            ), mock.patch(
                "training.foreground.train.BinaryMaskIoUCriterion",
                return_value=mock.Mock(),
            ), mock.patch(
                "training.foreground.train.create_optimizer",
                return_value=fake_optimizer,
            ), mock.patch(
                "training.foreground.train.make_scheduler",
                return_value=fake_scheduler,
            ), mock.patch(
                "training.foreground.train.train_one_epoch",
                side_effect=fake_train_one_epoch,
            ), mock.patch(
                "training.foreground.train.evaluate_loader",
                mock.Mock(side_effect=fake_resume_after_missing_csv),
            ), mock.patch(
                "training.foreground.train.save_checkpoint",
                side_effect=fake_save_checkpoint,
            ):
                rerun_result = run_training(config=config, paths=paths, device="cpu", resume=resume_path)

            self.assertEqual(rerun_result["start_epoch"], 96)
            self.assertTrue(metrics_path.is_file())

    def test_run_training_rejects_fresh_collisions_and_symlinked_artifacts(self):
        from training.foreground.train import TrainingPaths, run_training

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            config = self.load_selected_config()
            synthetic_dir = root / "synthetic"
            real_dir = root / "real"
            synthetic_dir.mkdir()
            real_dir.mkdir()
            axis_model = root / "axis.pt"
            legend_model = root / "legend.pt"
            axis_model.write_bytes(b"axis")
            legend_model.write_bytes(b"legend")

            symlink_target = root / "real_output"
            symlink_target.mkdir()
            symlink_output = root / "output_symlink"
            symlink_output.symlink_to(symlink_target, target_is_directory=True)
            symlink_paths = TrainingPaths(
                synthetic_dir=synthetic_dir,
                real_dir=real_dir,
                output_dir=symlink_output,
                axis_model=axis_model,
                legend_model=legend_model,
            )
            with self.assertRaisesRegex(ValueError, "output_dir"):
                run_training(config=config, paths=symlink_paths, device="cpu", resume=None)

            collision_output = root / "output_collision"
            collision_output.mkdir()
            (collision_output / "last.pt").write_bytes(b"x")
            collision_paths = TrainingPaths(
                synthetic_dir=synthetic_dir,
                real_dir=real_dir,
                output_dir=collision_output,
                axis_model=axis_model,
                legend_model=legend_model,
            )
            with self.assertRaisesRegex(ValueError, "Fresh runs require an empty reserved artifact path"):
                run_training(config=config, paths=collision_paths, device="cpu", resume=None)

            (collision_output / "last.pt").unlink()
            outside = root / "outside.pt"
            outside.write_bytes(b"x")
            (collision_output / "best.pt").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "best.pt must not be a symlink"):
                run_training(config=config, paths=collision_paths, device="cpu", resume=None)

    def test_resume_rejects_untrusted_and_malformed_checkpoint_payloads(self):
        from training.foreground.train import TrainingPaths, run_training

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            config = self.load_selected_config()
            synthetic_dir = root / "synthetic"
            real_dir = root / "real"
            output_dir = root / "output"
            synthetic_dir.mkdir()
            real_dir.mkdir()
            output_dir.mkdir()
            axis_model = root / "axis.pt"
            legend_model = root / "legend.pt"
            axis_model.write_bytes(b"axis")
            legend_model.write_bytes(b"legend")
            paths = TrainingPaths(
                synthetic_dir=synthetic_dir,
                real_dir=real_dir,
                output_dir=output_dir,
                axis_model=axis_model,
                legend_model=legend_model,
            )
            resume_path = output_dir / "last.pt"

            malformed_payload = {
                "model": UntrustedCheckpointObject(),
                "optimizer": {"optimizer_state": 1},
                "scheduler": {"scheduler_state": 2},
                "epoch": 0,
                "config": copy.deepcopy(config),
                "best_score": 0.1,
                "metrics": {},
            }
            torch.save(malformed_payload, resume_path)
            with self.assertRaisesRegex(ValueError, "loaded safely"):
                run_training(config=config, paths=paths, device="cpu", resume=resume_path)

            wrong_schema_payload = {
                "bad": 1,
            }
            torch.save(wrong_schema_payload, resume_path)
            with self.assertRaisesRegex(ValueError, "Checkpoint keys"):
                run_training(config=config, paths=paths, device="cpu", resume=resume_path)

    def test_resume_after_last_checkpoint_crash_repairs_best_and_regenerates_history(self):
        from training.foreground.train import TrainingPaths, run_training

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            config = self.load_selected_config()
            synthetic_dir = root / "synthetic"
            real_dir = root / "real"
            output_dir = root / "output"
            axis_model = root / "axis.pt"
            legend_model = root / "legend.pt"
            synthetic_dir.mkdir()
            real_dir.mkdir()
            axis_model.write_bytes(b"axis")
            legend_model.write_bytes(b"legend")
            paths = TrainingPaths(
                synthetic_dir=synthetic_dir,
                real_dir=real_dir,
                output_dir=output_dir,
                axis_model=axis_model,
                legend_model=legend_model,
            )

            synthetic_records = self.make_record_list(root / "fixtures", "synthetic", 5, "synthetic")
            real_records = self.make_record_list(root / "fixtures", "real", 4, "real")
            synthetic_train = synthetic_records[1:]
            synthetic_val = synthetic_records[:1]
            real_train = real_records[2:]
            real_val = real_records[1:2]
            real_test = real_records[:1]
            train_records = synthetic_train + real_train * 64

            class FakeStatefulModel:
                def __init__(self):
                    self.state_tag = "init"

                def to(self, device):
                    del device
                    return self

                def state_dict(self):
                    return {"module.tag": self.state_tag}

                def load_state_dict(self, state_dict):
                    self.state_tag = state_dict["tag"]

            def dataset_factory(**kwargs):
                return types.SimpleNamespace(
                    kwargs=kwargs,
                    label=(
                        "train"
                        if kwargs["is_train"]
                        else "synthetic_val"
                        if kwargs["records"] == synthetic_val
                        else "real_val"
                        if kwargs["records"] == real_val
                        else "real_test"
                    ),
                )

            def dataloader_factory(dataset, batch_size, shuffle, num_workers, generator=None, worker_init_fn=None):
                return types.SimpleNamespace(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    generator=generator,
                    worker_init_fn=worker_init_fn,
                )

            def fake_train_one_epoch(model, loader, criterion, optimizer, device, epoch):
                del loader, criterion, optimizer, device
                model.state_tag = "epoch_{0}".format(epoch)
                return {"train_loss": 0.9 - epoch * 0.001}

            phase = {"run": 0}
            initial_real_scores = iter([0.4, 0.9])
            resumed_real_scores = iter([0.3] * 98)

            def fake_evaluate_loader(model, loader, criterion, device, binary_threshold):
                del criterion, device
                self.assertEqual(binary_threshold, 0.5)
                if loader.dataset.label == "synthetic_val":
                    return {
                        "val_loss": 0.6,
                        "binary_iou": 0.2,
                        "binary_precision": 0.2,
                        "binary_recall": 0.2,
                        "binary_f1": 0.2,
                        "pred_fg_ratio": 0.2,
                        "gt_fg_ratio": 0.2,
                        "score": 0.2,
                    }
                if loader.dataset.label == "real_val":
                    score = next(initial_real_scores if phase["run"] == 0 else resumed_real_scores)
                    return {
                        "val_loss": 1.0 - score,
                        "binary_iou": score,
                        "binary_precision": score,
                        "binary_recall": score,
                        "binary_f1": score,
                        "pred_fg_ratio": 0.1,
                        "gt_fg_ratio": 0.1,
                        "score": score,
                    }
                return {
                    "val_loss": 0.2,
                    "binary_iou": 0.88,
                    "binary_precision": 0.88,
                    "binary_recall": 0.88,
                    "binary_f1": 0.88,
                    "pred_fg_ratio": 0.1,
                    "gt_fg_ratio": 0.1,
                    "score": 0.88,
                }

            optimizer = types.SimpleNamespace(
                state_dict=lambda: {"optimizer": "state"},
                load_state_dict=lambda state: None,
            )
            scheduler = types.SimpleNamespace(
                state_dict=lambda: {"scheduler": "state"},
                load_state_dict=lambda state: None,
                step=lambda: None,
            )

            write_metrics_call_count = {"value": 0}
            import training.foreground.train as train_module
            real_write_metrics = train_module._write_metrics_csv

            def crashing_write_metrics(path, rows):
                write_metrics_call_count["value"] += 1
                if phase["run"] == 0 and write_metrics_call_count["value"] == 2:
                    raise RuntimeError("crash-before-csv")
                return real_write_metrics(path, rows)

            common_patches = [
                mock.patch("training.foreground.train.collect_pairs", side_effect=[synthetic_records, real_records]),
                mock.patch("training.foreground.train.split_synthetic_pairs", return_value=(synthetic_train, synthetic_val)),
                mock.patch("training.foreground.train.split_real_pairs", return_value=(real_train, real_val, real_test)),
                mock.patch("training.foreground.train.build_train_records", return_value=train_records),
                mock.patch("training.foreground.train.build_region_map", return_value={}),
                mock.patch("training.foreground.train.UnifiedChartDataset", side_effect=dataset_factory),
                mock.patch("training.foreground.train.DataLoader", side_effect=dataloader_factory),
                mock.patch(
                    "training.foreground.train._load_model_helpers",
                    return_value=(
                        mock.Mock(return_value=FakeStatefulModel()),
                        mock.Mock(side_effect=lambda state: {key.replace("module.", "", 1): value for key, value in state.items()}),
                    ),
                ),
                mock.patch("training.foreground.train.BinaryMaskIoUCriterion", return_value=mock.Mock()),
                mock.patch("training.foreground.train.create_optimizer", return_value=optimizer),
                mock.patch("training.foreground.train.make_scheduler", return_value=scheduler),
                mock.patch("training.foreground.train.train_one_epoch", side_effect=fake_train_one_epoch),
                mock.patch("training.foreground.train.evaluate_loader", side_effect=fake_evaluate_loader),
                mock.patch("training.foreground.train._write_metrics_csv", side_effect=crashing_write_metrics),
            ]

            with contextlib.ExitStack() as stack:
                for patcher in common_patches:
                    stack.enter_context(patcher)
                with self.assertRaisesRegex(RuntimeError, "crash-before-csv"):
                    run_training(config=config, paths=paths, device="cpu", resume=None)

            last_payload = torch.load(output_dir / "last.pt", map_location="cpu", weights_only=True)
            history = last_payload["metrics"]["history"]
            self.assertEqual([row["epoch"] for row in history], [0, 1])
            self.assertTrue((output_dir / "metrics.csv").is_file())
            with (output_dir / "metrics.csv").open("r", encoding="utf-8", newline="") as handle:
                stale_rows = list(csv.DictReader(handle))
            self.assertEqual(len(stale_rows), 1)
            (output_dir / "best.pt").unlink()

            phase["run"] = 1
            with contextlib.ExitStack() as stack:
                for patcher in common_patches:
                    stack.enter_context(patcher)
                result = run_training(config=config, paths=paths, device="cpu", resume=output_dir / "last.pt")

            self.assertEqual(result["start_epoch"], 2)
            self.assertTrue((output_dir / "best.pt").is_file())
            self.assertTrue((output_dir / "metrics.csv").is_file())
            with (output_dir / "metrics.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 100)
            self.assertEqual(rows[0]["real_binary_iou"], "0.4")
            self.assertEqual(rows[1]["real_binary_iou"], "0.9")
            self.assertEqual(rows[1]["best_score"], "0.9")

    def test_rng_state_restore_reproduces_next_epoch_loader_order_and_augmentation(self):
        import training.foreground.train as train_module

        class RandomizedDataset(Dataset):
            def __len__(self):
                return 8

            def __getitem__(self, index):
                return {
                    "signature": torch.tensor(
                        [
                            index,
                            random.randint(0, 10_000),
                            int(np.random.randint(0, 10_000)),
                            int(torch.randint(0, 10_000, (1,)).item()),
                        ],
                        dtype=torch.int64,
                    )
                }

        def collect_epoch(loader):
            batches = []
            for batch in loader:
                batches.extend(tuple(row.tolist()) for row in batch["signature"])
            return batches

        dataset = RandomizedDataset()
        train_module._set_seed(42)
        generator_a = torch.Generator()
        generator_a.manual_seed(42)
        loader_a = train_module._make_loader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            generator=generator_a,
            worker_init_fn=train_module._seed_data_worker,
        )
        _ = collect_epoch(loader_a)
        saved_rng_state = train_module._capture_rng_state(generator_a)
        uninterrupted_next_epoch = collect_epoch(loader_a)

        train_module._set_seed(999)
        generator_b = torch.Generator()
        generator_b.manual_seed(999)
        train_module._restore_rng_state(saved_rng_state, generator_b)
        loader_b = train_module._make_loader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            generator=generator_b,
            worker_init_fn=train_module._seed_data_worker,
        )
        resumed_next_epoch = collect_epoch(loader_b)

        self.assertEqual(uninterrupted_next_epoch, resumed_next_epoch)

    def test_rng_state_restore_rejects_missing_or_malformed_nested_payloads(self):
        import training.foreground.train as train_module

        generator = torch.Generator()
        generator.manual_seed(42)
        valid_state = train_module._capture_rng_state(generator)

        for missing_key in ("python", "numpy", "torch", "train_loader_generator"):
            invalid_state = copy.deepcopy(valid_state)
            del invalid_state[missing_key]
            with self.assertRaisesRegex(ValueError, "rng_state"):
                train_module._restore_rng_state(invalid_state, generator)

        invalid_numpy_state = copy.deepcopy(valid_state)
        invalid_numpy_state["numpy"]["state"] = "bad"
        with self.assertRaisesRegex(ValueError, "rng_state.numpy.state"):
            train_module._restore_rng_state(invalid_numpy_state, generator)

        invalid_numpy_pos = copy.deepcopy(valid_state)
        invalid_numpy_pos["numpy"]["pos"] = "bad"
        with self.assertRaisesRegex(ValueError, "rng_state.numpy.pos"):
            train_module._restore_rng_state(invalid_numpy_pos, generator)

        invalid_torch_cpu = copy.deepcopy(valid_state)
        invalid_torch_cpu["torch"]["cpu"] = "bad"
        with self.assertRaisesRegex(ValueError, "rng_state.torch.cpu"):
            train_module._restore_rng_state(invalid_torch_cpu, generator)

        invalid_train_loader_generator = copy.deepcopy(valid_state)
        invalid_train_loader_generator["train_loader_generator"] = "bad"
        with self.assertRaisesRegex(ValueError, "rng_state.train_loader_generator"):
            train_module._restore_rng_state(invalid_train_loader_generator, generator)

        invalid_cuda_container = copy.deepcopy(valid_state)
        invalid_cuda_container["torch"]["cuda"] = {"bad": 1}
        with self.assertRaisesRegex(ValueError, "rng_state.torch.cuda"):
            train_module._restore_rng_state(invalid_cuda_container, generator)

        invalid_cuda_entry = copy.deepcopy(valid_state)
        invalid_cuda_entry["torch"]["cuda"] = [torch.zeros((2, 2), dtype=torch.uint8)]
        with self.assertRaisesRegex(ValueError, "rng_state.torch.cuda\\[0\\]"):
            train_module._restore_rng_state(invalid_cuda_entry, generator)

    def test_module_help_runs_cleanly_with_warnings_as_errors(self):
        python = Path(sys.executable)
        result = subprocess.run(
            [str(python), "-m", "training.foreground.train", "--help"],
            cwd=str(ROOT),
            env=dict(os.environ, PYTHONWARNINGS="error"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--config", result.stdout)
        self.assertIn("--synthetic-dir", result.stdout)
        self.assertIn("--dry-run", result.stdout)
        self.assertEqual(result.stderr, "")

    def test_packaging_includes_train_module_and_imports_from_built_wheel(self):
        python = Path(sys.executable)
        with tempfile.TemporaryDirectory() as tempdir:
            temp_root = Path(tempdir)
            repo_copy = temp_root / "repo"
            repo_copy.mkdir()
            shutil.copy2(ROOT / "pyproject.toml", repo_copy / "pyproject.toml")
            shutil.copy2(ROOT / "README.md", repo_copy / "README.md")
            shutil.copytree(ROOT / "training", repo_copy / "training")
            shutil.copytree(ROOT / "src", repo_copy / "src")
            dist_dir = temp_root / "dist"

            build_result = subprocess.run(
                [
                    str(python),
                    "-m",
                    "build",
                    "--sdist",
                    "--wheel",
                    "--no-isolation",
                    "--outdir",
                    str(dist_dir),
                ],
                cwd=str(repo_copy),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(build_result.returncode, 0, msg=build_result.stderr)

            wheel_path = next(dist_dir.glob("*.whl"))
            extracted_wheel = temp_root / "wheel_extract"
            subprocess.run(
                [str(python), "-c", "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])", str(wheel_path), str(extracted_wheel)],
                check=True,
                capture_output=True,
                text=True,
            )

            import_result = subprocess.run(
                [
                    str(python),
                    "-c",
                    (
                        "import sys; "
                        "sys.path.insert(0, r'{0}'); "
                        "import training, training.foreground.train; "
                        "print(training.__file__); "
                        "print(training.foreground.train.__file__)"
                    ).format(extracted_wheel),
                ],
                cwd=str(temp_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(import_result.returncode, 0, msg=import_result.stderr)
            self.assertIn("training/foreground/train.py", import_result.stdout)


if __name__ == "__main__":
    unittest.main()
