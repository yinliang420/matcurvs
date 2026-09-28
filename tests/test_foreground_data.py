import json
import math
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from training.foreground.data import (
    ChartRecord,
    UnifiedChartDataset,
    apply_region_masking,
    apply_regions_to_sample,
    assert_non_overlapping_records,
    build_keep_mask_from_regions,
    build_region_map,
    build_train_records,
    build_yolo_region_map,
    clamp_box,
    collect_pairs,
    detect_yolo_regions,
    deterministic_split,
    explicit_split,
    keep_mask_from_regions,
    rasterize_annotation,
    split_real_pairs,
    split_synthetic_pairs,
)


ROOT = Path(__file__).resolve().parents[1]


class _FakeBoxes:
    def __init__(self, boxes):
        self.xyxy = np.array([entry["xyxy"] for entry in boxes], dtype=np.float32)
        self.conf = np.array([entry.get("conf", 1.0) for entry in boxes], dtype=np.float32)
        self.cls = np.array([entry.get("cls", 0) for entry in boxes], dtype=np.float32)


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = _FakeBoxes(boxes)


class _FakeYOLO:
    def __init__(self, weights_path):
        self.weights_path = Path(weights_path)

    def predict(self, source, conf, verbose=False):
        if self.weights_path.name == "axis.pt":
            return [
                _FakeResult(
                    [
                        {"xyxy": [3, 3, 12, 18], "conf": 0.8, "cls": 0},
                        {"xyxy": [2, 2, 28, 28], "conf": 0.7, "cls": 0},
                        {"xyxy": [20, 20, 10, 10], "conf": 0.9, "cls": 0},
                    ]
                )
            ]
        if self.weights_path.name == "legend.pt":
            return [
                _FakeResult(
                    [
                        {"xyxy": [1, 1, 5, 5], "conf": 0.6, "cls": 0},
                        {"xyxy": [18, 18, 24, 24], "conf": 0.9, "cls": 1},
                    ]
                )
            ]
        raise AssertionError("unexpected fake weight path: {0}".format(self.weights_path))


class ForegroundDataTest(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def write_image(self, relative_path, size=(32, 32), color=255):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.full((size[1], size[0], 3), color, dtype=np.uint8)
        image[:, :, 0] = np.arange(size[0], dtype=np.uint8)[None, :]
        image[:, :, 1] = np.arange(size[1], dtype=np.uint8)[:, None]
        if not cv2.imwrite(str(path), image):
            raise AssertionError("failed to write image fixture")
        return path

    def write_annotation(self, relative_path, payload):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def make_line_annotation(self):
        return {
            "lines": {
                "Line_1": {
                    "display_mode": "line",
                    "points": [
                        {"pixel": [4, 4]},
                        {"pixel": [12, 12]},
                        {"pixel": [24, 16]},
                    ],
                },
                "Line_2": {
                    "render_mode": "scatter_only",
                    "marker_size": 10,
                    "points": [
                        {"pixel_coords": [22, 8]},
                    ],
                },
                "Line_3": {
                    "display_mode": "line_marker",
                    "marker_size": 8,
                    "marker_step": 1,
                    "points": [
                        [5, 24],
                        [14, 24],
                        [23, 24],
                    ],
                },
            }
        }

    def make_record(self, stem, source="synthetic"):
        image_path = self.write_image("{0}.png".format(stem))
        annotation_path = self.write_annotation(
            "{0}_annotation.json".format(stem),
            self.make_line_annotation(),
        )
        return ChartRecord(image_path=image_path, annotation_path=annotation_path, source=source)

    def test_collect_pairs_flat_and_recursive(self):
        self.write_image("flat/chart_a.png")
        self.write_annotation("flat/chart_a_annotation.json", self.make_line_annotation())
        self.write_image("nested/deep/chart_b.jpg")
        self.write_annotation("nested/deep/chart_b_annotation.json", self.make_line_annotation())

        flat_records = collect_pairs(self.root / "flat", source="synthetic", recursive=False)
        recursive_records = collect_pairs(self.root, source="real", recursive=True)

        self.assertEqual([record.image_path.name for record in flat_records], ["chart_a.png"])
        self.assertEqual(
            [record.image_path.name for record in recursive_records],
            ["chart_a.png", "chart_b.jpg"],
        )
        self.assertEqual({record.source for record in recursive_records}, {"real"})

    def test_collect_pairs_rejects_missing_root_missing_pair_duplicate_stem_and_empty(self):
        with self.assertRaises(FileNotFoundError):
            collect_pairs(self.root / "missing", source="synthetic", recursive=False)

        empty_root = self.root / "empty"
        empty_root.mkdir()
        with self.assertRaisesRegex(ValueError, "No annotation/image pairs"):
            collect_pairs(empty_root, source="synthetic", recursive=False)

        self.write_annotation("orphans/chart_a_annotation.json", self.make_line_annotation())
        with self.assertRaisesRegex(ValueError, "Missing paired image"):
            collect_pairs(self.root / "orphans", source="synthetic", recursive=False)

        self.write_image("image_only/chart_image_only.png")
        with self.assertRaisesRegex(ValueError, "Missing paired annotation"):
            collect_pairs(self.root / "image_only", source="synthetic", recursive=False)

        self.write_image("dupe/a/chart_c.png")
        self.write_annotation("dupe/a/chart_c_annotation.json", self.make_line_annotation())
        self.write_image("dupe/b/chart_c.jpg")
        self.write_annotation("dupe/b/chart_c_annotation.json", self.make_line_annotation())
        with self.assertRaisesRegex(ValueError, "Duplicate chart stem"):
            collect_pairs(self.root / "dupe", source="real", recursive=True)

    def test_collect_pairs_rejects_unknown_source(self):
        with self.assertRaisesRegex(ValueError, "source"):
            collect_pairs(self.root, source="other", recursive=False)

    def test_split_functions_are_deterministic_and_validate_sizes(self):
        synthetic_records = [self.make_record("synthetic_{0}".format(index)) for index in range(10)]

        train_a, val_a = split_synthetic_pairs(synthetic_records, val_ratio=0.2, seed=42)
        train_b, val_b = split_synthetic_pairs(synthetic_records, val_ratio=0.2, seed=42)
        train_c, val_c = split_synthetic_pairs(synthetic_records, val_ratio=0.2, seed=43)

        self.assertEqual([record.image_path.name for record in train_a], [record.image_path.name for record in train_b])
        self.assertEqual([record.image_path.name for record in val_a], [record.image_path.name for record in val_b])
        self.assertNotEqual(
            [record.image_path.name for record in train_a + val_a],
            [record.image_path.name for record in train_c + val_c],
        )

        real_records = [self.make_record("real_{0}".format(index), source="real") for index in range(65)]
        real_train, real_val, real_test = split_real_pairs(real_records, n_test=40, n_val=20, seed=42)
        self.assertEqual((len(real_train), len(real_val), len(real_test)), (5, 20, 40))

        undersized_real_records = [self.make_record("real_small_{0}".format(index), source="real") for index in range(10)]
        undersized_train, undersized_val, undersized_test = split_real_pairs(
            undersized_real_records,
            n_test=8,
            n_val=8,
            seed=42,
        )
        undersized_indices = np.random.RandomState(42).permutation(len(undersized_real_records))
        expected_test = [undersized_real_records[index].image_path.name for index in undersized_indices[:8]]
        expected_val = [undersized_real_records[index].image_path.name for index in undersized_indices[8:10]]
        expected_train = [undersized_real_records[index].image_path.name for index in undersized_indices[10:]]
        self.assertEqual([record.image_path.name for record in undersized_train], expected_train)
        self.assertEqual([record.image_path.name for record in undersized_val], expected_val)
        self.assertEqual([record.image_path.name for record in undersized_test], expected_test)

        shuffled = build_train_records(real_train, train_a, real_repeat=2, seed=7)
        expected_length = len(train_a) + len(real_train) * 2
        self.assertEqual(len(shuffled), expected_length)
        self.assertEqual(
            [record.image_path.name for record in shuffled],
            [record.image_path.name for record in build_train_records(real_train, train_a, real_repeat=2, seed=7)],
        )
        self.assertNotEqual(
            [record.image_path.name for record in shuffled],
            [record.image_path.name for record in build_train_records(real_train, train_a, real_repeat=2, seed=8)],
        )

        with self.assertRaisesRegex(ValueError, "at least 2"):
            split_synthetic_pairs(synthetic_records[:1], val_ratio=0.5, seed=42)
        with self.assertRaisesRegex(ValueError, "n_test must be an integer"):
            split_real_pairs(real_records, n_test=1.5, n_val=20, seed=42)
        with self.assertRaisesRegex(ValueError, "n_val must be >= 0"):
            split_real_pairs(real_records, n_test=40, n_val=-1, seed=42)
        with self.assertRaisesRegex(ValueError, "overlap"):
            assert_non_overlapping_records(train=train_a, val=train_a[:1])
        with self.assertRaisesRegex(ValueError, "real_repeat must be >= 1"):
            build_train_records(real_train, train_a, real_repeat=0, seed=42)
        with self.assertRaisesRegex(ValueError, "real_repeat must be an integer"):
            build_train_records(real_train, train_a, real_repeat=1.5, seed=42)
        with self.assertRaisesRegex(ValueError, "training record"):
            build_train_records([], [], real_repeat=1, seed=42)

    def test_rasterize_annotation_rejects_empty_malformed_out_of_bounds_and_nonfinite_points(self):
        with self.assertRaisesRegex(ValueError, "must contain"):
            rasterize_annotation({}, image_shape=(32, 32), line_width=3, max_instances=4)

        with self.assertRaisesRegex(ValueError, "empty"):
            rasterize_annotation({"lines": {"Line_1": {"points": []}}}, image_shape=(32, 32), line_width=3, max_instances=4)

        with self.assertRaisesRegex(ValueError, "point"):
            rasterize_annotation(
                {"lines": {"Line_1": {"points": [{"pixel": [40, 8]}]}}},
                image_shape=(32, 32),
                line_width=3,
                max_instances=4,
            )

        with self.assertRaisesRegex(ValueError, "finite"):
            rasterize_annotation(
                {"lines": {"Line_1": {"points": [{"pixel": [math.nan, 8]}]}}},
                image_shape=(32, 32),
                line_width=3,
                max_instances=4,
            )

        with self.assertRaisesRegex(ValueError, "points"):
            rasterize_annotation(
                {"data_series": [{"pixel": [4, 4]}]},
                image_shape=(32, 32),
                line_width=3,
                max_instances=4,
            )

    def test_rasterize_annotation_draws_line_scatter_and_markers(self):
        annotation = self.make_line_annotation()
        binary_mask, instance_masks, valid_flags = rasterize_annotation(
            annotation,
            image_shape=(32, 32),
            line_width=3,
            max_instances=4,
        )

        self.assertEqual(binary_mask.shape, (32, 32))
        self.assertEqual(instance_masks.shape, (4, 32, 32))
        self.assertEqual(valid_flags.tolist(), [True, True, True, False])
        self.assertGreater(int(binary_mask.sum()), 0)
        self.assertGreater(int(instance_masks[1, 8, 22]), 0)
        self.assertGreater(int(instance_masks[2].sum()), int(instance_masks[0].sum()))

    def test_clamp_keep_mask_and_apply_regions(self):
        self.assertEqual(clamp_box([2, 3, 12, 13], width=16, height=16), (2, 3, 12, 13))
        self.assertIsNone(clamp_box([10, 10, 9, 11], width=16, height=16))

        keep_mask = build_keep_mask_from_regions(
            (16, 16),
            axis_box=(2, 2, 14, 14),
            legend_boxes=[(6, 6, 10, 10)],
        )
        self.assertFalse(bool(keep_mask[0, 0]))
        self.assertTrue(bool(keep_mask[3, 3]))
        self.assertFalse(bool(keep_mask[7, 7]))

        image = np.zeros((16, 16, 3), dtype=np.uint8)
        binary_mask = np.ones((16, 16), dtype=np.uint8)
        instance_masks = np.ones((2, 16, 16), dtype=np.uint8)
        region = {"axis_box": [2, 2, 14, 14], "legend_boxes": [[6, 6, 10, 10]]}

        masked_image, masked_binary, masked_instances = apply_regions_to_sample(
            image,
            binary_mask,
            instance_masks,
            region,
        )

        self.assertTrue(np.array_equal(masked_image[0, 0], np.array([255, 255, 255], dtype=np.uint8)))
        self.assertTrue(np.array_equal(masked_image[7, 7], np.array([255, 255, 255], dtype=np.uint8)))
        self.assertEqual(int(masked_binary[0, 0]), 0)
        self.assertEqual(int(masked_binary[7, 7]), 0)
        self.assertEqual(int(masked_binary[3, 3]), 1)
        self.assertEqual(int(masked_instances[0, 7, 7]), 0)

    def test_build_yolo_region_map_uses_largest_axis_box_and_cache(self):
        real_record = self.make_record("real_yolo", source="real")
        axis_weights = self.root / "axis.pt"
        legend_weights = self.root / "legend.pt"
        axis_weights.write_text("axis", encoding="utf-8")
        legend_weights.write_text("legend", encoding="utf-8")
        cache_path = self.root / "cache" / "regions.json"

        fake_module = types.SimpleNamespace(YOLO=_FakeYOLO)
        with mock.patch.dict(sys.modules, {"ultralytics": fake_module}):
            region_map = build_yolo_region_map(
                [real_record],
                axis_weights=axis_weights,
                legend_weights=legend_weights,
                axis_confidence=0.5,
                legend_confidence=0.3,
                cache_path=cache_path,
            )

        region = region_map[str(real_record.image_path.resolve())]
        self.assertEqual(region["axis_box"], [2, 2, 28, 28])
        self.assertEqual(region["legend_boxes"], [[1, 1, 5, 5], [18, 18, 24, 24]])
        self.assertTrue(cache_path.is_file())

        axis_weights.unlink()
        legend_weights.unlink()
        region_map_from_cache = build_yolo_region_map(
            [real_record],
            axis_weights=axis_weights,
            legend_weights=legend_weights,
            cache_path=cache_path,
        )
        self.assertEqual(region_map_from_cache, region_map)

        with self.assertRaises(FileNotFoundError):
            build_yolo_region_map(
                [real_record],
                axis_weights=axis_weights,
                legend_weights=legend_weights,
                cache_path=self.root / "missing-cache.json",
            )

    def test_build_yolo_region_map_validates_cache_schema_and_canonicalizes_keys(self):
        real_record = self.make_record("real_cache", source="real")
        other_record = self.make_record("real_other", source="real")
        axis_weights = self.root / "axis.pt"
        legend_weights = self.root / "legend.pt"
        axis_weights.write_text("axis", encoding="utf-8")
        legend_weights.write_text("legend", encoding="utf-8")

        good_cache_path = self.root / "cache-good.json"
        good_cache_path.write_text(
            json.dumps(
                {
                    str(real_record.image_path.parent / "." / real_record.image_path.name): {
                        "axis_box": [1, 2, 20, 21],
                        "legend_boxes": [[3, 4, 5, 6]],
                    }
                }
            ),
            encoding="utf-8",
        )
        normalized = build_yolo_region_map(
            [real_record],
            axis_weights=axis_weights,
            legend_weights=legend_weights,
            cache_path=good_cache_path,
        )
        self.assertEqual(
            normalized,
            {
                str(real_record.image_path.resolve()): {
                    "axis_box": [1, 2, 20, 21],
                    "legend_boxes": [[3, 4, 5, 6]],
                }
            },
        )

        stale_cache_path = self.root / "cache-stale.json"
        stale_cache_path.write_text(
            json.dumps(
                {
                    str(other_record.image_path): {
                        "axis_box": [1, 2, 20, 21],
                        "legend_boxes": [],
                    }
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "missing required real-record entry"):
            build_yolo_region_map(
                [real_record],
                axis_weights=axis_weights,
                legend_weights=legend_weights,
                cache_path=stale_cache_path,
            )

        bad_schema_cache_path = self.root / "cache-schema.json"
        bad_schema_cache_path.write_text(
            json.dumps(
                {
                    str(real_record.image_path): {
                        "axis_box": [1, 2, 3],
                        "legend_boxes": [],
                    }
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "axis_box must be a 4-item box"):
            build_yolo_region_map(
                [real_record],
                axis_weights=axis_weights,
                legend_weights=legend_weights,
                cache_path=bad_schema_cache_path,
            )

    def test_alias_helpers_match_canonical_region_helpers(self):
        real_record = self.make_record("real_alias", source="real")
        synthetic_records = [self.make_record("synthetic_alias_{0}".format(index)) for index in range(4)]
        axis_weights = self.root / "axis.pt"
        legend_weights = self.root / "legend.pt"
        axis_weights.write_text("axis", encoding="utf-8")
        legend_weights.write_text("legend", encoding="utf-8")
        fake_module = types.SimpleNamespace(YOLO=_FakeYOLO)

        with mock.patch.dict(sys.modules, {"ultralytics": fake_module}):
            region = detect_yolo_regions(
                real_record.image_path,
                axis_weights=axis_weights,
                legend_weights=legend_weights,
            )
            region_map = build_region_map(
                [real_record],
                axis_weights=axis_weights,
                legend_weights=legend_weights,
            )
            canonical_map = build_yolo_region_map(
                [real_record],
                axis_weights=self.root / "axis.pt",
                legend_weights=self.root / "legend.pt",
                cache_path=self.root / "alias-cache.json",
            )

        image = np.zeros((16, 16, 3), dtype=np.uint8)
        binary_mask = np.ones((16, 16), dtype=np.uint8)
        masked_image, masked_binary = apply_region_masking(
            image,
            binary_mask,
            {"axis_box": [2, 2, 14, 14], "legend_boxes": [[6, 6, 10, 10]]},
        )
        _, canonical_binary, _ = apply_regions_to_sample(
            image,
            binary_mask,
            np.ones((1, 16, 16), dtype=np.uint8),
            {"axis_box": [2, 2, 14, 14], "legend_boxes": [[6, 6, 10, 10]]},
        )
        split_train, split_val = deterministic_split(synthetic_records, val_fraction=0.25, seed=42)
        canonical_train, canonical_val = split_synthetic_pairs(synthetic_records, val_ratio=0.25, seed=42)
        explicit_train, explicit_val = explicit_split(split_train, split_val)
        alias_keep = keep_mask_from_regions(
            (16, 16),
            axis_box=(2, 2, 14, 14),
            legend_boxes=[(6, 6, 10, 10)],
        )
        canonical_keep = build_keep_mask_from_regions(
            (16, 16),
            axis_box=(2, 2, 14, 14),
            legend_boxes=[(6, 6, 10, 10)],
        )

        self.assertEqual(region, canonical_map[str(real_record.image_path.resolve())])
        self.assertEqual(region_map, canonical_map)
        self.assertTrue(np.array_equal(masked_image[0, 0], np.array([255, 255, 255], dtype=np.uint8)))
        self.assertTrue(np.array_equal(masked_binary, canonical_binary))
        self.assertEqual(split_train, canonical_train)
        self.assertEqual(split_val, canonical_val)
        self.assertEqual(explicit_train, split_train)
        self.assertEqual(explicit_val, split_val)
        self.assertTrue(np.array_equal(alias_keep, canonical_keep))

    def test_dataset_shapes_strict_unreadable_image_real_weight_and_region_application(self):
        synthetic_record = self.make_record("synthetic_ok", source="synthetic")
        real_record = self.make_record("real_ok", source="real")
        region_map = {
            str(real_record.image_path.parent / "." / real_record.image_path.name): {
                "axis_box": [4, 4, 28, 28],
                "legend_boxes": [[20, 6, 28, 14]],
            }
        }
        dataset = UnifiedChartDataset(
            records=[synthetic_record, real_record],
            image_size=32,
            max_instances=4,
            is_train=False,
            line_width=3,
            real_weight=3.0,
            region_map=region_map,
        )

        synthetic_sample = dataset[0]
        real_sample = dataset[1]

        self.assertEqual(tuple(synthetic_sample["image"].shape), (3, 32, 32))
        self.assertEqual(tuple(synthetic_sample["binary_mask"].shape), (32, 32))
        self.assertEqual(tuple(synthetic_sample["instance_masks"].shape), (4, 32, 32))
        self.assertEqual(tuple(synthetic_sample["valid_flags"].shape), (4,))
        self.assertEqual(float(synthetic_sample["sample_weight"]), 1.0)
        self.assertEqual(float(synthetic_sample["is_real"]), 0.0)
        self.assertEqual(float(real_sample["sample_weight"]), 3.0)
        self.assertEqual(float(real_sample["is_real"]), 1.0)
        self.assertLess(float(real_sample["binary_mask"].sum()), float(synthetic_sample["binary_mask"].sum()))

        bad_image_path = self.root / "broken.png"
        bad_image_path.write_text("not-an-image", encoding="utf-8")
        bad_annotation = self.write_annotation("broken_annotation.json", self.make_line_annotation())
        bad_dataset = UnifiedChartDataset(
            records=[ChartRecord(image_path=bad_image_path, annotation_path=bad_annotation, source="real")],
            image_size=32,
            max_instances=4,
            is_train=False,
            line_width=3,
            real_weight=3.0,
        )
        with self.assertRaisesRegex(ValueError, "Unreadable image"):
            _ = bad_dataset[0]

        with self.assertRaisesRegex(ValueError, "missing required real-record entry"):
            UnifiedChartDataset(
                records=[real_record],
                image_size=32,
                max_instances=4,
                is_train=False,
                line_width=3,
                real_weight=3.0,
                region_map={str(self.root / "other.png"): {"axis_box": None, "legend_boxes": []}},
            )

    def test_dataset_train_augmentations_follow_patched_randomness(self):
        record = self.make_record("aug", source="synthetic")
        no_aug_dataset = UnifiedChartDataset(
            records=[record],
            image_size=32,
            max_instances=4,
            is_train=False,
            line_width=3,
            real_weight=3.0,
        )
        baseline = no_aug_dataset[0]

        flip_dataset = UnifiedChartDataset(
            records=[record],
            image_size=32,
            max_instances=4,
            is_train=True,
            line_width=3,
            real_weight=3.0,
        )
        with mock.patch("training.foreground.data.random.random", side_effect=[0.4, 0.9]):
            flipped = flip_dataset[0]

        self.assertTrue(np.array_equal(flipped["binary_mask"].numpy(), np.fliplr(baseline["binary_mask"].numpy())))

        brightness_dataset = UnifiedChartDataset(
            records=[record],
            image_size=32,
            max_instances=4,
            is_train=True,
            line_width=3,
            real_weight=3.0,
        )
        with mock.patch("training.foreground.data.random.random", side_effect=[0.9, 0.2]), mock.patch(
            "training.foreground.data.random.uniform",
            side_effect=[0.2, 10.0],
        ):
            bright = brightness_dataset[0]

        self.assertTrue(np.array_equal(bright["binary_mask"].numpy(), baseline["binary_mask"].numpy()))
        self.assertFalse(np.allclose(bright["image"].numpy(), baseline["image"].numpy()))

    def test_packaging_includes_foreground_package_files_and_yaml(self):
        with tempfile.TemporaryDirectory() as tempdir:
            temp_root = Path(tempdir)
            repo_copy = temp_root / "repo"
            repo_copy.mkdir()
            shutil.copy2(ROOT / "pyproject.toml", repo_copy / "pyproject.toml")
            shutil.copy2(ROOT / "README.md", repo_copy / "README.md")
            shutil.copytree(ROOT / "training", repo_copy / "training")
            shutil.copytree(ROOT / "src", repo_copy / "src")
            dist_dir = temp_root / "dist"

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--sdist",
                    "--wheel",
                    "--no-isolation",
                    "--outdir",
                    str(dist_dir),
                ],
                cwd=repo_copy,
                check=True,
                capture_output=True,
                text=True,
            )

            wheel_path = next(dist_dir.glob("*.whl"))
            with zipfile.ZipFile(wheel_path) as wheel_file:
                wheel_names = set(wheel_file.namelist())
            self.assertIn("training/foreground/__init__.py", wheel_names)
            self.assertIn("training/foreground/data.py", wheel_names)
            self.assertIn("training/foreground/configs/paper_q12_d3_h256.yaml", wheel_names)

            sdist_path = next(dist_dir.glob("*.tar.gz"))
            output = subprocess.run(
                ["tar", "-tf", str(sdist_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            sdist_names = set(output.stdout.splitlines())
            self.assertTrue(
                any(name.endswith("training/foreground/__init__.py") for name in sdist_names)
            )
            self.assertTrue(any(name.endswith("training/foreground/data.py") for name in sdist_names))
            self.assertTrue(
                any(name.endswith("training/foreground/configs/paper_q12_d3_h256.yaml") for name in sdist_names)
            )


if __name__ == "__main__":
    unittest.main()
