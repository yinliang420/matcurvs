import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import yaml

from training.yolo.train import (
    _load_yolo_class,
    build_argument_parser,
    build_train_kwargs,
    load_recipe,
    main,
    run_training,
    verify_dataset_classes,
)


class YoloTrainingTest(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def write_yaml(self, relative_path, payload):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        return path

    def write_text(self, relative_path, content):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def axis_recipe(self):
        return {
            "schema_version": "1.0",
            "role": "axis",
            "base_model": "yolo11x.pt",
            "classes": ["axis"],
            "train": {
                "workers": 8,
                "seed": 0,
                "deterministic": True,
                "pretrained": True,
                "optimizer": "auto",
                "amp": True,
                "save": True,
                "val": True,
                "plots": True,
                "epochs": 100,
                "patience": 50,
                "batch": 16,
                "imgsz": 640,
                "save_period": 10,
            },
            "hyperparameters": {
                "close_mosaic": 10,
                "lr0": 0.01,
                "lrf": 0.01,
                "momentum": 0.937,
                "weight_decay": 0.0005,
                "warmup_epochs": 3.0,
                "warmup_momentum": 0.8,
                "warmup_bias_lr": 0.1,
                "box": 7.5,
                "cls": 0.5,
                "dfl": 1.5,
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
                "degrees": 0.0,
                "translate": 0.1,
                "scale": 0.5,
                "shear": 0.0,
                "perspective": 0.0,
                "flipud": 0.0,
                "fliplr": 0.5,
                "mosaic": 1.0,
                "mixup": 0.0,
                "cutmix": 0.0,
            },
        }

    def dataset_payload(self, names):
        return {
            "path": "DATASET_ROOT",
            "train": "images/train",
            "val": "images/val",
            "nc": len(names) if isinstance(names, list) else len(names),
            "names": names,
        }

    def test_run_training_axis_recipe_passes_exact_expected_kwargs(self):
        recipe_path = self.write_yaml("configs/axis.yaml", self.axis_recipe())
        dataset_path = self.write_yaml("datasets/axis.yaml", self.dataset_payload(["axis"]))
        base_model = self.write_text("weights/yolo11x.pt", "weights")
        output_directory = self.root / "runs" / "axis"
        output_directory.parent.mkdir(parents=True, exist_ok=True)

        fake_model = mock.Mock()
        fake_model.train.return_value = {"status": "ok"}
        fake_yolo_class = mock.Mock(return_value=fake_model)

        with mock.patch("training.yolo.train._load_yolo_class", return_value=fake_yolo_class):
            result = run_training(recipe_path, dataset_path, base_model, output_directory)

        self.assertEqual(result, {"status": "ok"})
        fake_yolo_class.assert_called_once_with(str(base_model))
        fake_model.train.assert_called_once_with(
            data=str(dataset_path),
            project=str(output_directory),
            device="0",
            name="axis",
            exist_ok=False,
            workers=8,
            seed=0,
            deterministic=True,
            pretrained=True,
            optimizer="auto",
            amp=True,
            save=True,
            val=True,
            plots=True,
            epochs=100,
            patience=50,
            batch=16,
            imgsz=640,
            save_period=10,
            close_mosaic=10,
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            cutmix=0.0,
        )

    def test_verify_dataset_classes_accepts_list_names(self):
        dataset_path = self.write_yaml("datasets/list.yaml", self.dataset_payload(["axis"]))
        self.assertEqual(verify_dataset_classes(dataset_path, ["axis"]), ["axis"])

    def test_verify_dataset_classes_accepts_integer_keyed_dict_names(self):
        dataset_path = self.write_yaml(
            "datasets/dict.yaml",
            self.dataset_payload({0: "legend", 1: "clutter"}),
        )
        self.assertEqual(
            verify_dataset_classes(dataset_path, ["legend", "clutter"]),
            ["legend", "clutter"],
        )

    def test_verify_dataset_classes_rejects_legend_class_mismatch(self):
        dataset_path = self.write_yaml(
            "datasets/legend_bad.yaml",
            self.dataset_payload(["legend", "noise"]),
        )
        with self.assertRaisesRegex(ValueError, "Expected classes"):
            verify_dataset_classes(dataset_path, ["legend", "clutter"])

    def test_verify_dataset_classes_rejects_subfigure_class_mismatch(self):
        dataset_path = self.write_yaml(
            "datasets/subfigure_bad.yaml",
            self.dataset_payload(["subfigure", "extra"]),
        )
        with self.assertRaisesRegex(ValueError, "Expected classes"):
            verify_dataset_classes(dataset_path, ["subfigure"])

    def test_run_training_requires_existing_recipe_file(self):
        dataset_path = self.write_yaml("datasets/axis.yaml", self.dataset_payload(["axis"]))
        base_model = self.write_text("weights/yolo11x.pt", "weights")
        with self.assertRaises(FileNotFoundError):
            run_training(self.root / "missing.yaml", dataset_path, base_model, self.root / "runs")

    def test_run_training_requires_existing_dataset_file(self):
        recipe_path = self.write_yaml("configs/axis.yaml", self.axis_recipe())
        base_model = self.write_text("weights/yolo11x.pt", "weights")
        with self.assertRaises(FileNotFoundError):
            run_training(recipe_path, self.root / "missing.yaml", base_model, self.root / "runs")

    def test_run_training_requires_existing_base_model_file(self):
        recipe_path = self.write_yaml("configs/axis.yaml", self.axis_recipe())
        dataset_path = self.write_yaml("datasets/axis.yaml", self.dataset_payload(["axis"]))
        with self.assertRaises(FileNotFoundError):
            run_training(recipe_path, dataset_path, self.root / "weights" / "yolo11x.pt", self.root / "runs")

    def test_run_training_requires_expected_base_model_filename(self):
        recipe_path = self.write_yaml("configs/axis.yaml", self.axis_recipe())
        dataset_path = self.write_yaml("datasets/axis.yaml", self.dataset_payload(["axis"]))
        base_model = self.write_text("weights/yolo11n.pt", "weights")
        with self.assertRaisesRegex(ValueError, "base model filename"):
            run_training(recipe_path, dataset_path, base_model, self.root / "runs")

    def test_load_recipe_rejects_unknown_root_key(self):
        payload = self.axis_recipe()
        payload["extra"] = True
        recipe_path = self.write_yaml("configs/unknown.yaml", payload)
        with self.assertRaisesRegex(ValueError, "Unknown root keys"):
            load_recipe(recipe_path)

    def test_load_recipe_rejects_embedded_absolute_path(self):
        payload = self.axis_recipe()
        payload["train"]["optimizer"] = "/tmp/not-allowed"
        recipe_path = self.write_yaml("configs/path.yaml", payload)
        with self.assertRaisesRegex(ValueError, "path-like value"):
            load_recipe(recipe_path)

    def test_build_train_kwargs_rejects_reserved_key_collision(self):
        recipe_path = self.write_yaml("configs/collision.yaml", self.axis_recipe())
        dataset_path = self.write_yaml("datasets/axis.yaml", self.dataset_payload(["axis"]))
        recipe = load_recipe(recipe_path)
        recipe["train"]["project"] = "not-allowed"
        with self.assertRaisesRegex(ValueError, "reserved"):
            build_train_kwargs(recipe, dataset_path, self.root / "runs", "0")

    def test_verify_dataset_classes_rejects_malformed_dataset_class_map(self):
        dataset_path = self.write_yaml(
            "datasets/malformed.yaml",
            {
                "path": "DATASET_ROOT",
                "train": "images/train",
                "val": "images/val",
                "nc": 2,
                "names": {"zero": "legend", "one": "clutter"},
            },
        )
        with self.assertRaisesRegex(ValueError, "names"):
            verify_dataset_classes(dataset_path, ["legend", "clutter"])

    def test_build_argument_parser_requires_core_arguments(self):
        parser = build_argument_parser()
        namespace = parser.parse_args(
            [
                "--recipe",
                "recipe.yaml",
                "--dataset",
                "dataset.yaml",
                "--base-model",
                "yolo11x.pt",
                "--output",
                "runs",
            ]
        )
        self.assertEqual(namespace.recipe, Path("recipe.yaml"))
        self.assertEqual(namespace.dataset, Path("dataset.yaml"))
        self.assertEqual(namespace.base_model, Path("yolo11x.pt"))
        self.assertEqual(namespace.output, Path("runs"))
        self.assertEqual(namespace.device, "0")
        self.assertIsNone(namespace.run_name)

    def test_main_invokes_run_training_with_parsed_arguments(self):
        recipe_path = Path("recipe.yaml")
        dataset_path = Path("dataset.yaml")
        base_model = Path("yolo11x.pt")
        output_path = Path("runs")

        with mock.patch.object(
            sys,
            "argv",
            [
                "train.py",
                "--recipe",
                str(recipe_path),
                "--dataset",
                str(dataset_path),
                "--base-model",
                str(base_model),
                "--output",
                str(output_path),
                "--device",
                "cpu",
                "--run-name",
                "custom",
            ],
        ):
            with mock.patch("training.yolo.train.run_training") as run_training_mock:
                main()

        run_training_mock.assert_called_once_with(
            recipe_path,
            dataset_path,
            base_model,
            output_path,
            device="cpu",
            run_name="custom",
        )

    def test_load_yolo_class_imports_yolo_lazily(self):
        fake_yolo = object()
        fake_module = types.SimpleNamespace(YOLO=fake_yolo)
        with mock.patch.dict(sys.modules, {"ultralytics": fake_module}, clear=False):
            self.assertIs(_load_yolo_class(), fake_yolo)

    def test_dataset_examples_match_placeholders(self):
        root = Path(__file__).resolve().parents[1]
        expected = {
            "axis": ["axis"],
            "legend_clutter": ["legend", "clutter"],
            "subfigure": ["subfigure"],
        }
        for role, names in expected.items():
            path = root / "training" / "yolo" / "datasets" / (role + ".dataset.yaml.example")
            with path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
            self.assertEqual(
                set(data),
                {"path", "train", "val", "nc", "names"},
            )
            self.assertEqual(data["path"], "DATASET_ROOT")
            self.assertEqual(data["train"], "images/train")
            self.assertEqual(data["val"], "images/val")
            self.assertEqual(data["nc"], len(names))
            self.assertEqual(data["names"], names)

    def test_module_help_runs_cleanly_with_warnings_as_errors(self):
        root = Path(__file__).resolve().parents[1]
        python = Path(sys.executable)
        result = subprocess.run(
            [str(python), "-m", "training.yolo.train", "--help"],
            cwd=str(root),
            env=dict(os.environ, PYTHONWARNINGS="error"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--recipe", result.stdout)
        self.assertEqual(result.stderr, "")

    def test_build_artifacts_include_training_package_and_examples(self):
        root = Path(__file__).resolve().parents[1]
        python = Path(sys.executable)
        with tempfile.TemporaryDirectory() as tempdir:
            temp_root = Path(tempdir)
            repo_copy = temp_root / "repo"
            dist_dir = temp_root / "dist"
            self._copy_buildable_tree(root, repo_copy)

            result = subprocess.run(
                [str(python), "-m", "build", "--sdist", "--wheel", "--outdir", str(dist_dir)],
                cwd=str(repo_copy),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            wheel_path = self._single_path(dist_dir, "*.whl")
            sdist_path = self._single_path(dist_dir, "*.tar.gz")

            with zipfile.ZipFile(wheel_path) as wheel_zip:
                wheel_names = set(wheel_zip.namelist())
            self.assertIn("training/yolo/train.py", wheel_names)
            self.assertIn("training/yolo/configs/axis.yaml", wheel_names)
            self.assertIn("training/yolo/configs/legend_clutter.yaml", wheel_names)
            self.assertIn("training/yolo/configs/subfigure.yaml", wheel_names)
            self.assertIn("training/yolo/datasets/axis.dataset.yaml.example", wheel_names)
            self.assertIn("training/yolo/datasets/legend_clutter.dataset.yaml.example", wheel_names)
            self.assertIn("training/yolo/datasets/subfigure.dataset.yaml.example", wheel_names)
            self.assertIn("training/MANIFEST.json", wheel_names)

            with tarfile.open(sdist_path, "r:gz") as sdist_tar:
                sdist_names = set(sdist_tar.getnames())
            prefix = sdist_path.name[:-7] + "/"
            self.assertIn(prefix + "training/yolo/train.py", sdist_names)
            self.assertIn(prefix + "training/yolo/configs/axis.yaml", sdist_names)
            self.assertIn(prefix + "training/yolo/datasets/axis.dataset.yaml.example", sdist_names)
            self.assertIn(prefix + "training/MANIFEST.json", sdist_names)

            extracted_wheel = temp_root / "wheel_extract"
            with zipfile.ZipFile(wheel_path) as wheel_zip:
                wheel_zip.extractall(extracted_wheel)

            import_result = subprocess.run(
                [
                    str(python),
                    "-c",
                    (
                        "import sys; "
                        "sys.path.insert(0, r'{0}'); "
                        "import training, training.yolo.train; "
                        "print(training.__file__); "
                        "print(training.yolo.train.__file__)"
                    ).format(extracted_wheel),
                ],
                cwd=str(temp_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(import_result.returncode, 0, msg=import_result.stderr)
            self.assertIn("training/__init__.py", import_result.stdout)
            self.assertIn("training/yolo/train.py", import_result.stdout)

            self._cleanup_build_artifacts(repo_copy)
            self.assertFalse((repo_copy / "build").exists())
            self.assertFalse((repo_copy / "matdeplot.egg-info").exists())

    def _copy_buildable_tree(self, source_root, destination_root):
        destination_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / "pyproject.toml", destination_root / "pyproject.toml")
        shutil.copy2(source_root / "README.md", destination_root / "README.md")
        shutil.copytree(source_root / "src", destination_root / "src")
        shutil.copytree(source_root / "training", destination_root / "training", ignore=shutil.ignore_patterns("__pycache__"))

    def _single_path(self, directory, pattern):
        matches = sorted(directory.glob(pattern))
        self.assertEqual(len(matches), 1, msg="Expected one match for {0}, found {1}".format(pattern, matches))
        return matches[0]

    def _cleanup_build_artifacts(self, root):
        for relative_path in ("build", "matdeplot.egg-info"):
            path = root / relative_path
            if path.exists():
                shutil.rmtree(path)


if __name__ == "__main__":
    unittest.main()
