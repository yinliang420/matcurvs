import re
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(relative_path: str):
    path = ROOT / relative_path
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(relative_path: str):
    import json

    path = ROOT / relative_path
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class TrainingConfigsTest(unittest.TestCase):
    maxDiff = None

    def test_yolo_selected_recipes(self):
        common_train = {
            "workers": 8,
            "seed": 0,
            "deterministic": True,
            "pretrained": True,
            "optimizer": "auto",
            "amp": True,
            "save": True,
            "val": True,
            "plots": True,
        }
        expected_by_role = {
            "axis": {
                "epochs": 100,
                "batch": 16,
                "imgsz": 640,
                "classes": ["axis"],
                "patience": 50,
                "save_period": 10,
            },
            "legend_clutter": {
                "epochs": 500,
                "batch": 32,
                "imgsz": 640,
                "classes": ["legend", "clutter"],
                "patience": 100,
                "save_period": -1,
            },
            "subfigure": {
                "epochs": 200,
                "batch": 32,
                "imgsz": 640,
                "classes": ["subfigure"],
                "patience": 100,
                "save_period": -1,
            },
        }
        expected_hyperparameters = {
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
        }

        for role, expected in expected_by_role.items():
            data = load_yaml(f"training/yolo/configs/{role}.yaml")
            self.assertEqual(
                set(data),
                {"schema_version", "role", "base_model", "classes", "train", "hyperparameters"},
            )
            self.assertEqual(data["schema_version"], "1.0")
            self.assertEqual(data["role"], role)
            self.assertEqual(data["base_model"], "yolo11x.pt")
            self.assertEqual(data["classes"], expected["classes"])
            self.assertEqual(
                data["train"],
                common_train
                | {
                    "epochs": expected["epochs"],
                    "patience": expected["patience"],
                    "batch": expected["batch"],
                    "imgsz": expected["imgsz"],
                    "save_period": expected["save_period"],
                },
            )
            self.assertEqual(data["hyperparameters"], expected_hyperparameters)

    def test_foreground_selected_recipe(self):
        data = load_yaml("training/foreground/configs/paper_q12_d3_h256.yaml")
        self.assertEqual(
            set(data),
            {"schema_version", "selected_experiment", "model", "training", "splits", "preprocessing", "selection"},
        )
        self.assertEqual(data["schema_version"], "1.0")
        self.assertEqual(data["selected_experiment"], "v11_raw40k_q12_d3_h256")
        self.assertEqual(set(data["model"]), {"encoder", "hidden_dim", "n_queries", "decoder_layers"})
        self.assertEqual(
            data["model"],
            {
                "encoder": "mit_b3",
                "hidden_dim": 256,
                "n_queries": 12,
                "decoder_layers": 3,
            },
        )
        self.assertEqual(
            set(data["training"]),
            {
                "epochs",
                "batch_size",
                "image_size",
                "line_width",
                "max_instances",
                "real_weight",
                "real_repeat",
                "learning_rate",
                "warmup_epochs",
                "binary_threshold",
                "weight_decay",
                "num_workers",
                "seed",
            },
        )
        self.assertEqual(
            data["training"],
            {
                "epochs": 100,
                "batch_size": 2,
                "image_size": 640,
                "line_width": 3,
                "max_instances": 14,
                "real_weight": 3.0,
                "real_repeat": 64,
                "learning_rate": 1e-4,
                "warmup_epochs": 5,
                "binary_threshold": 0.5,
                "weight_decay": 0.05,
                "num_workers": 8,
                "seed": 42,
            },
        )
        self.assertEqual(set(data["splits"]), {"synthetic_val_ratio", "real_test_size", "real_val_size"})
        self.assertEqual(
            data["splits"],
            {
                "synthetic_val_ratio": 0.05,
                "real_test_size": 40,
                "real_val_size": 20,
            },
        )
        self.assertEqual(
            set(data["preprocessing"]),
            {"use_yolo_real_mask", "yolo_axis_confidence", "yolo_legend_confidence"},
        )
        self.assertEqual(
            data["preprocessing"],
            {
                "use_yolo_real_mask": True,
                "yolo_axis_confidence": 0.5,
                "yolo_legend_confidence": 0.3,
            },
        )
        self.assertEqual(set(data["selection"]), {"best_on", "metric"})
        self.assertEqual(
            data["selection"],
            {
                "best_on": "real",
                "metric": "real_binary_iou",
            },
        )

    def test_manifest_declares_external_artifacts(self):
        data = load_json("training/MANIFEST.json")
        self.assertEqual(set(data), {"schema_version", "weights_included", "training_data_included", "artifacts"})
        self.assertEqual(data["schema_version"], "1.0")
        self.assertFalse(data["weights_included"])
        self.assertFalse(data["training_data_included"])

        artifacts = {entry["role"]: entry for entry in data["artifacts"]}
        self.assertEqual(set(artifacts), {"foreground", "axis", "legend", "subfigure"})

        expected = {
            "foreground": "847aba64be1507c9061c98756d396721a8ff1e47f0d1a9bac41e3af02eee17a9",
            "axis": "cfec499d888fdf28f4475b2f2a040fc6a3275d2b0c34fabee6d9ebe816512569",
            "legend": "9e1bbdfcabeff45a0faa19eb276b97a96d9ba3f96b0f561eae48832df0403310",
            "subfigure": "ae40f5e7fab0a10d8b56537e301f90ff81b3204ebd2535359f00b5d98cb37742",
        }
        expected_filenames = {
            "foreground": "mask2foremer_q12_d3_h256_best.pt",
            "axis": "axis_yolo11_best.pt",
            "legend": "legend_yolo11_best.pt",
            "subfigure": "subfigure_yolo11x_best.pt",
        }
        digest_re = re.compile(r"^[0-9a-f]{64}$")

        for role, sha256 in expected.items():
            artifact = artifacts[role]
            self.assertEqual(set(artifact), {"role", "filename", "sha256", "availability"})
            self.assertEqual(artifact["role"], role)
            self.assertEqual(artifact["filename"], expected_filenames[role])
            self.assertNotIn("/", artifact["filename"])
            self.assertEqual(artifact["sha256"], sha256)
            self.assertRegex(artifact["sha256"], digest_re)
            self.assertEqual(artifact["availability"], "not included")


if __name__ == "__main__":
    unittest.main()
