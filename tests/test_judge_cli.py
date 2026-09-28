import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from matdeplot.judge import JudgeVerdict


def load_script_module(module_name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"failed to load spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop(module_name, None)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class RunDocumentCliTest(unittest.TestCase):
    def setUp(self):
        self.module = load_script_module(
            "test_run_document_script",
            "scripts/run_document.py",
        )

    def test_omitted_judge_passes_none_into_document_pipeline(self):
        workflow = mock.Mock()

        with mock.patch.object(
            self.module,
            "DocumentPipeline",
            return_value=workflow,
        ) as pipeline_cls, mock.patch.object(
            self.module,
            "QwenPanelClassifier",
        ) as classifier_cls:
            self.module.main(
                [
                    "--input",
                    "input_dir",
                    "--models",
                    "models_dir",
                    "--output",
                    "out_dir",
                ]
            )

        classifier_cls.assert_called_once_with()
        pipeline_cls.assert_called_once_with(
            Path("models_dir"),
            classifier=classifier_cls.return_value,
            judge=None,
        )
        workflow.run.assert_called_once_with(
            input_path=Path("input_dir"),
            output_directory=Path("out_dir"),
            curve_count_source="silhouette",
            curve_count_cache=None,
            assume_line_charts=False,
            calibrate=True,
        )

    def test_explicit_judge_constructs_model_and_passes_object(self):
        workflow = mock.Mock()
        judge = object()

        with mock.patch.object(
            self.module,
            "DocumentPipeline",
            return_value=workflow,
        ) as pipeline_cls, mock.patch.object(
            self.module,
            "QwenPanelClassifier",
        ) as classifier_cls, mock.patch.object(
            self.module,
            "QwenExtractionJudge",
            return_value=judge,
        ) as judge_cls:
            self.module.main(
                [
                    "--input",
                    "input_dir",
                    "--models",
                    "models_dir",
                    "--output",
                    "out_dir",
                    "--judge",
                    "qwen-vl-max",
                    "--curve-count-source",
                    "cached",
                    "--curve-count-cache",
                    "counts.csv",
                    "--assume-line-charts",
                    "--pixel-only",
                ]
            )

        classifier_cls.assert_not_called()
        judge_cls.assert_called_once_with(model="qwen-vl-max")
        pipeline_cls.assert_called_once_with(
            Path("models_dir"),
            classifier=None,
            judge=judge,
        )
        workflow.run.assert_called_once_with(
            input_path=Path("input_dir"),
            output_directory=Path("out_dir"),
            curve_count_source="cached",
            curve_count_cache=Path("counts.csv"),
            assume_line_charts=True,
            calibrate=False,
        )


class JudgeExtractionCliTest(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.module = load_script_module(
            "test_judge_extraction_script",
            "scripts/judge_extraction.py",
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def write_image(self) -> Path:
        path = self.root / "chart.png"
        path.write_bytes(b"png")
        return path

    def write_extraction(self) -> Path:
        path = self.root / "extraction.json"
        path.write_text("{}", encoding="utf-8")
        return path

    def read_json(self, path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    def make_args(self, output: Path):
        return [
            "--image",
            str(self.write_image()),
            "--extraction",
            str(self.write_extraction()),
            "--output",
            str(output),
        ]

    def test_standalone_accepts_and_writes_structured_verdict(self):
        output = self.root / "out" / "verdict.json"
        verdict = JudgeVerdict(
            model="qwen-vl-max",
            admitted=True,
            n_left=1,
            n_right=1,
            curve_count_match=True,
            shape_match=True,
            range_plausible=True,
            reason="clean match",
        )
        judge = mock.Mock()
        judge.review.return_value = verdict

        with mock.patch.object(
            self.module,
            "load_curves_from_extraction",
            return_value=["curve"],
        ) as load_mock, mock.patch.object(
            self.module,
            "QwenExtractionJudge",
            return_value=judge,
        ) as judge_cls:
            exit_code = self.module.main(self.make_args(output))

        self.assertEqual(exit_code, 0)
        judge_cls.assert_called_once_with(model="qwen-vl-max")
        load_mock.assert_called_once()
        judge.review.assert_called_once()
        self.assertEqual(self.read_json(output), verdict.to_dict())

    def test_standalone_rejects_quality_and_exits_one(self):
        output = self.root / "out" / "verdict.json"
        verdict = JudgeVerdict(
            model="qwen-vl-max",
            admitted=False,
            n_left=2,
            n_right=2,
            curve_count_match=True,
            shape_match=False,
            range_plausible=True,
            reason="shape mismatch",
        )
        judge = mock.Mock()
        judge.review.return_value = verdict

        with mock.patch.object(
            self.module,
            "load_curves_from_extraction",
            return_value=["curve"],
        ), mock.patch.object(
            self.module,
            "QwenExtractionJudge",
            return_value=judge,
        ):
            exit_code = self.module.main(self.make_args(output))

        self.assertEqual(exit_code, 1)
        self.assertEqual(self.read_json(output), verdict.to_dict())

    def test_standalone_operational_verdict_exits_two_and_writes_json(self):
        output = self.root / "out" / "verdict.json"
        verdict = JudgeVerdict(
            model="qwen-vl-max",
            admitted=False,
            reason="provider unavailable",
            error="provider_error",
        )
        judge = mock.Mock()
        judge.review.return_value = verdict

        with mock.patch.object(
            self.module,
            "load_curves_from_extraction",
            return_value=["curve"],
        ), mock.patch.object(
            self.module,
            "QwenExtractionJudge",
            return_value=judge,
        ):
            exit_code = self.module.main(self.make_args(output))

        self.assertEqual(exit_code, 2)
        self.assertEqual(self.read_json(output), verdict.to_dict())

    def test_loader_exception_writes_structured_error_verdict(self):
        output = self.root / "out" / "verdict.json"
        judge = mock.Mock()
        judge.review.return_value = JudgeVerdict(model="qwen-vl-max", admitted=True)

        with mock.patch.object(
            self.module,
            "load_curves_from_extraction",
            side_effect=ValueError("raw schema details should not leak"),
        ), mock.patch.object(
            self.module,
            "QwenExtractionJudge",
            return_value=judge,
        ):
            exit_code = self.module.main(self.make_args(output))

        self.assertEqual(exit_code, 2)
        self.assertEqual(
            self.read_json(output),
            {
                "model": "qwen-vl-max",
                "admitted": False,
                "n_left": None,
                "n_right": None,
                "curve_count_match": None,
                "shape_match": None,
                "range_plausible": None,
                "reason": "Extraction payload could not be loaded.",
                "latency_seconds": 0.0,
                "error": "extraction:ValueError",
            },
        )
        judge.review.assert_not_called()

    def test_atomic_write_cleans_up_temporary_file_on_replace_failure(self):
        output = self.root / "out" / "verdict.json"
        verdict = JudgeVerdict(
            model="qwen-vl-max",
            admitted=True,
            n_left=1,
            n_right=1,
            curve_count_match=True,
            shape_match=True,
            range_plausible=True,
            reason="clean match",
        )
        judge = mock.Mock()
        judge.review.return_value = verdict
        real_replace = self.module.os.replace
        seen = []

        def failing_replace(src, dst):
            seen.append(Path(src))
            raise OSError("replace failed")

        with mock.patch.object(
            self.module,
            "load_curves_from_extraction",
            return_value=["curve"],
        ), mock.patch.object(
            self.module,
            "QwenExtractionJudge",
            return_value=judge,
        ), mock.patch.object(
            self.module.os,
            "replace",
            side_effect=failing_replace,
        ):
            with self.assertRaisesRegex(OSError, "replace failed"):
                self.module.main(self.make_args(output))

        self.assertEqual(len(seen), 1)
        self.assertFalse(seen[0].exists())
        self.assertFalse(output.exists())
        self.assertEqual(sorted(output.parent.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
