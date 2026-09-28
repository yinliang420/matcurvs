import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from matdeplot.document_pipeline import DocumentPipeline, PanelClassification
from matdeplot.judge import JudgeVerdict
from matdeplot.schema import CurveRecord, ExtractionResult, FailureRecord


class DocumentJudgePipelineTest(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.model_directory = self.root / "models"
        self.model_directory.mkdir()
        self.figure_dir = self.root / "figures"
        self.figure_dir.mkdir()
        self.panel_path = self.write_image("panels/panel.png")
        self.write_image("figures/figure.png")

    def tearDown(self):
        self.tempdir.cleanup()

    def write_image(self, relative_path: str, size=(32, 24), value=255) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.full((size[1], size[0], 3), value, dtype=np.uint8)
        image[:, :, 0] = np.arange(size[0], dtype=np.uint8)[None, :]
        image[:, :, 1] = np.arange(size[1], dtype=np.uint8)[:, None]
        ok = cv2.imwrite(str(path), image)
        if not ok:
            raise AssertionError("failed to write image fixture")
        return path

    def make_line_chart(self) -> PanelClassification:
        return PanelClassification(
            chart_type="line chart",
            first_level="Experimental Data",
            second_level="Performance",
            confidence=0.9,
            reasoning="continuous connected curves",
        )

    def make_non_line_chart(self) -> PanelClassification:
        return PanelClassification(
            chart_type="scatter plot",
            first_level="Experimental Data",
            second_level="Performance",
            confidence=0.8,
            reasoning="isolated markers only",
        )

    def make_curve(
        self,
        index: int = 0,
        pixel_points=None,
        real_points=None,
    ) -> CurveRecord:
        return CurveRecord(
            curve_index=index,
            pixel_points=pixel_points or [(1.0, 1.0), (2.0, 2.0)],
            real_points=real_points or [(10.0, 10.0), (11.0, 11.0)],
            confidence=0.9,
            abstain=False,
        )

    def make_result(
        self,
        *,
        success: bool = True,
        curves=None,
        failures=None,
        image_id: str = "panel",
    ) -> ExtractionResult:
        return ExtractionResult(
            image_id=image_id,
            image_size=(32, 24),
            success=success,
            used_fallback=False,
            curve_count_source="silhouette",
            requested_curve_count=1,
            foreground_fraction=0.1,
            curves=list(curves or []),
            failures=list(failures or []),
        )

    def read_manifest(self, path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    def assert_raw_record_exists(self, output_directory: Path, panel_stem: str = "panel") -> None:
        self.assertTrue(
            (output_directory / "curves" / panel_stem / "extraction.json").is_file()
        )

    def test_judge_disabled_preserves_backward_compatibility_and_writes_raw_outputs(self):
        classifier = mock.Mock()
        classifier.classify.return_value = self.make_line_chart()
        pipeline = DocumentPipeline(self.model_directory, classifier=classifier)
        extraction = self.make_result(curves=[self.make_curve()])
        output_directory = self.root / "out"

        with mock.patch(
            "matdeplot.document_pipeline.crop_subfigures",
            return_value=[self.panel_path],
        ), mock.patch.object(
            pipeline.chart_pipeline,
            "extract",
            return_value=extraction,
        ) as extract_mock:
            result = pipeline.run(self.figure_dir, output_directory)

        self.assertEqual(set(result), {"manifest", "panels"})
        self.assertEqual(result["panels"], 1)
        self.assertFalse((output_directory / "accepted_manifest.json").exists())
        manifest = self.read_manifest(Path(result["manifest"]))
        self.assertEqual(manifest[0]["admitted"], None)
        self.assertNotIn("judge", manifest[0])
        self.assertIn("outputs", manifest[0])
        extract_mock.assert_called_once()
        self.assert_raw_record_exists(output_directory)

    def test_judge_enabled_writes_accepted_manifest_for_only_admitted_rows(self):
        classifier = mock.Mock()
        classifier.classify.side_effect = [self.make_line_chart(), self.make_line_chart()]
        judge = mock.Mock()
        judge.model = "qwen-vl-max"
        accepted_verdict = JudgeVerdict(
            model=judge.model,
            admitted=True,
            n_left=1,
            n_right=1,
            curve_count_match=True,
            shape_match=True,
            range_plausible=True,
            reason="clean match",
        )
        rejected_verdict = JudgeVerdict(
            model=judge.model,
            admitted=False,
            n_left=1,
            n_right=1,
            curve_count_match=True,
            shape_match=False,
            range_plausible=True,
            reason="shape mismatch",
        )
        judge.review.side_effect = [accepted_verdict, rejected_verdict]
        pipeline = DocumentPipeline(
            self.model_directory,
            classifier=classifier,
            judge=judge,
        )
        output_directory = self.root / "out"
        panel_two = self.write_image("panels/panel_two.png")
        first = self.make_result(curves=[self.make_curve()], image_id="panel")
        second = self.make_result(curves=[self.make_curve(index=1)], image_id="panel_two")

        with mock.patch(
            "matdeplot.document_pipeline.crop_subfigures",
            return_value=[self.panel_path, panel_two],
        ), mock.patch.object(
            pipeline.chart_pipeline,
            "extract",
            side_effect=[first, second],
        ):
            result = pipeline.run(self.figure_dir, output_directory)

        self.assertEqual(
            set(result),
            {"manifest", "accepted_manifest", "panels", "accepted"},
        )
        self.assertEqual(result["panels"], 2)
        self.assertEqual(result["accepted"], 1)
        manifest = self.read_manifest(Path(result["manifest"]))
        accepted_manifest = self.read_manifest(Path(result["accepted_manifest"]))
        self.assertEqual([row["admitted"] for row in manifest], [True, False])
        self.assertEqual(
            [row["judge"]["reason"] for row in manifest],
            ["clean match", "shape mismatch"],
        )
        self.assertEqual(accepted_manifest, [manifest[0]])
        self.assertEqual(
            [call.args[0].name for call in judge.review.call_args_list],
            ["panel.png", "panel_two.png"],
        )
        self.assert_raw_record_exists(output_directory, "panel")
        self.assert_raw_record_exists(output_directory, "panel_two")

    def test_judge_fails_closed_without_hosted_call_for_failed_extraction(self):
        classifier = mock.Mock()
        classifier.classify.return_value = self.make_line_chart()
        judge = mock.Mock()
        judge.model = "qwen-vl-max"
        pipeline = DocumentPipeline(
            self.model_directory,
            classifier=classifier,
            judge=judge,
        )
        failed = self.make_result(
            success=False,
            curves=[],
            failures=[FailureRecord(stage="foreground", message="mask2former failed")],
        )
        output_directory = self.root / "out"

        with mock.patch(
            "matdeplot.document_pipeline.crop_subfigures",
            return_value=[self.panel_path],
        ), mock.patch.object(
            pipeline.chart_pipeline,
            "extract",
            return_value=failed,
        ):
            result = pipeline.run(self.figure_dir, output_directory)

        manifest = self.read_manifest(Path(result["manifest"]))
        accepted_manifest = self.read_manifest(Path(result["accepted_manifest"]))
        self.assertEqual(manifest[0]["admitted"], False)
        self.assertEqual(manifest[0]["judge"]["model"], judge.model)
        self.assertEqual(manifest[0]["judge"]["error"], "extraction_failed")
        self.assertEqual(manifest[0]["judge"]["reason"], "extraction failed")
        self.assertEqual(accepted_manifest, [])
        judge.review.assert_not_called()
        self.assert_raw_record_exists(output_directory)

    def test_judge_fails_closed_without_hosted_call_when_calibrated_curves_unavailable(self):
        classifier = mock.Mock()
        classifier.classify.return_value = self.make_line_chart()
        judge = mock.Mock()
        judge.model = "qwen-vl-max"
        pipeline = DocumentPipeline(
            self.model_directory,
            classifier=classifier,
            judge=judge,
        )
        output_directory = self.root / "out"
        invalid_result = self.make_result(
            curves=[
                self.make_curve(
                    pixel_points=[(1.0, 1.0), (2.0, 2.0)],
                    real_points=[(10.0, 10.0)],
                )
            ]
        )

        with mock.patch(
            "matdeplot.document_pipeline.crop_subfigures",
            return_value=[self.panel_path],
        ), mock.patch.object(
            pipeline.chart_pipeline,
            "extract",
            return_value=invalid_result,
        ):
            result = pipeline.run(self.figure_dir, output_directory)

        manifest = self.read_manifest(Path(result["manifest"]))
        accepted_manifest = self.read_manifest(Path(result["accepted_manifest"]))
        self.assertEqual(manifest[0]["admitted"], False)
        self.assertEqual(
            manifest[0]["judge"]["error"],
            "calibrated_curves_unavailable",
        )
        self.assertEqual(
            manifest[0]["judge"]["reason"],
            "calibrated curves unavailable",
        )
        self.assertEqual(accepted_manifest, [])
        judge.review.assert_not_called()
        self.assert_raw_record_exists(output_directory)

    def test_judge_runtime_exception_is_structured_and_does_not_abort_manifest_writes(self):
        classifier = mock.Mock()
        classifier.classify.return_value = self.make_line_chart()
        judge = mock.Mock()
        judge.model = "qwen-vl-max"
        judge.review.side_effect = RuntimeError("provider down")
        pipeline = DocumentPipeline(
            self.model_directory,
            classifier=classifier,
            judge=judge,
        )
        output_directory = self.root / "out"
        extraction = self.make_result(curves=[self.make_curve()])

        with mock.patch(
            "matdeplot.document_pipeline.crop_subfigures",
            return_value=[self.panel_path],
        ), mock.patch.object(
            pipeline.chart_pipeline,
            "extract",
            return_value=extraction,
        ):
            result = pipeline.run(self.figure_dir, output_directory)

        manifest_path = Path(result["manifest"])
        accepted_manifest_path = Path(result["accepted_manifest"])
        self.assertTrue(manifest_path.is_file())
        self.assertTrue(accepted_manifest_path.is_file())
        manifest = self.read_manifest(manifest_path)
        accepted_manifest = self.read_manifest(accepted_manifest_path)
        self.assertEqual(manifest[0]["admitted"], False)
        self.assertEqual(manifest[0]["judge"]["error"], "judge:RuntimeError")
        self.assertEqual(
            manifest[0]["judge"]["reason"],
            "Final quality judge failed operationally.",
        )
        self.assertEqual(accepted_manifest, [])
        self.assert_raw_record_exists(output_directory)

    def test_non_line_chart_skips_judge_and_keeps_admitted_none(self):
        classifier = mock.Mock()
        classifier.classify.return_value = self.make_non_line_chart()
        judge = mock.Mock()
        judge.model = "qwen-vl-max"
        pipeline = DocumentPipeline(
            self.model_directory,
            classifier=classifier,
            judge=judge,
        )
        output_directory = self.root / "out"

        with mock.patch(
            "matdeplot.document_pipeline.crop_subfigures",
            return_value=[self.panel_path],
        ), mock.patch.object(pipeline.chart_pipeline, "extract") as extract_mock:
            result = pipeline.run(self.figure_dir, output_directory)

        manifest = self.read_manifest(Path(result["manifest"]))
        accepted_manifest = self.read_manifest(Path(result["accepted_manifest"]))
        self.assertEqual(manifest[0]["admitted"], None)
        self.assertEqual(manifest[0]["skipped"], "not a line chart")
        self.assertNotIn("judge", manifest[0])
        self.assertEqual(accepted_manifest, [])
        extract_mock.assert_not_called()
        judge.review.assert_not_called()
        self.assertFalse((output_directory / "curves").exists())


if __name__ == "__main__":
    unittest.main()
