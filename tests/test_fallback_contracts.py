import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from matdeplot.config import SeparationConfig, load_paper_config
from matdeplot.calibration import (
    CalibrationResult,
    calibrate_axes,
    generate_equidistant_ticks,
)
from matdeplot.curve_count import CurveCountResult, silhouette_curve_count
from matdeplot.document_pipeline import (
    DocumentPipeline,
    MinerUPDFParser,
    crop_subfigures,
)
from matdeplot.pipeline import MatDeplotPipeline
from matdeplot.schema import AxisDetection, AxisMapping, AxisTick, Box
from matdeplot.separation import prepare_pixel_mask


class _DummyMaskModel(nn.Module):
    def __init__(self, objectness_logits, mask_logits):
        super().__init__()
        self._parameter = nn.Parameter(torch.zeros(()))
        self._objectness_logits = objectness_logits
        self._mask_logits = mask_logits

    def forward(self, images):
        batch_size = images.shape[0]
        return {
            "objectness_logits": self._objectness_logits.repeat(batch_size, 1),
            "mask_logits": self._mask_logits.repeat(batch_size, 1, 1, 1),
        }


class FallbackContractsTest(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.model_directory = self.root / "models"
        self.model_directory.mkdir()
        self.image_path = self.write_image("chart.png")
        self.panel_path = self.write_image("panel.png")

    def tearDown(self):
        self.tempdir.cleanup()

    def write_image(self, relative_path: str, size=(32, 24), value=255) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.full((size[1], size[0], 3), value, dtype=np.uint8)
        image[:, :, 0] = np.arange(size[0], dtype=np.uint8)[None, :]
        image[:, :, 1] = np.arange(size[1], dtype=np.uint8)[:, None]
        image[4:-4, 6:-6, 2] = 128
        ok = cv2.imwrite(str(path), image)
        if not ok:
            raise AssertionError("failed to write image fixture")
        return path

    def make_pipeline(self) -> MatDeplotPipeline:
        return MatDeplotPipeline(self.model_directory)

    def make_axes(self) -> AxisDetection:
        return AxisDetection(
            x_axis=(2.0, 21.0, 29.0, 21.0),
            y_axis=(2.0, 2.0, 2.0, 21.0),
            plot_box=Box(x1=1.0, y1=1.0, x2=31.0, y2=23.0),
        )

    def make_axis_diagnostics(self):
        return {
            "primary_method": "axis_yolo",
            "primary_status": "success",
            "primary_error": None,
            "fallback_triggered": False,
            "fallback_method": "cv_vlm_axes",
            "fallback_status": "not_triggered",
            "fallback_error": None,
            "selected_method": "axis_yolo",
        }

    def make_curve_mask(self) -> np.ndarray:
        mask = np.zeros((24, 32), dtype=bool)
        mask[6:18, 10:22] = True
        return mask

    def test_paper_config_enables_six_point_synthetic_tick_fallback(self):
        calibration = load_paper_config().calibration
        self.assertTrue(calibration.synthetic_fallback)
        self.assertEqual(calibration.synthetic_tick_count, 6)

    def test_generate_equidistant_x_ticks_matches_six_point_contract(self):
        ticks = generate_equidistant_ticks((10, 80, 110, 80), "x", count=6)
        self.assertEqual(
            [tick.pixel for tick in ticks],
            [10, 30, 50, 70, 90, 110],
        )
        self.assertEqual(
            [tick.value for tick in ticks],
            [10, 30, 50, 70, 90, 110],
        )

    def test_generate_equidistant_y_ticks_reverses_values(self):
        ticks = generate_equidistant_ticks((20, 10, 20, 110), "y", count=6)
        self.assertEqual(
            [tick.pixel for tick in ticks],
            [10, 30, 50, 70, 90, 110],
        )
        self.assertEqual(
            [tick.value for tick in ticks],
            [110, 90, 70, 50, 30, 10],
        )

    def test_calibrate_axes_uses_synthetic_ticks_for_sparse_primary_ticks(self):
        primary = [
            [],
            [AxisTick(pixel=10.0, value=0.0)],
        ]
        with mock.patch(
            "matdeplot.calibration.extract_numeric_ticks",
            side_effect=primary,
        ):
            result = calibrate_axes(
                np.zeros((24, 32, 3), dtype=np.uint8),
                self.make_axes(),
                load_paper_config().calibration,
                ocr=mock.Mock(),
            )

        self.assertTrue(result.calibrated)
        self.assertEqual(result.diagnostics["x"]["primary_tick_count"], 0)
        self.assertEqual(result.diagnostics["y"]["primary_tick_count"], 1)
        self.assertTrue(result.diagnostics["x"]["synthetic_ticks_used"])
        self.assertTrue(result.diagnostics["y"]["synthetic_ticks_used"])
        self.assertEqual(len(result.x_mapping.ticks), 6)
        self.assertEqual(len(result.y_mapping.ticks), 6)

    def test_calibrate_axes_keeps_two_primary_ticks_without_fallback(self):
        ticks = [
            AxisTick(pixel=0.0, value=0.0),
            AxisTick(pixel=10.0, value=1.0),
        ]
        with mock.patch(
            "matdeplot.calibration.extract_numeric_ticks",
            side_effect=[ticks, ticks],
        ):
            result = calibrate_axes(
                np.zeros((24, 32, 3), dtype=np.uint8),
                self.make_axes(),
                load_paper_config().calibration,
                ocr=mock.Mock(),
            )

        self.assertTrue(result.calibrated)
        self.assertFalse(result.diagnostics["x"]["fallback_triggered"])
        self.assertFalse(result.diagnostics["y"]["fallback_triggered"])
        self.assertEqual(result.diagnostics["x"]["selected_method"], "ocr_ticks")

    def test_calibrate_axes_does_not_relabel_ocr_exception_as_sparse_ticks(self):
        with mock.patch(
            "matdeplot.calibration.extract_numeric_ticks",
            side_effect=RuntimeError("ocr unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "ocr unavailable"):
                calibrate_axes(
                    np.zeros((24, 32, 3), dtype=np.uint8),
                    self.make_axes(),
                    load_paper_config().calibration,
                    ocr=mock.Mock(),
                )

    def test_axis_fallback_selects_colored_horizontal_and_vertical_axes(self):
        from matdeplot.axis_fallback import detect_axes_cv_vlm

        client = mock.Mock()
        client.select_candidates.side_effect = [
            [(4, 3, 4, 21), (28, 3, 28, 21)],
            [(4, 21, 28, 21), (4, 3, 28, 3)],
        ]
        client.select_axes.return_value = {"x_axis": "red", "y_axis": "green"}
        candidates = [
            (4, 3, 4, 21),
            (28, 3, 28, 21),
            (4, 21, 28, 21),
            (4, 3, 28, 3),
        ]

        with mock.patch(
            "matdeplot.axis_fallback._hough_axis_candidates",
            return_value=candidates,
        ):
            result = detect_axes_cv_vlm(self.image_path, client=client)

        self.assertEqual(result.x_axis, (4.0, 21.0, 28.0, 21.0))
        self.assertEqual(result.y_axis, (4.0, 3.0, 4.0, 21.0))
        self.assertEqual(
            (
                result.plot_box.x1,
                result.plot_box.y1,
                result.plot_box.x2,
                result.plot_box.y2,
            ),
            (4.0, 3.0, 28.0, 21.0),
        )

    def test_axis_fallback_rejects_missing_orientations_before_vlm(self):
        from matdeplot.axis_fallback import detect_axes_cv_vlm

        client = mock.Mock()
        with mock.patch(
            "matdeplot.axis_fallback._hough_axis_candidates",
            return_value=[(4, 21, 28, 21)],
        ):
            with self.assertRaisesRegex(RuntimeError, "insufficient candidates"):
                detect_axes_cv_vlm(self.image_path, client=client)
        client.select_candidates.assert_not_called()

    def test_axis_fallback_malformed_response_does_not_expose_raw_text(self):
        from matdeplot.axis_fallback import AxisSelectionClient

        selector = object.__new__(AxisSelectionClient)
        selector.model = "test-model"
        selector.client = mock.Mock()
        choice = mock.Mock()
        choice.message.content = "private-provider-response"
        selector.client.chat.completions.create.return_value = mock.Mock(
            choices=[choice]
        )

        with self.assertRaisesRegex(RuntimeError, "not valid JSON") as raised:
            selector._complete(np.zeros((8, 8, 3), dtype=np.uint8), "prompt")
        self.assertNotIn("private-provider-response", str(raised.exception))

    def test_axis_fallback_rejects_unknown_selected_color(self):
        from matdeplot.axis_fallback import detect_axes_cv_vlm

        client = mock.Mock()
        client.select_candidates.side_effect = [
            [(4, 3, 4, 21)],
            [(4, 21, 28, 21)],
        ]
        client.select_axes.return_value = {"x_axis": "purple", "y_axis": "green"}
        candidates = [(4, 3, 4, 21), (4, 21, 28, 21)]

        with mock.patch(
            "matdeplot.axis_fallback._hough_axis_candidates",
            return_value=candidates,
        ):
            with self.assertRaisesRegex(RuntimeError, "unknown candidate"):
                detect_axes_cv_vlm(self.image_path, client=client)

    def test_axis_route_keeps_valid_yolo_result_without_fallback(self):
        from matdeplot.detectors import detect_axes_with_fallback

        fallback = mock.Mock()
        with mock.patch(
            "matdeplot.detectors.detect_axes",
            return_value=self.make_axes(),
        ):
            axes, diagnostics = detect_axes_with_fallback(
                self.image_path,
                self.model_directory / "axis.pt",
                fallback=fallback,
            )

        self.assertEqual(axes, self.make_axes())
        fallback.assert_not_called()
        self.assertEqual(diagnostics["selected_method"], "axis_yolo")
        self.assertFalse(diagnostics["fallback_triggered"])

    def test_axis_route_recovers_after_yolo_exception(self):
        from matdeplot.detectors import detect_axes_with_fallback

        fallback = mock.Mock(return_value=self.make_axes())
        with mock.patch(
            "matdeplot.detectors.detect_axes",
            side_effect=RuntimeError("axis detector failed"),
        ):
            axes, diagnostics = detect_axes_with_fallback(
                self.image_path,
                self.model_directory / "axis.pt",
                fallback=fallback,
            )

        self.assertEqual(axes, self.make_axes())
        fallback.assert_called_once_with(self.image_path)
        self.assertEqual(diagnostics["primary_status"], "failed")
        self.assertEqual(diagnostics["fallback_status"], "rescued")
        self.assertEqual(diagnostics["selected_method"], "cv_vlm_axes")

    def test_axis_route_recovers_after_empty_yolo_result(self):
        from matdeplot.detectors import detect_axes_with_fallback

        empty = AxisDetection(x_axis=None, y_axis=None, plot_box=None)
        fallback = mock.Mock(return_value=self.make_axes())
        with mock.patch("matdeplot.detectors.detect_axes", return_value=empty):
            axes, diagnostics = detect_axes_with_fallback(
                self.image_path,
                self.model_directory / "axis.pt",
                fallback=fallback,
            )

        self.assertEqual(axes, self.make_axes())
        self.assertEqual(diagnostics["primary_error"], "invalid_or_missing_axes")
        self.assertEqual(diagnostics["fallback_status"], "rescued")

    def test_pipeline_records_axis_failure_diagnostics_when_both_paths_fail(self):
        pipeline = self.make_pipeline()
        diagnostics = self.make_axis_diagnostics()
        diagnostics.update(
            primary_status="failed",
            primary_error="invalid_or_missing_axes",
            fallback_triggered=True,
            fallback_status="failed",
            fallback_error="RuntimeError: no valid axis candidates",
            selected_method=None,
        )

        with mock.patch(
            "matdeplot.pipeline.detect_axes_with_fallback",
            return_value=(None, diagnostics),
        ):
            result = pipeline.extract(self.image_path, calibrate=False)

        self.assertFalse(result.success)
        self.assertEqual(result.failures[-1].stage, "axis_detection")
        self.assertEqual(result.metadata["axis_detection"], diagnostics)

    def test_pipeline_records_synthetic_calibration_and_real_points(self):
        pipeline = self.make_pipeline()
        curve_mask = self.make_curve_mask()
        mapping = AxisMapping(
            kind="linear",
            coefficients=(1.0, 0.0),
            ticks=[
                AxisTick(pixel=0.0, value=0.0),
                AxisTick(pixel=10.0, value=10.0),
            ],
            calibrated=True,
        )
        calibration = CalibrationResult(
            x_mapping=mapping,
            y_mapping=mapping,
            diagnostics={
                "x": {"synthetic_ticks_used": True},
                "y": {"synthetic_ticks_used": True},
                "calibrated": True,
                "used_synthetic_ticks": True,
            },
        )

        with mock.patch(
            "matdeplot.pipeline.detect_axes_with_fallback",
            return_value=(self.make_axes(), self.make_axis_diagnostics()),
        ), mock.patch(
            "matdeplot.pipeline.detect_legends",
            return_value=[],
        ), mock.patch(
            "matdeplot.pipeline.get_line_prob_multiscale",
            return_value=np.full((24, 32), 0.8, dtype=np.float32),
        ), mock.patch.object(
            pipeline,
            "_resolve_curve_count",
            return_value=CurveCountResult(value=1, source="silhouette"),
        ), mock.patch(
            "matdeplot.pipeline.separate_curves",
            return_value=[curve_mask],
        ), mock.patch(
            "matdeplot.pipeline.mask_to_polyline",
            return_value=[(1.0, 2.0), (3.0, 4.0)],
        ), mock.patch(
            "matdeplot.pipeline.calibrate_axes",
            return_value=calibration,
        ):
            result = pipeline.extract(self.image_path, calibrate=True)

        self.assertTrue(result.success)
        self.assertEqual(
            result.metadata["axis_detection"]["selected_method"],
            "axis_yolo",
        )
        self.assertTrue(result.metadata["calibration"]["used_synthetic_ticks"])
        self.assertNotEqual(result.curves[0].real_points, [])
        self.assertIsInstance(json.dumps(result.to_dict()), str)

    def test_crop_subfigures_empty_detector_returns_full_image_crop(self):
        output_dir = self.root / "crops"
        with mock.patch(
            "matdeplot.document_pipeline.detect_subfigures",
            return_value=[],
        ):
            panels = crop_subfigures(
                [self.image_path],
                checkpoint=self.model_directory / "subfigure.pt",
                output_directory=output_dir,
            )

        self.assertEqual(len(panels), 1)
        crop = cv2.imread(str(panels[0]), cv2.IMREAD_COLOR)
        original = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        self.assertEqual(tuple(crop.shape), tuple(original.shape))
        self.assertTrue(np.array_equal(crop, original))

    def test_crop_subfigures_detector_exception_propagates(self):
        with mock.patch(
            "matdeplot.document_pipeline.detect_subfigures",
            side_effect=RuntimeError("subfigure detector failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "subfigure detector failed"):
                crop_subfigures(
                    [self.image_path],
                    checkpoint=self.model_directory / "subfigure.pt",
                    output_directory=self.root / "crops",
                )

    def test_document_pipeline_pdf_mineru_unavailable_is_terminal(self):
        parser = MinerUPDFParser()
        pdf_path = self.root / "input.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%stub\n")

        import builtins

        real_import = builtins.__import__
        magic_pdf_keys = {
            key: None for key in list(sys.modules) if key == "magic_pdf" or key.startswith("magic_pdf.")
        }
        magic_pdf_keys.setdefault("magic_pdf", None)

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.startswith("magic_pdf"):
                raise ImportError("missing MinerU")
            return real_import(name, globals, locals, fromlist, level)

        with mock.patch.dict(sys.modules, magic_pdf_keys, clear=False):
            with mock.patch("builtins.__import__", side_effect=fake_import):
                with self.assertRaisesRegex(RuntimeError, "MinerU is not installed"):
                    parser.parse(pdf_path, self.root / "mineru")

    def test_document_pipeline_pdf_parse_failure_is_terminal(self):
        pipeline = DocumentPipeline(self.model_directory, classifier=mock.Mock())
        pdf_path = self.root / "input.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%stub\n")

        with mock.patch(
            "matdeplot.document_pipeline.MinerUPDFParser.parse",
            side_effect=RuntimeError("mineru parse failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "mineru parse failed"):
                pipeline.run(pdf_path, self.root / "out")

    def test_document_pipeline_missing_classifier_is_terminal_without_assume_flag(self):
        pipeline = DocumentPipeline(self.model_directory, classifier=None)
        figure_dir = self.root / "figures"
        figure_dir.mkdir()
        figure_path = self.write_image("figures/figure.png")

        with mock.patch(
            "matdeplot.document_pipeline.crop_subfigures",
            return_value=[figure_path],
        ):
            with self.assertRaisesRegex(RuntimeError, "Stage-2 classifier is required"):
                pipeline.run(figure_dir, self.root / "out", assume_line_charts=False)

    def test_document_pipeline_assume_line_charts_bypasses_classifier(self):
        classifier = mock.Mock()
        pipeline = DocumentPipeline(self.model_directory, classifier=classifier)
        figure_dir = self.root / "figures"
        figure_dir.mkdir(exist_ok=True)
        self.write_image("figures/figure.png")
        extraction = mock.Mock(success=True)

        with mock.patch(
            "matdeplot.document_pipeline.crop_subfigures",
            return_value=[self.panel_path],
        ), mock.patch.object(
            pipeline.chart_pipeline,
            "extract",
            return_value=extraction,
        ) as extract_mock, mock.patch(
            "matdeplot.document_pipeline.write_extraction",
            return_value={"record": self.root / "out" / "curves" / "panel" / "extraction.json"},
        ):
            result = pipeline.run(
                figure_dir,
                self.root / "out",
                assume_line_charts=True,
            )

        classifier.classify.assert_not_called()
        extract_mock.assert_called_once()
        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest[0]["classification"]["reasoning"], "Classification bypassed by explicit CLI option.")
        self.assertEqual(result["panels"], 1)

    def test_document_pipeline_classifier_failure_is_terminal(self):
        classifier = mock.Mock()
        classifier.classify.side_effect = RuntimeError("classifier down")
        pipeline = DocumentPipeline(self.model_directory, classifier=classifier)
        figure_dir = self.root / "figures"
        figure_dir.mkdir(exist_ok=True)
        self.write_image("figures/figure.png")

        with mock.patch(
            "matdeplot.document_pipeline.crop_subfigures",
            return_value=[self.panel_path],
        ):
            with self.assertRaisesRegex(RuntimeError, "classifier down"):
                pipeline.run(figure_dir, self.root / "out")

    def test_extract_legend_exception_records_failure_and_continues(self):
        pipeline = self.make_pipeline()
        curve_mask = self.make_curve_mask()

        with mock.patch(
            "matdeplot.pipeline.detect_axes_with_fallback",
            return_value=(self.make_axes(), self.make_axis_diagnostics()),
        ), mock.patch(
            "matdeplot.pipeline.detect_legends",
            side_effect=RuntimeError("legend unavailable"),
        ), mock.patch(
            "matdeplot.pipeline.get_line_prob_multiscale",
            return_value=np.full((24, 32), 0.8, dtype=np.float32),
        ), mock.patch.object(
            pipeline,
            "_resolve_curve_count",
            return_value=CurveCountResult(value=1, source="silhouette"),
        ), mock.patch(
            "matdeplot.pipeline.separate_curves",
            return_value=[curve_mask],
        ), mock.patch(
            "matdeplot.pipeline.mask_to_polyline",
            return_value=[(1.0, 2.0), (3.0, 4.0)],
        ):
            result = pipeline.extract(self.image_path, calibrate=False)

        self.assertTrue(result.success)
        self.assertEqual(result.legend_boxes, [])
        self.assertEqual(
            [(failure.stage, failure.exception_type) for failure in result.failures],
            [("legend_detection", "RuntimeError")],
        )
        self.assertEqual(len(result.curves), 1)

    def test_extract_axis_exception_returns_failure_result(self):
        pipeline = self.make_pipeline()
        diagnostics = self.make_axis_diagnostics()
        diagnostics.update(
            primary_status="failed",
            primary_error="RuntimeError: axis detector failed",
            fallback_triggered=True,
            fallback_status="failed",
            fallback_error="RuntimeError: axis fallback failed",
            selected_method=None,
        )

        with mock.patch(
            "matdeplot.pipeline.detect_axes_with_fallback",
            return_value=(None, diagnostics),
        ):
            result = pipeline.extract(self.image_path, calibrate=False)

        self.assertFalse(result.success)
        self.assertEqual(result.failures[-1].stage, "axis_detection")
        self.assertEqual(result.failures[-1].exception_type, "RuntimeError")
        self.assertEqual(result.metadata["axis_detection"], diagnostics)

    def test_extract_foreground_exception_returns_failure_result(self):
        pipeline = self.make_pipeline()

        with mock.patch(
            "matdeplot.pipeline.detect_axes_with_fallback",
            return_value=(self.make_axes(), self.make_axis_diagnostics()),
        ), mock.patch(
            "matdeplot.pipeline.detect_legends",
            return_value=[],
        ), mock.patch(
            "matdeplot.pipeline.get_line_prob_multiscale",
            side_effect=RuntimeError("foreground model failed"),
        ):
            result = pipeline.extract(self.image_path, calibrate=False)

        self.assertFalse(result.success)
        self.assertEqual(result.failures[-1].stage, "foreground")
        self.assertEqual(result.failures[-1].exception_type, "RuntimeError")

    def test_mask2former_low_foreground_uses_paper_nonwhite_pixel_fallback(self):
        """Mask2Former stays primary; only low foreground activates the fallback."""
        pipeline = self.make_pipeline()
        curve_mask = self.make_curve_mask()
        fallback_mask = np.ones((24, 32), dtype=bool)

        with mock.patch(
            "matdeplot.pipeline.detect_axes_with_fallback",
            return_value=(self.make_axes(), self.make_axis_diagnostics()),
        ), mock.patch(
            "matdeplot.pipeline.detect_legends",
            return_value=[],
        ), mock.patch(
            "matdeplot.pipeline.get_line_prob_multiscale",
            return_value=np.zeros((24, 32), dtype=np.float32),
        ) as mask2former_mock, mock.patch(
            "matdeplot.pipeline.prepare_pixel_mask",
            return_value=fallback_mask,
        ) as prepare_mock, mock.patch.object(
            pipeline,
            "_resolve_curve_count",
            return_value=CurveCountResult(value=1, source="silhouette"),
        ), mock.patch(
            "matdeplot.pipeline.separate_curves",
            return_value=[curve_mask],
        ) as separation_mock, mock.patch(
            "matdeplot.pipeline.mask_to_polyline",
            return_value=[(1.0, 1.0), (2.0, 2.0)],
        ):
            result = pipeline.extract(self.image_path, calibrate=False)

        self.assertTrue(result.success)
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.foreground_fraction, 0.0)
        self.assertEqual(pipeline.config.foreground.fallback_fraction, 0.003)
        mask2former_mock.assert_called_once()
        self.assertTrue(prepare_mock.call_args.args[3])
        separation_mock.assert_called_once()
        self.assertTrue(
            np.array_equal(separation_mock.call_args.args[1], fallback_mask)
        )

    def test_prepare_pixel_mask_matches_direct_boolean_branches_without_morphology(self):
        image = np.array(
            [
                [[255, 255, 255], [230, 230, 230], [250, 250, 250]],
                [[10, 10, 10], [241, 241, 241], [100, 100, 100]],
            ],
            dtype=np.uint8,
        )
        foreground_mask = np.array(
            [
                [True, False, True],
                [False, True, True],
            ],
            dtype=bool,
        )
        valid_region = np.array(
            [
                [True, True, False],
                [True, False, True],
            ],
            dtype=bool,
        )
        config = SeparationConfig(
            mode="v4",
            spatial_weight=0.5,
            kmeans_n_init=5,
            random_state=42,
            morph_close_kernel=0,
            dilation_kernel=0,
            min_cluster_area=1,
            delta_e_threshold=12.0,
        )

        fallback_mask = prepare_pixel_mask(
            image,
            foreground_mask,
            valid_region,
            True,
            config,
        )
        direct_fallback = valid_region & (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) < 240)
        self.assertTrue(np.array_equal(fallback_mask, direct_fallback))

        primary_mask = prepare_pixel_mask(
            image,
            foreground_mask,
            valid_region,
            False,
            config,
        )
        self.assertTrue(np.array_equal(primary_mask, foreground_mask & valid_region))

    def test_silhouette_curve_count_returns_too_few_foreground_pixels_fallback(self):
        config = load_paper_config()
        image = np.zeros((5, 5, 3), dtype=np.uint8)
        pixel_mask = np.zeros((5, 5), dtype=bool)
        pixel_mask[:2, :5] = True
        pixel_mask[2, :4] = True

        result = silhouette_curve_count(
            image,
            pixel_mask,
            config.curve_count,
            config.separation,
        )

        self.assertEqual(result.value, 1)
        self.assertEqual(result.source, "silhouette")
        self.assertEqual(result.fallback_reason, "too_few_foreground_pixels")

    def test_curve_count_exception_uses_silhouette_fallback_and_records_reason(self):
        pipeline = self.make_pipeline()
        curve_mask = self.make_curve_mask()

        with mock.patch(
            "matdeplot.pipeline.detect_axes_with_fallback",
            return_value=(self.make_axes(), self.make_axis_diagnostics()),
        ), mock.patch(
            "matdeplot.pipeline.detect_legends",
            return_value=[],
        ), mock.patch(
            "matdeplot.pipeline.get_line_prob_multiscale",
            return_value=np.full((24, 32), 0.8, dtype=np.float32),
        ), mock.patch(
            "matdeplot.pipeline.prepare_pixel_mask",
            return_value=np.ones((24, 32), dtype=bool),
        ), mock.patch.object(
            pipeline,
            "_resolve_curve_count",
            side_effect=RuntimeError("count provider failed"),
        ), mock.patch(
            "matdeplot.pipeline.silhouette_curve_count",
            return_value=CurveCountResult(value=2, source="silhouette"),
        ) as silhouette_mock, mock.patch(
            "matdeplot.pipeline.separate_curves",
            return_value=[curve_mask],
        ), mock.patch(
            "matdeplot.pipeline.mask_to_polyline",
            return_value=[(1.0, 1.0), (2.0, 2.0)],
        ):
            result = pipeline.extract(self.image_path, curve_count_source="gemini", calibrate=False)

        silhouette_mock.assert_called_once()
        self.assertTrue(result.success)
        self.assertEqual(result.curve_count_source, "silhouette")
        self.assertEqual(result.metadata["curve_count_fallback_reason"], "RuntimeError: count provider failed")
        self.assertEqual(result.failures[-1].stage, "curve_count")

    def test_calibration_failure_keeps_pixel_points_and_empty_real_points(self):
        pipeline = self.make_pipeline()
        curve_mask = self.make_curve_mask()
        points = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]

        with mock.patch(
            "matdeplot.pipeline.detect_axes_with_fallback",
            return_value=(self.make_axes(), self.make_axis_diagnostics()),
        ), mock.patch(
            "matdeplot.pipeline.detect_legends",
            return_value=[],
        ), mock.patch(
            "matdeplot.pipeline.get_line_prob_multiscale",
            return_value=np.full((24, 32), 0.8, dtype=np.float32),
        ), mock.patch.object(
            pipeline,
            "_resolve_curve_count",
            return_value=CurveCountResult(value=1, source="silhouette"),
        ), mock.patch(
            "matdeplot.pipeline.separate_curves",
            return_value=[curve_mask],
        ), mock.patch(
            "matdeplot.pipeline.mask_to_polyline",
            return_value=points,
        ), mock.patch(
            "matdeplot.pipeline.calibrate_axes",
            side_effect=RuntimeError("ocr unavailable"),
        ):
            result = pipeline.extract(self.image_path, calibrate=True)

        self.assertTrue(result.success)
        self.assertEqual(result.curves[0].pixel_points, points)
        self.assertEqual(result.curves[0].real_points, [])
        self.assertEqual(result.failures[-1].stage, "calibration")
        self.assertEqual(result.failures[-1].exception_type, "RuntimeError")

    def test_uncalibrated_calibration_result_adds_failure_and_keeps_pixel_points(self):
        pipeline = self.make_pipeline()
        curve_mask = self.make_curve_mask()
        points = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
        unavailable = AxisMapping(
            kind="unavailable",
            coefficients=(1.0, 0.0),
            ticks=[],
            calibrated=False,
        )
        calibration = CalibrationResult(
            x_mapping=unavailable,
            y_mapping=unavailable,
        )

        with mock.patch(
            "matdeplot.pipeline.detect_axes_with_fallback",
            return_value=(self.make_axes(), self.make_axis_diagnostics()),
        ), mock.patch(
            "matdeplot.pipeline.detect_legends",
            return_value=[],
        ), mock.patch(
            "matdeplot.pipeline.get_line_prob_multiscale",
            return_value=np.full((24, 32), 0.8, dtype=np.float32),
        ), mock.patch.object(
            pipeline,
            "_resolve_curve_count",
            return_value=CurveCountResult(value=1, source="silhouette"),
        ), mock.patch(
            "matdeplot.pipeline.separate_curves",
            return_value=[curve_mask],
        ), mock.patch(
            "matdeplot.pipeline.mask_to_polyline",
            return_value=points,
        ), mock.patch(
            "matdeplot.pipeline.calibrate_axes",
            return_value=calibration,
        ):
            result = pipeline.extract(self.image_path, calibrate=True)

        self.assertTrue(result.success)
        self.assertEqual(result.curves[0].pixel_points, points)
        self.assertEqual(result.curves[0].real_points, [])
        self.assertEqual(result.failures[-1].stage, "calibration")
        self.assertEqual(
            result.failures[-1].message,
            "Fewer than two reliable ticks were found on an axis",
        )

    def test_confidence_below_threshold_sets_abstain(self):
        pipeline = self.make_pipeline()
        curve_mask = self.make_curve_mask()

        with mock.patch(
            "matdeplot.pipeline.detect_axes_with_fallback",
            return_value=(self.make_axes(), self.make_axis_diagnostics()),
        ), mock.patch(
            "matdeplot.pipeline.detect_legends",
            return_value=[],
        ), mock.patch(
            "matdeplot.pipeline.get_line_prob_multiscale",
            return_value=np.full((24, 32), 0.8, dtype=np.float32),
        ), mock.patch.object(
            pipeline,
            "_resolve_curve_count",
            return_value=CurveCountResult(value=1, source="silhouette"),
        ), mock.patch(
            "matdeplot.pipeline.separate_curves",
            return_value=[curve_mask],
        ), mock.patch(
            "matdeplot.pipeline.mask_to_polyline",
            return_value=[(1.0, 1.0), (2.0, 2.0)],
        ), mock.patch.object(
            pipeline,
            "_curve_confidence",
            return_value=0.2,
        ):
            result = pipeline.extract(self.image_path, calibrate=False)

        self.assertEqual(load_paper_config().abstention.threshold, 0.3)
        self.assertTrue(result.curves[0].abstain)
        self.assertEqual(result.curves[0].confidence, 0.2)

    def test_get_line_binary_mask_keeps_max_objectness_query_when_none_cross_threshold(self):
        from matdeplot.foreground import get_line_binary_mask

        objectness = torch.tensor([[-6.0, -5.0]], dtype=torch.float32)
        mask_logits = torch.tensor(
            [
                [
                    [[-10.0, -10.0], [-10.0, -10.0]],
                    [[10.0, -10.0], [-10.0, -10.0]],
                ]
            ],
            dtype=torch.float32,
        )
        image = np.zeros((2, 2, 3), dtype=np.uint8)

        with mock.patch(
            "matdeplot.foreground.load_model",
            return_value=_DummyMaskModel(objectness, mask_logits),
        ):
            mask = get_line_binary_mask(
                image,
                ckpt_path=self.model_directory / "foreground.pt",
                infer_size=2,
                threshold=0.5,
                obj_threshold=0.5,
                device="cpu",
            )

        expected = np.array([[True, False], [False, False]])
        self.assertTrue(np.array_equal(mask, expected))

    def test_failure_and_fallback_doc_exists_and_states_non_accuracy_boundary(self):
        path = ROOT / "docs" / "FAILURE_AND_FALLBACKS.md"
        self.assertTrue(path.is_file(), "docs/FAILURE_AND_FALLBACKS.md must exist")

        text = path.read_text(encoding="utf-8")
        required_snippets = [
            "fallback_fraction=0.003",
            "abstention.threshold=0.3",
            "axis_yolo",
            "cv_vlm_axes",
            "equidistant_ticks",
            "synthetic_ticks_used",
            "code paths for recovery, explicit failure reporting,",
            "Their presence does not establish an accuracy",
        ]
        for snippet in required_snippets:
            self.assertIn(snippet, text)


if __name__ == "__main__":
    unittest.main()
