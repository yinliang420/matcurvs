import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.matdeplot.schema import CurveRecord

from src.matdeplot.judge import (
    CRITERION_NAMES,
    JUDGE_PROMPT,
    JudgeVerdict,
    QwenExtractionJudge,
    load_curves_from_extraction,
    parse_judge_response,
    render_extracted_curves,
)


_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO"
    "WgmWQAAAAASUVORK5CYII="
)
_JUDGE_PROMPT_SHA256 = "0dfe918cdad5d7327b11194ae087ab2f60867163e27b0d4bbd501f3391c372f5"


class _FakeResponse:
    def __init__(self, content):
        self.choices = [mock.Mock(message=mock.Mock(content=content))]


class _FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []
        self.chat = mock.Mock()
        self.chat.completions = mock.Mock()
        self.chat.completions.create = self._create

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class JudgeCoreTest(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def make_curve(self, index=0, real_points=None, pixel_points=None):
        return CurveRecord(
            curve_index=index,
            pixel_points=pixel_points or [(0.0, 0.0), (1.0, 1.0)],
            real_points=real_points or [(10.0, 20.0), (11.0, 21.0)],
            confidence=0.9,
            abstain=False,
        )

    def write_image(self, name="chart.png"):
        path = self.root / name
        path.write_bytes(_PNG_1X1)
        return path

    def test_parse_accepts_all_three_true_and_ignores_model_decision(self):
        verdict = parse_judge_response(
            json.dumps(
                {
                    "n_left": 2,
                    "n_right": 2,
                    "decision": "bad",
                    "curve_count_match": True,
                    "shape_match": True,
                    "range_plausible": True,
                    "reason": "all criteria passed",
                }
            ),
            model="qwen-vl-max",
            latency_seconds=1.25,
        )

        self.assertTrue(verdict.admitted)
        self.assertEqual(verdict.n_left, 2)
        self.assertEqual(verdict.n_right, 2)
        self.assertEqual(verdict.reason, "all criteria passed")
        self.assertEqual(verdict.latency_seconds, 1.25)
        self.assertIsNone(verdict.error)
        self.assertEqual(
            CRITERION_NAMES,
            ("curve_count_match", "shape_match", "range_plausible"),
        )
        self.assertEqual(
            verdict.to_dict(),
            {
                "model": "qwen-vl-max",
                "admitted": True,
                "n_left": 2,
                "n_right": 2,
                "curve_count_match": True,
                "shape_match": True,
                "range_plausible": True,
                "reason": "all criteria passed",
                "latency_seconds": 1.25,
                "error": None,
            },
        )

    def test_parse_rejects_any_false_even_if_model_says_good(self):
        verdict = parse_judge_response(
            json.dumps(
                {
                    "n_left": 2,
                    "n_right": 2,
                    "decision": "good",
                    "curve_count_match": True,
                    "shape_match": False,
                    "range_plausible": True,
                    "reason": "shape mismatch",
                }
            ),
            model="qwen-vl-max",
        )

        self.assertFalse(verdict.admitted)
        self.assertFalse(verdict.shape_match)
        self.assertIsNone(verdict.error)

    def test_parse_recomputes_curve_count_match_locally(self):
        verdict = parse_judge_response(
            json.dumps(
                {
                    "n_left": 2,
                    "n_right": 3,
                    "decision": "good",
                    "curve_count_match": True,
                    "shape_match": True,
                    "range_plausible": True,
                    "reason": "provider contradicted the counts",
                }
            ),
            model="qwen-vl-max",
        )

        self.assertFalse(verdict.admitted)
        self.assertFalse(verdict.curve_count_match)
        self.assertEqual(verdict.n_left, 2)
        self.assertEqual(verdict.n_right, 3)

    def test_parse_rejects_provider_false_curve_count_even_when_counts_match(self):
        verdict = parse_judge_response(
            json.dumps(
                {
                    "n_left": 2,
                    "n_right": 2,
                    "decision": "good",
                    "curve_count_match": False,
                    "shape_match": True,
                    "range_plausible": True,
                    "reason": "provider flagged a count mismatch",
                }
            ),
            model="qwen-vl-max",
        )

        self.assertFalse(verdict.admitted)
        self.assertFalse(verdict.curve_count_match)
        self.assertEqual(verdict.n_left, 2)
        self.assertEqual(verdict.n_right, 2)

    def test_parse_accepts_markdown_fenced_json(self):
        verdict = parse_judge_response(
            """```json
            {"n_left": 1, "n_right": 1, "decision": "good",
             "curve_count_match": true, "shape_match": true,
             "range_plausible": true,
             "reason": "ok"}
            ```""",
            model="qwen-vl-max",
        )

        self.assertTrue(verdict.admitted)
        self.assertEqual(verdict.reason, "ok")

    def test_parse_fails_closed_on_empty_no_json_and_invalid_json(self):
        cases = [
            ("", "empty_response"),
            ("no structured output", "no_json"),
            ('{"n_left": 1,', "invalid_json"),
        ]
        for raw, error in cases:
            with self.subTest(error=error):
                verdict = parse_judge_response(raw, model="qwen-vl-max")
                self.assertFalse(verdict.admitted)
                self.assertEqual(verdict.error, error)

    def test_parse_fails_closed_on_missing_or_invalid_fields(self):
        cases = [
            {"n_right": 1, "decision": "good", "curve_count_match": True, "shape_match": True, "range_plausible": True, "reason": "missing left"},
            {"n_left": "2", "n_right": 1, "decision": "good", "curve_count_match": True, "shape_match": True, "range_plausible": True, "reason": "left string"},
            {"n_left": -1, "n_right": 1, "decision": "good", "curve_count_match": False, "shape_match": True, "range_plausible": True, "reason": "negative left"},
            {"n_left": 1, "n_right": 1, "decision": "good", "curve_count_match": "yes", "shape_match": True, "range_plausible": True, "reason": "non-bool"},
            {"n_left": 1, "n_right": 1, "decision": "good", "curve_count_match": True, "shape_match": True, "reason": "missing criterion"},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                verdict = parse_judge_response(json.dumps(payload), model="qwen-vl-max")
                self.assertFalse(verdict.admitted)
                self.assertEqual(verdict.error, "missing_or_invalid_fields")

    def test_public_judge_prompt_matches_canonical_release_bytes(self):
        self.assertEqual(
            hashlib.sha256(JUDGE_PROMPT.encode("utf-8")).hexdigest(),
            _JUDGE_PROMPT_SHA256,
        )
        self.assertIn("2θ", JUDGE_PROMPT)
        self.assertIn("5°-90°", JUDGE_PROMPT)
        self.assertIn("cm⁻¹", JUDGE_PROMPT)
        self.assertIn("spurious spikes", JUDGE_PROMPT)

    def test_render_extracted_curves_returns_png_bytes(self):
        rendered = render_extracted_curves(
            [
                self.make_curve(
                    index=0,
                    pixel_points=[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
                    real_points=[(1.0, 4.0), (2.0, 5.0), (3.0, 6.0)],
                ),
                self.make_curve(
                    index=1,
                    pixel_points=[(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
                    real_points=[(1.0, 1.0), (2.0, 1.5), (3.0, 2.0)],
                ),
            ]
        )

        self.assertTrue(rendered.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertGreater(len(rendered), 100)

    def test_render_extracted_curves_rejects_invalid_calibrated_curves(self):
        cases = [
            ([], "at least one curve"),
            ([self.make_curve(real_points=[(1.0, 2.0)])], "at least two real points"),
            ([self.make_curve(real_points=[(1.0, 2.0), (2.0, 3.0)], pixel_points=[(0.0, 0.0)])], "same length"),
            ([self.make_curve(real_points=[(1.0, 2.0), (float("nan"), 3.0)])], "finite"),
        ]
        for curves, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    render_extracted_curves(curves)

    def test_load_curves_from_extraction_reads_schema_records(self):
        extraction_path = self.root / "extraction.json"
        extraction_path.write_text(
            json.dumps(
                {
                    "image_id": "panel_1",
                    "curves": [
                        {
                            "curve_index": 0,
                            "pixel_points": [[1.0, 2.0], [3.0, 4.0]],
                            "real_points": [[10.0, 20.0], [30.0, 40.0]],
                            "confidence": 0.8,
                            "abstain": False,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        curves = load_curves_from_extraction(extraction_path)

        self.assertEqual(len(curves), 1)
        self.assertIsInstance(curves[0], CurveRecord)
        self.assertEqual(curves[0].curve_index, 0)
        self.assertEqual(curves[0].real_points, [(10.0, 20.0), (30.0, 40.0)])

    def test_load_curves_from_extraction_rejects_malformed_top_level_and_missing_keys(self):
        cases = [
            ([{"curve_index": 0}], "top-level payload must be an object"),
            ({"curves": "not-a-list"}, "curves must be a list"),
            ({"curves": [1]}, "each curve entry must be an object"),
            (
                {
                    "curves": [
                        {
                            "curve_index": 0,
                            "pixel_points": [[1.0, 2.0], [3.0, 4.0]],
                            "real_points": [[10.0, 20.0], [30.0, 40.0]],
                            "confidence": 0.8,
                        }
                    ]
                },
                r"missing required key\(s\): abstain",
            ),
        ]
        for payload, message in cases:
            with self.subTest(message=message):
                extraction_path = self.root / "bad_extraction.json"
                extraction_path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_curves_from_extraction(extraction_path)

    def test_load_curves_from_extraction_rejects_stringly_typed_fields(self):
        base_curve = {
            "curve_index": 0,
            "pixel_points": [[1.0, 2.0], [3.0, 4.0]],
            "real_points": [[10.0, 20.0], [30.0, 40.0]],
            "confidence": 0.8,
            "abstain": False,
        }
        cases = [
            ({"curve_index": "3"}, "curve_index must be an int"),
            ({"confidence": "0.75"}, "confidence must be a finite number"),
            ({"abstain": "false"}, "abstain must be a bool"),
            ({"pixel_points": [["1.0", 2.0], [3.0, 4.0]]}, "pixel_points must contain finite numeric"),
            ({"real_points": [[10.0, 20.0], [30.0, "40.0"]]}, "real_points must contain finite numeric"),
        ]
        for patch, message in cases:
            with self.subTest(message=message):
                payload = {"curves": [dict(base_curve, **patch)]}
                extraction_path = self.root / "typed_extraction.json"
                extraction_path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_curves_from_extraction(extraction_path)

    def test_review_returns_missing_key_before_rendering(self):
        image_path = self.write_image()
        judge = QwenExtractionJudge(api_key=None)
        curve = self.make_curve(real_points=[])

        with mock.patch("src.matdeplot.judge.os.environ", {}), mock.patch(
            "src.matdeplot.judge.render_extracted_curves",
            side_effect=AssertionError("render should not run"),
        ):
            verdict = judge.review(image_path, [curve])

        self.assertFalse(verdict.admitted)
        self.assertEqual(verdict.error, "missing_DASHSCOPE_API_KEY")

    def test_review_returns_structured_input_errors(self):
        missing_path = self.root / "missing.png"
        unsupported_path = self.root / "chart.gif"
        unsupported_path.write_bytes(_PNG_1X1)
        judge = QwenExtractionJudge(api_key="secret")

        missing_verdict = judge.review(
            missing_path,
            [self.make_curve(real_points=[(1.0, 1.0), (2.0, 2.0)])],
        )
        unsupported_verdict = judge.review(
            unsupported_path,
            [self.make_curve(real_points=[(1.0, 1.0), (2.0, 2.0)])],
        )

        self.assertFalse(missing_verdict.admitted)
        self.assertEqual(missing_verdict.error, "input:FileNotFoundError")
        self.assertFalse(unsupported_verdict.admitted)
        self.assertEqual(unsupported_verdict.error, "input:ValueError")

    def test_review_returns_structured_render_error(self):
        image_path = self.write_image()
        judge = QwenExtractionJudge(api_key="secret")

        with mock.patch(
            "src.matdeplot.judge.render_extracted_curves",
            side_effect=ValueError("bad curves"),
        ):
            verdict = judge.review(
                image_path,
                [self.make_curve(real_points=[(1.0, 1.0), (2.0, 2.0)])],
            )

        self.assertFalse(verdict.admitted)
        self.assertEqual(verdict.error, "render:ValueError")
        self.assertEqual(verdict.reason, "bad curves")

    def test_review_returns_accepted_verdict_with_mocked_api(self):
        image_path = self.write_image()
        fake_client = _FakeClient(
            response=_FakeResponse(
                json.dumps(
                    {
                        "n_left": 2,
                        "n_right": 2,
                        "decision": "good",
                        "curve_count_match": True,
                        "shape_match": True,
                        "range_plausible": True,
                        "reason": "looks consistent",
                    }
                )
            )
        )
        judge = QwenExtractionJudge(api_key="secret", client=fake_client, timeout=12, retries=0)

        verdict = judge.review(
            image_path,
            [
                self.make_curve(index=0, real_points=[(1.0, 1.0), (2.0, 2.0)]),
                self.make_curve(index=1, real_points=[(1.0, 2.0), (2.0, 3.0)]),
            ],
        )

        self.assertTrue(verdict.admitted)
        self.assertEqual(verdict.reason, "looks consistent")
        self.assertEqual(len(fake_client.calls), 1)
        request = fake_client.calls[0]
        self.assertEqual(request["model"], "qwen-vl-max")
        content = request["messages"][0]["content"]
        self.assertEqual([entry["type"] for entry in content], ["image_url", "image_url", "text"])
        self.assertTrue(content[0]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_review_retries_parse_failures_until_valid_response(self):
        image_path = self.write_image()
        fake_client = _FakeClient(
            response=None,
        )
        fake_client.responses = [
            _FakeResponse("not json"),
            _FakeResponse(
                json.dumps(
                    {
                        "n_left": 1,
                        "n_right": 1,
                        "decision": "good",
                        "curve_count_match": True,
                        "shape_match": True,
                        "range_plausible": True,
                        "reason": "retry succeeded",
                    }
                )
            ),
        ]

        def create_with_sequence(**kwargs):
            fake_client.calls.append(kwargs)
            return fake_client.responses.pop(0)

        fake_client.chat.completions.create = create_with_sequence
        judge = QwenExtractionJudge(api_key="secret", client=fake_client, retries=1)

        verdict = judge.review(
            image_path,
            [self.make_curve(real_points=[(1.0, 1.0), (2.0, 2.0)])],
        )

        self.assertTrue(verdict.admitted)
        self.assertEqual(verdict.reason, "retry succeeded")
        self.assertEqual(len(fake_client.calls), 2)

    def test_review_fails_closed_on_provider_error(self):
        image_path = self.write_image()
        fake_client = _FakeClient(error=RuntimeError("provider down"))
        judge = QwenExtractionJudge(api_key="secret", client=fake_client, retries=0)

        verdict = judge.review(
            image_path,
            [self.make_curve(real_points=[(1.0, 1.0), (2.0, 2.0)])],
        )

        self.assertFalse(verdict.admitted)
        self.assertEqual(verdict.error, "provider_error")
        self.assertEqual(verdict.reason, "provider down")


if __name__ == "__main__":
    unittest.main()
