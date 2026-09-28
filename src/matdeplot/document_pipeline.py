"""Paper-aligned PDF-to-curve workflow for MatDeplot Stages 1--3."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
from openai import OpenAI

from .curve_count import load_cached_counts
from .detectors import detect_subfigures, read_image
from .judge import JudgeVerdict, render_extracted_curves
from .output import write_extraction
from .pipeline import MatDeplotPipeline
from .schema import Box, ExtractionResult


CHART_TYPE_PROMPT = """
Classify the scientific panel by its visual encoding. First inspect axes,
marks, legends, and connectivity, then return strict JSON:
{
  "analysis": "brief visual evidence",
  "chart_type": "line chart"
}
The chart_type must be exactly one of: line chart, bar chart, pie chart,
scatter plot, heatmap, other. XRD patterns and continuous electrochemical
curves are line charts. Microscopy and elemental maps are other.
""".strip()

DATA_ORIGIN_PROMPT = """
Classify the scientific panel using this hierarchy.
First level: Experimental Data, Computational Data, or
Schematics / Illustrations.
Second level:
- Experimental Data: Microscopy, Structural Characterization,
  Compositional Analysis, Performance
- Computational Data: DFT, MD, FEM
- Schematics / Illustrations: Structural diagrams, Process flow diagrams,
  Conceptual illustrations
Return strict JSON with first_level, second_level, confidence, and a brief
reasoning field.
""".strip()


@dataclass(frozen=True)
class PanelClassification:
    chart_type: str
    first_level: str
    second_level: str
    confidence: float
    reasoning: str

    @property
    def is_line_chart(self) -> bool:
        return self.chart_type == "line chart"


class QwenPanelClassifier:
    """Stage-2 qwen-vl-max classifier through DashScope's official API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-vl-max",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ) -> None:
        key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            raise RuntimeError("DASHSCOPE_API_KEY is required for Stage 2")
        self.model = model
        self.client = OpenAI(api_key=key, base_url=base_url)

    def classify(self, image_path: Path) -> PanelClassification:
        chart = self._call(image_path, CHART_TYPE_PROMPT)
        origin = self._call(image_path, DATA_ORIGIN_PROMPT)
        return PanelClassification(
            chart_type=str(chart.get("chart_type", "other")).strip().lower(),
            first_level=str(origin.get("first_level", "Unknown")),
            second_level=str(origin.get("second_level", "Unknown")),
            confidence=float(origin.get("confidence", 0.0)),
            reasoning=str(origin.get("reasoning", "")),
        )

    def _call(self, image_path: Path, prompt: str) -> Dict[str, object]:
        suffix = image_path.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{encoded}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        text = response.choices[0].message.content or ""
        start, end = text.find("{"), text.rfind("}") + 1
        if start < 0 or end <= start:
            raise RuntimeError("Stage-2 classifier did not return JSON")
        return json.loads(text[start:end])


class MinerUPDFParser:
    """Thin, optional wrapper around the MinerU API used in Stage 1."""

    def parse(self, pdf_path: Path, output_directory: Path) -> List[Path]:
        try:
            from magic_pdf.config.enums import SupportedPdfParseMethod
            from magic_pdf.data.data_reader_writer import (
                FileBasedDataReader,
                FileBasedDataWriter,
            )
            from magic_pdf.data.dataset import PymuDocDataset
            from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
        except ImportError as exc:
            raise RuntimeError(
                "MinerU is not installed; install it only for PDF inputs"
            ) from exc

        pdf_path = Path(pdf_path)
        output_directory = Path(output_directory)
        images_directory = output_directory / "images"
        images_directory.mkdir(parents=True, exist_ok=True)
        image_writer = FileBasedDataWriter(str(images_directory))
        document_writer = FileBasedDataWriter(str(output_directory))

        reader = FileBasedDataReader("")
        dataset = PymuDocDataset(reader.read(str(pdf_path)))
        use_ocr = dataset.classify() == SupportedPdfParseMethod.OCR
        inference = dataset.apply(doc_analyze, ocr=use_ocr)
        parsed = (
            inference.pipe_ocr_mode(image_writer)
            if use_ocr
            else inference.pipe_txt_mode(image_writer)
        )
        document_id = pdf_path.stem
        parsed.dump_md(document_writer, f"{document_id}.md", "images")
        parsed.dump_content_list(
            document_writer,
            f"{document_id}_content_list.json",
            "images",
        )
        parsed.dump_middle_json(document_writer, f"{document_id}_middle.json")
        return sorted(
            path
            for path in images_directory.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )


def crop_subfigures(
    figure_paths: Iterable[Path],
    checkpoint: Path,
    output_directory: Path,
    confidence: float = 0.25,
) -> List[Path]:
    """Apply the Stage-1 YOLO11x detector and save reading-order crops."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    panels: List[Path] = []
    for figure_path in sorted(Path(path) for path in figure_paths):
        image = read_image(figure_path)
        height, width = image.shape[:2]
        boxes = detect_subfigures(figure_path, checkpoint, confidence)
        if not boxes:
            boxes = [Box(x1=0, y1=0, x2=width, y2=height)]
        for index, box in enumerate(boxes, 1):
            x1 = max(0, min(width, int(box.x1)))
            y1 = max(0, min(height, int(box.y1)))
            x2 = max(0, min(width, int(box.x2)))
            y2 = max(0, min(height, int(box.y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            panel_path = output_directory / f"{figure_path.stem}_{index}.png"
            if not cv2.imwrite(str(panel_path), image[y1:y2, x1:x2]):
                raise RuntimeError(f"Failed to write panel crop: {panel_path}")
            panels.append(panel_path)
    return panels


class DocumentPipeline:
    """Run the three paper stages without any benchmark evaluation code."""

    def __init__(
        self,
        model_directory: Path,
        classifier: Optional[QwenPanelClassifier] = None,
        judge: Optional[object] = None,
    ) -> None:
        self.model_directory = Path(model_directory)
        self.chart_pipeline = MatDeplotPipeline(self.model_directory)
        self.classifier = classifier
        self.judge = judge

    def _fail_closed_judge_verdict(
        self,
        result: ExtractionResult,
    ) -> Optional[JudgeVerdict]:
        if self.judge is None:
            return None
        if not result.success:
            return JudgeVerdict(
                model=self.judge.model,
                admitted=False,
                reason="extraction failed",
                error="extraction_failed",
            )
        try:
            render_extracted_curves(result.curves)
        except Exception:
            return JudgeVerdict(
                model=self.judge.model,
                admitted=False,
                reason="calibrated curves unavailable",
                error="calibrated_curves_unavailable",
            )
        return None

    def run(
        self,
        input_path: Path,
        output_directory: Path,
        curve_count_source: str = "silhouette",
        curve_count_cache: Optional[Path] = None,
        assume_line_charts: bool = False,
        calibrate: bool = True,
    ) -> Dict[str, object]:
        input_path = Path(input_path)
        output_directory = Path(output_directory)
        stage1_directory = output_directory / "stage1"
        if input_path.is_file() and input_path.suffix.lower() == ".pdf":
            figures = MinerUPDFParser().parse(
                input_path,
                stage1_directory / "mineru",
            )
        elif input_path.is_dir():
            figures = sorted(
                path
                for path in input_path.iterdir()
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
        else:
            raise ValueError("input must be a PDF or a directory of figure images")

        panels = crop_subfigures(
            figures,
            self.chart_pipeline.model_paths["subfigure"],
            stage1_directory / "panels",
        )
        cached_counts = (
            load_cached_counts(curve_count_cache) if curve_count_cache else None
        )
        manifest: List[Dict[str, object]] = []
        for panel_path in panels:
            if assume_line_charts:
                classification = PanelClassification(
                    chart_type="line chart",
                    first_level="Unknown",
                    second_level="Unknown",
                    confidence=0.0,
                    reasoning="Classification bypassed by explicit CLI option.",
                )
            else:
                if self.classifier is None:
                    raise RuntimeError(
                        "Stage-2 classifier is required unless "
                        "--assume-line-charts is set"
                    )
                classification = self.classifier.classify(panel_path)

            row: Dict[str, object] = {
                "panel": panel_path.name,
                "classification": asdict(classification),
                "admitted": None,
            }
            if classification.is_line_chart:
                result = self.chart_pipeline.extract(
                    panel_path,
                    curve_count_source=curve_count_source,
                    cached_counts=cached_counts,
                    calibrate=calibrate,
                )
                panel_output = output_directory / "curves" / panel_path.stem
                row["outputs"] = {
                    key: str(path.relative_to(output_directory))
                    for key, path in write_extraction(result, panel_output).items()
                }
                row["success"] = result.success
                if self.judge is not None:
                    verdict = self._fail_closed_judge_verdict(result)
                    if verdict is None:
                        try:
                            verdict = self.judge.review(panel_path, result.curves)
                        except Exception as exc:
                            verdict = JudgeVerdict(
                                model=self.judge.model,
                                admitted=False,
                                reason="Final quality judge failed operationally.",
                                error=f"judge:{type(exc).__name__}",
                            )
                    row["judge"] = verdict.to_dict()
                    row["admitted"] = verdict.admitted
            else:
                row["success"] = True
                row["skipped"] = "not a line chart"
            manifest.append(row)

        output_directory.mkdir(parents=True, exist_ok=True)
        manifest_path = output_directory / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if self.judge is None:
            return {"manifest": str(manifest_path), "panels": len(manifest)}

        accepted_manifest = [row for row in manifest if row["admitted"] is True]
        accepted_manifest_path = output_directory / "accepted_manifest.json"
        accepted_manifest_path.write_text(
            json.dumps(accepted_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return {
            "manifest": str(manifest_path),
            "accepted_manifest": str(accepted_manifest_path),
            "panels": len(manifest),
            "accepted": len(accepted_manifest),
        }
