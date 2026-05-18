from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.docx.docx_reader import read_paragraphs
from src.docx.docx_writer import write_paragraphs_like
from src.inference.corrector import CorrectionResult, Corrector
from src.validation.diff_analyzer import Edit


class DocxCorrector(Protocol):
    def correct(self, text: str) -> CorrectionResult:
        ...


def correct_docx(input_path: str | Path, output_path: str | Path, corrector: DocxCorrector | None = None) -> list[Edit]:
    if corrector is None:
        corrector = Corrector()
    paragraphs = read_paragraphs(input_path)
    corrected: list[str] = []
    edits: list[Edit] = []

    for paragraph in paragraphs:
        result = corrector.correct(paragraph) if paragraph.strip() else None
        if result is None:
            corrected.append(paragraph)
            continue
        corrected.append(result.corrected_text)
        edits.extend(result.edits)

    write_paragraphs_like(input_path, output_path, corrected)
    return edits
