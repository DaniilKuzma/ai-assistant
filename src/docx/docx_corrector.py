from __future__ import annotations

from pathlib import Path

from src.docx.docx_reader import read_paragraphs
from src.docx.docx_writer import write_paragraphs_like
from src.inference.corrector import Corrector
from src.validation.diff_analyzer import Edit


def correct_docx(input_path: str | Path, output_path: str | Path, corrector: Corrector | None = None) -> list[Edit]:
    corrector = corrector or Corrector()
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
