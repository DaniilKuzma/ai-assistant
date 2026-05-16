from __future__ import annotations

from pathlib import Path

from docx import Document


def write_paragraphs_like(source_path: str | Path, output_path: str | Path, paragraphs: list[str]) -> None:
    document = Document(source_path)
    for paragraph, text in zip(document.paragraphs, paragraphs, strict=False):
        if paragraph.runs:
            paragraph.runs[0].text = text
            for run in paragraph.runs[1:]:
                run.text = ""
        else:
            paragraph.text = text
    document.save(output_path)
