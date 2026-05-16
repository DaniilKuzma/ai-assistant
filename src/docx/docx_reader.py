from __future__ import annotations

from pathlib import Path

from docx import Document


def read_paragraphs(path: str | Path) -> list[str]:
    document = Document(path)
    return [paragraph.text for paragraph in document.paragraphs]
