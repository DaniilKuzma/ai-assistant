from __future__ import annotations

from dataclasses import dataclass

from src.alignment.aligner import Aligner


@dataclass(frozen=True)
class EditLabel:
    source: str
    target: str
    edit_type: str


def build_edit_labels(source: str, target: str) -> list[EditLabel]:
    alignment = Aligner().align(source, target)
    return [EditLabel(edit.source, edit.replacement, edit.edit_type) for edit in alignment.edits]
