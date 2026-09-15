from __future__ import annotations

from dataclasses import dataclass

from src.validation.diff_analyzer import DiffAnalyzer, Edit
from src.schema.edit_types import is_allowed_edit_type


@dataclass(frozen=True)
class AlignmentResult:
    source: str
    target: str
    edits: list[Edit]
    is_supported: bool
    reason: str = ""


class Aligner:
    """Convert source-target pairs into supported edit operations."""

    def __init__(self) -> None:
        self.diff_analyzer = DiffAnalyzer()

    def align(self, source: str, target: str) -> AlignmentResult:
        edits = self.diff_analyzer.analyze(source, target)
        unsupported = [edit for edit in edits if not is_allowed_edit_type(edit.edit_type)]
        if unsupported:
            return AlignmentResult(source, target, edits, False, "contains unsupported edits")
        return AlignmentResult(source, target, edits, True)
