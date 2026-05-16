from __future__ import annotations

from dataclasses import dataclass

from src.preprocessing.protected_spans import ProtectedSpan, find_protected_spans
from src.validation.diff_analyzer import DiffAnalyzer, Edit
from src.validation.edit_classifier import is_allowed_edit_type


@dataclass
class ValidationResult:
    source: str
    target: str
    edits: list[Edit]

    @property
    def accepted_edits(self) -> list[Edit]:
        return [edit for edit in self.edits if edit.status == "accepted"]

    @property
    def rejected_edits(self) -> list[Edit]:
        return [edit for edit in self.edits if edit.status == "rejected"]

    def apply_accepted(self) -> str:
        if not self.rejected_edits:
            return self.target

        text = self.source
        for edit in self.accepted_edits:
            if edit.edit_type == "final_punctuation":
                text = text.rstrip()
                if text and text[-1] in ".!?":
                    text = text[:-1]
                text += edit.replacement
            elif edit.edit_type in {"split_word", "join_words", "hyphen_change", "spelling_replace", "case_change"}:
                if edit.source:
                    text = text.replace(edit.source, edit.replacement, 1)
            elif edit.edit_type == "punctuation_insert":
                if edit.start >= 0:
                    text = text[: edit.start] + edit.replacement + text[edit.start :]
                elif edit.replacement == ",":
                    text = _insert_comma_before_common_subordinator(text)
            elif edit.edit_type == "punctuation_delete" and edit.start >= 0 and edit.end >= edit.start:
                text = text[: edit.start] + text[edit.end :]
            elif edit.edit_type == "punctuation_replace" and edit.start >= 0 and edit.end >= edit.start:
                text = text[: edit.start] + edit.replacement + text[edit.end :]
        return text


class StrictValidator:
    def __init__(self) -> None:
        self.diff_analyzer = DiffAnalyzer()

    def pre_validate(self, text: str) -> list[ProtectedSpan]:
        return find_protected_spans(text)

    def validate(self, source: str, target: str) -> ValidationResult:
        edits = self.diff_analyzer.analyze(source, target)
        protected = self.pre_validate(source)
        validated: list[Edit] = []

        for edit in edits:
            if is_allowed_edit_type(edit.edit_type) and not _touches_protected(edit, protected):
                validated.append(edit.with_status("accepted", "allowed strict-scope edit"))
            else:
                validated.append(edit.with_status("rejected", "outside strict spelling/punctuation scope"))

        return ValidationResult(source=source, target=target, edits=validated)


def _touches_protected(edit: Edit, protected: list[ProtectedSpan]) -> bool:
    if edit.start < 0 or edit.end < 0:
        return False
    return any(edit.start < span.end and span.start < edit.end for span in protected)


def _insert_comma_before_common_subordinator(text: str) -> str:
    for marker in (" что ", " чтобы ", " когда ", " если "):
        if marker in text and "," + marker[:-1] not in text:
            return text.replace(marker, "," + marker, 1)
    return text
