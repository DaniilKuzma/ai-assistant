from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.candidates.frequent_errors import CONTEXT_DEPENDENT_WHITELIST
from src.preprocessing.protected_spans import ProtectedSpan, find_protected_spans
from src.preprocessing.tokenizer import tokenize_words
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
        positioned_edits: list[Edit] = []
        fallback_edits: list[Edit] = []
        for edit in self.accepted_edits:
            if _is_positioned_edit(edit):
                positioned_edits.append(edit)
            else:
                fallback_edits.append(edit)

        for edit in sorted(positioned_edits, key=lambda item: (item.start, item.end), reverse=True):
            text = _apply_positioned_edit(text, edit)

        for edit in fallback_edits:
            if edit.edit_type == "punctuation_insert" and edit.replacement == ",":
                text = _insert_comma_before_common_subordinator(text)
        return text


class StrictValidator:
    def __init__(self, context_pair_threshold: float = 0.98) -> None:
        self.diff_analyzer = DiffAnalyzer()
        self.context_pair_threshold = context_pair_threshold

    def pre_validate(self, text: str) -> list[ProtectedSpan]:
        return find_protected_spans(text)

    def validate(self, source: str, target: str, trusted_edits: list[Any] | None = None) -> ValidationResult:
        edits = self.diff_analyzer.analyze(source, target)
        protected = self.pre_validate(source)
        trusted_context_keys = _trusted_context_pair_keys(trusted_edits or [], self.context_pair_threshold)
        validated: list[Edit] = []

        for edit in edits:
            if _is_context_dependent_edit(edit):
                if _is_trusted_context_pair(source, edit, trusted_context_keys) and not _touches_protected(edit, protected):
                    validated.append(edit.with_status("accepted", "trusted high-confidence context pair"))
                else:
                    validated.append(edit.with_status("rejected", "context-dependent pair requires trusted model confidence"))
            elif is_allowed_edit_type(edit.edit_type) and not _touches_protected(edit, protected):
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


def _is_context_dependent_edit(edit: Edit) -> bool:
    return CONTEXT_DEPENDENT_WHITELIST.get(edit.source.lower()) == edit.replacement.lower()


def _trusted_context_pair_keys(trusted_edits: list[Any], threshold: float) -> set[tuple[int, int, str, str]]:
    keys: set[tuple[int, int, str, str]] = set()
    for edit in trusted_edits:
        source = str(getattr(edit, "source", "")).lower()
        replacement = str(getattr(edit, "replacement", "")).lower()
        confidence = float(getattr(edit, "confidence", 0.0))
        if confidence < threshold:
            continue
        if CONTEXT_DEPENDENT_WHITELIST.get(source) != replacement:
            continue
        keys.add((int(getattr(edit, "start", -1)), int(getattr(edit, "end", -1)), source, replacement))
    return keys


def _is_trusted_context_pair(source_text: str, edit: Edit, trusted_keys: set[tuple[int, int, str, str]]) -> bool:
    key = (edit.start, edit.end, edit.source.lower(), edit.replacement.lower())
    return key in trusted_keys and _passes_context_pair_guard(source_text, edit)


def _passes_context_pair_guard(source_text: str, edit: Edit) -> bool:
    source = edit.source.lower()
    replacement = edit.replacement.lower()
    next_word = _next_word_after(source_text, edit.end)
    if source == "также" and replacement == "так же":
        return next_word == "как"
    if source == "что бы" and replacement == "чтобы":
        return _looks_like_infinitive(next_word)
    return False


def _next_word_after(text: str, position: int) -> str:
    for word in tokenize_words(text):
        if word.start >= position:
            return word.text.lower()
    return ""


def _looks_like_infinitive(word: str) -> bool:
    return word.endswith(("ть", "ться", "ти", "чь"))


def _is_positioned_edit(edit: Edit) -> bool:
    if edit.start < 0 or edit.end < edit.start:
        return False
    return edit.edit_type in {
        "final_punctuation",
        "split_word",
        "join_words",
        "hyphen_change",
        "spelling_replace",
        "case_change",
        "punctuation_insert",
        "punctuation_delete",
        "punctuation_replace",
    }


def _apply_positioned_edit(text: str, edit: Edit) -> str:
    start = max(0, min(edit.start, len(text)))
    end = max(start, min(edit.end, len(text)))
    if edit.edit_type == "final_punctuation":
        return text[:start] + edit.replacement + text[end:]
    if edit.edit_type in {"split_word", "join_words", "hyphen_change", "spelling_replace", "case_change"}:
        return text[:start] + edit.replacement + text[end:]
    if edit.edit_type == "punctuation_insert":
        return text[:start] + edit.replacement + text[start:]
    if edit.edit_type == "punctuation_delete":
        return text[:start] + text[end:]
    if edit.edit_type == "punctuation_replace":
        return text[:start] + edit.replacement + text[end:]
    return text
