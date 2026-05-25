from __future__ import annotations

from dataclasses import dataclass
import re
from collections.abc import Sequence

from src.runtime.edit_realizer import apply_runtime_edits
from src.schema import RuntimeEdit


RUSSIAN_WORD_RE = re.compile(r"[А-Яа-яЁё]+", re.UNICODE)
ALLOWED_EDIT_TYPES = frozenset({"spelling", "split_join", "hyphen", "punctuation", "casing"})


@dataclass(frozen=True)
class ScopeGuard:
    max_length_ratio: float = 1.35

    def validate_edit(self, source_text: str, edit: RuntimeEdit) -> bool:
        if edit.edit_type not in ALLOWED_EDIT_TYPES:
            return False
        if edit.start < 0 or edit.end < edit.start or edit.end > len(source_text):
            return False

        span = source_text[edit.start : edit.end]
        if edit.source and span != edit.source and edit.source not in source_text:
            return False
        if edit.source == edit.replacement:
            return False
        if edit.replacement == "" and _contains_word(edit.source) and edit.edit_type != "punctuation":
            return False
        if edit.source == "" and _contains_word(edit.replacement):
            return False
        if edit.rule_id == "spacing_normalization":
            return _word_sequence(edit.source) == _word_sequence(edit.replacement)
        if edit.edit_type in {"split_join", "hyphen"}:
            return _letters_only(edit.source) == _letters_only(edit.replacement)
        if edit.edit_type == "casing":
            return edit.source.lower() == edit.replacement.lower()
        if edit.edit_type == "punctuation":
            return not _adds_word(edit)
        if edit.edit_type == "spelling":
            return bool(edit.source) and bool(edit.replacement)
        return True

    def validate_result(
        self,
        source_text: str,
        corrected_text: str,
        edits: Sequence[RuntimeEdit],
    ) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        for edit in edits:
            if not self.validate_edit(source_text, edit):
                reasons.append(f"invalid_edit:{edit.rule_id}:{edit.start}:{edit.end}")

        if len(corrected_text) > max(len(source_text) + 32, int(len(source_text) * self.max_length_ratio)):
            reasons.append("result_too_long")
        if _semantic_word_delta_too_large(source_text, corrected_text, edits):
            reasons.append("semantic_word_delta")

        if not reasons:
            return True, []

        hard_reasons = [reason for reason in reasons if not reason.startswith("invalid_edit:") or "punctuation" in reason]
        if hard_reasons:
            return False, reasons
        projected = apply_runtime_edits(source_text, edits)
        if projected == corrected_text:
            return True, reasons
        return True, [*reasons, "projection_mismatch_stage_relative_offsets"]


def _contains_word(value: str) -> bool:
    return bool(RUSSIAN_WORD_RE.search(value))


def _adds_word(edit: RuntimeEdit) -> bool:
    return _contains_word(edit.replacement) and not _contains_word(edit.source)


def _letters_only(value: str) -> str:
    return "".join(RUSSIAN_WORD_RE.findall(value.lower()))


def _word_sequence(value: str) -> list[str]:
    return RUSSIAN_WORD_RE.findall(value.lower())


def _semantic_word_delta_too_large(source_text: str, corrected_text: str, edits: Sequence[RuntimeEdit]) -> bool:
    source_words = _word_sequence(source_text)
    corrected_words = _word_sequence(corrected_text)
    allowed_delta = sum(1 for edit in edits if edit.edit_type == "split_join")
    return abs(len(corrected_words) - len(source_words)) > max(1, allowed_delta + 1)


__all__ = ["ScopeGuard"]
