from __future__ import annotations

from dataclasses import dataclass

from src.preprocessing.tokenizer import Token, tokenize_words
from src.validation.diff_analyzer import DiffAnalyzer, Edit


PUNCT_LABELS = {
    ",": "COMMA",
    ".": "DOT",
    "?": "QUESTION",
    "!": "EXCLAMATION",
    ":": "COLON",
    "—": "DASH",
    ";": "SEMICOLON",
    "…": "ELLIPSIS",
    "«": "QUOTE_OPEN",
    "»": "QUOTE_CLOSE",
    "(": "BRACKET_OPEN",
    ")": "BRACKET_CLOSE",
}


@dataclass(frozen=True)
class PunctuationGapLabel:
    gap_index: int
    label: str


def build_punctuation_gap_labels(source: str, target: str) -> list[PunctuationGapLabel]:
    words = tokenize_words(source)
    if not words:
        return []

    labels = [PunctuationGapLabel(index, "NONE") for index in range(len(words))]
    mutable = list(labels)

    for edit in DiffAnalyzer().analyze(source, target):
        _apply_punctuation_edit_label(mutable, words, edit)

    return mutable


def _apply_punctuation_edit_label(labels: list[PunctuationGapLabel], words: list[Token], edit: Edit) -> None:
    if edit.edit_type == "final_punctuation" and edit.replacement in PUNCT_LABELS:
        gap = len(labels) - 1
        labels[gap] = PunctuationGapLabel(gap, PUNCT_LABELS[edit.replacement])
        return

    if edit.edit_type not in {"punctuation_insert", "punctuation_replace"}:
        return
    if edit.replacement not in PUNCT_LABELS:
        return

    gap = _gap_index_for_position(words, edit.start)
    labels[gap] = PunctuationGapLabel(gap, PUNCT_LABELS[edit.replacement])


def _gap_index_for_position(words: list[Token], position: int) -> int:
    if position < 0:
        return 0
    gap = 0
    for index, word in enumerate(words):
        if word.end <= position:
            gap = index
        elif word.start > position:
            break
    return max(0, min(gap, len(words) - 1))
