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


@dataclass(frozen=True)
class PunctuationGapActionLabel:
    gap_index: int
    action: str


PUNCT_ACTION_LABELS = {
    "KEEP_NONE": 0,
    "KEEP_EXISTING": 1,
    "INSERT": 2,
    "DELETE": 3,
    "REPLACE": 4,
}


def build_punctuation_gap_labels(source: str, target: str) -> list[PunctuationGapLabel]:
    words = tokenize_words(source)
    if not words:
        return []

    labels = _labels_from_source_punctuation(source, words)
    mutable = list(labels)

    for edit in DiffAnalyzer().punctuation_edits(source, target):
        _apply_punctuation_edit_label(mutable, words, edit)

    return mutable


def build_punctuation_gap_action_labels(source: str, target: str) -> list[PunctuationGapActionLabel]:
    words = tokenize_words(source)
    if not words:
        return []

    labels = _action_labels_from_source_punctuation(source, words)
    mutable = list(labels)

    for edit in DiffAnalyzer().punctuation_edits(source, target):
        _apply_punctuation_action_label(mutable, words, edit)

    return mutable


def _apply_punctuation_edit_label(labels: list[PunctuationGapLabel], words: list[Token], edit: Edit) -> None:
    if edit.edit_type == "final_punctuation" and edit.replacement in PUNCT_LABELS:
        gap = len(labels) - 1
        labels[gap] = PunctuationGapLabel(gap, PUNCT_LABELS[edit.replacement])
        return

    if edit.edit_type not in {"punctuation_insert", "punctuation_replace"}:
        if edit.edit_type == "punctuation_delete":
            gap = _gap_index_for_position(words, edit.start)
            labels[gap] = PunctuationGapLabel(gap, "NONE")
        return
    if edit.replacement not in PUNCT_LABELS:
        return

    gap = _gap_index_for_position(words, edit.start)
    labels[gap] = PunctuationGapLabel(gap, PUNCT_LABELS[edit.replacement])


def _apply_punctuation_action_label(labels: list[PunctuationGapActionLabel], words: list[Token], edit: Edit) -> None:
    if edit.edit_type == "final_punctuation":
        gap = len(labels) - 1
        action = "REPLACE" if edit.source else "INSERT"
        labels[gap] = PunctuationGapActionLabel(gap, action)
        return
    if edit.edit_type == "punctuation_insert":
        gap = _gap_index_for_position(words, edit.start)
        labels[gap] = PunctuationGapActionLabel(gap, "INSERT")
        return
    if edit.edit_type == "punctuation_delete":
        gap = _gap_index_for_position(words, edit.start)
        labels[gap] = PunctuationGapActionLabel(gap, "DELETE")
        return
    if edit.edit_type == "punctuation_replace":
        gap = _gap_index_for_position(words, edit.start)
        labels[gap] = PunctuationGapActionLabel(gap, "REPLACE")


def _labels_from_source_punctuation(text: str, words: list[Token]) -> list[PunctuationGapLabel]:
    labels = [PunctuationGapLabel(index, "NONE") for index in range(len(words))]
    for index, word in enumerate(words):
        punctuation = _punctuation_after_word(text, word.end)
        if punctuation in PUNCT_LABELS:
            labels[index] = PunctuationGapLabel(index, PUNCT_LABELS[punctuation])
    return labels


def _action_labels_from_source_punctuation(text: str, words: list[Token]) -> list[PunctuationGapActionLabel]:
    labels = [PunctuationGapActionLabel(index, "KEEP_NONE") for index in range(len(words))]
    for index, word in enumerate(words):
        punctuation = _punctuation_after_word(text, word.end)
        if punctuation in PUNCT_LABELS:
            labels[index] = PunctuationGapActionLabel(index, "KEEP_EXISTING")
    return labels


def _punctuation_after_word(text: str, position: int) -> str:
    index = position
    while index < len(text) and text[index].isspace():
        index += 1
    if index < len(text):
        return text[index]
    return ""


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
