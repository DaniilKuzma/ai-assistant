from __future__ import annotations

from collections.abc import MutableSequence, Sequence

from src.grammar_gen.rules.base import GenerationMode
from src.schema import GeneratedExample, WordToken


PUNCTUATION_TO_GAP_LABEL = {
    ",": "COMMA",
    "-": "DASH",
    "\u2014": "DASH",
    ":": "COLON",
    ";": "SEMICOLON",
    ".": "DOT",
    "?": "QUESTION",
    "!": "EXCLAMATION",
    "...": "ELLIPSIS",
    "\u2026": "ELLIPSIS",
}
REMOVABLE_PUNCTUATION = frozenset(PUNCTUATION_TO_GAP_LABEL)


def make_clean_identity_example(
    text: str,
    tokens: Sequence[WordToken],
    rule_id: str = "clean_identity",
) -> GeneratedExample:
    token_list = list(tokens)
    return GeneratedExample(
        source_text=text,
        target_text=text,
        source_tokens=token_list,
        token_edit_labels=token_labels_all_keep(token_list),
        gap_labels=gap_labels_from_text(text, token_list),
        rule_ids=[rule_id] * len(token_list),
        primary_rule_id=rule_id,
        mode=GenerationMode.CLEAN_IDENTITY.value,
        explanation_ids=[],
        metadata={},
    )


def token_labels_all_keep(tokens: Sequence[WordToken]) -> list[str]:
    return ["KEEP"] * len(tokens)


def gap_labels_from_text(text: str, tokens: Sequence[WordToken]) -> list[str]:
    labels: list[str] = []
    for token in tokens:
        punctuation = _punctuation_after(text, token.end)
        labels.append(PUNCTUATION_TO_GAP_LABEL.get(punctuation, "NONE"))
    return labels


def gap_labels_none(tokens: Sequence[WordToken]) -> list[str]:
    return ["NONE"] * len(tokens)


def find_token_sequence(tokens: Sequence[WordToken], sequence: Sequence[str]) -> int:
    if not sequence:
        return 0
    if len(sequence) > len(tokens):
        return -1

    token_texts = [token.text for token in tokens]
    for start in range(0, len(token_texts) - len(sequence) + 1):
        if token_texts[start : start + len(sequence)] == list(sequence):
            return start
    return -1


def label_span(
    labels: MutableSequence[str],
    start: int,
    end: int,
    first_label: str,
    rest_label: str = "SKIP_MERGED",
) -> MutableSequence[str]:
    if start < 0 or end > len(labels) or start >= end:
        raise ValueError(f"Invalid label span: {start}:{end} for {len(labels)} labels.")

    labels[start] = first_label
    for index in range(start + 1, end):
        labels[index] = rest_label
    return labels


def replace_once_checked(text: str, old: str, new: str) -> str:
    if not old:
        raise ValueError("old must not be empty.")
    count = text.count(old)
    if count != 1:
        raise ValueError(f"Expected exactly one occurrence of {old!r}, found {count}.")
    return text.replace(old, new, 1)


def remove_punctuation_before(text: str, marker: str) -> str:
    marker_start = _single_marker_start(text, marker)
    before = text[:marker_start].rstrip()
    after = text[marker_start:].lstrip()
    if not before:
        raise ValueError(f"No text before marker {marker!r}.")

    if before.endswith("..."):
        before = before[:-3].rstrip()
    elif before[-1] in REMOVABLE_PUNCTUATION:
        before = before[:-1].rstrip()
    else:
        raise ValueError(f"No removable punctuation before marker {marker!r}.")

    return _join_before_marker(before, after)


def insert_punctuation_before(text: str, marker: str, punctuation: str) -> str:
    if punctuation not in REMOVABLE_PUNCTUATION:
        raise ValueError(f"Unsupported punctuation: {punctuation!r}.")

    marker_start = _single_marker_start(text, marker)
    before = text[:marker_start].rstrip()
    after = text[marker_start:].lstrip()
    if not before:
        raise ValueError(f"No text before marker {marker!r}.")
    if before.endswith("...") or before[-1] in REMOVABLE_PUNCTUATION:
        raise ValueError(f"Punctuation already exists before marker {marker!r}.")

    if punctuation in {"-", "\u2014"}:
        return f"{before} {punctuation} {after}"
    return f"{before}{punctuation} {after}"


def _punctuation_after(text: str, position: int) -> str:
    index = position
    while index < len(text) and text[index].isspace():
        index += 1
    if text.startswith("...", index):
        return "..."
    if index < len(text):
        return text[index]
    return ""


def _single_marker_start(text: str, marker: str) -> int:
    if not marker:
        raise ValueError("marker must not be empty.")
    count = text.count(marker)
    if count != 1:
        raise ValueError(f"Expected exactly one occurrence of marker {marker!r}, found {count}.")
    return text.index(marker)


def _join_before_marker(before: str, after: str) -> str:
    if not before:
        return after
    if not after:
        return before
    return f"{before} {after}"


__all__ = [
    "find_token_sequence",
    "gap_labels_from_text",
    "gap_labels_none",
    "insert_punctuation_before",
    "label_span",
    "make_clean_identity_example",
    "remove_punctuation_before",
    "replace_once_checked",
    "token_labels_all_keep",
]
