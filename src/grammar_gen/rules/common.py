from __future__ import annotations

from collections.abc import MutableSequence, Sequence
from typing import Any

from src.grammar_gen.ast import NounPhrase
from src.grammar_gen.rules.base import GenerationMode
from src.grammar_gen.safety import safety_clauses_for_ast
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
    metadata: dict[str, Any] | None = None,
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
        metadata=metadata or {},
    )


def make_punctuation_example(
    *,
    source: str,
    target: str,
    source_tokens: Sequence[WordToken],
    target_tokens: Sequence[WordToken],
    rule_id: str,
    mode: GenerationMode,
    active_gap_labels: frozenset[str],
    metadata: dict[str, str] | None = None,
) -> GeneratedExample:
    token_list = list(source_tokens)
    gap_labels = gap_labels_from_target_text(target, target_tokens, token_list)
    return GeneratedExample(
        source_text=source,
        target_text=target,
        source_tokens=token_list,
        token_edit_labels=token_labels_all_keep(token_list),
        gap_labels=gap_labels,
        rule_ids=rule_ids_for_active_gap_labels(gap_labels, rule_id, active_gap_labels),
        primary_rule_id=rule_id,
        mode=mode.value,
        explanation_ids=[rule_id],
        metadata=metadata or {},
    )


def metadata_with_safety_clauses(ast, lexicon, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    result = dict(metadata or {})
    result["uses_safety_clauses"] = True
    result["safety_clauses"] = safety_clauses_for_ast(ast, lexicon)
    return result


def metadata_without_safety_clauses(metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    result = dict(metadata or {})
    result["uses_safety_clauses"] = False
    return result


def noun_phrase_from_entry(entry: Any, *, case: str = "nomn", adjective_lemmas: tuple[str, ...] = ()) -> NounPhrase:
    number = "plur" if entry.gender == "plur" else "sing"
    return NounPhrase(
        noun_lemma=entry.lemma,
        gender=entry.gender,
        animacy=entry.animacy,
        number=number,
        case=case,
        adjective_lemmas=adjective_lemmas,
        semantic_class=entry.semantic_class,
    )


def varied_np(
    builder: Any,
    rng: Any,
    classes: tuple[str, ...],
    *,
    case: str = "nomn",
    adjective_probability: float = 0.35,
) -> NounPhrase:
    entry = builder.lexicon.random_noun_for_classes(classes, rng)
    adjectives = varied_adjectives(builder, rng, probability=adjective_probability)
    return noun_phrase_from_entry(entry, case=case, adjective_lemmas=adjectives)


def varied_adjectives(builder: Any, rng: Any, *, probability: float = 0.35) -> tuple[str, ...]:
    if not rng.chance(probability):
        return ()
    first = builder.lexicon.random_adjective(rng).lemma
    if rng.chance(0.12):
        second = builder.lexicon.random_adjective(rng).lemma
        if second != first:
            return (first, second)
    return (first,)


def capitalize_first(text: str) -> str:
    if not text:
        return text
    return f"{text[0].upper()}{text[1:]}"


def token_labels_all_keep(tokens: Sequence[WordToken]) -> list[str]:
    return ["KEEP"] * len(tokens)


def gap_labels_from_text(text: str, tokens: Sequence[WordToken]) -> list[str]:
    labels: list[str] = []
    for token in tokens:
        punctuation = _punctuation_after(text, token.end)
        labels.append(PUNCTUATION_TO_GAP_LABEL.get(punctuation, "NONE"))
    return labels


def gap_labels_from_target_text(
    target_text: str,
    target_tokens: Sequence[WordToken],
    source_tokens: Sequence[WordToken],
) -> list[str]:
    labels = gap_labels_from_text(target_text, target_tokens)
    if len(labels) != len(source_tokens):
        raise ValueError(
            "Target punctuation labels must match source token count: "
            f"{len(labels)} != {len(source_tokens)}."
        )
    return labels


def gap_labels_none(tokens: Sequence[WordToken]) -> list[str]:
    return ["NONE"] * len(tokens)


def rule_ids_for_active_gap_labels(
    gap_labels: Sequence[str],
    rule_id: str,
    active_gap_labels: frozenset[str],
) -> list[str]:
    rule_ids = [rule_id if label in active_gap_labels else "none" for label in gap_labels]
    if rule_id not in rule_ids and rule_ids:
        rule_ids[0] = rule_id
    return rule_ids


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
    "capitalize_first",
    "gap_labels_from_text",
    "gap_labels_from_target_text",
    "gap_labels_none",
    "insert_punctuation_before",
    "label_span",
    "make_clean_identity_example",
    "make_punctuation_example",
    "metadata_with_safety_clauses",
    "metadata_without_safety_clauses",
    "noun_phrase_from_entry",
    "remove_punctuation_before",
    "replace_once_checked",
    "rule_ids_for_active_gap_labels",
    "token_labels_all_keep",
    "varied_adjectives",
    "varied_np",
]
