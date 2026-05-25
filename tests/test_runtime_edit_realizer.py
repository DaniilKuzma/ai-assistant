from __future__ import annotations

from src.runtime.edit_realizer import apply_gap_labels, apply_token_edit_labels
from src.runtime.tokenization import tokenize_runtime_words


def test_split_ne_verb_applies_to_token_span() -> None:
    text = "Он незнал ответ"
    tokens = tokenize_runtime_words(text)

    corrected, edits = apply_token_edit_labels(
        text,
        tokens,
        ["KEEP", "SPLIT_NE_VERB", "KEEP"],
        [1.0, 0.95, 1.0],
        threshold=0.7,
    )

    assert corrected == "Он не знал ответ"
    assert [(edit.source, edit.replacement, edit.rule_id) for edit in edits] == [
        ("незнал", "не знал", "ne_verb")
    ]


def test_merge_tak_zhe_consumes_second_token() -> None:
    text = "Он так же пришел"
    tokens = tokenize_runtime_words(text)

    corrected, edits = apply_token_edit_labels(
        text,
        tokens,
        ["KEEP", "MERGE_TAK_ZHE_TO_TAKZHE", "SKIP_MERGED", "KEEP"],
        [1.0, 0.96, 1.0, 1.0],
        threshold=0.7,
    )

    assert corrected == "Он также пришел"
    assert [(edit.source, edit.replacement, edit.rule_id) for edit in edits] == [
        ("так же", "также", "takzhe_tak_zhe")
    ]


def test_gap_comma_insertion_preserves_spaces() -> None:
    text = "Он знал что делать"
    tokens = tokenize_runtime_words(text)

    corrected, edits = apply_gap_labels(
        text,
        tokens,
        ["NONE", "COMMA", "NONE", "NONE"],
        [1.0, 0.93, 1.0, 1.0],
        threshold=0.7,
    )

    assert corrected == "Он знал, что делать"
    assert [(edit.source, edit.replacement, edit.rule_id) for edit in edits] == [
        ("", ",", "comma_subordinate")
    ]


def test_gap_comma_insertion_does_not_duplicate_existing_comma() -> None:
    text = "Он знал, что делать"
    tokens = tokenize_runtime_words(text)

    corrected, edits = apply_gap_labels(
        text,
        tokens,
        ["NONE", "COMMA", "NONE", "NONE"],
        [1.0, 0.93, 1.0, 1.0],
        threshold=0.7,
    )

    assert corrected == text
    assert edits == []
