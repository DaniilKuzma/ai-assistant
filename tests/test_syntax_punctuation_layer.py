from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from src.config.load_config import load_config
from src.grammar_gen.audit import audit_batch
from src.grammar_gen.factory import online_generator_from_config
from src.grammar_gen.generator import OnlineExampleGenerator
from src.grammar_gen.rules.base import GenerationMode
from src.grammar_gen.safety import validate_generated_pair
from src.runtime.edit_realizer import apply_gap_labels
from src.runtime.scope_guard import ScopeGuard
from src.schema import RuntimeEdit, rule_id_to_label, rule_tag_to_id


ROOT = Path(__file__).resolve().parents[1]
SYNTAX_RULE_IDS = (
    "punct_final_marks",
    "punct_dash_syntax",
    "punct_homogeneous_extended",
    "punct_detached_definitions",
    "punct_detached_adverbials",
    "punct_comparative_turns",
    "punct_introductory_extended",
    "punct_address_interjection",
    "punct_complex_sentences",
    "punct_bsp",
    "punct_fixed_expression_guards",
)
EDIT_GAP_LABELS = {
    "COMMA",
    "DASH",
    "COLON",
    "SEMICOLON",
    "DOT",
    "QUESTION",
    "EXCLAMATION",
    "ELLIPSIS",
    "DELETE_PUNCTUATION",
}


def test_syntax_punctuation_rule_ids_are_registered() -> None:
    for rule_id in SYNTAX_RULE_IDS:
        assert rule_id_to_label(rule_tag_to_id(rule_id)) == rule_id


def test_syntax_punctuation_loader_covers_all_groups_and_modes() -> None:
    from src.rule_layers.syntax_punctuation import load_syntax_punctuation_specs

    specs = load_syntax_punctuation_specs(ROOT / "lexicon" / "layers")
    by_rule = {spec.rule_id: spec for spec in specs}

    assert set(by_rule) == set(SYNTAX_RULE_IDS)
    for rule_id in SYNTAX_RULE_IDS:
        modes = {case.mode for case in by_rule[rule_id].cases}
        assert "positive" in modes, rule_id
        assert "hard_negative" in modes, rule_id
        assert by_rule[rule_id].family == "syntax_punctuation"
        assert by_rule[rule_id].layer == "syntax_punctuation"


@pytest.mark.parametrize("rule_id", SYNTAX_RULE_IDS)
@pytest.mark.parametrize("mode", (GenerationMode.POSITIVE, GenerationMode.HARD_NEGATIVE))
def test_syntax_punctuation_groups_generate_valid_examples(rule_id: str, mode: GenerationMode) -> None:
    generator = _syntax_generator()

    example = generator.sample(rule_id=rule_id, mode=mode)

    assert example.primary_rule_id == rule_id
    assert example.metadata["layer"] == "syntax_punctuation"
    assert len(example.source_tokens) == len(example.gap_labels) == len(example.rule_ids)
    assert set(example.token_edit_labels) == {"KEEP"}
    assert validate_generated_pair(example) == []

    rule_owned_gap_count = sum(
        1
        for label, gap_rule_id in zip(example.gap_labels, example.rule_ids, strict=True)
        if gap_rule_id == example.primary_rule_id and label in EDIT_GAP_LABELS
    )
    expected_gap = int(example.metadata["expected_gap_edit_count"])
    if mode is GenerationMode.POSITIVE:
        assert example.source_text != example.target_text
        assert expected_gap >= 1
        assert rule_owned_gap_count == expected_gap
    else:
        assert example.source_text == example.target_text
        assert example.metadata["expected_gap_edit_count"] == 0
        assert "DELETE_PUNCTUATION" not in example.gap_labels


def test_delete_punctuation_examples_validate_and_apply_runtime_deletion() -> None:
    generator = _syntax_generator()
    example = _first_example_with_label(
        generator,
        rule_id="punct_fixed_expression_guards",
        label="DELETE_PUNCTUATION",
    )

    corrected, edits = apply_gap_labels(
        example.source_text,
        example.source_tokens,
        example.gap_labels,
        [0.99] * len(example.gap_labels),
        threshold=0.70,
        rule_ids=example.rule_ids,
    )

    assert example.metadata["expected_gap_edit_count"] == 1
    assert validate_generated_pair(example) == []
    assert corrected == example.target_text
    assert [(edit.source, edit.replacement, edit.edit_type, edit.rule_id) for edit in edits] == [
        (",", "", "punctuation", "punct_fixed_expression_guards")
    ]


def test_extra_dash_examples_use_delete_punctuation_and_apply_runtime_deletion() -> None:
    generator = _syntax_generator()
    example = _first_example_with_label(
        generator,
        rule_id="punct_dash_syntax",
        label="DELETE_PUNCTUATION",
    )

    corrected, edits = apply_gap_labels(
        example.source_text,
        example.source_tokens,
        example.gap_labels,
        [0.99] * len(example.gap_labels),
        threshold=0.70,
        rule_ids=example.rule_ids,
    )

    assert example.metadata["operation"] == "delete_extra_dash"
    assert example.metadata["expected_gap_edit_count"] == 1
    assert corrected == example.target_text
    assert [(edit.source, edit.replacement, edit.edit_type, edit.rule_id) for edit in edits] == [
        (" — ", " ", "punctuation", "punct_dash_syntax")
    ]


def test_introductory_medial_examples_produce_two_gap_labels() -> None:
    generator = _syntax_generator()
    example = _first_example_matching(
        generator,
        rule_id="punct_introductory_extended",
        predicate=lambda item: item.metadata.get("position") == "medial",
    )

    assert example.metadata["expected_gap_edit_count"] == 2
    assert example.gap_labels.count("COMMA") >= 2
    assert validate_generated_pair(example) == []


def test_subject_predicate_guard_has_no_comma_or_dash_edit() -> None:
    generator = _syntax_generator()
    example = _first_example_matching(
        generator,
        rule_id="punct_dash_syntax",
        mode=GenerationMode.HARD_NEGATIVE,
        predicate=lambda item: item.metadata.get("guard") == "no_dash_subject_predicate",
    )

    assert example.source_text == example.target_text
    assert example.metadata["expected_gap_edit_count"] == 0
    assert "DASH" not in example.gap_labels
    assert "COMMA" not in example.gap_labels


def test_final_marks_allow_missing_source_final_mark() -> None:
    generator = _syntax_generator()

    examples = [
        generator.sample(rule_id="punct_final_marks", mode=GenerationMode.POSITIVE)
        for _ in range(200)
    ]
    labels = {example.gap_labels[-1] for example in examples}

    assert labels == {"DOT"}
    assert all(example.target_text.endswith(".") for example in examples)
    assert all(example.source_text[-1] not in ".!?\u2026" for example in examples)
    assert all(validate_generated_pair(example) == [] for example in examples)


def test_scope_guard_accepts_punctuation_only_insertions_and_deletions() -> None:
    source = "Редактор проверил отчёт"
    guard = ScopeGuard()
    insertion_point = len(source)

    assert guard.validate_edit(
        source,
        RuntimeEdit(insertion_point, insertion_point, "", ",", "punctuation", "punct_complex_sentences", 0.95),
    )
    assert guard.validate_edit(
        "Редактор проверил, отчёт",
        RuntimeEdit(16, 17, ",", "", "punctuation", "punct_fixed_expression_guards", 0.95),
    )
    assert not guard.validate_edit(
        source,
        RuntimeEdit(24, 24, "", " слово", "punctuation", "punct_complex_sentences", 0.95),
    )


def test_1000_syntax_punctuation_examples_have_reasonable_duplicate_rate_and_clean_audit() -> None:
    generator = _syntax_generator()
    examples = [generator.sample_by_index(index) for index in range(1000)]
    unique = {
        (example.source_text, example.target_text, example.primary_rule_id, example.mode)
        for example in examples
    }
    duplicate_rate = 1.0 - (len(unique) / len(examples))
    audit = audit_batch(examples)
    distribution = Counter(example.primary_rule_id for example in examples)

    assert set(distribution) == set(SYNTAX_RULE_IDS)
    assert duplicate_rate < 0.75
    assert audit["failed_examples_count"] == 0


def _syntax_generator() -> OnlineExampleGenerator:
    config = load_config(ROOT / "configs" / "config.yaml")
    config["generation"]["enabled_rule_groups"] = ["syntax_punctuation"]
    config["generation"]["mix"] = {"syntax_punctuation": 1.0}
    config["generation"]["grammar"]["max_generation_retries"] = 60
    config["generation"]["rule_layers"]["groups"]["syntax_punctuation"] = {
        "enabled": True,
        "layers": ["syntax_punctuation"],
    }
    return online_generator_from_config(config, seed=13)


def _first_example_with_label(
    generator: OnlineExampleGenerator,
    *,
    rule_id: str,
    label: str,
):
    return _first_example_matching(
        generator,
        rule_id=rule_id,
        predicate=lambda item: label in item.gap_labels,
    )


def _first_example_matching(
    generator: OnlineExampleGenerator,
    *,
    rule_id: str,
    predicate,
    mode: GenerationMode = GenerationMode.POSITIVE,
):
    for _ in range(200):
        example = generator.sample(rule_id=rule_id, mode=mode)
        if predicate(example):
            return example
    raise AssertionError(f"No matching example found for {rule_id}.")
