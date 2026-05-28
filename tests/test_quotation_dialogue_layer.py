from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from src.config.load_config import load_config
from src.grammar_gen.factory import online_generator_from_config
from src.grammar_gen.generator import OnlineExampleGenerator
from src.grammar_gen.rules.base import GenerationMode
from src.grammar_gen.safety import validate_generated_pair
from src.runtime.edit_realizer import apply_boundary_and_token_edit_labels, apply_gap_labels
from src.runtime.tokenization import tokenize_runtime_words
from src.schema import GeneratedExample, rule_id_to_label, rule_tag_to_id


ROOT = Path(__file__).resolve().parents[1]
QUOTATION_RULE_IDS = (
    "quote_pairing",
    "quote_normalization",
    "quote_extra_marks",
    "dialogue_author_before",
    "dialogue_speech_before_author",
    "dialogue_author_inside_speech",
    "dialogue_bracket_guards",
)
POSITIVE_QUOTATION_RULE_IDS = (
    "quote_pairing",
    "quote_normalization",
    "quote_extra_marks",
    "dialogue_author_before",
    "dialogue_speech_before_author",
    "dialogue_author_inside_speech",
)


def test_quotation_dialogue_rule_ids_are_registered() -> None:
    for rule_id in QUOTATION_RULE_IDS:
        assert rule_id_to_label(rule_tag_to_id(rule_id)) == rule_id


def test_quotation_dialogue_loader_covers_rules_and_modes() -> None:
    from src.rule_layers.quotation_dialogue import load_quotation_dialogue_specs

    specs = load_quotation_dialogue_specs(ROOT / "lexicon" / "layers")
    by_rule = {spec.rule_id: spec for spec in specs}

    assert set(by_rule) == set(QUOTATION_RULE_IDS)
    for rule_id, spec in by_rule.items():
        modes = {case.mode for case in spec.cases}
        assert spec.layer == "quotation_dialogue"
        assert spec.family == "quotation_dialogue"
        assert "positive" in modes or rule_id == "dialogue_bracket_guards"
        assert {"hard_negative", "clean_identity"} & modes
        for case in spec.cases:
            assert case.metadata["layer"] == "quotation_dialogue"
            assert case.metadata["family"] == "quotation_dialogue"
            assert case.metadata["construction_family"] == "quotation_dialogue"
            assert case.metadata["uses_construction_bank"] is True


@pytest.mark.parametrize(
    "rule_id",
    POSITIVE_QUOTATION_RULE_IDS,
)
def test_quotation_dialogue_positive_examples_generate_and_validate(rule_id: str) -> None:
    generator = _quotation_generator()

    example = generator.sample(rule_id=rule_id, mode=GenerationMode.POSITIVE)

    assert example.primary_rule_id == rule_id
    assert example.metadata["layer"] == "quotation_dialogue"
    assert example.source_text != example.target_text
    assert validate_generated_pair(example) == []
    assert GeneratedExample.from_dict(example.to_dict()) == example
    assert _apply_runtime_labels(example) == example.target_text


@pytest.mark.parametrize("mode", (GenerationMode.HARD_NEGATIVE, GenerationMode.CLEAN_IDENTITY))
def test_quotation_dialogue_identity_modes_are_stable(mode: GenerationMode) -> None:
    generator = _quotation_generator()
    examples = [
        generator.sample(rule_id=rule_id, mode=mode)
        for rule_id in QUOTATION_RULE_IDS
        if generator.registry.get_rule(rule_id) and generator.registry.get_rule(rule_id).can_generate(mode)
    ]

    assert examples
    for example in examples:
        assert example.source_text == example.target_text
        assert example.metadata["expected_token_edit_count"] == 0
        assert example.metadata["expected_gap_edit_count"] == 0
        assert example.metadata["expected_boundary_edit_count"] == 0
        assert validate_generated_pair(example) == []


def test_author_before_dialogue_counts_and_runtime_dot_after_closing_quote() -> None:
    generator = _quotation_generator()
    example = _first_example_matching(
        generator,
        "dialogue_author_before",
        lambda item: item.metadata.get("sub_rule_id") == "author_before_missing_colon_quotes",
    )

    assert example.metadata["expected_token_edit_count"] == 1
    assert example.metadata["expected_gap_edit_count"] == 2
    assert example.metadata["expected_boundary_edit_count"] == 2
    assert example.metadata["expected_total_edit_count"] == 5
    assert "CAPITALIZE" in example.token_edit_labels
    assert "COLON" in example.gap_labels
    assert "DOT" in example.gap_labels
    assert _apply_runtime_labels(example) == example.target_text


def test_speech_before_author_comma_dash_after_closing_quote() -> None:
    generator = _quotation_generator()
    example = _first_example_matching(
        generator,
        "dialogue_speech_before_author",
        lambda item: item.metadata.get("sub_rule_id") == "speech_before_author_missing_quotes_comma_dash",
    )

    assert example.metadata["expected_gap_edit_count"] == 1
    assert example.metadata["expected_boundary_edit_count"] == 2
    assert "COMMA_DASH" in example.gap_labels
    assert _apply_runtime_labels(example) == example.target_text


def test_generated_example_without_boundary_fields_defaults_to_none() -> None:
    text = "Редактор проверил отчёт."
    tokens = tokenize_runtime_words(text)
    example = GeneratedExample(
        source_text=text,
        target_text=text,
        source_tokens=tokens,
        token_edit_labels=["KEEP"] * len(tokens),
        gap_labels=["NONE", "NONE", "DOT"],
        rule_ids=["clean_identity"] * len(tokens),
        primary_rule_id="clean_identity",
        mode=GenerationMode.CLEAN_IDENTITY.value,
        explanation_ids=[],
        metadata={},
    )

    payload = example.to_dict()
    payload.pop("boundary_before_labels")
    payload.pop("boundary_after_labels")
    restored = GeneratedExample.from_dict(payload)

    assert restored.boundary_before_labels == ["NONE"] * len(tokens)
    assert restored.boundary_after_labels == ["NONE"] * len(tokens)
    assert validate_generated_pair(restored) == []


def test_quotation_examples_have_low_duplicate_rate() -> None:
    generator = _quotation_generator()
    examples = [generator.sample_by_index(index) for index in range(1000)]
    unique = {
        (example.source_text, example.target_text, example.primary_rule_id, example.mode)
        for example in examples
    }
    duplicate_rate = 1.0 - (len(unique) / len(examples))
    distribution = Counter(example.primary_rule_id for example in examples)

    assert set(distribution) == set(POSITIVE_QUOTATION_RULE_IDS)
    assert duplicate_rate <= 0.20
    assert all(validate_generated_pair(example) == [] for example in examples)


def _quotation_generator() -> OnlineExampleGenerator:
    config = load_config(ROOT / "configs" / "config.yaml")
    config["generation"]["enabled_rule_groups"] = ["quotation_dialogue"]
    config["generation"]["mix"] = {"quotation_dialogue": 1.0}
    config["generation"]["grammar"]["max_generation_retries"] = 80
    config["generation"]["rule_layers"]["groups"]["quotation_dialogue"] = {
        "enabled": True,
        "layers": ["quotation_dialogue"],
    }
    return online_generator_from_config(config, seed=13)


def _first_example_matching(generator: OnlineExampleGenerator, rule_id: str, predicate):
    for _ in range(250):
        example = generator.sample(rule_id=rule_id, mode=GenerationMode.POSITIVE)
        if predicate(example):
            return example
    raise AssertionError(f"No matching example found for {rule_id}.")


def _apply_runtime_labels(example: GeneratedExample) -> str:
    token_confidences = [0.99] * len(example.source_tokens)
    boundary_confidences = [0.99] * len(example.source_tokens)
    after_boundary_text, _boundary_edits = apply_boundary_and_token_edit_labels(
        example.source_text,
        example.source_tokens,
        example.token_edit_labels,
        token_confidences,
        threshold=0.70,
        boundary_before_labels=example.boundary_before_labels,
        boundary_after_labels=example.boundary_after_labels,
        boundary_before_confidences=boundary_confidences,
        boundary_after_confidences=boundary_confidences,
        rule_ids=example.rule_ids,
    )
    updated_tokens = tokenize_runtime_words(after_boundary_text)
    corrected, _gap_edits = apply_gap_labels(
        after_boundary_text,
        updated_tokens,
        example.gap_labels,
        [0.99] * len(example.gap_labels),
        threshold=0.70,
        rule_ids=example.rule_ids,
    )
    return corrected
