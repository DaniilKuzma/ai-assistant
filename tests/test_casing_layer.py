from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from src.config.load_config import load_config
from src.grammar_gen.factory import online_generator_from_config
from src.grammar_gen.generator import GenerationError, OnlineExampleGenerator
from src.grammar_gen.rules.base import GenerationMode
from src.grammar_gen.safety import validate_generated_pair
from src.runtime.edit_realizer import apply_token_edit_labels
from src.runtime.scope_guard import ScopeGuard
from src.schema import GeneratedExample, rule_id_to_label, rule_tag_to_id


ROOT = Path(__file__).resolve().parents[1]
CASING_RULE_IDS = (
    "casing_sentence_start",
    "casing_person_names",
    "casing_geo_names",
    "casing_organizations",
    "casing_documents_events",
    "casing_common_lowercase",
    "casing_formal_you_guard",
)
CORRECTION_CAPABLE_RULE_IDS = tuple(
    rule_id for rule_id in CASING_RULE_IDS if rule_id != "casing_formal_you_guard"
)
EXPECTED_SUB_RULE_IDS = {
    "casing_sentence_start": {
        "sentence_start_basic",
        "sentence_after_period_basic",
        "already_correct_start_guard",
        "abbreviation_start_guard",
        "numeric_start_guard",
    },
    "casing_person_names": {
        "person_ivan_petrov",
        "person_maria_ivanova",
        "person_anna_kuznetsova",
        "person_already_correct_guard",
    },
    "casing_geo_names": {
        "geo_city_single",
        "geo_country_single",
        "geo_country_multiword",
        "geo_already_correct_guard",
        "geo_common_noun_guard",
    },
    "casing_organizations": {
        "organization_rgrtu",
        "organization_ministry",
        "ordinary_ministry_guard",
        "ordinary_university_guard",
    },
    "casing_documents_events": {
        "document_federal_law_numbered",
        "event_great_patriotic_war",
        "event_victory_day",
        "ordinary_law_guard",
        "ordinary_event_words_guard",
    },
    "casing_common_lowercase": {
        "common_role_mid_sentence",
        "common_president_title",
        "common_weekday_month",
        "sentence_start_uppercase_guard",
        "proper_name_uppercase_guard",
        "abbreviation_uppercase_guard",
    },
    "casing_formal_you_guard": {
        "formal_you_uppercase_guard",
        "formal_your_lowercase_acceptable_guard",
        "formal_dative_lowercase_acceptable_guard",
    },
}


def test_casing_rule_ids_are_registered() -> None:
    for rule_id in CASING_RULE_IDS:
        assert rule_id_to_label(rule_tag_to_id(rule_id)) == rule_id


def test_casing_loader_covers_rules_subrules_and_capabilities() -> None:
    from src.rule_layers.casing import load_casing_specs

    specs = load_casing_specs(ROOT / "lexicon" / "layers")
    by_rule = {spec.rule_id: spec for spec in specs}

    assert set(by_rule) == set(CASING_RULE_IDS)
    for rule_id, spec in by_rule.items():
        assert spec.layer == "casing"
        assert spec.family == "casing"
        assert {case.sub_rule_id for case in spec.cases} >= EXPECTED_SUB_RULE_IDS[rule_id]
        assert spec.metadata["rule_kind"] in {"correction", "guard"}
        assert spec.metadata["supports_positive"] is (rule_id != "casing_formal_you_guard")
        for case in spec.cases:
            assert case.family == "casing"
            assert case.metadata["layer"] == "casing"
            assert case.metadata["family"] == "casing"
            assert case.metadata["source_file"].endswith(".yaml")
            assert case.metadata["case_id"] == case.sub_rule_id
            assert case.metadata["phenomenon"]
            assert case.metadata["operation"]
            assert case.metadata["casing_class"]
            assert case.metadata["rule_kind"] in {"correction", "guard"}
            assert case.metadata["supports_positive"] is (rule_id != "casing_formal_you_guard")


def test_casing_layer_uses_only_capitalize_and_lowercase() -> None:
    from src.rule_layers.casing import load_casing_specs

    labels = [
        operation.label
        for spec in load_casing_specs(ROOT / "lexicon" / "layers")
        for case in spec.cases
        for operation in case.token_operations
    ]

    assert labels
    assert "UPPERCASE" not in labels
    assert set(labels) <= {"CAPITALIZE", "LOWERCASE"}
    proper_geo_labels = [
        operation.label
        for spec in load_casing_specs(ROOT / "lexicon" / "layers")
        if spec.rule_id in {"casing_person_names", "casing_geo_names"}
        for case in spec.cases
        for operation in case.token_operations
    ]
    common_labels = [
        operation.label
        for spec in load_casing_specs(ROOT / "lexicon" / "layers")
        if spec.rule_id == "casing_common_lowercase"
        for case in spec.cases
        for operation in case.token_operations
    ]
    assert proper_geo_labels and set(proper_geo_labels) == {"CAPITALIZE"}
    assert common_labels and set(common_labels) == {"LOWERCASE"}


def test_casing_positive_and_identity_case_invariants() -> None:
    from src.rule_layers.casing import load_casing_specs

    for spec in load_casing_specs(ROOT / "lexicon" / "layers"):
        modes = {case.mode for case in spec.cases}
        if spec.rule_id == "casing_formal_you_guard":
            assert modes == {"hard_negative", "clean_identity"}
        else:
            assert "positive" in modes
            assert {"hard_negative", "clean_identity"} & modes

        for case in spec.cases:
            if case.mode == "positive":
                assert case.source_text != case.target_text
                assert case.expected_token_edit_count >= 1
            else:
                assert case.source_text == case.target_text
                assert case.expected_token_edit_count == 0
                assert not case.token_operations
            assert case.expected_gap_edit_count == 0
            assert not case.gap_operations
            assert not case.boundary_operations


@pytest.mark.parametrize("rule_id", CORRECTION_CAPABLE_RULE_IDS)
def test_casing_positive_examples_generate_validate_and_realize(rule_id: str) -> None:
    generator = _casing_generator()

    example = generator.sample(rule_id=rule_id, mode=GenerationMode.POSITIVE)

    assert example.primary_rule_id == rule_id
    assert example.metadata["layer"] == "casing"
    assert example.source_text != example.target_text
    assert validate_generated_pair(example) == []
    assert GeneratedExample.from_dict(example.to_dict()) == example
    assert _apply_token_labels(example) == example.target_text
    assert _scope_guard_accepts(example)


@pytest.mark.parametrize("mode", (GenerationMode.HARD_NEGATIVE, GenerationMode.CLEAN_IDENTITY))
def test_casing_identity_modes_generate_and_validate(mode: GenerationMode) -> None:
    generator = _casing_generator()
    examples = [
        generator.sample(rule_id=rule_id, mode=mode)
        for rule_id in CASING_RULE_IDS
        if generator.registry.get_rule(rule_id) and generator.registry.get_rule(rule_id).can_generate(mode)
    ]

    assert examples
    for example in examples:
        assert example.source_text == example.target_text
        assert example.metadata["expected_token_edit_count"] == 0
        assert example.metadata["expected_gap_edit_count"] == 0
        assert validate_generated_pair(example) == []
        assert all(label == "KEEP" for label in example.token_edit_labels)


def test_casing_formal_you_guard_is_guard_only() -> None:
    generator = _casing_generator()
    rule = generator.registry.get_rule("casing_formal_you_guard")

    assert rule is not None
    assert not rule.can_generate(GenerationMode.POSITIVE)
    assert rule.can_generate(GenerationMode.HARD_NEGATIVE)
    assert rule.can_generate(GenerationMode.CLEAN_IDENTITY)

    with pytest.raises(GenerationError, match="does not support mode 'positive'"):
        generator.sample(rule_id="casing_formal_you_guard", mode=GenerationMode.POSITIVE)


def test_formal_you_guard_examples_are_identity() -> None:
    generator = _casing_generator()
    examples = [
        generator.sample(rule_id="casing_formal_you_guard", mode=GenerationMode.HARD_NEGATIVE),
        generator.sample(rule_id="casing_formal_you_guard", mode=GenerationMode.CLEAN_IDENTITY),
    ]

    assert any("ваш документ" in example.source_text for example in examples)
    for example in examples:
        assert example.source_text == example.target_text
        assert example.metadata["rule_kind"] == "guard"
        assert example.metadata["supports_positive"] is False
        assert all(label == "KEEP" for label in example.token_edit_labels)


def test_representative_casing_runtime_targets_are_exact() -> None:
    generator = _casing_generator()
    expectations = {
        "casing_sentence_start": lambda item: item.target_text.startswith("Редактор "),
        "casing_person_names": lambda item: "Иван Петров" in item.target_text,
        "casing_geo_names": lambda item: "Российская Федерация" in item.target_text,
        "casing_organizations": lambda item: "Рязанский государственный радиотехнический университет" in item.target_text,
        "casing_documents_events": lambda item: "Великая Отечественная война" in item.target_text,
        "casing_common_lowercase": lambda item: "директора утром" in item.target_text,
    }

    for rule_id, predicate in expectations.items():
        example = _first_example_matching(generator, rule_id, predicate)
        assert _apply_token_labels(example) == example.target_text
        assert _scope_guard_accepts(example)


def test_casing_examples_have_low_duplicate_rate() -> None:
    generator = _casing_generator()
    examples = [generator.sample_by_index(index) for index in range(1000)]
    unique = {
        (example.source_text, example.target_text, example.primary_rule_id, example.mode)
        for example in examples
    }
    duplicate_rate = 1.0 - (len(unique) / len(examples))
    distribution = Counter(example.primary_rule_id for example in examples)

    assert set(distribution) <= set(CORRECTION_CAPABLE_RULE_IDS)
    assert set(distribution) >= set(CORRECTION_CAPABLE_RULE_IDS)
    assert duplicate_rate <= 0.20
    assert all(validate_generated_pair(example) == [] for example in examples)


def _casing_generator() -> OnlineExampleGenerator:
    config = load_config(ROOT / "configs" / "config.yaml")
    config["generation"]["enabled_rule_groups"] = ["casing"]
    config["generation"]["mix"] = {"casing": 1.0}
    config["generation"]["grammar"]["max_generation_retries"] = 100
    config["generation"]["rule_layers"]["groups"]["casing"] = {
        "enabled": True,
        "layers": ["casing"],
    }
    return online_generator_from_config(config, seed=13)


def _first_example_matching(generator: OnlineExampleGenerator, rule_id: str, predicate):
    for _ in range(500):
        example = generator.sample(rule_id=rule_id, mode=GenerationMode.POSITIVE)
        if predicate(example):
            return example
    raise AssertionError(f"No matching example found for {rule_id}.")


def _apply_token_labels(example: GeneratedExample) -> str:
    corrected, _edits = apply_token_edit_labels(
        example.source_text,
        example.source_tokens,
        example.token_edit_labels,
        [0.99] * len(example.source_tokens),
        threshold=0.70,
        rule_ids=example.rule_ids,
    )
    return corrected


def _scope_guard_accepts(example: GeneratedExample) -> bool:
    corrected, edits = apply_token_edit_labels(
        example.source_text,
        example.source_tokens,
        example.token_edit_labels,
        [0.99] * len(example.source_tokens),
        threshold=0.70,
        rule_ids=example.rule_ids,
    )
    valid, reasons = ScopeGuard().validate_result(example.source_text, corrected, edits)
    assert reasons == []
    return valid
