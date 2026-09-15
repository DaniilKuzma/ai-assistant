from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.config.load_config import load_config
from src.grammar_gen.audit import audit_batch
from src.grammar_gen.factory import online_generator_from_config
from src.grammar_gen.randomness import RandomSource
from src.grammar_gen.rules.base import GenerationMode
from src.grammar_gen.rules.registry import default_rule_registry
from src.grammar_gen.safety import validate_generated_pair
from src.orthography_gen.compiler import OrthographicScenarioCompiler
from src.runtime.edit_realizer import apply_token_edit_labels
from src.runtime.orthographic_lexicon import OrthographicCorrectionLexicon
from src.schema import GeneratedExample


ROOT = Path(__file__).resolve().parents[1]
LEGACY_RULE_IDS = {"suffix_its_ets", "suffix_enn_yan", "n_nn_basic"}
MORPHEME_RULE_IDS = {
    "morpheme_hissing_vowels",
    "morpheme_soft_hard_signs",
    "morpheme_root_vowels",
    "morpheme_prefixes",
    "morpheme_suffixes",
    "morpheme_n_nn",
    "morpheme_consonants",
    "morpheme_endings",
}
ALL_MORPHEME_RULE_IDS = LEGACY_RULE_IDS | MORPHEME_RULE_IDS


def test_compiler_registers_all_morpheme_specs_and_cards() -> None:
    compiler = OrthographicScenarioCompiler.default()

    assert ALL_MORPHEME_RULE_IDS <= set(compiler.specs)
    assert ALL_MORPHEME_RULE_IDS <= set(compiler.cards_by_rule)
    for rule_id in ALL_MORPHEME_RULE_IDS:
        assert compiler.has_mode(rule_id, GenerationMode.POSITIVE), rule_id


def test_dynamic_rule_programs_include_new_and_legacy_morpheme_rules() -> None:
    registry_ids = {rule.info.rule_id for rule in default_rule_registry().all_rules()}

    assert ALL_MORPHEME_RULE_IDS <= registry_ids


def test_each_morpheme_rule_generates_positive_dict_replace_metadata() -> None:
    compiler = OrthographicScenarioCompiler.default()
    lexicon = OrthographicCorrectionLexicon.default()

    for index, rule_id in enumerate(sorted(ALL_MORPHEME_RULE_IDS), start=100):
        example = compiler.compile_example(rule_id, GenerationMode.POSITIVE, RandomSource(seed=index))

        assert example.primary_rule_id == rule_id
        assert example.token_edit_labels.count("DICT_REPLACE") == 1
        assert validate_generated_pair(example) == []
        assert GeneratedExample.from_dict(example.to_dict()) == example

        metadata = example.metadata
        assert metadata["layer"] == "morpheme"
        assert metadata["sub_rule_id"]
        assert metadata["site_type"] in {"root", "prefix", "suffix", "ending", "sign", "consonant"}
        assert isinstance(metadata["orthography_site"], dict)
        assert metadata["correct_form"] in example.target_text
        assert metadata["wrong_form"] in example.source_text
        assert metadata["replacement"] == {
            "source": metadata["wrong_form"],
            "target": metadata["correct_form"],
        }

        entries = lexicon.lookup(metadata["wrong_form"], rule_id=rule_id, operation="dict_replace")
        assert {entry.target for entry in entries} == {metadata["correct_form"]}


def test_large_morpheme_rules_have_hard_negatives() -> None:
    compiler = OrthographicScenarioCompiler.default()

    for index, rule_id in enumerate(sorted(MORPHEME_RULE_IDS), start=500):
        example = compiler.compile_example(rule_id, GenerationMode.HARD_NEGATIVE, RandomSource(seed=index))

        assert example.source_text == example.target_text
        assert set(example.token_edit_labels) == {"KEEP"}
        assert example.metadata["expected_edit_count"] == 0
        assert example.metadata["layer"] == "morpheme"
        assert example.metadata["wrong_form"] == example.metadata["correct_form"]
        assert validate_generated_pair(example) == []


def test_generated_dict_replace_applies_through_runtime_lexicon() -> None:
    compiler = OrthographicScenarioCompiler.default()
    lexicon = OrthographicCorrectionLexicon.default()

    for index, rule_id in enumerate(sorted(MORPHEME_RULE_IDS), start=800):
        example = compiler.compile_example(rule_id, GenerationMode.POSITIVE, RandomSource(seed=index))
        confidences = [1.0] * len(example.source_tokens)

        corrected, edits = apply_token_edit_labels(
            example.source_text,
            example.source_tokens,
            example.token_edit_labels,
            confidences,
            threshold=0.7,
            rule_ids=example.rule_ids,
            orthographic_lexicon=lexicon,
        )

        assert corrected == example.target_text
        assert len(edits) == 1
        assert edits[0].rule_id == rule_id


def test_morpheme_only_generator_audit_and_duplicate_threshold() -> None:
    config = load_config(ROOT / "configs" / "config.yaml")
    config["generation"]["enabled_rule_groups"] = ["orthography_morphemic"]
    config["generation"]["mix"] = {"orthography_morphemic": 1.0}
    config["generation"]["grammar"]["max_generation_retries"] = 50
    generator = online_generator_from_config(config, seed=303)

    examples = [generator.sample_by_index(index) for index in range(1000)]
    audit = audit_batch(examples)
    pair_counts = Counter((example.source_text, example.target_text) for example in examples)

    assert audit["failed_examples_count"] == 0
    assert MORPHEME_RULE_IDS <= {example.primary_rule_id for example in examples}
    assert max(pair_counts.values()) <= 20


def test_morpheme_examples_use_varied_natural_contexts_for_finite_cards() -> None:
    config = load_config(ROOT / "configs" / "config.yaml")
    config["generation"]["enabled_rule_groups"] = ["orthography_morphemic"]
    config["generation"]["mix"] = {"orthography_morphemic": 1.0}
    config["generation"]["grammar"]["max_generation_retries"] = 50
    generator = online_generator_from_config(config, seed=515)

    examples = [
        generator.sample(rule_id="morpheme_n_nn", mode=GenerationMode.POSITIVE)
        for _ in range(180)
    ]
    unique_pairs = {(example.source_text, example.target_text) for example in examples}

    assert len(unique_pairs) >= 145
    assert not any("пример номер" in example.source_text.casefold() for example in examples)


def test_config_enabled_morpheme_specs_have_executable_cards() -> None:
    config = load_config(ROOT / "configs" / "config.yaml")
    enabled = set(config["generation"]["orthography_morphemic"]["rules"])
    compiler = OrthographicScenarioCompiler.default()

    assert ALL_MORPHEME_RULE_IDS <= enabled
    for rule_id in enabled:
        assert rule_id in compiler.specs
        assert compiler.has_mode(rule_id, GenerationMode.POSITIVE)
