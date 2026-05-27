from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.config.load_config import load_config
from src.grammar_gen.audit import audit_batch
from src.grammar_gen.diversity import duplicate_pair_rate, sample_diverse_examples
from src.grammar_gen.factory import online_generator_from_config
from src.schema import GeneratedExample, WordToken


ROOT = Path(__file__).resolve().parents[1]
MAX_DUPLICATE_PAIR_RATE = 0.12
ENABLED_LAYER_MARKERS = {
    "compound_spelling",
    "morpheme",
    "dictionary_typo",
    "syntax_punctuation",
}


def test_audit_batch_reports_duplicate_and_template_stats() -> None:
    config = load_config(ROOT / "configs" / "config.yaml")
    generator = online_generator_from_config(config, seed=17)
    examples = sample_diverse_examples(
        generator,
        count=200,
        stream_name="unit-audit",
        max_attempts=12,
        max_duplicate_pair_rate=MAX_DUPLICATE_PAIR_RATE,
    )

    audit = audit_batch(examples)
    diversity = audit["diversity"]

    assert audit["failed_examples_count"] == 0
    assert diversity["count"] == 200
    assert diversity["unique_source_target_pairs"] >= 176
    assert diversity["duplicate_pair_rate"] <= MAX_DUPLICATE_PAIR_RATE
    assert diversity["average_token_count"] > 3.0
    assert diversity["token_edit_count_distribution"]
    assert diversity["gap_edit_count_distribution"]
    assert "duplicate_rate_by_rule_id" in diversity
    assert "duplicate_rate_by_sub_rule_id" in diversity
    assert "template_distribution" in diversity


def test_generated_2000_examples_stay_under_duplicate_pair_threshold_and_cover_layers() -> None:
    config = load_config(ROOT / "configs" / "config.yaml")
    generator = online_generator_from_config(config, seed=config["generation"]["seed"])

    examples = sample_diverse_examples(
        generator,
        count=2000,
        stream_name="unit-diversity",
        max_attempts=16,
        max_duplicate_pair_rate=MAX_DUPLICATE_PAIR_RATE,
    )
    audit = audit_batch(examples)
    pair_rate = duplicate_pair_rate(examples)
    observed_layers = {
        _layer_marker(example)
        for example in examples
        if _layer_marker(example)
    }

    assert audit["failed_examples_count"] == 0
    assert pair_rate <= MAX_DUPLICATE_PAIR_RATE, audit["diversity"]["duplicate_diagnostics"]
    assert ENABLED_LAYER_MARKERS <= observed_layers
    assert len(Counter(example.primary_rule_id for example in examples)) >= 35


def test_diverse_generation_is_deterministic_without_mutating_sample_by_index() -> None:
    config = load_config(ROOT / "configs" / "config.yaml")
    first_generator = online_generator_from_config(config, seed=31)
    second_generator = online_generator_from_config(config, seed=31)

    first = sample_diverse_examples(first_generator, count=300, stream_name="deterministic", max_attempts=8)
    second = sample_diverse_examples(second_generator, count=300, stream_name="deterministic", max_attempts=8)

    assert [example.stable_id() for example in first] == [example.stable_id() for example in second]
    assert first_generator.sample_by_index(17) == first_generator.sample_by_index(17)
    assert first_generator.sample_by_index(17) == second_generator.sample_by_index(17)


def test_different_seed_changes_diverse_generation_meaningfully() -> None:
    config = load_config(ROOT / "configs" / "config.yaml")
    first_generator = online_generator_from_config(config, seed=101)
    second_generator = online_generator_from_config(config, seed=102)

    first_ids = {
        example.stable_id()
        for example in sample_diverse_examples(first_generator, count=200, stream_name="seed-check")
    }
    second_ids = {
        example.stable_id()
        for example in sample_diverse_examples(second_generator, count=200, stream_name="seed-check")
    }

    assert len(first_ids & second_ids) <= 20


def test_audit_rejects_numbered_example_shells() -> None:
    example = GeneratedExample(
        source_text="Пример номер 4344: он проверил отчёт.",
        target_text="Пример номер 4344: он проверил отчёт.",
        source_tokens=[
            WordToken("Пример", 0, 6),
            WordToken("номер", 7, 12),
            WordToken("он", 19, 21),
            WordToken("проверил", 22, 30),
            WordToken("отчёт", 31, 36),
        ],
        token_edit_labels=["KEEP", "KEEP", "KEEP", "KEEP", "KEEP"],
        gap_labels=["NONE", "COLON", "NONE", "NONE", "DOT"],
        rule_ids=["none", "none", "none", "none", "none"],
        primary_rule_id="none",
        mode="clean_identity",
        explanation_ids=["none"],
        metadata={
            "expected_edit_count": 0,
            "expected_token_edit_count": 0,
            "expected_gap_edit_count": 0,
            "uses_safety_clauses": False,
        },
    )

    audit = audit_batch([example])

    assert audit["failed_examples_count"] == 1
    assert audit["failure_reasons"] == {"forbidden_numbered_example_shell": 1}


def _layer_marker(example) -> str:
    if example.metadata.get("layer") in ENABLED_LAYER_MARKERS:
        return str(example.metadata["layer"])
    if example.metadata.get("construction_family") == "orthography_morphemic":
        return "morpheme"
    if example.primary_rule_id.startswith("punct_"):
        return "syntax_punctuation"
    return ""
