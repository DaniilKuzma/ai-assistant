from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.grammar_gen.randomness import RandomSource
from src.grammar_gen.rules.base import GenerationMode
from src.grammar_gen.safety import assert_json_safe_metadata, validate_generated_pair
from src.orthography_gen.compiler import OrthographicScenarioCompiler
from src.runtime.orthographic_lexicon import OrthographicCorrectionLexicon
from src.schema import GeneratedExample


def test_suffix_its_ets_positive_generates_dict_replace_metadata() -> None:
    compiler = OrthographicScenarioCompiler.default()

    examples = [
        compiler.compile_example("suffix_its_ets", GenerationMode.POSITIVE, RandomSource(seed=index))
        for index in range(30)
    ]
    example = next(item for item in examples if "платьеце" in item.source_text)

    assert "платьеце" in example.source_text
    assert "платьице" in example.target_text
    assert example.token_edit_labels.count("DICT_REPLACE") == 1
    assert example.metadata["replacement"] == {"source": "платьеце", "target": "платьице"}
    assert example.primary_rule_id == "suffix_its_ets"
    assert example.metadata["orthography_site"]["site_type"] == "suffix"
    assert example.metadata["orthography_site"]["stress_position"] == "before_suffix"


def test_suffix_enn_yan_positive_generates_known_replacement_and_site_metadata() -> None:
    compiler = OrthographicScenarioCompiler.default()

    examples = [
        compiler.compile_example("suffix_enn_yan", GenerationMode.POSITIVE, RandomSource(seed=index))
        for index in range(60)
    ]
    matching = [
        item
        for item in examples
        if ("коженный" in item.source_text and "кожаный" in item.target_text)
        or ("буквяный" in item.source_text and "буквенный" in item.target_text)
    ]

    assert matching
    example = matching[0]
    assert example.token_edit_labels.count("DICT_REPLACE") == 1
    assert example.metadata["orthography_site"]["site_type"] == "suffix"
    assert example.metadata["orthography_rule_spec"]["rule_id"] == "suffix_enn_yan"


def test_maslyany_masleny_contextual_pairs_have_directional_targets() -> None:
    compiler = OrthographicScenarioCompiler.default()

    examples = compiler.compile_batch(
        200,
        rules=["suffix_enn_yan"],
        rng=RandomSource(seed=310),
    )
    oil = next(item for item in examples if item.metadata.get("context_class") == "oil")
    food = next(item for item in examples if item.metadata.get("context_class") in {"food", "smeared"})

    assert "масленый" in oil.source_text
    assert "масляный" in oil.target_text
    assert oil.metadata["replacement"] == {"source": "масленый", "target": "масляный"}
    assert "масляный" in food.source_text
    assert "масленый" in food.target_text
    assert food.metadata["replacement"] == {"source": "масляный", "target": "масленый"}


def test_n_nn_hard_negative_and_dependent_word_positive() -> None:
    compiler = OrthographicScenarioCompiler.default()

    hard_negatives = [
        compiler.compile_example("n_nn_basic", GenerationMode.HARD_NEGATIVE, RandomSource(seed=index))
        for index in range(30)
    ]
    hard_negative = next(item for item in hard_negatives if "крашеный забор" in item.source_text.lower())

    assert hard_negative.source_text == hard_negative.target_text
    assert set(hard_negative.token_edit_labels) == {"KEEP"}
    assert hard_negative.metadata["expected_edit_count"] == 0

    positives = [
        compiler.compile_example("n_nn_basic", GenerationMode.POSITIVE, RandomSource(seed=index))
        for index in range(60)
    ]
    positive = next(item for item in positives if "крашеный вчера забор" in item.source_text.lower())

    assert "крашенный вчера забор" in positive.target_text.lower()
    assert positive.token_edit_labels.count("DICT_REPLACE") == 1
    assert positive.metadata["replacement"] == {"source": "крашеный", "target": "крашенный"}


def test_generated_examples_validate_and_metadata_is_json_safe() -> None:
    compiler = OrthographicScenarioCompiler.default()

    examples = [
        compiler.compile_example(rule_id, mode, RandomSource(seed=index))
        for index, (rule_id, mode) in enumerate(
            [
                ("suffix_its_ets", GenerationMode.POSITIVE),
                ("suffix_enn_yan", GenerationMode.POSITIVE),
                ("n_nn_basic", GenerationMode.POSITIVE),
                ("n_nn_basic", GenerationMode.HARD_NEGATIVE),
            ],
            start=100,
        )
    ]

    for example in examples:
        assert GeneratedExample.from_json(example.to_json()) == example
        assert validate_generated_pair(example) == []
        assert_json_safe_metadata(example.metadata)
        json.dumps(example.metadata, ensure_ascii=False)


def test_runtime_orthographic_lexicon_lookups_and_ambiguity(tmp_path: Path) -> None:
    lexicon = OrthographicCorrectionLexicon.default()

    kozha = lexicon.lookup("коженный")
    platye = lexicon.lookup("платьеце")

    assert any(entry.target == "кожаный" and entry.rule_id == "suffix_enn_yan" for entry in kozha)
    assert any(entry.target == "платьице" and entry.rule_id == "suffix_its_ets" for entry in platye)

    fixture_dir = tmp_path / "orthography"
    fixture_dir.mkdir()
    (fixture_dir / "ambiguous.yaml").write_text(
        yaml.safe_dump(
            {
                "rule_spec": {
                    "rule_id": "suffix_enn_yan",
                    "orfogrammka_id": "fixture",
                    "family": "suffix_enn_yan",
                    "pos": "ADJF",
                    "site_type": "suffix",
                    "requires": ["semantic_context"],
                    "correct_patterns": ["fixture"],
                    "wrong_patterns": ["fixture"],
                    "explanation_id": "fixture",
                    "model_role": "context_disambiguation",
                },
                "lexeme_cards": [
                    {
                        "rule_id": "suffix_enn_yan",
                        "correct_lemma": "масляный",
                        "wrong_lemma": "масленый",
                        "pos": "ADJF",
                        "correct_site": "ян",
                        "wrong_site": "ен",
                        "forms": {"nomn_masc_sing": {"correct": "масляный", "wrong": "масленый"}},
                        "safe_contexts": [{"context_class": "oil", "noun": "соус"}],
                        "explanation_id": "fixture",
                    },
                    {
                        "rule_id": "suffix_enn_yan",
                        "correct_lemma": "масленый",
                        "wrong_lemma": "масляный",
                        "pos": "ADJF",
                        "correct_site": "ен",
                        "wrong_site": "ян",
                        "forms": {"nomn_masc_sing": {"correct": "масленый", "wrong": "масляный"}},
                        "safe_contexts": [{"context_class": "food", "noun": "блин"}],
                        "explanation_id": "fixture",
                    },
                ],
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    ambiguous = OrthographicCorrectionLexicon.from_dir(fixture_dir)

    assert len(ambiguous.lookup("масленый")) == 1
    assert len(ambiguous.lookup("масляный")) == 1
    assert {entry.target for entry in ambiguous.lookup_any(["масленый", "масляный"])} == {"масляный", "масленый"}
