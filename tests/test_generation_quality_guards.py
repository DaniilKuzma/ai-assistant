from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.config.load_config import load_config
from src.grammar_gen import GrammarBuilder, Lexicon, MorphologyEngine, RandomSource, Realizer
from src.grammar_gen.ast import Clause, NounPhrase, VerbPhrase
from src.grammar_gen.audit import audit_batch, audit_example
from src.grammar_gen.factory import online_generator_from_config
from src.grammar_gen.generator import GenerationError, OnlineExampleGenerator
from src.grammar_gen.rules.base import GenerationMode, RuleInfo, RuleProgram
from src.grammar_gen.rules.common import gap_labels_from_text, token_labels_all_keep
from src.grammar_gen.rules.registry import RuleRegistry, default_rule_registry
from src.grammar_gen.safety import (
    assert_json_safe_metadata,
    count_logical_token_edits,
    validate_generated_pair,
)
from src.schema import GeneratedExample


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "text",
    (
        "Девочка пошёл гулять.",
        "Комиссия решил вопрос.",
        "Студент пошла домой.",
        "Он пошёл во огород.",
    ),
)
def test_known_bad_agreement_and_vo_phrases_fail(text: str) -> None:
    example = _manual_example(text, metadata={"uses_safety_clauses": False})

    assert validate_generated_pair(example) != []


@pytest.mark.parametrize(
    "text",
    (
        "Школьница далеко провести событие.",
        "Отчёт содержать ошибка.",
        "Банк провести чистое собрание, что продавец открыть инструкция.",
        "Инженер отправил тихюю таблица.",
        "Городской цитата — сообщение.",
    ),
)
def test_bad_target_morphology_is_rejected(text: str) -> None:
    example = _manual_example(text, metadata={"uses_safety_clauses": False})

    assert validate_generated_pair(example) != []


@pytest.mark.parametrize(
    "text",
    (
        "Сосед открыл файл.",
        "Отчёт содержит ошибку.",
        "Банк провёл собрание.",
        "Инженер отправил тихую таблицу.",
        "Городская цитата — сообщение.",
    ),
)
def test_good_target_morphology_is_accepted(text: str) -> None:
    example = _manual_example(text, metadata={"uses_safety_clauses": False})

    assert validate_generated_pair(example) == []


def test_audit_includes_generated_pair_validation_reasons() -> None:
    example = _manual_example(
        "Городской цитата — сообщение.",
        metadata={"uses_safety_clauses": False},
    )

    reasons = audit_example(example)

    assert any(reason.startswith("pair_validation:") for reason in reasons)


@pytest.mark.parametrize(
    ("text", "frame_id", "verb_lemma", "subject_lemma", "subject_class", "object_lemma", "object_class"),
    (
        ("Дом купил продукты.", "buy_goods", "купить", "дом", "building", "продукты", "food"),
        ("Окно объяснило решение.", "explain_issue", "объяснить", "окно", "object", "решение", "decision"),
        ("Документ съел яблоко.", "eat_food", "съесть", "документ", "document", "яблоко", "food"),
        ("Отчёт сказал факт.", "say_fact", "сказать", "отчёт", "report", "факт", "fact"),
        ("Стол решил вопрос.", "solve_task", "решить", "стол", "object", "вопрос", "question"),
        ("Продукты прочитали документ.", "read_text", "прочитать", "продукты", "food", "документ", "document"),
    ),
)
def test_semantic_nonsense_with_safety_clause_fails(
    text: str,
    frame_id: str,
    verb_lemma: str,
    subject_lemma: str,
    subject_class: str,
    object_lemma: str,
    object_class: str,
) -> None:
    example = _manual_example(
        text,
        metadata={
            "uses_safety_clauses": True,
            "safety_clauses": [
                _safety_clause(
                    frame_id=frame_id,
                    verb_lemma=verb_lemma,
                    subject_lemma=subject_lemma,
                    subject_class=subject_class,
                    object_lemma=object_lemma,
                    object_class=object_class,
                )
            ],
        },
    )

    assert validate_generated_pair(example) != []


@pytest.mark.parametrize(
    ("text", "frame_id", "verb_lemma", "subject_lemma", "subject_class", "object_lemma", "object_class"),
    (
        ("Студент купил продукты.", "buy_goods", "купить", "студент", "person", "продукты", "food"),
        ("Комиссия утвердила решение.", "approve_decision", "утвердить", "комиссия", "organization", "решение", "decision"),
        ("Отчёт содержит ошибку.", "contain_info", "содержать", "отчёт", "report", "ошибка", "error"),
    ),
)
def test_valid_frame_safety_clauses_pass(
    text: str,
    frame_id: str,
    verb_lemma: str,
    subject_lemma: str,
    subject_class: str,
    object_lemma: str,
    object_class: str,
) -> None:
    example = _manual_example(
        text,
        metadata={
            "uses_safety_clauses": True,
            "safety_clauses": [
                _safety_clause(
                    frame_id=frame_id,
                    verb_lemma=verb_lemma,
                    subject_lemma=subject_lemma,
                    subject_class=subject_class,
                    object_lemma=object_lemma,
                    object_class=object_class,
                )
            ],
        },
    )

    assert validate_generated_pair(example) == []


def test_safety_clause_metadata_is_json_serializable_for_generated_example() -> None:
    example = _generator().sample(rule_id="ne_verb", mode=GenerationMode.POSITIVE)

    json.dumps(example.metadata["safety_clauses"], ensure_ascii=False)
    assert_json_safe_metadata(example.metadata)


def test_raw_dataclass_in_metadata_fails_json_safety() -> None:
    example = _manual_example(
        "Студент купил продукты.",
        metadata={"uses_safety_clauses": False, "bad": _RawMetadata("not json safe")},
    )

    assert "metadata_not_json_safe" in validate_generated_pair(example)


def test_missing_required_safety_clauses_fails() -> None:
    example = _manual_example(
        "Студент купил продукты.",
        metadata={"uses_safety_clauses": True},
    )

    assert "missing_safety_clauses" in validate_generated_pair(example)


def test_skip_merged_counts_only_as_merge_continuation() -> None:
    assert count_logical_token_edits(["KEEP", "MERGE_TAK_ZHE_TO_TAKZHE", "SKIP_MERGED", "KEEP"]) == 1
    assert count_logical_token_edits(["KEEP", "HYPHENATE_PARTICLE_TO", "SKIP_MERGED"]) == 1
    assert count_logical_token_edits(["KEEP", "SPLIT_NE_VERB", "KEEP"]) == 1

    with pytest.raises(ValueError, match="SKIP_MERGED"):
        count_logical_token_edits(["KEEP", "SKIP_MERGED"])


def test_online_generator_1000_examples_have_no_safety_failures() -> None:
    generator = _generator()

    failures = [
        (index, example.primary_rule_id, validate_generated_pair(example), example.source_text, example.target_text)
        for index in range(1000)
        for example in [generator.sample_by_index(index)]
        if validate_generated_pair(example)
    ]

    assert failures == []


@pytest.mark.slow
def test_online_generator_5000_examples_have_no_semantic_audit_failures() -> None:
    generator = _generator()

    failures = [
        (index, example.primary_rule_id, validate_generated_pair(example), example.source_text, example.target_text)
        for index in range(5000)
        for example in [generator.sample_by_index(index)]
        if validate_generated_pair(example)
    ]

    assert failures == []


def test_production_generator_5000_examples_quality_gate() -> None:
    generator = _generator()
    examples = [generator.sample_by_index(index) for index in range(5000)]
    audit = audit_batch(examples)
    validation_failures = [
        (index, example.primary_rule_id, validate_generated_pair(example), example.source_text, example.target_text)
        for index, example in enumerate(examples)
        if validate_generated_pair(example)
    ]
    bad_target_patterns = (
        " провести ",
        " открыть ",
        " найти ",
        " закрыть ",
        " съесть ",
        " произойти ",
        "громкее",
        "тихюю",
        "городской цитата",
    )
    bad_targets = [
        (index, example.target_text)
        for index, example in enumerate(examples)
        if any(pattern in f" {example.target_text.lower()} " for pattern in bad_target_patterns)
    ]
    unique_pairs = {
        (example.source_text, example.target_text, example.primary_rule_id, example.mode)
        for example in examples
    }
    duplicate_ratio = 1.0 - (len(unique_pairs) / len(examples))

    assert audit["failed_examples_count"] == 0
    assert validation_failures == []
    assert bad_targets == []
    assert duplicate_ratio < 0.35


def test_ne_verb_positive_expected_edit_count_is_logical_one() -> None:
    example = _generator().sample(rule_id="ne_verb", mode=GenerationMode.POSITIVE)

    assert example.metadata["expected_edit_count"] == 1
    assert example.metadata["expected_token_edit_count"] == 1
    assert example.metadata["expected_gap_edit_count"] == 0
    assert validate_generated_pair(example) == []


def test_comma_introductory_medial_can_expect_two_gap_edits() -> None:
    generator = _generator()
    examples = [
        generator.sample(rule_id="comma_introductory", mode=GenerationMode.POSITIVE)
        for _ in range(100)
    ]
    medial = next(example for example in examples if example.metadata.get("introductory_position") == "medial")

    assert medial.metadata["expected_edit_count"] == 2
    assert medial.metadata["expected_gap_edit_count"] == 2
    assert validate_generated_pair(medial) == []


def test_generated_example_with_safety_clauses_has_serialized_clause_data() -> None:
    example = _generator().sample(rule_id="comma_subordinate", mode=GenerationMode.POSITIVE)

    assert example.metadata["uses_safety_clauses"] is True
    assert example.metadata["safety_clauses"]
    first_clause = example.metadata["safety_clauses"][0]
    assert first_clause["predicate"]["tense"] in {"past", "present"}
    assert first_clause["predicate"]["verb_lemma"]
    assert first_clause["predicate"]["rendered_verb"]
    assert first_clause["subject"]["surface"]
    assert first_clause["subject"]["case"] == "nomn"
    if first_clause.get("object") is not None:
        assert first_clause["object"]["surface"]
        assert first_clause["object"]["case"] in {"accs", "gent", "datv", "ablt", "loct", "nomn"}
    assert_json_safe_metadata(example.metadata)


def test_curated_morphology_covers_generation_lexicon() -> None:
    lexicon = Lexicon.default()
    required_noun_forms = {"nom_sg", "gen_sg", "dat_sg", "acc_sg", "ins_sg", "loc_sg", "nom_pl", "acc_pl"}
    required_adjective_forms = {
        "masc_nom",
        "fem_nom",
        "neut_nom",
        "plur_nom",
        "fem_acc",
        "masc_acc_inanim",
        "neut_acc",
    }
    required_verb_forms = {
        "past_masc",
        "past_fem",
        "past_neut",
        "past_plur",
        "present_3sg",
        "infinitive",
    }

    noun_failures = [
        noun.lemma
        for noun in lexicon.nouns
        if not required_noun_forms <= set(noun.forms)
    ]
    adjective_failures = [
        adjective.lemma
        for adjective in lexicon.adjectives
        if not required_adjective_forms <= set(adjective.forms)
    ]
    frame_verbs = {frame.verb_lemma for frame in lexicon.frames.frames}
    verb_failures = [
        verb.lemma
        for verb in lexicon.verbs
        if verb.lemma in frame_verbs and not required_verb_forms <= set(verb.forms)
    ]

    assert noun_failures == []
    assert adjective_failures == []
    assert verb_failures == []


def test_generation_error_reports_last_invalid_pair_details() -> None:
    registry = RuleRegistry()
    registry.register_rule(_AlwaysBadRule())
    generator = OnlineExampleGenerator(
        registry,
        Lexicon.default(),
        MorphologyEngine(use_pymorphy=False),
        {"generation": {"mix": {"orthography_contextual": 1.0}, "grammar": {"max_generation_retries": 1}}},
        seed=13,
    )

    with pytest.raises(GenerationError) as exc_info:
        generator.sample(rule_id="ne_verb", mode=GenerationMode.POSITIVE)

    message = str(exc_info.value)
    assert "rule_id='ne_verb'" in message
    assert "mode='positive'" in message
    assert "last_source='Девочка пошёл гулять.'" in message
    assert "last_target='Девочка пошёл гулять.'" in message
    assert "bad_pair_devochka_poshel" in message


@dataclass(frozen=True)
class _RawMetadata:
    value: str


class _AlwaysBadRule(RuleProgram):
    info = RuleInfo(
        rule_id="ne_verb",
        family="orthography_contextual",
        description="always bad",
        explanation="always bad",
    )
    supported_modes = (GenerationMode.POSITIVE,)

    def generate(
        self,
        builder: GrammarBuilder,
        realizer: Realizer,
        rng: RandomSource,
        mode: GenerationMode,
    ) -> GeneratedExample:
        del builder, rng, mode
        text = "Девочка пошёл гулять."
        tokens = realizer.tokenize_words_with_offsets(text)
        return GeneratedExample(
            source_text=text,
            target_text=text,
            source_tokens=tokens,
            token_edit_labels=token_labels_all_keep(tokens),
            gap_labels=gap_labels_from_text(text, tokens),
            rule_ids=[self.info.rule_id] * len(tokens),
            primary_rule_id=self.info.rule_id,
            mode=GenerationMode.POSITIVE.value,
            explanation_ids=[self.info.rule_id],
            metadata={"uses_safety_clauses": False},
        )


def _generator() -> OnlineExampleGenerator:
    config = load_config(ROOT / "configs" / "config.yaml")
    return online_generator_from_config(config, seed=config["generation"]["seed"])


def _manual_example(text: str, *, metadata: dict) -> GeneratedExample:
    realizer = Realizer(Lexicon.default(), MorphologyEngine(use_pymorphy=False))
    tokens = realizer.tokenize_words_with_offsets(text)
    metadata = {
        "expected_edit_count": 0,
        "expected_token_edit_count": 0,
        "expected_gap_edit_count": 0,
        **metadata,
    }
    return GeneratedExample(
        source_text=text,
        target_text=text,
        source_tokens=tokens,
        token_edit_labels=token_labels_all_keep(tokens),
        gap_labels=gap_labels_from_text(text, tokens),
        rule_ids=["clean_identity"] * len(tokens),
        primary_rule_id="clean_identity",
        mode=GenerationMode.CLEAN_IDENTITY.value,
        explanation_ids=[],
        metadata=metadata,
    )


def _safety_clause(
    *,
    frame_id: str,
    verb_lemma: str,
    subject_lemma: str,
    subject_class: str,
    object_lemma: str | None,
    object_class: str | None,
) -> dict:
    frame = next((frame for frame in Lexicon.default().frames.frames if frame.frame_id == frame_id), None)
    allowed_subject_classes = list(frame.subject_classes) if frame is not None else []
    allowed_object_classes = list(frame.object_classes) if frame is not None else []
    return {
        "frame_id": frame_id,
        "verb_lemma": verb_lemma,
        "frame_family": frame.frame_family if frame is not None else "",
        "subject": {
            "lemma": subject_lemma,
            "semantic_class": subject_class,
            "gender": "masc",
            "number": "sing",
            "animacy": "inanim",
        },
        "object": None
        if object_lemma is None
        else {
            "lemma": object_lemma,
            "semantic_class": object_class,
            "gender": "masc",
            "number": "sing",
            "animacy": "inanim",
        },
        "allowed_subject_classes": allowed_subject_classes,
        "allowed_object_classes": allowed_object_classes,
        "prep_slots": [],
    }
