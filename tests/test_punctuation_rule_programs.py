from __future__ import annotations

import pytest

from src.grammar_gen import Lexicon, MorphologyEngine, Realizer
from src.grammar_gen.generator import OnlineExampleGenerator
from src.grammar_gen.rules.base import GenerationMode
from src.grammar_gen.rules.registry import default_rule_registry
from src.grammar_gen.safety import validate_generated_pair, validate_surface
from src.schema import GeneratedExample, WordToken


EXPECTED_POSITIVE_GAPS = {
    "comma_subordinate": {"COMMA"},
    "comma_introductory": {"COMMA"},
    "comma_homogeneous": {"COMMA"},
    "comma_adversative": {"COMMA"},
    "dash_subject_predicate": {"DASH"},
    "final_punctuation": {"DOT", "QUESTION", "EXCLAMATION"},
}
RULE_IDS = tuple(EXPECTED_POSITIVE_GAPS)


@pytest.mark.parametrize("rule_id", RULE_IDS)
@pytest.mark.parametrize(
    "mode",
    (
        GenerationMode.POSITIVE,
        GenerationMode.HARD_NEGATIVE,
        GenerationMode.CLEAN_IDENTITY,
    ),
)
def test_punctuation_rules_generate_valid_examples(rule_id: str, mode: GenerationMode) -> None:
    generator = _generator(seed=17)
    rule = default_rule_registry().get_rule(rule_id)

    assert rule is not None
    assert rule.can_generate(mode)

    example = generator.sample(rule_id=rule_id, mode=mode)

    assert GeneratedExample.from_dict(example.to_dict()) == example
    assert len(example.gap_labels) == len(example.source_tokens)
    assert set(example.token_edit_labels) == {"KEEP"}
    assert validate_generated_pair(example) == []
    assert validate_surface(example.target_text) == []

    if mode is GenerationMode.POSITIVE:
        assert example.source_text != example.target_text
        assert set(example.gap_labels) & EXPECTED_POSITIVE_GAPS[rule_id]
    else:
        assert example.source_text == example.target_text

    if mode is GenerationMode.HARD_NEGATIVE:
        unexpected = {"COMMA", "DASH"} & set(example.gap_labels)
        assert unexpected == set()


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_punctuation_rule_samples_are_surface_safe(rule_id: str) -> None:
    generator = _generator(seed=100 + RULE_IDS.index(rule_id))

    for index in range(50):
        mode = (
            GenerationMode.POSITIVE,
            GenerationMode.HARD_NEGATIVE,
            GenerationMode.CLEAN_IDENTITY,
        )[index % 3]
        example = generator.sample(rule_id=rule_id, mode=mode)

        assert validate_generated_pair(example) == []
        assert validate_surface(example.target_text) == []
        if not (rule_id == "final_punctuation" and mode is GenerationMode.POSITIVE):
            assert validate_surface(example.source_text) == []


def test_comma_subordinate_hard_negative_has_no_comma_before_chto_to() -> None:
    generator = _generator(seed=23)

    examples = [
        generator.sample(rule_id="comma_subordinate", mode=GenerationMode.HARD_NEGATIVE)
        for _ in range(20)
    ]
    trap = next(example for example in examples if "что-то" in example.source_text.lower())
    trap_tokens = [token.text.lower() for token in trap.source_tokens]
    trap_index = trap_tokens.index("что-то")

    assert trap.gap_labels[trap_index - 1] != "COMMA"


def test_final_punctuation_positive_generates_dot_question_and_exclamation() -> None:
    generator = _generator(seed=31)
    labels = {
        generator.sample(rule_id="final_punctuation", mode=GenerationMode.POSITIVE).gap_labels[-1]
        for _ in range(30)
    }

    assert {"DOT", "QUESTION", "EXCLAMATION"}.issubset(labels)


def test_final_punctuation_positive_pair_safety_allows_expected_missing_final_mark() -> None:
    example = _manual_example(
        source="Редактор проверил отчёт",
        target="Редактор проверил отчёт.",
        primary_rule_id="final_punctuation",
        mode=GenerationMode.POSITIVE,
        gap_labels=["NONE", "NONE", "DOT"],
        rule_ids=["none", "none", "final_punctuation"],
        metadata={"expected_error": "missing_final_punctuation"},
    )

    assert validate_surface(example.source_text) == ["missing_final_punctuation"]
    assert validate_generated_pair(example) == []


def test_final_punctuation_positive_pair_safety_allows_expected_missing_ellipsis() -> None:
    example = _manual_example(
        source="Редактор проверил отчёт",
        target="Редактор проверил отчёт…",
        primary_rule_id="final_punctuation",
        mode=GenerationMode.POSITIVE,
        gap_labels=["NONE", "NONE", "ELLIPSIS"],
        rule_ids=["none", "none", "final_punctuation"],
        metadata={"expected_error": "missing_final_punctuation"},
    )

    assert validate_generated_pair(example) == []


def test_pair_safety_rejects_missing_final_mark_for_other_rules() -> None:
    example = _manual_example(
        source="Редактор непроверил отчёт",
        target="Редактор не проверил отчёт.",
        primary_rule_id="ne_verb",
        mode=GenerationMode.POSITIVE,
        gap_labels=["NONE", "NONE", "NONE"],
        rule_ids=["none", "ne_verb", "none"],
    )

    assert "missing_final_punctuation" in validate_generated_pair(example)


@pytest.mark.parametrize("mode", (GenerationMode.HARD_NEGATIVE, GenerationMode.CLEAN_IDENTITY))
def test_pair_safety_rejects_final_punctuation_non_positive_missing_final_mark(
    mode: GenerationMode,
) -> None:
    example = _manual_example(
        source="Редактор проверил отчёт",
        target="Редактор проверил отчёт",
        primary_rule_id="final_punctuation",
        mode=mode,
        gap_labels=["NONE", "NONE", "NONE"],
        rule_ids=["final_punctuation", "none", "none"],
        metadata={"expected_error": "missing_final_punctuation"},
    )

    assert "missing_final_punctuation" in validate_generated_pair(example)


def test_pair_safety_rejects_final_punctuation_positive_with_other_surface_failures() -> None:
    example = _manual_example(
        source="Редактор проверил {отчёт",
        target="Редактор проверил {отчёт.",
        primary_rule_id="final_punctuation",
        mode=GenerationMode.POSITIVE,
        gap_labels=["NONE", "NONE", "DOT"],
        rule_ids=["none", "none", "final_punctuation"],
        metadata={"expected_error": "missing_final_punctuation"},
    )

    failures = validate_generated_pair(example)

    assert "template_brace" in failures


def test_pair_safety_rejects_final_punctuation_positive_with_non_final_difference() -> None:
    example = _manual_example(
        source="Редактор проверил отчёт",
        target="Редактор исправил отчёт.",
        primary_rule_id="final_punctuation",
        mode=GenerationMode.POSITIVE,
        gap_labels=["NONE", "NONE", "DOT"],
        rule_ids=["none", "none", "final_punctuation"],
        metadata={"expected_error": "missing_final_punctuation"},
    )

    assert "missing_final_punctuation" in validate_generated_pair(example)


def test_pair_safety_rejects_final_punctuation_positive_without_last_gap_label() -> None:
    example = _manual_example(
        source="Редактор проверил отчёт",
        target="Редактор проверил отчёт.",
        primary_rule_id="final_punctuation",
        mode=GenerationMode.POSITIVE,
        gap_labels=["NONE", "NONE", "NONE"],
        rule_ids=["none", "none", "final_punctuation"],
        metadata={"expected_error": "missing_final_punctuation"},
    )

    assert "missing_final_punctuation" in validate_generated_pair(example)


def _generator(seed: int) -> OnlineExampleGenerator:
    lexicon = Lexicon.default()
    morphology = MorphologyEngine(use_pymorphy=False)
    config = {
        "generation": {
            "mix": {"punctuation": 1.0},
            "grammar": {"max_generation_retries": 30},
        }
    }
    return OnlineExampleGenerator(default_rule_registry(), lexicon, morphology, config, seed=seed)


def _manual_example(
    *,
    source: str,
    target: str,
    primary_rule_id: str,
    mode: GenerationMode,
    gap_labels: list[str],
    rule_ids: list[str],
    metadata: dict[str, str] | None = None,
) -> GeneratedExample:
    tokens = _tokens(source)
    return GeneratedExample(
        source_text=source,
        target_text=target,
        source_tokens=tokens,
        token_edit_labels=["KEEP"] * len(tokens),
        gap_labels=gap_labels,
        rule_ids=rule_ids,
        primary_rule_id=primary_rule_id,
        mode=mode.value,
        explanation_ids=[primary_rule_id],
        metadata=metadata or {},
    )


def _tokens(text: str) -> list[WordToken]:
    realizer = Realizer(Lexicon.default(), MorphologyEngine(use_pymorphy=False))
    return realizer.tokenize_words_with_offsets(text)
