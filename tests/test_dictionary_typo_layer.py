from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from src.config.load_config import load_config
from src.grammar_gen import Lexicon, MorphologyEngine, Realizer
from src.grammar_gen.audit import audit_batch
from src.grammar_gen.factory import online_generator_from_config
from src.grammar_gen.randomness import RandomSource
from src.grammar_gen.rules.base import GenerationMode
from src.grammar_gen.safety import validate_generated_pair
from src.rule_layers.dictionary_typo import (
    RUSSIAN_KEYBOARD_NEIGHBORS,
    generate_glued_phrase_typos,
    generate_keyboard_neighbor_typos,
    generate_typos_for_word,
    generate_word_split_typos,
    is_protected_token,
    load_dictionary_typo_corrections,
    load_dictionary_typo_specs,
)
from src.rule_layers.example_builders import build_generated_example_from_case
from src.runtime.edit_realizer import apply_token_edit_labels
from src.runtime.orthographic_lexicon import CorrectionEntry, OrthographicCorrectionLexicon
from src.runtime.tokenization import tokenize_runtime_words
from src.schema import GeneratedExample


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RULE_IDS = {
    "dictionary_normative_words",
    "dictionary_borrowed_words",
    "dictionary_domain_terms",
    "dictionary_common_misspellings",
    "typo_character_noise",
    "typo_keyboard_neighbor",
    "typo_space_noise",
}
REQUIRED_WORDS = {
    "аккуратный",
    "искусственный",
    "территория",
    "коллектив",
    "апелляция",
    "грамматика",
    "библиотека",
    "периодически",
    "обожаю",
    "здесь",
    "лестница",
    "сердце",
    "праздник",
    "честный",
    "чувствовать",
    "агентство",
    "маркетинг",
    "менеджмент",
    "дистрибьютор",
    "риелтор",
    "промоушен",
    "аккаунт",
    "офлайн",
    "онлайн",
    "блогер",
    "интерфейс",
    "фреймворк",
    "алгоритм",
    "модель",
    "корпус",
    "датасет",
    "токенизация",
    "документация",
    "ассистент",
    "пунктуация",
    "орфография",
    "корректура",
    "впоследствии",
    "вперемешку",
    "наперекор",
    "поодиночке",
}


def test_dictionary_typo_specs_cover_required_groups_modes_and_words() -> None:
    specs = load_dictionary_typo_specs(ROOT / "lexicon" / "layers", seed=17)
    by_rule = {spec.rule_id: spec for spec in specs}
    correct_words = {
        str(case.metadata.get("target") or case.metadata.get("correct") or "")
        for spec in specs
        for case in spec.cases
    }

    assert EXPECTED_RULE_IDS <= set(by_rule)
    assert REQUIRED_WORDS <= correct_words

    for rule_id in EXPECTED_RULE_IDS:
        cases = by_rule[rule_id].cases
        modes = {case.mode for case in cases}
        assert GenerationMode.POSITIVE.value in modes, rule_id
        assert GenerationMode.HARD_NEGATIVE.value in modes, rule_id
        assert GenerationMode.CLEAN_IDENTITY.value in modes, rule_id


def test_typo_generator_is_reproducible_by_seed() -> None:
    first = load_dictionary_typo_corrections(ROOT / "lexicon" / "layers", seed=101)
    second = load_dictionary_typo_corrections(ROOT / "lexicon" / "layers", seed=101)
    third = load_dictionary_typo_corrections(ROOT / "lexicon" / "layers", seed=102)

    first_generated = [asdict(item) for item in first if item.metadata.get("generated")]
    second_generated = [asdict(item) for item in second if item.metadata.get("generated")]
    third_generated = [asdict(item) for item in third if item.metadata.get("generated")]

    assert first_generated
    assert first_generated == second_generated
    assert first_generated != third_generated


def test_keyboard_neighbor_map_works_on_russian_layout() -> None:
    assert "д" in RUSSIAN_KEYBOARD_NEIGHBORS["л"]
    assert "модедь" in generate_keyboard_neighbor_typos("модель")


def test_space_typo_helpers_cover_split_and_glue_operations() -> None:
    assert "орфо графия" in generate_word_split_typos("орфография", seed=3, max_count=50)
    assert "вотчёте" in generate_glued_phrase_typos("в отчёте")
    assert generate_glued_phrase_typos("AB-123 отчёт") == ()


def test_no_generated_positive_without_runtime_correction_target() -> None:
    specs = load_dictionary_typo_specs(ROOT / "lexicon" / "layers", seed=13)
    lexicon = OrthographicCorrectionLexicon.from_root(ROOT / "lexicon")

    for spec in specs:
        for case in spec.cases:
            if case.mode != GenerationMode.POSITIVE.value:
                continue
            source = str(case.metadata["source"])
            target = str(case.metadata["target"])
            operation = str(case.metadata["runtime_operation"])
            entries = lexicon.lookup(source, rule_id=case.rule_id, operation=operation)
            assert {entry.target for entry in entries} == {target}, case.metadata


def test_dictionary_misspelling_uses_dict_replace_and_runtime_lexicon() -> None:
    example = _example_for("dictionary_common_misspellings", source="граматика")
    lexicon = OrthographicCorrectionLexicon.from_root(ROOT / "lexicon")

    corrected, edits = apply_token_edit_labels(
        example.source_text,
        example.source_tokens,
        example.token_edit_labels,
        [1.0] * len(example.source_tokens),
        threshold=0.7,
        rule_ids=example.rule_ids,
        orthographic_lexicon=lexicon,
    )

    assert example.token_edit_labels.count("DICT_REPLACE") == 1
    assert corrected == example.target_text
    assert [(edit.source, edit.replacement, edit.rule_id, edit.edit_type) for edit in edits] == [
        ("граматика", "грамматика", "dictionary_common_misspellings", "spelling")
    ]


def test_split_and_glue_typos_use_span_replace_by_lexicon() -> None:
    lexicon = OrthographicCorrectionLexicon.from_root(ROOT / "lexicon")
    for source in ("орфо графия", "вотчёте"):
        example = _example_for("typo_space_noise", source=source)

        corrected, edits = apply_token_edit_labels(
            example.source_text,
            example.source_tokens,
            example.token_edit_labels,
            [1.0] * len(example.source_tokens),
            threshold=0.7,
            rule_ids=example.rule_ids,
            orthographic_lexicon=lexicon,
        )

        assert example.token_edit_labels.count("SPAN_REPLACE_BY_LEXICON") == 1
        assert corrected == example.target_text
        assert len(edits) == 1
        assert edits[0].rule_id == "typo_space_noise"
        assert edits[0].edit_type == "split_join"


def test_runtime_noop_for_ambiguous_dictionary_typo_correction() -> None:
    text = "В отчёте указано слово «тест»."
    tokens = tokenize_runtime_words(text)
    lexicon = OrthographicCorrectionLexicon(
        [
            CorrectionEntry("тест", "текст", "dictionary_common_misspellings", operation="dict_replace"),
            CorrectionEntry("тест", "тесто", "dictionary_common_misspellings", operation="dict_replace"),
        ]
    )

    corrected, edits = apply_token_edit_labels(
        text,
        tokens,
        ["KEEP", "KEEP", "KEEP", "KEEP", "DICT_REPLACE"],
        [1.0] * len(tokens),
        threshold=0.7,
        rule_ids=["none", "none", "none", "none", "dictionary_common_misspellings"],
        orthographic_lexicon=lexicon,
    )

    assert corrected == text
    assert edits == []


def test_generated_dictionary_typo_examples_validate() -> None:
    specs = load_dictionary_typo_specs(ROOT / "lexicon" / "layers", seed=31)
    realizer = _realizer()

    for spec in specs:
        for case in spec.cases[:12]:
            example = build_generated_example_from_case(case, realizer, layer=spec.layer)
            assert validate_generated_pair(example) == []
            assert GeneratedExample.from_dict(example.to_dict()) == example


def test_dictionary_typo_generator_audit_and_duplicate_rate() -> None:
    config = load_config(ROOT / "configs" / "config.yaml")
    config["generation"]["enabled_rule_groups"] = ["dictionary_typo"]
    config["generation"]["mix"] = {"dictionary_typo": 1.0}
    config["generation"]["grammar"]["max_generation_retries"] = 50
    config["generation"]["rule_layers"]["groups"]["dictionary_typo"]["enabled"] = True
    config["generation"]["rule_layers"]["groups"]["dictionary_typo"]["layers"] = ["dictionary_typo"]
    generator = online_generator_from_config(config, seed=404)

    examples = [generator.sample_by_index(index) for index in range(1000)]
    audit = audit_batch(examples)
    pair_counts = Counter((example.source_text, example.target_text) for example in examples)

    assert audit["failed_examples_count"] == 0
    assert EXPECTED_RULE_IDS <= {example.primary_rule_id for example in examples}
    assert max(pair_counts.values()) <= 18


def test_protected_tokens_are_not_corrupted() -> None:
    protected = ("https://example.ru", "user@example.ru", "12345", "AB-123", "США", "ИИ")

    for token in protected:
        assert is_protected_token(token)
        assert generate_typos_for_word(token, seed=1) == ()


def test_generated_typo_sources_do_not_collide_with_protected_lexicon() -> None:
    corrections = load_dictionary_typo_corrections(ROOT / "lexicon" / "layers", seed=55)
    targets_by_source: dict[str, set[str]] = defaultdict(set)

    for correction in corrections:
        targets_by_source[correction.source].add(correction.target)
        assert correction.source != correction.target
        assert not is_protected_token(correction.source)

    ambiguous = {source: targets for source, targets in targets_by_source.items() if len(targets) > 1}
    assert ambiguous == {}


def _example_for(rule_id: str, *, source: str) -> GeneratedExample:
    specs = load_dictionary_typo_specs(ROOT / "lexicon" / "layers", seed=13)
    realizer = _realizer()
    for spec in specs:
        if spec.rule_id != rule_id:
            continue
        for case in spec.cases:
            if case.mode == GenerationMode.POSITIVE.value and case.metadata.get("source") == source:
                return build_generated_example_from_case(case, realizer, layer=spec.layer)
    raise AssertionError(f"Missing positive case for {rule_id}:{source}")


def _realizer() -> Realizer:
    return Realizer(Lexicon.default(), MorphologyEngine(use_pymorphy=False))
