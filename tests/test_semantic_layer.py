from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.config.load_config import load_config
from src.grammar_gen import Lexicon, MorphologyEngine, Realizer
from src.grammar_gen.factory import online_generator_from_config
from src.grammar_gen.rules.base import GenerationMode
from src.grammar_gen.safety import validate_generated_pair
from src.rule_layers.example_builders import build_generated_example_from_case
from src.rule_layers.semantic import load_semantic_specs
from src.runtime.corrector import Corrector
from src.runtime.edit_realizer import apply_token_edit_labels
from src.runtime.orthographic_lexicon import OrthographicCorrectionLexicon
from src.runtime.tokenization import tokenize_runtime_words


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RULE_IDS = {
    "semantic_service_words",
    "semantic_derived_prepositions",
    "semantic_ne_ni",
    "semantic_introductory_context",
    "semantic_comparative_context",
}
EXPECTED_SUB_RULE_IDS = {
    "takzhe_additive_merge",
    "takzhe_comparison_split",
    "takzhe_comparison_guard",
    "tozhe_additive_merge",
    "tozhe_demonstrative_split",
    "tozhe_demonstrative_guard",
    "zato_adversative_merge",
    "zato_preposition_split",
    "zato_preposition_guard",
    "chtoby_conjunction_merge",
    "chtoby_particle_split",
    "potomu_chto_merge",
    "potomu_chto_guard",
    "ottogo_merge",
    "ottogo_guard",
    "otchego_merge",
    "otchego_guard",
    "zachem_merge",
    "zachem_guard",
    "v_techenie_temporal",
    "v_techenie_literal_guard",
    "vsledstvie_causal",
    "vsledstvie_literal_guard",
    "naschet_about",
    "naschet_literal_guard",
    "vvidu_causal",
    "vvidu_literal_guard",
    "nesmotrya_concessive",
    "nesmotrya_literal_guard",
    "ne_kto_inoy_guard",
    "nikto_inoy_guard",
    "ne_chto_inoe_guard",
    "nichto_inoe_guard",
    "introductory_homonym_guard",
    "comparative_kak_role_guard",
    "comparative_kak_appositive",
}
POSITIVE_SUB_RULE_IDS = {
    "takzhe_additive_merge",
    "takzhe_comparison_split",
    "tozhe_additive_merge",
    "tozhe_demonstrative_split",
    "zato_adversative_merge",
    "zato_preposition_split",
    "chtoby_conjunction_merge",
    "potomu_chto_merge",
    "ottogo_merge",
    "otchego_merge",
    "zachem_merge",
    "v_techenie_temporal",
    "vsledstvie_causal",
    "naschet_about",
    "vvidu_causal",
    "nesmotrya_concessive",
    "comparative_kak_appositive",
}
GUARD_SUB_RULE_IDS = EXPECTED_SUB_RULE_IDS - POSITIVE_SUB_RULE_IDS
SPECIALIZED_LABELS = {
    "takzhe_additive_merge": "MERGE_TAK_ZHE_TO_TAKZHE",
    "takzhe_comparison_split": "SPLIT_TAKZHE_TO_TAK_ZHE",
    "tozhe_additive_merge": "MERGE_TO_ZHE_TO_TOZHE",
    "tozhe_demonstrative_split": "SPLIT_TOZHE_TO_TO_ZHE",
    "zato_adversative_merge": "MERGE_ZA_TO_TO_ZATO",
    "zato_preposition_split": "SPLIT_ZATO_TO_ZA_TO",
}
GENERIC_SPAN_SUB_RULE_IDS = {
    "chtoby_conjunction_merge",
    "potomu_chto_merge",
    "ottogo_merge",
    "otchego_merge",
    "zachem_merge",
    "v_techenie_temporal",
    "vsledstvie_causal",
    "naschet_about",
    "vvidu_causal",
    "nesmotrya_concessive",
}
ALLOWED_CASE_TYPES = {
    "additive",
    "comparison",
    "demonstrative",
    "causal",
    "temporal",
    "literal",
    "concession",
    "guard",
}
REQUIRED_METADATA_KEYS = {
    "semantic_case_type",
    "ambiguity_pair",
    "semantic_signal",
    "operation",
    "source",
    "target",
    "legacy_overlap",
    "compound_overlap",
}


def test_semantic_specs_cover_required_rules_subrules_and_metadata() -> None:
    specs = load_semantic_specs(ROOT / "lexicon" / "layers")
    cases = [case for spec in specs for case in spec.cases]
    by_sub_rule = Counter(case.sub_rule_id for case in cases)
    positive_counts = Counter(case.sub_rule_id for case in cases if case.mode == "positive")
    guard_counts = Counter(case.sub_rule_id for case in cases if case.mode == "hard_negative")

    assert EXPECTED_RULE_IDS == {spec.rule_id for spec in specs}
    assert EXPECTED_SUB_RULE_IDS <= set(by_sub_rule)
    for sub_rule_id in POSITIVE_SUB_RULE_IDS:
        assert positive_counts[sub_rule_id] >= 5, sub_rule_id
    for sub_rule_id in GUARD_SUB_RULE_IDS:
        assert guard_counts[sub_rule_id] >= 4, sub_rule_id

    for case in cases:
        assert REQUIRED_METADATA_KEYS <= set(case.metadata), case.sub_rule_id
        assert case.metadata["semantic_case_type"] in ALLOWED_CASE_TYPES
        assert isinstance(case.metadata["legacy_overlap"], bool), case.sub_rule_id
        assert isinstance(case.metadata["compound_overlap"], bool), case.sub_rule_id


def test_semantic_positive_examples_validate_and_use_expected_labels() -> None:
    realizer = _realizer()
    cases = [case for spec in _specs() for case in spec.cases if case.mode == "positive"]

    for case in cases:
        example = build_generated_example_from_case(case, realizer, layer="semantic")
        assert validate_generated_pair(example) == [], case.sub_rule_id

        if case.sub_rule_id in SPECIALIZED_LABELS:
            assert SPECIALIZED_LABELS[case.sub_rule_id] in example.token_edit_labels
            assert "SPAN_REPLACE_BY_LEXICON" not in example.token_edit_labels
        if case.sub_rule_id in GENERIC_SPAN_SUB_RULE_IDS:
            assert example.token_edit_labels.count("SPAN_REPLACE_BY_LEXICON") == 1


def test_semantic_hard_negatives_are_identity_guards() -> None:
    realizer = _realizer()
    hard_negatives = [case for spec in _specs() for case in spec.cases if case.mode == "hard_negative"]

    for case in hard_negatives:
        example = build_generated_example_from_case(case, realizer, layer="semantic")
        assert example.source_text == example.target_text
        assert set(example.token_edit_labels) == {"KEEP"}
        assert example.metadata["expected_token_edit_count"] == 0
        assert example.metadata["expected_gap_edit_count"] == 0
        assert all(
            gap_label == "NONE" or rule_id != case.rule_id
            for gap_label, rule_id in zip(example.gap_labels, example.rule_ids, strict=True)
        )
        assert validate_generated_pair(example) == [], case.sub_rule_id


def test_semantic_runtime_uses_specialized_takzhe_label() -> None:
    text = "Студент проверил отчёт, так же исправил ошибку."
    tokens = tokenize_runtime_words(text)

    corrected, edits = apply_token_edit_labels(
        text,
        tokens,
        ["KEEP", "KEEP", "KEEP", "MERGE_TAK_ZHE_TO_TAKZHE", "KEEP", "KEEP", "KEEP"],
        [1.0, 1.0, 1.0, 0.99, 1.0, 1.0, 1.0],
        threshold=0.70,
        rule_ids=["none", "none", "none", "semantic_service_words", "none", "none", "none"],
    )

    assert corrected == "Студент проверил отчёт, также исправил ошибку."
    assert [(edit.source, edit.replacement, edit.rule_id, edit.edit_type) for edit in edits] == [
        ("так же", "также", "semantic_service_words", "split_join")
    ]


def test_semantic_span_replace_positive_cases_have_lexicon_entries() -> None:
    lexicon = OrthographicCorrectionLexicon.from_root(ROOT / "lexicon")
    span_cases = [
        case
        for spec in _specs()
        for case in spec.cases
        if case.mode == "positive"
        for operation in case.token_operations
        if operation.label == "SPAN_REPLACE_BY_LEXICON"
    ]

    assert span_cases
    for case in span_cases:
        operation = next(item for item in case.token_operations if item.label == "SPAN_REPLACE_BY_LEXICON")
        entries = lexicon.lookup(
            operation.source_pattern,
            rule_id=case.rule_id,
            sub_rule_id=case.sub_rule_id,
        )
        assert {entry.target for entry in entries} == {operation.target_pattern}, case.sub_rule_id


def test_semantic_runtime_uses_lexicon_for_naschet_span() -> None:
    text = "Редактор спросил на счёт оплаты."
    tokens = tokenize_runtime_words(text)
    lexicon = OrthographicCorrectionLexicon.from_root(ROOT / "lexicon")

    corrected, edits = apply_token_edit_labels(
        text,
        tokens,
        ["KEEP", "KEEP", "SPAN_REPLACE_BY_LEXICON", "KEEP", "KEEP"],
        [1.0, 1.0, 0.99, 1.0, 1.0],
        threshold=0.70,
        rule_ids=["none", "none", "semantic_derived_prepositions", "none", "none"],
        orthographic_lexicon=lexicon,
    )

    assert corrected == "Редактор спросил насчёт оплаты."
    assert [(edit.source, edit.replacement, edit.rule_id, edit.edit_type) for edit in edits] == [
        ("на счёт", "насчёт", "semantic_derived_prepositions", "split_join")
    ]


def test_semantic_entries_do_not_run_as_deterministic_lexicon_corrections() -> None:
    config = load_config(ROOT / "configs" / "config.yaml")
    config["runtime"] = {
        **config.get("runtime", {}),
        "deterministic_lexicon": True,
        "deterministic_final_punctuation": False,
        "neural_token_edits": False,
        "neural_punctuation": False,
    }
    corrector = Corrector(neural_backend=None, config=config)

    result = corrector.correct("Совещание отменили вследствии болезни директора.")

    assert result.corrected_text == "Совещание отменили вследствии болезни директора."
    assert result.edits == []


def test_semantic_generator_duplicate_pair_rate_stays_below_threshold() -> None:
    config = load_config(ROOT / "configs" / "config.yaml")
    config["generation"]["enabled_rule_groups"] = ["semantic"]
    config["generation"]["mix"] = {"semantic": 1.0}
    config["generation"]["grammar"]["max_generation_retries"] = 60
    generator = online_generator_from_config(config, seed=404)

    examples = [generator.sample() for _ in range(2000)]
    unique_pairs = {(example.source_text, example.target_text) for example in examples}
    duplicate_rate = 1.0 - (len(unique_pairs) / len(examples))

    assert duplicate_rate <= 0.15
    assert not any("пример номер" in example.source_text.casefold() for example in examples)


def _specs():
    return load_semantic_specs(ROOT / "lexicon" / "layers")


def _realizer() -> Realizer:
    return Realizer(Lexicon.default(), MorphologyEngine(use_pymorphy=False))
