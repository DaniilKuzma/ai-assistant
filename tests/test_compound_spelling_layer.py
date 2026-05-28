from __future__ import annotations

from pathlib import Path

from src.config.load_config import load_config
from src.grammar_gen import Lexicon, MorphologyEngine, Realizer
from src.grammar_gen.audit import audit_batch
from src.grammar_gen.factory import online_generator_from_config
from src.grammar_gen.rules.base import GenerationMode
from src.rule_layers.compound_spelling import load_compound_spelling_specs
from src.rule_layers.example_builders import build_generated_example_from_case
from src.runtime.edit_realizer import apply_token_edit_labels
from src.runtime.orthographic_lexicon import OrthographicCorrectionLexicon
from src.runtime.tokenization import tokenize_runtime_words
from src.schema import GeneratedExample


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RULE_IDS = {
    "compound_service_words",
    "compound_prepositions",
    "compound_pronouns_particles",
    "compound_adverbs",
    "compound_nouns_adjectives",
    "compound_ne_spellings",
    "compound_pol_polu",
}
EXPECTED_SUB_RULE_IDS = {
    "takzhe_tak_zhe",
    "tozhe_to_zhe",
    "chto_by_chto_bi",
    "zato_za_to",
    "potomu_chto_po_tomu_chto",
    "ottogo_ot_togo",
    "otchego_ot_chego",
    "zachem_za_chem",
    "v_techenie_v_techenii",
    "v_prodolzhenie_v_prodolzhenii",
    "vsledstvie_v_sledstvii",
    "naschet_na_schet",
    "vvidu_v_vidu",
    "napodobie_na_podobie",
    "navstrechu_na_vstrechu",
    "nesmotrya_na_ne_smotrya_na",
    "nevziraya_na_ne_vziraya_na",
    "koe_pronouns",
    "postfix_to_libo_nibud",
    "nu_ka",
    "vse_taki",
    "po_adverb_omu",
    "po_adverb_ski",
    "repeated_adverbs",
    "ordinal_adverbs",
    "compound_nouns",
    "hyphen_appositions",
    "compound_first_parts",
    "compound_adjectives_social",
    "compound_adjectives_directional",
    "compound_adjectives_color",
    "pol_before_vowel",
    "pol_before_l",
    "pol_before_capital",
    "polu_compounds",
    "ne_verb",
    "ne_nouns",
    "ne_adjectives",
    "ne_adverbs",
    "ne_participles",
}


def test_compound_spelling_specs_cover_required_groups_and_modes() -> None:
    specs = load_compound_spelling_specs(ROOT / "lexicon" / "layers")
    by_rule = {spec.rule_id: spec for spec in specs}
    sub_rule_ids = {case.sub_rule_id for spec in specs for case in spec.cases}

    assert EXPECTED_RULE_IDS <= set(by_rule)
    assert EXPECTED_SUB_RULE_IDS <= sub_rule_ids

    for sub_rule_id in EXPECTED_SUB_RULE_IDS:
        cases = [case for spec in specs for case in spec.cases if case.sub_rule_id == sub_rule_id]
        positives = [case for case in cases if case.mode == GenerationMode.POSITIVE.value]
        hard_negatives = [case for case in cases if case.mode == GenerationMode.HARD_NEGATIVE.value]
        assert len(positives) >= 3, sub_rule_id
        assert len(hard_negatives) >= 3, sub_rule_id


def test_compound_spelling_legacy_overlap_uses_specialized_direct_labels() -> None:
    example = _example_for("compound_service_words", "takzhe_tak_zhe", GenerationMode.POSITIVE)

    assert example.primary_rule_id == "compound_service_words"
    assert "MERGE_TAK_ZHE_TO_TAKZHE" in example.token_edit_labels
    assert "SKIP_MERGED" in example.token_edit_labels
    assert "SPAN_REPLACE_BY_LEXICON" not in example.token_edit_labels
    assert example.metadata["family"] == "compound_spelling"
    assert example.metadata["sub_rule_id"] == "takzhe_tak_zhe"


def test_compound_spelling_generic_cases_use_span_replace_metadata() -> None:
    example = _example_for("compound_prepositions", "naschet_na_schet", GenerationMode.POSITIVE)

    assert example.primary_rule_id == "compound_prepositions"
    assert example.token_edit_labels.count("SPAN_REPLACE_BY_LEXICON") == 1
    assert example.metadata["operation"] == "merge"
    assert example.metadata["source"] == "на счёт"
    assert example.metadata["target"] == "насчёт"
    assert GeneratedExample.from_dict(example.to_dict()) == example


def test_compound_spelling_hard_negatives_are_identity_examples() -> None:
    specs = load_compound_spelling_specs(ROOT / "lexicon" / "layers")
    realizer = _realizer()

    for spec in specs:
        for case in spec.cases:
            if case.mode != GenerationMode.HARD_NEGATIVE.value:
                continue
            example = build_generated_example_from_case(case, realizer, layer=spec.layer)
            assert example.source_text == example.target_text
            assert set(example.token_edit_labels) == {"KEEP"}
            assert GeneratedExample.from_dict(example.to_dict()) == example


def test_compound_spelling_postfix_hard_negatives_do_not_hide_real_hyphen_errors() -> None:
    specs = load_compound_spelling_specs(ROOT / "lexicon" / "layers")
    bad_fragment = "\u043a\u0442\u043e \u0442\u043e \u0441\u0434\u0435\u043b\u0430\u043b"

    hard_negatives = [
        case
        for spec in specs
        for case in spec.cases
        if case.sub_rule_id == "postfix_to_libo_nibud" and case.mode == GenerationMode.HARD_NEGATIVE.value
    ]

    assert hard_negatives
    assert all(bad_fragment not in case.source_text.casefold() for case in hard_negatives)


def test_compound_spelling_generator_can_sample_batch_and_audit() -> None:
    config = load_config(ROOT / "configs" / "config.yaml")
    generator = online_generator_from_config(config, seed=101)

    explicit = generator.sample(rule_id="compound_prepositions", mode=GenerationMode.POSITIVE)
    batch = [generator.sample() for _ in range(60)]
    audit = audit_batch([explicit, *batch])

    assert explicit.primary_rule_id == "compound_prepositions"
    assert explicit.metadata["layer"] == "compound_spelling"
    assert "compound_spelling" in config["generation"]["enabled_rule_groups"]
    assert audit["failed_examples_count"] == 0


def test_compound_spelling_runtime_uses_layer_corrections_yaml() -> None:
    text = "Он спросил на счёт оплаты."
    tokens = tokenize_runtime_words(text)
    lexicon = OrthographicCorrectionLexicon.from_root(ROOT / "lexicon")

    corrected, edits = apply_token_edit_labels(
        text,
        tokens,
        ["KEEP", "KEEP", "SPAN_REPLACE_BY_LEXICON", "KEEP", "KEEP"],
        [1.0, 1.0, 0.99, 1.0, 1.0],
        threshold=0.7,
        rule_ids=["none", "none", "compound_prepositions", "none", "none"],
        orthographic_lexicon=lexicon,
    )

    assert corrected == "Он спросил насчёт оплаты."
    assert [(edit.source, edit.replacement, edit.rule_id, edit.edit_type) for edit in edits] == [
        ("на счёт", "насчёт", "compound_prepositions", "split_join")
    ]


def test_compound_spelling_runtime_keeps_ambiguous_span_noop() -> None:
    text = "по прежнему"
    tokens = tokenize_runtime_words(text)
    lexicon = OrthographicCorrectionLexicon.from_root(ROOT / "lexicon")

    corrected, edits = apply_token_edit_labels(
        text,
        tokens,
        ["SPAN_REPLACE_BY_LEXICON", "KEEP"],
        [0.99, 1.0],
        threshold=0.7,
        rule_ids=["compound_adverbs", "none"],
        orthographic_lexicon=lexicon,
    )

    assert corrected == text
    assert edits == []


def test_compound_spelling_runtime_accepts_replace_entries_for_dict_replace() -> None:
    text = "Редактор сделал коекак."
    tokens = tokenize_runtime_words(text)
    lexicon = OrthographicCorrectionLexicon.from_root(ROOT / "lexicon")

    corrected, edits = apply_token_edit_labels(
        text,
        tokens,
        ["KEEP", "KEEP", "DICT_REPLACE"],
        [1.0, 1.0, 0.99],
        threshold=0.7,
        rule_ids=["none", "none", "compound_adverbs"],
        orthographic_lexicon=lexicon,
    )

    assert corrected == "Редактор сделал кое-как."
    assert [(edit.source, edit.replacement, edit.rule_id, edit.edit_type) for edit in edits] == [
        ("коекак", "кое-как", "compound_adverbs", "spelling")
    ]


def _example_for(rule_id: str, sub_rule_id: str, mode: GenerationMode) -> GeneratedExample:
    specs = load_compound_spelling_specs(ROOT / "lexicon" / "layers")
    realizer = _realizer()
    for spec in specs:
        if spec.rule_id != rule_id:
            continue
        for case in spec.cases:
            if case.sub_rule_id == sub_rule_id and case.mode == mode.value:
                return build_generated_example_from_case(case, realizer, layer=spec.layer)
    raise AssertionError(f"Missing case {rule_id}:{sub_rule_id}:{mode.value}")


def _realizer() -> Realizer:
    return Realizer(Lexicon.default(), MorphologyEngine(use_pymorphy=False))
