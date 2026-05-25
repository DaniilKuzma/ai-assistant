from __future__ import annotations

from pathlib import Path

import pytest

from src.grammar_gen import Lexicon, MorphologyEngine
from src.grammar_gen.generator import OnlineExampleGenerator
from src.grammar_gen.rules.base import GenerationMode
from src.grammar_gen.rules.registry import default_rule_registry
from src.grammar_gen.safety import validate_surface
from src.schema import GeneratedExample


REQUIRED_POSITIVE_LABELS = {
    "ne_verb": {"SPLIT_NE_VERB"},
    "takzhe_tak_zhe": {"MERGE_TAK_ZHE_TO_TAKZHE", "SKIP_MERGED"},
    "tozhe_to_zhe": {"MERGE_TO_ZHE_TO_TOZHE", "SKIP_MERGED"},
    "zato_za_to": {"MERGE_ZA_TO_TO_ZATO", "SKIP_MERGED"},
    "hyphen_particles": {"SKIP_MERGED"},
    "hyphen_koe": {"HYPHENATE_KOE", "SKIP_MERGED"},
    "hyphen_po_adverb": {"HYPHENATE_PO_ADVERB", "SKIP_MERGED"},
    "tsya_ttsya": set(),
}

ANY_POSITIVE_LABELS = {
    "hyphen_particles": {
        "HYPHENATE_PARTICLE_TO",
        "HYPHENATE_PARTICLE_LIBO",
        "HYPHENATE_PARTICLE_NIBUD",
    },
    "tsya_ttsya": {"FIX_TSYA_TO_TTSYA", "FIX_TTSYA_TO_TSYA"},
}

RULE_IDS = tuple(REQUIRED_POSITIVE_LABELS)


@pytest.mark.parametrize("rule_id", RULE_IDS)
@pytest.mark.parametrize(
    "mode",
    (
        GenerationMode.POSITIVE,
        GenerationMode.HARD_NEGATIVE,
        GenerationMode.CLEAN_IDENTITY,
    ),
)
def test_orthography_rules_generate_valid_examples(rule_id: str, mode: GenerationMode) -> None:
    generator = _generator(seed=11)
    rule = default_rule_registry().get_rule(rule_id)

    assert rule is not None
    assert rule.can_generate(mode)

    example = generator.sample(rule_id=rule_id, mode=mode)

    assert example.source_text.strip()
    assert example.target_text.strip()
    assert GeneratedExample.from_dict(example.to_dict()) == example
    assert validate_surface(example.source_text) == []
    assert validate_surface(example.target_text) == []

    if mode is GenerationMode.POSITIVE:
        assert example.source_text != example.target_text
        labels = set(example.token_edit_labels)
        assert REQUIRED_POSITIVE_LABELS[rule_id].issubset(labels)
        if rule_id in ANY_POSITIVE_LABELS:
            assert labels & ANY_POSITIVE_LABELS[rule_id]
    else:
        assert example.source_text == example.target_text
        assert set(example.token_edit_labels) == {"KEEP"}


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_orthography_rule_samples_are_surface_safe(rule_id: str) -> None:
    generator = _generator(seed=100 + RULE_IDS.index(rule_id))
    forbidden = ("Девочка пошёл", "во огород", "{", "}", "  ")

    examples = [
        generator.sample(
            rule_id=rule_id,
            mode=(
                GenerationMode.POSITIVE,
                GenerationMode.HARD_NEGATIVE,
                GenerationMode.CLEAN_IDENTITY,
            )[index % 3],
        )
        for index in range(50)
    ]

    for example in examples:
        for text in (example.source_text, example.target_text):
            assert validate_surface(text) == []
            assert not any(bad in text for bad in forbidden), text


def test_takzhe_hard_negative_keeps_comparison_tak_zhe() -> None:
    example = _generator(seed=21).sample(
        rule_id="takzhe_tak_zhe",
        mode=GenerationMode.HARD_NEGATIVE,
    )

    assert "так же" in example.source_text.lower()
    assert example.source_text == example.target_text
    assert set(example.token_edit_labels) == {"KEEP"}
    assert example.metadata["trap_type"] == "comparison_tak_zhe"


def test_ne_verb_positive_merges_ne_only_in_source() -> None:
    example = _generator(seed=31).sample(rule_id="ne_verb", mode=GenerationMode.POSITIVE)

    assert "не " in example.target_text.lower()
    merged_tokens = [
        token.text.lower()
        for token, label in zip(example.source_tokens, example.token_edit_labels)
        if label == "SPLIT_NE_VERB"
    ]
    assert merged_tokens
    assert all(token.startswith("не") and not token.startswith("не ") for token in merged_tokens)


def test_tsya_ttsya_positive_generates_both_directions_across_samples() -> None:
    generator = _generator(seed=41)
    labels = {
        label
        for _ in range(40)
        for label in generator.sample(rule_id="tsya_ttsya", mode=GenerationMode.POSITIVE).token_edit_labels
    }

    assert {"FIX_TSYA_TO_TTSYA", "FIX_TTSYA_TO_TSYA"}.issubset(labels)


def test_orthography_rules_do_not_import_candidate_generator() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "grammar_gen" / "rules" / "orthography"
    offenders = [
        path
        for path in root.rglob("*.py")
        if "CandidateGenerator" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def _generator(seed: int) -> OnlineExampleGenerator:
    lexicon = Lexicon.default()
    morphology = MorphologyEngine(use_pymorphy=False)
    config = {
        "generation": {
            "mix": {"orthography_contextual": 1.0},
            "grammar": {"max_generation_retries": 30},
        }
    }
    return OnlineExampleGenerator(default_rule_registry(), lexicon, morphology, config, seed=seed)
