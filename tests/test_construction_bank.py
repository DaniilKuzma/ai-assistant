from __future__ import annotations

import re

from src.grammar_gen import GrammarBuilder, Lexicon, MorphologyEngine, RandomSource, Realizer
from src.grammar_gen.constructions import ConstructionBank
from src.grammar_gen.safety import validate_surface


def test_default_construction_bank_loads_patterns() -> None:
    bank = ConstructionBank.default()

    assert bank.patterns
    assert {pattern.family for pattern in bank.patterns} >= {
        "simple_transitive",
        "subordinate_that",
        "dash_nominal",
        "homogeneous_objects",
        "adversative",
        "introductory",
        "tsya_ttsya",
        "context_pairs",
    }


def test_every_pattern_has_required_contract_fields() -> None:
    bank = ConstructionBank.default()

    for pattern in bank.patterns:
        assert pattern.id
        assert pattern.family
        assert pattern.surface
        assert pattern.roles
        assert pattern.allowed_rule_ids
        for role_name, role in pattern.roles.items():
            assert role.name == role_name
            assert role.semantic_classes


def test_render_default_patterns_without_invalid_surface() -> None:
    bank = ConstructionBank.default()
    lexicon = Lexicon.default()
    morphology = MorphologyEngine(use_pymorphy=False, lexicon=lexicon)
    rng = RandomSource(seed=101)
    builder = GrammarBuilder(lexicon, morphology, rng, construction_bank=bank, production=True)
    realizer = Realizer(lexicon, morphology)

    rendered = [
        bank.render(rng.choice(bank.patterns), builder, realizer, rng)
        for _ in range(100)
    ]

    assert all(item.text for item in rendered)
    assert all("{" not in item.text and "}" not in item.text for item in rendered)
    assert all(validate_surface(item.text) == [] for item in rendered)
    assert all(item.tokens for item in rendered)
    assert all(item.metadata["uses_construction_bank"] is True for item in rendered)


def test_patterns_for_comma_subordinate_are_subordinate_safe() -> None:
    bank = ConstructionBank.default()

    patterns = bank.patterns_for_rule("comma_subordinate")

    assert patterns
    assert {pattern.family for pattern in patterns} == {"subordinate_that"}
    assert all("comma_subordinate" in pattern.allowed_rule_ids for pattern in patterns)


def test_subordinate_patterns_do_not_render_forbidden_main_clauses() -> None:
    bank = ConstructionBank.default()
    lexicon = Lexicon.default()
    morphology = MorphologyEngine(use_pymorphy=False, lexicon=lexicon)
    rng = RandomSource(seed=202)
    builder = GrammarBuilder(lexicon, morphology, rng, construction_bank=bank, production=True)
    realizer = Realizer(lexicon, morphology)
    forbidden = (
        re.compile(r"\bпров[её]л\b.*,\s+что\b", re.IGNORECASE),
        re.compile(r"\bотправил\b.*,\s+что\b", re.IGNORECASE),
        re.compile(r"\bсравнил\b.*,\s+что\b", re.IGNORECASE),
    )

    texts = [
        bank.render(bank.sample_for_rule("comma_subordinate", rng), builder, realizer, rng).text
        for _ in range(500)
    ]

    assert all(", что " in text for text in texts)
    assert [
        text
        for text in texts
        if any(pattern.search(text) for pattern in forbidden)
    ] == []
