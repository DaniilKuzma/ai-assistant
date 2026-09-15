from __future__ import annotations

from src.grammar_gen import Lexicon, MorphologyEngine
from src.grammar_gen.generator import OnlineExampleGenerator
from src.grammar_gen.rules.base import GenerationMode
from src.grammar_gen.rules.registry import RuleRegistry, default_rule_registry, register_layered_rules


def test_register_layered_rules_with_empty_specs_keeps_old_rules_working() -> None:
    registry = RuleRegistry()
    for rule in default_rule_registry().all_rules():
        registry.register_rule(rule)

    register_layered_rules(registry, ())
    generator = OnlineExampleGenerator(
        registry,
        Lexicon.default(),
        MorphologyEngine(use_pymorphy=False),
        {"generation": {"mix": {"orthography_contextual": 1.0}, "grammar": {"max_generation_retries": 30}}},
        seed=3,
    )

    example = generator.sample(rule_id="ne_verb", mode=GenerationMode.POSITIVE)

    assert example.primary_rule_id == "ne_verb"
    assert "SPLIT_NE_VERB" in example.token_edit_labels
