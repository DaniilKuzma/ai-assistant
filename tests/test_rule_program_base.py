from __future__ import annotations

from pathlib import Path

import pytest

from src.grammar_gen import GrammarBuilder, Lexicon, MorphologyEngine, RandomSource, Realizer
from src.grammar_gen.generator import GenerationError, OnlineExampleGenerator
from src.grammar_gen.rules.base import GenerationMode, RuleInfo, RuleProgram
from src.grammar_gen.rules.common import gap_labels_from_text, token_labels_all_keep
from src.grammar_gen.rules.registry import RuleRegistry
from src.schema import GeneratedExample


class DummyRule(RuleProgram):
    info = RuleInfo(
        rule_id="ne_verb",
        family="orthography_contextual",
        description="dummy positive rule",
        explanation="dummy explanation",
        deterministic=True,
        weight=2.0,
    )
    supported_modes = (GenerationMode.POSITIVE,)

    def generate(
        self,
        builder: GrammarBuilder,
        realizer: Realizer,
        rng: RandomSource,
        mode: GenerationMode,
    ) -> GeneratedExample:
        del builder, rng
        text = "\u041e\u043d \u0437\u043d\u0430\u043b \u043e\u0442\u0432\u0435\u0442."
        tokens = realizer.tokenize_words_with_offsets(text)
        return GeneratedExample(
            source_text=text,
            target_text=text,
            source_tokens=tokens,
            token_edit_labels=token_labels_all_keep(tokens),
            gap_labels=gap_labels_from_text(text, tokens),
            rule_ids=[self.info.rule_id] * len(tokens),
            primary_rule_id=self.info.rule_id,
            mode=mode.value,
            explanation_ids=["dummy_explanation"],
            metadata={"rule": self.info.rule_id},
        )


class InvalidRule(DummyRule):
    info = RuleInfo(
        rule_id="ne_verb",
        family="orthography_contextual",
        description="invalid dummy rule",
        explanation="invalid dummy explanation",
    )

    def generate(
        self,
        builder: GrammarBuilder,
        realizer: Realizer,
        rng: RandomSource,
        mode: GenerationMode,
    ) -> GeneratedExample:
        del builder, rng, mode
        text = "\u041e\u043d \u0437\u043d\u0430\u043b."
        tokens = realizer.tokenize_words_with_offsets(text)
        return GeneratedExample(
            source_text=text,
            target_text=text,
            source_tokens=tokens,
            token_edit_labels=["KEEP"],
            gap_labels=gap_labels_from_text(text, tokens),
            rule_ids=[self.info.rule_id] * len(tokens),
            primary_rule_id=self.info.rule_id,
            mode=GenerationMode.POSITIVE.value,
            explanation_ids=[],
            metadata={},
        )


class FlakyRule(DummyRule):
    def __init__(self) -> None:
        self.attempts = 0

    def generate(
        self,
        builder: GrammarBuilder,
        realizer: Realizer,
        rng: RandomSource,
        mode: GenerationMode,
    ) -> GeneratedExample:
        self.attempts += 1
        if self.attempts == 1:
            raise RuntimeError("transient failure")
        return super().generate(builder, realizer, rng, mode)


def test_registry_add_get_and_duplicate_protection() -> None:
    registry = RuleRegistry()
    rule = DummyRule()

    registry.register_rule(rule)

    assert registry.get_rule("ne_verb") is rule
    assert registry.get_rule("missing") is None
    with pytest.raises(ValueError, match="Duplicate rule id"):
        registry.register_rule(DummyRule())


def test_online_example_generator_samples_from_dummy_rule() -> None:
    generator = _generator_for(DummyRule(), {"orthography_contextual": 1.0})

    example = generator.sample()

    assert example.primary_rule_id == "ne_verb"
    assert example.mode == GenerationMode.POSITIVE.value
    assert example.source_text == example.target_text
    assert example.token_edit_labels == ["KEEP", "KEEP", "KEEP"]
    assert example.gap_labels == ["NONE", "NONE", "DOT"]


def test_online_example_generator_enforces_generated_example_invariants() -> None:
    generator = _generator_for(InvalidRule(), {"orthography_contextual": 1.0}, max_retries=2)

    with pytest.raises(GenerationError, match="Failed to generate example"):
        generator.sample(mode=GenerationMode.POSITIVE, rule_id="ne_verb")


def test_online_example_generator_retries_rule_failures() -> None:
    rule = FlakyRule()
    generator = _generator_for(rule, {"orthography_contextual": 1.0}, max_retries=3)

    example = generator.sample(mode=GenerationMode.POSITIVE, rule_id="ne_verb")

    assert rule.attempts == 2
    assert example.primary_rule_id == "ne_verb"


def test_grammar_gen_does_not_import_candidate_generator() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "grammar_gen"
    forbidden = "Candidate" + "Generator"

    offenders = [
        path
        for path in root.rglob("*.py")
        if forbidden in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def _generator_for(
    rule: RuleProgram,
    mix: dict[str, float],
    *,
    max_retries: int = 20,
) -> OnlineExampleGenerator:
    registry = RuleRegistry()
    registry.register_rule(rule)
    lexicon = Lexicon.default()
    morphology = MorphologyEngine(use_pymorphy=False)
    config = {
        "generation": {
            "mix": mix,
            "grammar": {"max_generation_retries": max_retries},
        }
    }
    return OnlineExampleGenerator(registry, lexicon, morphology, config, seed=7)
