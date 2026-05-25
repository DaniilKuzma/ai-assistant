from __future__ import annotations

from src.grammar_gen.builders import GrammarBuilder
from src.grammar_gen.randomness import RandomSource
from src.grammar_gen.realizer import Realizer
from src.grammar_gen.rules.base import GenerationMode, RuleInfo, RuleProgram
from src.grammar_gen.rules.common import make_punctuation_example, remove_punctuation_before
from src.schema import GeneratedExample


ACTIVE_GAPS = frozenset({"DASH"})


class DashSubjectPredicateRule(RuleProgram):
    info = RuleInfo(
        rule_id="dash_subject_predicate",
        family="punctuation",
        description="Dash between nominal subject and nominal predicate.",
        explanation="Между подлежащим и именным сказуемым в таких конструкциях ставится тире.",
        deterministic=True,
        weight=1.0,
    )
    supported_modes = (
        GenerationMode.POSITIVE,
        GenerationMode.HARD_NEGATIVE,
        GenerationMode.CLEAN_IDENTITY,
    )

    def generate(
        self,
        builder: GrammarBuilder,
        realizer: Realizer,
        rng: RandomSource,
        mode: GenerationMode,
    ) -> GeneratedExample:
        if mode is GenerationMode.POSITIVE:
            target = _dash_target(builder, realizer)
            source = remove_punctuation_before(target, _second_word(target))
            return _example(source, target, realizer, self.info.rule_id, mode)
        if mode is GenerationMode.HARD_NEGATIVE:
            text = rng.choice(
                (
                    "Комиссия проверила отчёт.",
                    "Студент прочитал книгу.",
                )
            )
            return _example(text, text, realizer, self.info.rule_id, mode, {"trap_type": "verbal_predicate"})
        if mode is GenerationMode.CLEAN_IDENTITY:
            text = _dash_target(builder, realizer)
            return _example(text, text, realizer, self.info.rule_id, mode)
        raise ValueError(f"Unsupported generation mode: {mode!r}")


def _dash_target(builder: GrammarBuilder, realizer: Realizer) -> str:
    return realizer.render_sentence(builder.dash_subject_predicate_sentence()).replace(" - ", " \u2014 ")


def _second_word(text: str) -> str:
    separator = "\u2014" if "\u2014" in text else "-"
    after_dash = text.split(separator, 1)[1].strip()
    tokens = after_dash.split()
    if not tokens:
        raise ValueError(f"Dash sentence must contain a predicate after the dash: {text!r}")
    return tokens[0].rstrip(".?!")


def _example(
    source: str,
    target: str,
    realizer: Realizer,
    rule_id: str,
    mode: GenerationMode,
    metadata: dict[str, str] | None = None,
) -> GeneratedExample:
    return make_punctuation_example(
        source=source,
        target=target,
        source_tokens=realizer.tokenize_words_with_offsets(source),
        target_tokens=realizer.tokenize_words_with_offsets(target),
        rule_id=rule_id,
        mode=mode,
        active_gap_labels=ACTIVE_GAPS,
        metadata=metadata,
    )


__all__ = ["DashSubjectPredicateRule"]
