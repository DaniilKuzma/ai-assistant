from __future__ import annotations

from src.grammar_gen.ast import IntroductorySentence
from src.grammar_gen.builders import GrammarBuilder
from src.grammar_gen.randomness import RandomSource
from src.grammar_gen.realizer import Realizer
from src.grammar_gen.rules.base import GenerationMode, RuleInfo, RuleProgram
from src.grammar_gen.rules.common import (
    make_punctuation_example,
    metadata_with_safety_clauses,
    metadata_without_safety_clauses,
    replace_once_checked,
)
from src.schema import GeneratedExample


ACTIVE_GAPS = frozenset({"COMMA"})


class CommaIntroductoryRule(RuleProgram):
    info = RuleInfo(
        rule_id="comma_introductory",
        family="punctuation",
        description="Commas around introductory word конечно.",
        explanation="Вводное слово конечно отделяется запятыми.",
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
            target, ast = _introductory_target(builder, realizer, rng)
            source = _remove_introductory_commas(target)
            return _example(
                source,
                target,
                realizer,
                self.info.rule_id,
                mode,
                metadata_with_safety_clauses(
                    ast,
                    builder.lexicon,
                    {"introductory_position": ast.position},
                ),
            )
        if mode is GenerationMode.HARD_NEGATIVE:
            text = rng.choice(
                (
                    "Документ точно включал факт.",
                    "Комиссия примерно проверила отчёт.",
                )
            )
            return _example(
                text,
                text,
                realizer,
                self.info.rule_id,
                mode,
                metadata_without_safety_clauses({"trap_type": "normal_adverb"}),
            )
        if mode is GenerationMode.CLEAN_IDENTITY:
            text, ast = _introductory_target(builder, realizer, rng)
            return _example(
                text,
                text,
                realizer,
                self.info.rule_id,
                mode,
                metadata_with_safety_clauses(
                    ast,
                    builder.lexicon,
                    {"introductory_position": ast.position},
                ),
            )
        raise ValueError(f"Unsupported generation mode: {mode!r}")


def _introductory_target(
    builder: GrammarBuilder,
    realizer: Realizer,
    rng: RandomSource,
) -> tuple[str, IntroductorySentence]:
    position = "medial" if rng.chance(0.5) else "initial"
    ast = IntroductorySentence(
        introductory="конечно",
        clause=builder.random_clause(transitive=True),
        position=position,
    )
    return realizer.render_sentence(ast), ast


def _remove_introductory_commas(target: str) -> str:
    if target.startswith("Конечно, "):
        return replace_once_checked(target, "Конечно, ", "Конечно ")
    return replace_once_checked(target, ", конечно,", " конечно")


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


__all__ = ["CommaIntroductoryRule"]
