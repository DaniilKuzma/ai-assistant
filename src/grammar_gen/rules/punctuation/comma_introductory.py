from __future__ import annotations

from src.grammar_gen.builders import GrammarBuilder
from src.grammar_gen.randomness import RandomSource
from src.grammar_gen.realizer import Realizer
from src.grammar_gen.rules.base import GenerationMode, RuleInfo, RuleProgram
from src.grammar_gen.rules.common import (
    make_punctuation_example,
    metadata_from_construction,
    replace_once_checked,
    render_construction_by_id,
    render_construction_for_rule,
)
from src.grammar_gen.safety import validate_target_ast_or_raise
from src.schema import GeneratedExample


ACTIVE_GAPS = frozenset({"COMMA"})
INTRODUCTORY_WORDS = (
    "конечно",
    "вероятно",
    "кстати",
    "по-видимому",
    "безусловно",
    "разумеется",
)


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
            rendered = render_construction_for_rule(builder, realizer, rng, self.info.rule_id, "introductory")
            target = rendered.text
            source = _remove_introductory_commas(target)
            example = _example(
                source,
                target,
                realizer,
                self.info.rule_id,
                mode,
                metadata_from_construction(rendered),
            )
            validate_target_ast_or_raise(rendered.ast, target, example)
            return example
        if mode is GenerationMode.HARD_NEGATIVE:
            rendered = render_construction_by_id(builder, realizer, rng, "introductory_normal_adverb_trap")
            text = rendered.text
            return _example(
                text,
                text,
                realizer,
                self.info.rule_id,
                mode,
                metadata_from_construction(rendered),
            )
        if mode is GenerationMode.CLEAN_IDENTITY:
            rendered = render_construction_for_rule(builder, realizer, rng, self.info.rule_id, "introductory")
            text = rendered.text
            example = _example(
                text,
                text,
                realizer,
                self.info.rule_id,
                mode,
                metadata_from_construction(rendered),
            )
            validate_target_ast_or_raise(rendered.ast, text, example)
            return example
        raise ValueError(f"Unsupported generation mode: {mode!r}")


def _remove_introductory_commas(target: str) -> str:
    for word in INTRODUCTORY_WORDS:
        initial = f"{word.capitalize()}, "
        if target.startswith(initial):
            return replace_once_checked(target, initial, f"{word.capitalize()} ")
        medial = f", {word},"
        if medial in target:
            return replace_once_checked(target, medial, f" {word}")
    raise ValueError("Target sentence does not contain a supported introductory comma pattern.")


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
