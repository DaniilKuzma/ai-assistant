from __future__ import annotations

from src.grammar_gen.builders import GrammarBuilder
from src.grammar_gen.randomness import RandomSource
from src.grammar_gen.realizer import Realizer
from src.grammar_gen.rules.base import GenerationMode, RuleInfo, RuleProgram
from src.grammar_gen.rules.common import (
    make_punctuation_example,
    metadata_from_construction,
    render_construction_for_rule,
)
from src.grammar_gen.safety import validate_target_ast_or_raise
from src.schema import GeneratedExample


ACTIVE_GAPS = frozenset({"DOT"})
FINAL_MARKS = (".",)


class FinalPunctuationRule(RuleProgram):
    info = RuleInfo(
        rule_id="final_punctuation",
        family="punctuation",
        description="Sentence-final dot punctuation.",
        explanation="В конце повествовательного предложения ставится точка.",
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
            rendered = _target_with_mark(builder, realizer, rng, self.info.rule_id)
            target = rendered.text
            source = target[:-1]
            example = _example(
                source,
                target,
                realizer,
                self.info.rule_id,
                mode,
                metadata_from_construction(rendered, {"expected_error": "missing_final_punctuation"}),
            )
            validate_target_ast_or_raise(rendered.ast, target, example)
            return example
        if mode is GenerationMode.HARD_NEGATIVE:
            rendered = _target_with_mark(builder, realizer, rng, self.info.rule_id)
            text = rendered.text
            example = _example(
                text,
                text,
                realizer,
                self.info.rule_id,
                mode,
                metadata_from_construction(rendered, {"trap_type": "already_final"}),
            )
            validate_target_ast_or_raise(rendered.ast, text, example)
            return example
        if mode is GenerationMode.CLEAN_IDENTITY:
            rendered = _target_with_mark(builder, realizer, rng, self.info.rule_id)
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


def _target_with_mark(
    builder: GrammarBuilder,
    realizer: Realizer,
    rng: RandomSource,
    rule_id: str,
):
    rendered = render_construction_for_rule(builder, realizer, rng, rule_id, "simple_transitive")
    rendered.text = f"{rendered.text[:-1]}{rng.choice(FINAL_MARKS)}"
    rendered.tokens = realizer.tokenize_words_with_offsets(rendered.text)
    return rendered


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


__all__ = ["FinalPunctuationRule"]
