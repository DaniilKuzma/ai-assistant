from __future__ import annotations

from src.grammar_gen.builders import GrammarBuilder
from src.grammar_gen.randomness import RandomSource
from src.grammar_gen.realizer import Realizer
from src.grammar_gen.rules.base import GenerationMode, RuleInfo, RuleProgram
from src.grammar_gen.ast import SimpleSentence
from src.grammar_gen.rules.common import (
    make_punctuation_example,
    metadata_with_safety_clauses,
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
            target, ast = _target_with_mark(builder, realizer, rng)
            source = target[:-1]
            example = _example(
                source,
                target,
                realizer,
                self.info.rule_id,
                mode,
                metadata_with_safety_clauses(
                    ast,
                    builder.lexicon,
                    {"expected_error": "missing_final_punctuation"},
                ),
            )
            validate_target_ast_or_raise(ast, target, example)
            return example
        if mode is GenerationMode.HARD_NEGATIVE:
            text, ast = _target_with_mark(builder, realizer, rng)
            example = _example(
                text,
                text,
                realizer,
                self.info.rule_id,
                mode,
                metadata_with_safety_clauses(ast, builder.lexicon, {"trap_type": "already_final"}),
            )
            validate_target_ast_or_raise(ast, text, example)
            return example
        if mode is GenerationMode.CLEAN_IDENTITY:
            text, ast = _target_with_mark(builder, realizer, rng)
            example = _example(
                text,
                text,
                realizer,
                self.info.rule_id,
                mode,
                metadata_with_safety_clauses(ast, builder.lexicon),
            )
            validate_target_ast_or_raise(ast, text, example)
            return example
        raise ValueError(f"Unsupported generation mode: {mode!r}")


def _target_with_mark(
    builder: GrammarBuilder,
    realizer: Realizer,
    rng: RandomSource,
) -> tuple[str, SimpleSentence]:
    ast = builder.simple_sentence()
    text = realizer.render_sentence(ast)
    return f"{text[:-1]}{rng.choice(FINAL_MARKS)}", ast


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
