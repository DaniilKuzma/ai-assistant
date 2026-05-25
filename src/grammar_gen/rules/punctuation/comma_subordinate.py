from __future__ import annotations

from src.grammar_gen.builders import GrammarBuilder
from src.grammar_gen.randomness import RandomSource
from src.grammar_gen.realizer import Realizer
from src.grammar_gen.rules.base import GenerationMode, RuleInfo, RuleProgram
from src.grammar_gen.rules.common import make_punctuation_example, remove_punctuation_before
from src.schema import GeneratedExample


ACTIVE_GAPS = frozenset({"COMMA"})


class CommaSubordinateRule(RuleProgram):
    info = RuleInfo(
        rule_id="comma_subordinate",
        family="punctuation",
        description="Comma before subordinate conjunction что.",
        explanation="Перед придаточной частью с союзом что ставится запятая.",
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
            target = realizer.render_sentence(builder.complex_subordinate_sentence("что"))
            source = remove_punctuation_before(target, "что")
            return _example(source, target, realizer, self.info.rule_id, mode)
        if mode is GenerationMode.HARD_NEGATIVE:
            text = rng.choice(
                (
                    "Редактор заметил что-то важное.",
                    "Редактор заметил кое-что важное.",
                    "Студент не то что проверил отчёт.",
                )
            )
            return _example(text, text, realizer, self.info.rule_id, mode, {"trap_type": "chto_pronoun"})
        if mode is GenerationMode.CLEAN_IDENTITY:
            text = realizer.render_sentence(builder.complex_subordinate_sentence("что"))
            return _example(text, text, realizer, self.info.rule_id, mode)
        raise ValueError(f"Unsupported generation mode: {mode!r}")


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


__all__ = ["CommaSubordinateRule"]
