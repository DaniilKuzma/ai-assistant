from __future__ import annotations

from src.grammar_gen.builders import GrammarBuilder
from src.grammar_gen.randomness import RandomSource
from src.grammar_gen.realizer import Realizer
from src.grammar_gen.rules.base import GenerationMode, RuleInfo, RuleProgram
from src.grammar_gen.rules.common import make_punctuation_example, remove_punctuation_before
from src.schema import GeneratedExample


ACTIVE_GAPS = frozenset({"COMMA"})


class CommaHomogeneousRule(RuleProgram):
    info = RuleInfo(
        rule_id="comma_homogeneous",
        family="punctuation",
        description="Comma between homogeneous objects before a final conjunction.",
        explanation="Однородные члены без повторяющегося союза разделяются запятой.",
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
        del builder
        if mode is GenerationMode.POSITIVE:
            target = rng.choice(
                (
                    "Студент прочитал книгу, отчёт и письмо.",
                    "Редактор проверил отчёт, письмо и документ.",
                )
            )
            source = remove_punctuation_before(target, "отчёт" if "книгу" in target else "письмо")
            return _example(source, target, realizer, self.info.rule_id, mode)
        if mode is GenerationMode.HARD_NEGATIVE:
            text = rng.choice(
                (
                    "Студент прочитал книгу и отчёт.",
                    "Редактор проверил отчёт и письмо.",
                )
            )
            return _example(text, text, realizer, self.info.rule_id, mode, {"trap_type": "single_conjunction"})
        if mode is GenerationMode.CLEAN_IDENTITY:
            text = rng.choice(
                (
                    "Студент прочитал книгу, отчёт и письмо.",
                    "Редактор проверил отчёт, письмо и документ.",
                )
            )
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


__all__ = ["CommaHomogeneousRule"]
