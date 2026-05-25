from __future__ import annotations

from src.grammar_gen.builders import GrammarBuilder
from src.grammar_gen.randomness import RandomSource
from src.grammar_gen.realizer import Realizer
from src.grammar_gen.rules.base import GenerationMode, RuleInfo, RuleProgram
from src.grammar_gen.rules.common import capitalize_first, make_punctuation_example, remove_punctuation_before, varied_np
from src.schema import GeneratedExample


ACTIVE_GAPS = frozenset({"COMMA"})


class CommaAdversativeRule(RuleProgram):
    info = RuleInfo(
        rule_id="comma_adversative",
        family="punctuation",
        description="Comma before adversative conjunctions но and а.",
        explanation="Перед противительными союзами но и а обычно ставится запятая.",
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
            target = _adversative_sentence(builder, realizer, rng)
            marker = "но не" if ", но " in target else "а не"
            source = remove_punctuation_before(target, marker)
            return _example(source, target, realizer, self.info.rule_id, mode)
        if mode is GenerationMode.HARD_NEGATIVE:
            text = rng.choice(
                (
                    "Студент проверил отчёт а также письмо.",
                    "Редактор прочитал книгу а также отчёт.",
                )
            )
            return _example(text, text, realizer, self.info.rule_id, mode, {"trap_type": "a_takzhe"})
        if mode is GenerationMode.CLEAN_IDENTITY:
            text = _adversative_sentence(builder, realizer, rng)
            return _example(text, text, realizer, self.info.rule_id, mode)
        raise ValueError(f"Unsupported generation mode: {mode!r}")


def _adversative_sentence(builder: GrammarBuilder, realizer: Realizer, rng: RandomSource) -> str:
    subject = varied_np(builder, rng, ("person", "organization"), adjective_probability=0.25)
    first_object = varied_np(
        builder,
        rng,
        ("document", "report", "text", "message", "file", "book", "plan"),
        case="accs",
    )
    second_object = varied_np(builder, rng, ("error", "problem", "issue"), case="accs")
    first_verb = realizer.morphology.inflect_verb_past("проверить", subject.gender, subject.number)
    second_verb = realizer.morphology.inflect_verb_past("исправить", subject.gender, subject.number)
    conjunction = rng.choice(("но", "а"))
    return capitalize_first(
        f"{realizer.render_np(subject)} {first_verb} {realizer.render_np(first_object)}, "
        f"{conjunction} не {second_verb} {realizer.render_np(second_object)}."
    )


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


__all__ = ["CommaAdversativeRule"]
