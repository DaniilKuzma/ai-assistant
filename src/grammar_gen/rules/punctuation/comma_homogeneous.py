from __future__ import annotations

from src.grammar_gen.builders import GrammarBuilder
from src.grammar_gen.randomness import RandomSource
from src.grammar_gen.realizer import Realizer
from src.grammar_gen.rules.base import GenerationMode, RuleInfo, RuleProgram
from src.grammar_gen.rules.common import (
    capitalize_first,
    make_punctuation_example,
    noun_phrase_from_entry,
    remove_punctuation_before,
    varied_adjectives,
    varied_np,
)
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
        if mode is GenerationMode.POSITIVE:
            target, marker = _homogeneous_sentence(builder, realizer, rng)
            source = remove_punctuation_before(target, marker)
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
            text, _marker = _homogeneous_sentence(builder, realizer, rng)
            return _example(text, text, realizer, self.info.rule_id, mode)
        raise ValueError(f"Unsupported generation mode: {mode!r}")


def _homogeneous_sentence(builder: GrammarBuilder, realizer: Realizer, rng: RandomSource) -> tuple[str, str]:
    subject = varied_np(builder, rng, ("person",), adjective_probability=0.25)
    verb_lemma = rng.choice(("проверить", "прочитать", "подписать", "открыть"))
    verb = realizer.morphology.inflect_verb_past(verb_lemma, subject.gender, subject.number)
    objects = [_object_for_verb(builder, rng, verb_lemma) for _ in range(3)]
    rendered = [realizer.render_np(obj) for obj in objects]
    text = f"{realizer.render_np(subject)} {verb} {rendered[0]}, {rendered[1]} и {rendered[2]}."
    return capitalize_first(text), rendered[1]


def _object_for_verb(builder: GrammarBuilder, rng: RandomSource, verb_lemma: str):
    allowed_lemmas = {
        "подписать": ("документ", "заявление", "протокол", "договор", "приказ", "отчёт", "доклад", "сводка"),
        "открыть": ("файл", "архив", "документ"),
    }.get(verb_lemma)
    if allowed_lemmas is not None:
        candidates = tuple(noun for noun in builder.lexicon.nouns if noun.lemma in allowed_lemmas)
        entry = rng.choice(candidates)
        return noun_phrase_from_entry(
            entry,
            case="accs",
            adjective_lemmas=varied_adjectives(builder, rng, noun_entry=entry, probability=0.15),
        )

    classes = {
        "проверить": ("document", "report", "text", "calculation", "task", "data"),
        "прочитать": ("book", "document", "text", "message", "report"),
    }[verb_lemma]
    return varied_np(builder, rng, classes, case="accs", adjective_probability=0.15)


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
