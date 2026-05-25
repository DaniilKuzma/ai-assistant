from __future__ import annotations

from src.grammar_gen.ast import Clause, NounPhrase, SimpleSentence, VerbPhrase
from src.grammar_gen.builders import GrammarBuilder
from src.grammar_gen.lexicon import NounEntry
from src.grammar_gen.randomness import RandomSource
from src.grammar_gen.realizer import Realizer
from src.grammar_gen.rules.base import GenerationMode, RuleInfo, RuleProgram
from src.grammar_gen.rules.common import (
    find_token_sequence,
    gap_labels_from_text,
    label_span,
    make_clean_identity_example,
    metadata_with_safety_clauses,
    metadata_without_safety_clauses,
    replace_once_checked,
    token_labels_all_keep,
)
from src.schema import GeneratedExample


class TozheRule(RuleProgram):
    info = RuleInfo(
        rule_id="tozhe_to_zhe",
        family="orthography_contextual",
        description="Merged spelling of тоже in additive meaning.",
        explanation="Союз тоже в значении добавления пишется слитно.",
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
            return self._positive(builder, realizer, rng)
        if mode is GenerationMode.HARD_NEGATIVE:
            return self._hard_negative(realizer, rng)
        if mode is GenerationMode.CLEAN_IDENTITY:
            return self._clean_identity(builder, realizer, rng)
        raise ValueError(f"Unsupported generation mode: {mode!r}")

    def _positive(
        self,
        builder: GrammarBuilder,
        realizer: Realizer,
        rng: RandomSource,
    ) -> GeneratedExample:
        target, sentence = _additive_sentence(builder, realizer, rng)
        source = replace_once_checked(target, "тоже", "то же")
        source_tokens = realizer.tokenize_words_with_offsets(source)
        labels = token_labels_all_keep(source_tokens)
        start = find_token_sequence(source_tokens, ("то", "же"))
        label_span(labels, start, start + 2, "MERGE_TO_ZHE_TO_TOZHE")
        return _example(
            source,
            target,
            source_tokens,
            labels,
            self.info.rule_id,
            GenerationMode.POSITIVE,
            metadata_with_safety_clauses(sentence, builder.lexicon),
        )

    def _hard_negative(
        self,
        realizer: Realizer,
        rng: RandomSource,
    ) -> GeneratedExample:
        text = rng.choice(
            (
                "Студент выбрал то же самое.",
                "Студент сделал то же, что эксперт.",
            )
        )
        return _identity_example(
            text,
            realizer,
            self.info.rule_id,
            GenerationMode.HARD_NEGATIVE,
            metadata_without_safety_clauses(),
        )

    def _clean_identity(
        self,
        builder: GrammarBuilder,
        realizer: Realizer,
        rng: RandomSource,
    ) -> GeneratedExample:
        if rng.chance(0.5):
            text, sentence = _additive_sentence(builder, realizer, rng)
            metadata = metadata_with_safety_clauses(sentence, builder.lexicon)
        else:
            text = "Студент выбрал то же самое."
            metadata = metadata_without_safety_clauses()
        tokens = realizer.tokenize_words_with_offsets(text)
        return make_clean_identity_example(text, tokens, rule_id=self.info.rule_id, metadata=metadata)


def _additive_sentence(builder: GrammarBuilder, realizer: Realizer, rng: RandomSource) -> tuple[str, SimpleSentence]:
    frame = next(frame for frame in builder.lexicon.frames.frames if frame.frame_id == "check_document")
    subject_entry = rng.choice(
        tuple(
            noun
            for noun in (_entry(builder, "студент"), _entry(builder, "комиссия"))
            if builder.lexicon.frames.validate_subject(frame, noun)
        )
    )
    object_entry = _entry(builder, "отчёт")
    sentence = SimpleSentence(
        Clause(
            subject=_np(subject_entry),
            predicate=VerbPhrase(
                verb_lemma=frame.verb_lemma,
                object_np=_np(object_entry, case="accs"),
                adverbs=("тоже",),
                frame_id=frame.frame_id,
            ),
        )
    )
    return realizer.render_sentence(sentence), sentence


def _entry(builder: GrammarBuilder, lemma: str) -> NounEntry:
    return next(noun for noun in builder.lexicon.nouns if noun.lemma == lemma)


def _np(entry: NounEntry, *, case: str = "nomn") -> NounPhrase:
    number = "plur" if entry.gender == "plur" else "sing"
    return NounPhrase(
        noun_lemma=entry.lemma,
        gender=entry.gender,
        animacy=entry.animacy,
        number=number,
        case=case,
        semantic_class=entry.semantic_class,
    )


def _identity_example(
    text: str,
    realizer: Realizer,
    rule_id: str,
    mode: GenerationMode,
    metadata: dict[str, str],
) -> GeneratedExample:
    tokens = realizer.tokenize_words_with_offsets(text)
    return _example(text, text, tokens, token_labels_all_keep(tokens), rule_id, mode, metadata)


def _example(
    source: str,
    target: str,
    source_tokens,
    token_edit_labels: list[str],
    rule_id: str,
    mode: GenerationMode,
    metadata: dict[str, str],
) -> GeneratedExample:
    return GeneratedExample(
        source_text=source,
        target_text=target,
        source_tokens=list(source_tokens),
        token_edit_labels=token_edit_labels,
        gap_labels=gap_labels_from_text(source, source_tokens),
        rule_ids=[rule_id] * len(source_tokens),
        primary_rule_id=rule_id,
        mode=mode.value,
        explanation_ids=[rule_id],
        metadata=metadata,
    )


__all__ = ["TozheRule"]
