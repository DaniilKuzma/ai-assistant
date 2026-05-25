from __future__ import annotations

from src.grammar_gen.ast import Clause, ComplexSentence, NounPhrase, VerbPhrase
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
    replace_once_checked,
    token_labels_all_keep,
)
from src.schema import GeneratedExample


class ZatoRule(RuleProgram):
    info = RuleInfo(
        rule_id="zato_za_to",
        family="orthography_contextual",
        description="Merged spelling of зато as a contrast conjunction.",
        explanation="Союз зато в противопоставительном значении пишется слитно.",
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
            return self._hard_negative(realizer)
        if mode is GenerationMode.CLEAN_IDENTITY:
            return self._clean_identity(builder, realizer, rng)
        raise ValueError(f"Unsupported generation mode: {mode!r}")

    def _positive(
        self,
        builder: GrammarBuilder,
        realizer: Realizer,
        rng: RandomSource,
    ) -> GeneratedExample:
        target = _contrast_sentence(builder, realizer, rng)
        source = replace_once_checked(target, "зато", "за то")
        source_tokens = realizer.tokenize_words_with_offsets(source)
        labels = token_labels_all_keep(source_tokens)
        start = find_token_sequence(source_tokens, ("за", "то"))
        label_span(labels, start, start + 2, "MERGE_ZA_TO_TO_ZATO")
        return _example(source, target, source_tokens, labels, self.info.rule_id, GenerationMode.POSITIVE, {})

    def _hard_negative(self, realizer: Realizer) -> GeneratedExample:
        text = "Комиссия голосовала за то решение."
        return _identity_example(text, realizer, self.info.rule_id, GenerationMode.HARD_NEGATIVE, {})

    def _clean_identity(
        self,
        builder: GrammarBuilder,
        realizer: Realizer,
        rng: RandomSource,
    ) -> GeneratedExample:
        text = _contrast_sentence(builder, realizer, rng) if rng.chance(0.5) else "Комиссия голосовала за то решение."
        tokens = realizer.tokenize_words_with_offsets(text)
        return make_clean_identity_example(text, tokens, rule_id=self.info.rule_id)


def _contrast_sentence(builder: GrammarBuilder, realizer: Realizer, rng: RandomSource) -> str:
    main = _clause_for_frame(builder, rng, "check_document")
    subordinate = _clause_for_frame(builder, rng, "correct_error")
    return realizer.render_sentence(
        ComplexSentence(
            main=main,
            conjunction="зато",
            subordinate=subordinate,
            comma_before_conjunction=True,
        )
    )


def _clause_for_frame(builder: GrammarBuilder, rng: RandomSource, frame_id: str) -> Clause:
    frame = next(frame for frame in builder.lexicon.frames.frames if frame.frame_id == frame_id)
    subject_entry = rng.choice(
        tuple(
            noun
            for noun in (_entry(builder, "студент"), _entry(builder, "комиссия"))
            if builder.lexicon.frames.validate_subject(frame, noun)
        )
    )
    object_entry = _entry(builder, "отчёт" if frame_id == "check_document" else "текст")
    return Clause(
        subject=_np(subject_entry),
        predicate=VerbPhrase(
            verb_lemma=frame.verb_lemma,
            object_np=_np(object_entry, case="accs"),
            frame_id=frame.frame_id,
        ),
    )


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


__all__ = ["ZatoRule"]
