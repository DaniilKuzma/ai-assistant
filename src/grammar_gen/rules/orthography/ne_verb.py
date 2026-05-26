from __future__ import annotations

from src.grammar_gen.ast import Clause, SimpleSentence, VerbPhrase
from src.grammar_gen.builders import GrammarBuilder
from src.grammar_gen.lexicon import NounEntry
from src.grammar_gen.randomness import RandomSource
from src.grammar_gen.realizer import Realizer
from src.grammar_gen.rules.base import GenerationMode, RuleInfo, RuleProgram
from src.grammar_gen.rules.common import (
    gap_labels_from_text,
    make_clean_identity_example,
    metadata_with_safety_clauses,
    metadata_without_safety_clauses,
    noun_phrase_from_entry,
    object_np_for_frame,
    replace_once_checked,
    token_labels_all_keep,
)
from src.grammar_gen.safety import validate_target_ast_or_raise
from src.grammar_gen.semantics import VerbFrame
from src.schema import GeneratedExample


class NeVerbRule(RuleProgram):
    info = RuleInfo(
        rule_id="ne_verb",
        family="orthography_contextual",
        description="Separated spelling of не with verbs.",
        explanation="Частица не с глаголами обычно пишется раздельно.",
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
        sentence = _sentence_with_negated_allowed_verb(builder, rng)
        target = realizer.render_sentence(sentence)
        target_tokens = realizer.tokenize_words_with_offsets(target)
        ne_index = _find_token(target_tokens, "не")
        if ne_index < 0 or ne_index + 1 >= len(target_tokens):
            raise ValueError("Generated target sentence does not contain separated не before a verb.")

        verb_form = target_tokens[ne_index + 1].text
        source = replace_once_checked(target, f"не {verb_form}", f"не{verb_form}")
        source_tokens = realizer.tokenize_words_with_offsets(source)
        labels = token_labels_all_keep(source_tokens)
        merged_index = _find_token(source_tokens, f"не{verb_form}")
        if merged_index < 0:
            raise ValueError("Merged не+verb token was not found in source sentence.")
        labels[merged_index] = "SPLIT_NE_VERB"

        example = _example(
            source=source,
            target=target,
            source_tokens=source_tokens,
            token_edit_labels=labels,
            rule_id=self.info.rule_id,
            mode=GenerationMode.POSITIVE,
            metadata=metadata_with_safety_clauses(
                sentence,
                builder.lexicon,
                {"phenomenon": "ne_verb"},
            ),
        )
        validate_target_ast_or_raise(sentence, target, example)
        return example

    def _hard_negative(
        self,
        realizer: Realizer,
        rng: RandomSource,
    ) -> GeneratedExample:
        text = rng.choice(
            (
                "Студент ненавидел шум.",
                "Комиссия негодовала после заседания.",
            )
        )
        return _identity_example(
            text,
            realizer,
            self.info.rule_id,
            GenerationMode.HARD_NEGATIVE,
            metadata_without_safety_clauses({"trap_type": "lexicalized_ne_verb"}),
        )

    def _clean_identity(
        self,
        builder: GrammarBuilder,
        realizer: Realizer,
        rng: RandomSource,
    ) -> GeneratedExample:
        sentence = _sentence_with_negated_allowed_verb(builder, rng)
        text = realizer.render_sentence(sentence)
        tokens = realizer.tokenize_words_with_offsets(text)
        example = make_clean_identity_example(
            text,
            tokens,
            rule_id=self.info.rule_id,
            metadata=metadata_with_safety_clauses(sentence, builder.lexicon),
        )
        validate_target_ast_or_raise(sentence, text, example)
        return example


def _sentence_with_negated_allowed_verb(builder: GrammarBuilder, rng: RandomSource) -> SimpleSentence:
    frame = _controlled_allowed_negative_frame(builder)
    subject = noun_phrase_from_entry(_controlled_subject(builder, frame, rng))
    object_np = object_np_for_frame(builder, rng, frame, case="accs", adjective_probability=0.25)
    predicate = VerbPhrase(
        verb_lemma=frame.verb_lemma,
        object_np=object_np,
        adverbs=(builder.lexicon.random_adverb(rng).lemma,) if rng.chance(0.25) else (),
        negated=True,
        frame_id=frame.frame_id,
    )
    return SimpleSentence(Clause(subject=subject, predicate=predicate))


def _controlled_allowed_negative_frame(builder: GrammarBuilder) -> VerbFrame:
    frame = next(frame for frame in builder.lexicon.frames.frames if frame.frame_id == "check_document")
    verb_entry = next(verb for verb in builder.lexicon.verbs if verb.lemma == frame.verb_lemma)
    if not frame.allow_ne or not verb_entry.allow_ne:
        raise ValueError("Controlled ne_verb frame does not allow separated не.")
    return frame


def _controlled_subject(builder: GrammarBuilder, frame: VerbFrame, rng: RandomSource) -> NounEntry:
    valid = tuple(noun for noun in builder.lexicon.nouns if builder.lexicon.frames.validate_subject(frame, noun))
    return rng.choice(valid)


def _entry(builder: GrammarBuilder, lemma: str) -> NounEntry:
    return next(noun for noun in builder.lexicon.nouns if noun.lemma == lemma)


def _find_token(tokens, text: str) -> int:
    lowered = text.lower()
    for index, token in enumerate(tokens):
        if token.text.lower() == lowered:
            return index
    return -1


def _identity_example(
    text: str,
    realizer: Realizer,
    rule_id: str,
    mode: GenerationMode,
    metadata: dict[str, str],
) -> GeneratedExample:
    tokens = realizer.tokenize_words_with_offsets(text)
    return _example(
        source=text,
        target=text,
        source_tokens=tokens,
        token_edit_labels=token_labels_all_keep(tokens),
        rule_id=rule_id,
        mode=mode,
        metadata=metadata,
    )


def _example(
    *,
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


__all__ = ["NeVerbRule"]
