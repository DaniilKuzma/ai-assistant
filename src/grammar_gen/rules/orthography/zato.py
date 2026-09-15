from __future__ import annotations

from src.grammar_gen.ast import Clause, ComplexSentence, VerbPhrase
from src.grammar_gen.builders import GrammarBuilder
from src.grammar_gen.randomness import RandomSource
from src.grammar_gen.realizer import Realizer
from src.grammar_gen.rules.base import GenerationMode, RuleInfo, RuleProgram
from src.grammar_gen.rules.common import (
    find_token_sequence,
    gap_labels_from_text,
    label_span,
    make_clean_identity_example,
    metadata_from_construction,
    metadata_with_safety_clauses,
    metadata_without_safety_clauses,
    object_np_for_frame,
    replace_once_checked,
    render_construction_by_id,
    token_labels_all_keep,
    varied_np,
)
from src.grammar_gen.safety import validate_target_ast_or_raise
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
            return self._hard_negative(builder, realizer, rng)
        if mode is GenerationMode.CLEAN_IDENTITY:
            return self._clean_identity(builder, realizer, rng)
        raise ValueError(f"Unsupported generation mode: {mode!r}")

    def _positive(
        self,
        builder: GrammarBuilder,
        realizer: Realizer,
        rng: RandomSource,
    ) -> GeneratedExample:
        rendered = render_construction_by_id(builder, realizer, rng, "zato_document_correction")
        target = rendered.text
        source = replace_once_checked(target, "зато", "за то")
        source_tokens = realizer.tokenize_words_with_offsets(source)
        labels = token_labels_all_keep(source_tokens)
        start = find_token_sequence(source_tokens, ("за", "то"))
        label_span(labels, start, start + 2, "MERGE_ZA_TO_TO_ZATO")
        example = _example(
            source,
            target,
            source_tokens,
            labels,
            self.info.rule_id,
            GenerationMode.POSITIVE,
            metadata_from_construction(rendered),
        )
        validate_target_ast_or_raise(rendered.ast, target, example)
        return example

    def _hard_negative(self, builder: GrammarBuilder, realizer: Realizer, rng: RandomSource) -> GeneratedExample:
        rendered = render_construction_by_id(builder, realizer, rng, "zato_preposition_trap")
        text = rendered.text
        return _identity_example(
            text,
            realizer,
            self.info.rule_id,
            GenerationMode.HARD_NEGATIVE,
            metadata_from_construction(rendered),
        )

    def _clean_identity(
        self,
        builder: GrammarBuilder,
        realizer: Realizer,
        rng: RandomSource,
    ) -> GeneratedExample:
        if rng.chance(0.5):
            rendered = render_construction_by_id(builder, realizer, rng, "zato_document_correction")
            text = rendered.text
            metadata = metadata_from_construction(rendered)
        else:
            rendered = render_construction_by_id(builder, realizer, rng, "zato_preposition_trap")
            text = rendered.text
            metadata = metadata_from_construction(rendered)
        tokens = realizer.tokenize_words_with_offsets(text)
        example = make_clean_identity_example(text, tokens, rule_id=self.info.rule_id, metadata=metadata)
        if "safety_clauses" in metadata:
            if rendered.ast is not None:
                validate_target_ast_or_raise(rendered.ast, text, example)
        return example


def _contrast_sentence(builder: GrammarBuilder, realizer: Realizer, rng: RandomSource) -> tuple[str, ComplexSentence]:
    main = _clause_for_frame(builder, rng, "check_document")
    subordinate = _clause_for_frame(builder, rng, "correct_error")
    sentence = ComplexSentence(
        main=main,
        conjunction="зато",
        subordinate=subordinate,
        comma_before_conjunction=True,
    )
    return realizer.render_sentence(sentence), sentence


def _clause_for_frame(builder: GrammarBuilder, rng: RandomSource, frame_id: str) -> Clause:
    frame = next(frame for frame in builder.lexicon.frames.frames if frame.frame_id == frame_id)
    subject_entry = rng.choice(tuple(noun for noun in builder.lexicon.nouns if builder.lexicon.frames.validate_subject(frame, noun)))
    return Clause(
        subject=varied_np(builder, rng, (subject_entry.semantic_class,), adjective_probability=0.20),
        predicate=VerbPhrase(
            verb_lemma=frame.verb_lemma,
            object_np=object_np_for_frame(builder, rng, frame, case="accs", adjective_probability=0.25),
            frame_id=frame.frame_id,
        ),
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
