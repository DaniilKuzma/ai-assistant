from __future__ import annotations

from src.grammar_gen.builders import GrammarBuilder
from src.grammar_gen.randomness import RandomSource
from src.grammar_gen.realizer import Realizer
from src.grammar_gen.rules.base import GenerationMode, RuleInfo, RuleProgram
from src.grammar_gen.rules.common import (
    capitalize_first,
    gap_labels_from_text,
    make_clean_identity_example,
    metadata_from_construction,
    render_construction_by_id,
    token_labels_all_keep,
    varied_np,
)
from src.schema import GeneratedExample


CONTROLLED_VERBS = (
    ("учиться", "учится"),
    ("готовиться", "готовится"),
    ("трудиться", "трудится"),
    ("ошибаться", "ошибается"),
    ("возвращаться", "возвращается"),
)


class TsyaTtsyaRule(RuleProgram):
    info = RuleInfo(
        rule_id="tsya_ttsya",
        family="orthography_contextual",
        description="Contextual тся/ться spelling for finite verbs and infinitives.",
        explanation="В инфинитиве пишется -ться, в форме 3-го лица без мягкого знака пишется -тся.",
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
            rendered = render_construction_by_id(builder, realizer, rng, "tsya_ttsya_controlled_person")
            infinitive = str(rendered.metadata["tsya_infinitive"])
            finite = str(rendered.metadata["tsya_finite"])
            if rendered.metadata["tsya_direction"] == "infinitive":
                return _labeled_example(
                    source=rendered.text.replace(infinitive, finite),
                    target=rendered.text,
                    labeled_token=finite,
                    label="FIX_TSYA_TO_TTSYA",
                    realizer=realizer,
                    rule_id=self.info.rule_id,
                    mode=GenerationMode.POSITIVE,
                    metadata=metadata_from_construction(rendered),
                )
            return _labeled_example(
                source=rendered.text.replace(finite, infinitive),
                target=rendered.text,
                labeled_token=infinitive,
                label="FIX_TTSYA_TO_TSYA",
                realizer=realizer,
                rule_id=self.info.rule_id,
                mode=GenerationMode.POSITIVE,
                metadata=metadata_from_construction(rendered),
            )
        if mode is GenerationMode.HARD_NEGATIVE:
            rendered = render_construction_by_id(builder, realizer, rng, "tsya_ttsya_controlled_person")
            return _identity_example(
                rendered.text,
                realizer,
                self.info.rule_id,
                GenerationMode.HARD_NEGATIVE,
                metadata_from_construction(rendered),
            )
        if mode is GenerationMode.CLEAN_IDENTITY:
            rendered = render_construction_by_id(builder, realizer, rng, "tsya_ttsya_controlled_person")
            text = rendered.text
            tokens = realizer.tokenize_words_with_offsets(text)
            return make_clean_identity_example(
                text,
                tokens,
                rule_id=self.info.rule_id,
                metadata=metadata_from_construction(rendered),
            )
        raise ValueError(f"Unsupported generation mode: {mode!r}")


def _labeled_example(
    *,
    source: str,
    target: str,
    labeled_token: str,
    label: str,
    realizer: Realizer,
    rule_id: str,
    mode: GenerationMode,
    metadata: dict[str, object],
) -> GeneratedExample:
    tokens = realizer.tokenize_words_with_offsets(source)
    labels = token_labels_all_keep(tokens)
    for index, token in enumerate(tokens):
        if token.text.lower() == labeled_token:
            labels[index] = label
            break
    else:
        raise ValueError(f"Token {labeled_token!r} was not found in source sentence.")
    return _example(source, target, tokens, labels, rule_id, mode, metadata)


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


__all__ = ["TsyaTtsyaRule"]
