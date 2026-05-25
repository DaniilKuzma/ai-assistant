from __future__ import annotations

from src.grammar_gen.builders import GrammarBuilder
from src.grammar_gen.randomness import RandomSource
from src.grammar_gen.realizer import Realizer
from src.grammar_gen.rules.base import GenerationMode, RuleInfo, RuleProgram
from src.grammar_gen.rules.common import (
    capitalize_first,
    gap_labels_from_text,
    make_clean_identity_example,
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
        infinitive, finite = rng.choice(CONTROLLED_VERBS)
        subject_np = varied_np(builder, rng, ("person",), adjective_probability=0.20)
        subject = capitalize_first(realizer.render_np(subject_np))
        want = realizer.morphology.inflect_verb_past("хотеть", subject_np.gender, subject_np.number)
        if mode is GenerationMode.POSITIVE:
            if rng.chance(0.5):
                return _labeled_example(
                    source=f"{subject} {want} {finite}.",
                    target=f"{subject} {want} {infinitive}.",
                    labeled_token=finite,
                    label="FIX_TSYA_TO_TTSYA",
                    realizer=realizer,
                    rule_id=self.info.rule_id,
                    mode=GenerationMode.POSITIVE,
                )
            return _labeled_example(
                source=f"{subject} {infinitive} утром.",
                target=f"{subject} {finite} утром.",
                labeled_token=infinitive,
                label="FIX_TTSYA_TO_TSYA",
                realizer=realizer,
                rule_id=self.info.rule_id,
                mode=GenerationMode.POSITIVE,
            )
        if mode is GenerationMode.HARD_NEGATIVE:
            text = (
                f"{subject} {want} {infinitive}."
                if rng.chance(0.5)
                else f"{subject} {finite} утром."
            )
            return _identity_example(text, realizer, self.info.rule_id, GenerationMode.HARD_NEGATIVE, {})
        if mode is GenerationMode.CLEAN_IDENTITY:
            text = (
                f"{subject} {want} {infinitive}."
                if rng.chance(0.5)
                else f"{subject} {finite} утром."
            )
            tokens = realizer.tokenize_words_with_offsets(text)
            return make_clean_identity_example(text, tokens, rule_id=self.info.rule_id)
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
) -> GeneratedExample:
    tokens = realizer.tokenize_words_with_offsets(source)
    labels = token_labels_all_keep(tokens)
    for index, token in enumerate(tokens):
        if token.text.lower() == labeled_token:
            labels[index] = label
            break
    else:
        raise ValueError(f"Token {labeled_token!r} was not found in source sentence.")
    return _example(source, target, tokens, labels, rule_id, mode, {})


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
