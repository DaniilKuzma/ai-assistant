from __future__ import annotations

from src.grammar_gen.builders import GrammarBuilder
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


PARTICLE_CASES = (
    ("Кто-то проверил отчёт.", "Кто то проверил отчёт.", ("Кто", "то"), "HYPHENATE_PARTICLE_TO"),
    ("Кто-либо проверил отчёт.", "Кто либо проверил отчёт.", ("Кто", "либо"), "HYPHENATE_PARTICLE_LIBO"),
    ("Кто-нибудь проверил отчёт.", "Кто нибудь проверил отчёт.", ("Кто", "нибудь"), "HYPHENATE_PARTICLE_NIBUD"),
)


class HyphenParticlesRule(RuleProgram):
    info = RuleInfo(
        rule_id="hyphen_particles",
        family="orthography_contextual",
        description="Hyphenated indefinite pronoun particles то, либо, нибудь.",
        explanation="Частицы то, либо, нибудь в неопределённых местоимениях пишутся через дефис.",
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
            target, source, sequence, label = rng.choice(PARTICLE_CASES)
            return _hyphen_positive(source, target, sequence, label, realizer, self.info.rule_id)
        if mode is GenerationMode.HARD_NEGATIVE:
            text = rng.choice(("То решение осталось важным.", "Либо эксперт, либо студент проверил отчёт."))
            return _identity_example(text, realizer, self.info.rule_id, GenerationMode.HARD_NEGATIVE, {})
        if mode is GenerationMode.CLEAN_IDENTITY:
            target, _, _, _ = rng.choice(PARTICLE_CASES)
            tokens = realizer.tokenize_words_with_offsets(target)
            return make_clean_identity_example(target, tokens, rule_id=self.info.rule_id)
        raise ValueError(f"Unsupported generation mode: {mode!r}")


class HyphenKoeRule(RuleProgram):
    info = RuleInfo(
        rule_id="hyphen_koe",
        family="orthography_contextual",
        description="Hyphenated indefinite pronouns with кое.",
        explanation="Кое в неопределённых местоимениях пишется через дефис.",
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
            return _hyphen_positive(
                "Кое кто проверил отчёт.",
                "Кое-кто проверил отчёт.",
                ("Кое", "кто"),
                "HYPHENATE_KOE",
                realizer,
                self.info.rule_id,
            )
        if mode is GenerationMode.HARD_NEGATIVE:
            return _identity_example(
                "Студент кое у кого спросил.",
                realizer,
                self.info.rule_id,
                GenerationMode.HARD_NEGATIVE,
                {},
            )
        if mode is GenerationMode.CLEAN_IDENTITY:
            text = "Кое-кто проверил отчёт." if rng.chance(0.5) else "Студент кое у кого спросил."
            tokens = realizer.tokenize_words_with_offsets(text)
            return make_clean_identity_example(text, tokens, rule_id=self.info.rule_id)
        raise ValueError(f"Unsupported generation mode: {mode!r}")


class HyphenPoAdverbRule(RuleProgram):
    info = RuleInfo(
        rule_id="hyphen_po_adverb",
        family="orthography_contextual",
        description="Hyphenated adverbs with по- and -ски.",
        explanation="Наречия на по-...-ски пишутся через дефис.",
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
            target = "Студент говорил по-русски."
            source = replace_once_checked(target, "по-русски", "по русски")
            return _hyphen_positive(
                source,
                target,
                ("по", "русски"),
                "HYPHENATE_PO_ADVERB",
                realizer,
                self.info.rule_id,
            )
        if mode is GenerationMode.HARD_NEGATIVE:
            return _identity_example(
                "Студент шёл по русской дороге.",
                realizer,
                self.info.rule_id,
                GenerationMode.HARD_NEGATIVE,
                {},
            )
        if mode is GenerationMode.CLEAN_IDENTITY:
            text = "Студент говорил по-русски." if rng.chance(0.5) else "Студент шёл по русской дороге."
            tokens = realizer.tokenize_words_with_offsets(text)
            return make_clean_identity_example(text, tokens, rule_id=self.info.rule_id)
        raise ValueError(f"Unsupported generation mode: {mode!r}")


def _hyphen_positive(
    source: str,
    target: str,
    sequence: tuple[str, str],
    first_label: str,
    realizer: Realizer,
    rule_id: str,
) -> GeneratedExample:
    source_tokens = realizer.tokenize_words_with_offsets(source)
    labels = token_labels_all_keep(source_tokens)
    start = find_token_sequence(source_tokens, sequence)
    label_span(labels, start, start + 2, first_label)
    return _example(source, target, source_tokens, labels, rule_id, GenerationMode.POSITIVE, {})


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


__all__ = ["HyphenKoeRule", "HyphenParticlesRule", "HyphenPoAdverbRule"]
