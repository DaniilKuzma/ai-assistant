from __future__ import annotations

from src.grammar_gen.builders import GrammarBuilder
from src.grammar_gen.randomness import RandomSource
from src.grammar_gen.realizer import Realizer
from src.grammar_gen.rules.base import GenerationMode, RuleInfo, RuleProgram
from src.grammar_gen.rules.common import (
    find_token_sequence,
    capitalize_first,
    gap_labels_from_text,
    label_span,
    make_clean_identity_example,
    metadata_from_construction,
    noun_phrase_from_entry,
    render_construction_by_id,
    varied_np,
    varied_adjectives,
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
        if mode is GenerationMode.POSITIVE:
            rendered = render_construction_by_id(builder, realizer, rng, "hyphen_particles_context")
            return _hyphen_positive(
                str(rendered.metadata["source_text"]),
                rendered.text,
                tuple(rendered.metadata["token_sequence"]),
                str(rendered.metadata["token_label"]),
                realizer,
                self.info.rule_id,
                metadata_from_construction(rendered),
            )
        if mode is GenerationMode.HARD_NEGATIVE:
            rendered = render_construction_by_id(builder, realizer, rng, "hyphen_particles_negative_context")
            text = rendered.text
            return _identity_example(
                text,
                realizer,
                self.info.rule_id,
                GenerationMode.HARD_NEGATIVE,
                metadata_from_construction(rendered),
            )
        if mode is GenerationMode.CLEAN_IDENTITY:
            rendered = render_construction_by_id(builder, realizer, rng, "hyphen_particles_context")
            target = rendered.text
            tokens = realizer.tokenize_words_with_offsets(target)
            return make_clean_identity_example(
                target,
                tokens,
                rule_id=self.info.rule_id,
                metadata=metadata_from_construction(rendered),
            )
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
        if mode is GenerationMode.POSITIVE:
            rendered = render_construction_by_id(builder, realizer, rng, "hyphen_koe_context")
            return _hyphen_positive(
                str(rendered.metadata["source_text"]),
                rendered.text,
                tuple(rendered.metadata["token_sequence"]),
                str(rendered.metadata["token_label"]),
                realizer,
                self.info.rule_id,
                metadata_from_construction(rendered),
            )
        if mode is GenerationMode.HARD_NEGATIVE:
            rendered = render_construction_by_id(builder, realizer, rng, "hyphen_koe_negative_context")
            return _identity_example(
                rendered.text,
                realizer,
                self.info.rule_id,
                GenerationMode.HARD_NEGATIVE,
                metadata_from_construction(rendered),
            )
        if mode is GenerationMode.CLEAN_IDENTITY:
            rendered = (
                render_construction_by_id(builder, realizer, rng, "hyphen_koe_context")
                if rng.chance(0.5)
                else render_construction_by_id(builder, realizer, rng, "hyphen_koe_negative_context")
            )
            text = rendered.text
            tokens = realizer.tokenize_words_with_offsets(text)
            return make_clean_identity_example(
                text,
                tokens,
                rule_id=self.info.rule_id,
                metadata=metadata_from_construction(rendered),
            )
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
        if mode is GenerationMode.POSITIVE:
            rendered = render_construction_by_id(builder, realizer, rng, "hyphen_po_adverb_context")
            return _hyphen_positive(
                str(rendered.metadata["source_text"]),
                rendered.text,
                tuple(rendered.metadata["token_sequence"]),
                str(rendered.metadata["token_label"]),
                realizer,
                self.info.rule_id,
                metadata_from_construction(rendered),
            )
        if mode is GenerationMode.HARD_NEGATIVE:
            rendered = render_construction_by_id(builder, realizer, rng, "hyphen_po_adverb_negative_context")
            return _identity_example(
                rendered.text,
                realizer,
                self.info.rule_id,
                GenerationMode.HARD_NEGATIVE,
                metadata_from_construction(rendered),
            )
        if mode is GenerationMode.CLEAN_IDENTITY:
            rendered = (
                render_construction_by_id(builder, realizer, rng, "hyphen_po_adverb_context")
                if rng.chance(0.5)
                else render_construction_by_id(builder, realizer, rng, "hyphen_po_adverb_negative_context")
            )
            text = rendered.text
            tokens = realizer.tokenize_words_with_offsets(text)
            return make_clean_identity_example(
                text,
                tokens,
                rule_id=self.info.rule_id,
                metadata=metadata_from_construction(rendered),
            )
        raise ValueError(f"Unsupported generation mode: {mode!r}")


def _particle_case(builder: GrammarBuilder, realizer: Realizer, rng: RandomSource) -> tuple[str, str, tuple[str, str], str]:
    target_subject, source_subject, sequence, label = rng.choice(PARTICLE_CASES)
    target_subject = target_subject.split()[0]
    source_subject = " ".join(source_subject.split()[:2])
    predicate = _pronoun_predicate(builder, realizer, rng)
    target = f"{target_subject} {predicate}"
    source = f"{source_subject} {predicate}"
    return target, source, sequence, label


def _koe_sentence(builder: GrammarBuilder, realizer: Realizer, rng: RandomSource) -> tuple[str, str]:
    predicate = _pronoun_predicate(builder, realizer, rng)
    return f"Кое-кто {predicate}", f"Кое кто {predicate}"


def _pronoun_predicate(builder: GrammarBuilder, realizer: Realizer, rng: RandomSource) -> str:
    verb = rng.choice(("проверил", "прочитал", "открыл", "подписал", "получил", "отправил"))
    obj = _pronoun_object_for_verb(builder, rng, verb)
    adverb = f" {builder.lexicon.random_adverb(rng).lemma}" if rng.chance(0.25) else ""
    return f"{verb} {realizer.render_np(obj)}{adverb}."


def _pronoun_object_for_verb(builder: GrammarBuilder, rng: RandomSource, verb: str):
    allowed_lemmas = {
        "открыл": ("файл", "архив", "документ"),
        "подписал": ("документ", "заявление", "протокол", "договор", "приказ", "отчёт", "доклад", "сводка"),
    }.get(verb)
    if allowed_lemmas is not None:
        candidates = tuple(noun for noun in builder.lexicon.nouns if noun.lemma in allowed_lemmas)
        entry = rng.choice(candidates)
        return noun_phrase_from_entry(
            entry,
            case="accs",
            adjective_lemmas=varied_adjectives(builder, rng, noun_entry=entry),
        )

    classes = {
        "проверил": ("document", "report", "text", "calculation", "task", "data"),
        "прочитал": ("book", "document", "text", "message", "report"),
        "получил": ("message", "file", "document"),
        "отправил": ("message", "file", "document"),
    }[verb]
    return varied_np(builder, rng, classes, case="accs")


def _po_adverb_sentence(builder: GrammarBuilder, realizer: Realizer, rng: RandomSource) -> str:
    subject = varied_np(builder, rng, ("person",), adjective_probability=0.25)
    verb = realizer.morphology.inflect_verb_past(
        rng.choice(("говорить", "писать", "ответить", "спросить")),
        subject.gender,
        subject.number,
    )
    adverbial = f" {builder.lexicon.random_adverb(rng).lemma}" if rng.chance(0.25) else ""
    return capitalize_first(f"{realizer.render_np(subject)} {verb} по-русски{adverbial}.")


def _hyphen_positive(
    source: str,
    target: str,
    sequence: tuple[str, str],
    first_label: str,
    realizer: Realizer,
    rule_id: str,
    metadata: dict[str, object],
) -> GeneratedExample:
    source_tokens = realizer.tokenize_words_with_offsets(source)
    labels = token_labels_all_keep(source_tokens)
    start = find_token_sequence(source_tokens, sequence)
    label_span(labels, start, start + 2, first_label)
    return _example(source, target, source_tokens, labels, rule_id, GenerationMode.POSITIVE, metadata)


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
