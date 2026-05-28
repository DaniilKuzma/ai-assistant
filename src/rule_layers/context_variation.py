from __future__ import annotations

from dataclasses import dataclass

from src.grammar_gen.randomness import RandomSource
from src.grammar_gen.rules.common import gap_labels_from_text
from src.runtime.tokenization import tokenize_runtime_words
from src.rule_layers.base import LayerDirectCase
from src.schema import GeneratedExample, WordToken


VARIED_FAMILIES = frozenset(
    {"compound_spelling", "dictionary_typo", "syntax_punctuation", "quotation_dialogue", "casing", "semantic"}
)

STYLE_EVERYDAY = "everyday"
STYLE_SCHOOL = "school"
STYLE_TECH = "tech"
STYLE_BUSINESS = "business"
STYLE_OFFICIAL = "editorial_official"


@dataclass(frozen=True)
class ContextShell:
    template: str
    style_bucket: str


@dataclass(frozen=True)
class RenderContext:
    shell: ContextShell
    values: dict[str, str]


DOCUMENTS: tuple[str, ...] = (
    "черновике",
    "отчёте",
    "протоколе",
    "инструкции",
    "заметке",
    "регламенте",
    "журнале правок",
    "учебном задании",
    "деловом письме",
    "текстовой сводке",
    "редакторской карточке",
    "рабочей памятке",
)
DOCUMENTS_GEN: tuple[str, ...] = (
    "черновика",
    "отчёта",
    "протокола",
    "инструкции",
    "заметки",
    "регламента",
    "журнала правок",
    "учебного задания",
    "делового письма",
    "текстовой сводки",
    "редакторской карточки",
    "рабочей памятки",
)
ACTORS: tuple[str, ...] = (
    "корректор",
    "редактор",
    "методист",
    "лингвист",
    "эксперт",
    "автор",
    "проверяющий",
    "куратор",
)
VERBS: tuple[str, ...] = (
    "оставил",
    "сверил",
    "проверил",
    "выделил",
    "подчеркнул",
    "обсудил",
    "заметил",
    "вынес в список",
)
PEOPLE: tuple[str, ...] = (
    "папа",
    "дедушка",
    "брат",
    "сосед",
    "друг",
    "сын",
    "отец",
    "дядя",
)
ITEMS: tuple[str, ...] = (
    "список покупок",
    "напоминание",
    "заметку",
    "сообщение",
    "записку",
    "план на вечер",
    "адрес",
    "расписание",
)
SCHOOL_ITEMS: tuple[str, ...] = (
    "тетради",
    "дневнике",
    "расписании",
    "задании",
    "конспекте",
    "плане кружка",
)
TECH_ITEMS: tuple[str, ...] = (
    "приложении",
    "настройках телефона",
    "чате поддержки",
    "уведомлении",
    "форме заказа",
)
BUSINESS_ITEMS: tuple[str, ...] = (
    "рабочем чате",
    "заявке",
    "письме клиенту",
    "таблице задач",
    "коротком отчёте",
)
SHELL_SPECS: tuple[ContextShell, ...] = (
    ContextShell("{person_cap} оставил на холодильнике {item}: {text}", STYLE_EVERYDAY),
    ContextShell("В телефоне сохранилось напоминание: {text}", STYLE_EVERYDAY),
    ContextShell("Дома в блокноте записали: {text}", STYLE_EVERYDAY),
    ContextShell("В семейном чате появилось сообщение: {text}", STYLE_EVERYDAY),
    ContextShell("{person_cap} перед выходом написал: {text}", STYLE_EVERYDAY),
    ContextShell("После прогулки {person} добавил в заметку: {text}", STYLE_EVERYDAY),
    ContextShell("У подъезда на листке осталось: {text}", STYLE_EVERYDAY),
    ContextShell("В списке покупок рядом с хлебом написали: {text}", STYLE_EVERYDAY),
    ContextShell("Перед поездкой в телефоне сохранили: {text}", STYLE_EVERYDAY),
    ContextShell("На кухне утром записали: {text}", STYLE_EVERYDAY),
    ContextShell("В {school_item} ученик написал: {text}", STYLE_SCHOOL),
    ContextShell("На уроке в задании осталось: {text}", STYLE_SCHOOL),
    ContextShell("После кружка в дневнике записали: {text}", STYLE_SCHOOL),
    ContextShell("В школьном чате появилось задание: {text}", STYLE_SCHOOL),
    ContextShell("На полях конспекта осталось: {text}", STYLE_SCHOOL),
    ContextShell("В {tech_item} видно сообщение: {text}", STYLE_TECH),
    ContextShell("На экране телефона появилось: {text}", STYLE_TECH),
    ContextShell("В приложении доставки сохранили строку: {text}", STYLE_TECH),
    ContextShell("В чате поддержки оставили сообщение: {text}", STYLE_TECH),
    ContextShell("В настройках профиля написали: {text}", STYLE_TECH),
    ContextShell("В {business_item} оставили строку: {text}", STYLE_BUSINESS),
    ContextShell("Перед созвоном в списке задач указали: {text}", STYLE_BUSINESS),
    ContextShell("В коротком письме клиенту написали: {text}", STYLE_BUSINESS),
    ContextShell("В рабочей заметке сохранили: {text}", STYLE_BUSINESS),
    ContextShell("В {doc} {actor} оставил строку: {text}", STYLE_OFFICIAL),
    ContextShell("{actor_cap} сверил фразу в {doc}: {text}", STYLE_OFFICIAL),
    ContextShell("На полях {doc_gen} осталась запись: {text}", STYLE_OFFICIAL),
    ContextShell("После проверки {doc_gen} сохранилась строка: {text}", STYLE_OFFICIAL),
    ContextShell("В рабочей версии {doc_gen} указали фразу: {text}", STYLE_OFFICIAL),
    ContextShell("{actor_cap} {verb} такой фрагмент: {text}", STYLE_OFFICIAL),
)
SHELLS: tuple[str, ...] = tuple(shell.template for shell in SHELL_SPECS)


def contextualize_case(case: LayerDirectCase, rng: RandomSource) -> LayerDirectCase:
    if not _should_contextualize(case):
        return case

    context = _sample_context(rng)
    source_text = _render_shell(context, text=case.source_text)
    target_text = _render_shell(context, text=case.target_text)
    metadata = dict(case.metadata)
    metadata["context_variation"] = True
    metadata["context_shell"] = context.shell.template.replace("{text}", "").strip()
    metadata["context_style_bucket"] = context.shell.style_bucket
    return LayerDirectCase(
        rule_id=case.rule_id,
        family=case.family,
        sub_rule_id=case.sub_rule_id,
        mode=case.mode,
        source_text=source_text,
        target_text=target_text,
        token_operations=case.token_operations,
        gap_operations=case.gap_operations,
        boundary_operations=case.boundary_operations,
        expected_token_edit_count=case.expected_token_edit_count,
        expected_gap_edit_count=case.expected_gap_edit_count,
        metadata=metadata,
        weight=case.weight,
        direct_token_labels=case.direct_token_labels,
        direct_gap_labels=case.direct_gap_labels,
        direct_boundary_before_labels=case.direct_boundary_before_labels,
        direct_boundary_after_labels=case.direct_boundary_after_labels,
        expected_boundary_edit_count=case.expected_boundary_edit_count,
    )


def contextualize_example(example: GeneratedExample, rng: RandomSource) -> GeneratedExample:
    if _truthy(example.metadata.get("disable_context_variation")):
        return example

    for _ in range(8):
        candidate = _contextualize_example_once(example, rng)
        if candidate == example:
            return example
        if not _validation_reasons(candidate):
            return candidate
    return example


def _contextualize_example_once(example: GeneratedExample, rng: RandomSource) -> GeneratedExample:
    prefix, shell_id, style_bucket = _sample_prefix(rng)
    prefix_tokens = tokenize_runtime_words(prefix)
    if not prefix_tokens:
        return example

    shifted_tokens = [
        WordToken(
            text=token.text,
            start=token.start + len(prefix),
            end=token.end + len(prefix),
            lemma=token.lemma,
            pos=token.pos,
            feats=dict(token.feats),
        )
        for token in example.source_tokens
    ]
    metadata = dict(example.metadata)
    metadata["context_variation"] = True
    metadata["context_shell"] = shell_id
    metadata["context_style_bucket"] = style_bucket
    return GeneratedExample(
        source_text=f"{prefix}{example.source_text}",
        target_text=f"{prefix}{example.target_text}",
        source_tokens=[*prefix_tokens, *shifted_tokens],
        token_edit_labels=["KEEP"] * len(prefix_tokens) + list(example.token_edit_labels),
        gap_labels=gap_labels_from_text(prefix, prefix_tokens) + list(example.gap_labels),
        boundary_before_labels=["NONE"] * len(prefix_tokens) + list(example.boundary_before_labels),
        boundary_after_labels=["NONE"] * len(prefix_tokens) + list(example.boundary_after_labels),
        rule_ids=["none"] * len(prefix_tokens) + list(example.rule_ids),
        primary_rule_id=example.primary_rule_id,
        mode=example.mode,
        explanation_ids=list(example.explanation_ids),
        metadata=metadata,
    )


def _validation_reasons(example: GeneratedExample) -> list[str]:
    from src.grammar_gen.safety import validate_generated_pair

    return validate_generated_pair(example)


def _should_contextualize(case: LayerDirectCase) -> bool:
    if case.family not in VARIED_FAMILIES:
        return False
    if _truthy(case.metadata.get("disable_context_variation")):
        return False
    return (
        not case.direct_token_labels
        and not case.direct_gap_labels
        and not case.direct_boundary_before_labels
        and not case.direct_boundary_after_labels
    )


def _render_shell(context: RenderContext, *, text: str) -> str:
    return context.shell.template.format(text=text, **context.values)


def _sample_prefix(rng: RandomSource) -> tuple[str, str, str]:
    context = _sample_context(rng)
    prefix = _render_shell(context, text="")
    return prefix, context.shell.template.replace("{text}", "").strip(), context.shell.style_bucket


def _sample_context(rng: RandomSource) -> RenderContext:
    doc_index = rng.randint(0, len(DOCUMENTS) - 1)
    actor = rng.choice(ACTORS)
    person = rng.choice(PEOPLE)
    shell = rng.choice(SHELL_SPECS)
    verb = rng.choice(VERBS)
    values = {
        "doc": DOCUMENTS[doc_index],
        "doc_gen": DOCUMENTS_GEN[doc_index],
        "actor": actor,
        "actor_cap": actor.capitalize(),
        "verb": verb,
        "person": person,
        "person_cap": person.capitalize(),
        "item": rng.choice(ITEMS),
        "school_item": rng.choice(SCHOOL_ITEMS),
        "tech_item": rng.choice(TECH_ITEMS),
        "business_item": rng.choice(BUSINESS_ITEMS),
    }
    return RenderContext(shell=shell, values=values)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


__all__ = ["contextualize_case", "contextualize_example"]
