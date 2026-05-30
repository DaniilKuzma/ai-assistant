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
    "проверочном листе",
    "служебной записке",
    "таблице замечаний",
    "карточке задания",
    "черновой версии",
    "письме редактору",
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
    "проверочного листа",
    "служебной записки",
    "таблицы замечаний",
    "карточки задания",
    "черновой версии",
    "письма редактору",
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
    "аналитик",
    "консультант",
    "секретарь",
    "документалист",
    "преподаватель",
    "администратор",
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
    "перенёс",
    "сохранил",
    "добавил в журнал",
    "пометил",
    "уточнил",
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
    "мама",
    "сестра",
    "коллега",
    "знакомый",
    "школьник",
    "учитель",
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
    "лист поручений",
    "заголовок",
    "короткую фразу",
    "черновую строку",
    "памятку",
    "список дел",
)
SCHOOL_ITEMS: tuple[str, ...] = (
    "тетради",
    "дневнике",
    "расписании",
    "задании",
    "конспекте",
    "плане кружка",
    "рабочей тетради",
    "карточке задания",
    "таблице ответов",
    "проверочном листе",
)
TECH_ITEMS: tuple[str, ...] = (
    "приложении",
    "настройках телефона",
    "чате поддержки",
    "уведомлении",
    "форме заказа",
    "журнале событий",
    "панели управления",
    "окне проверки",
    "настроечном файле",
)
BUSINESS_ITEMS: tuple[str, ...] = (
    "рабочем чате",
    "заявке",
    "письме клиенту",
    "таблице задач",
    "коротком отчёте",
    "протоколе встречи",
    "плане проекта",
    "служебной записке",
    "реестре поручений",
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
    ContextShell("В личной заметке оставили строку: {text}", STYLE_EVERYDAY),
    ContextShell("После звонка дома записали фразу: {text}", STYLE_EVERYDAY),
    ContextShell("В семейном списке дел появилась строка: {text}", STYLE_EVERYDAY),
    ContextShell("{person_cap} переписал на листок такую фразу: {text}", STYLE_EVERYDAY),
    ContextShell("В {school_item} ученик написал: {text}", STYLE_SCHOOL),
    ContextShell("На уроке в задании осталось: {text}", STYLE_SCHOOL),
    ContextShell("После кружка в дневнике записали: {text}", STYLE_SCHOOL),
    ContextShell("В школьном чате появилось задание: {text}", STYLE_SCHOOL),
    ContextShell("На полях конспекта осталось: {text}", STYLE_SCHOOL),
    ContextShell("В карточке упражнения указали: {text}", STYLE_SCHOOL),
    ContextShell("После диктанта учитель отметил строку: {text}", STYLE_SCHOOL),
    ContextShell("В таблице ответов сохранилась запись: {text}", STYLE_SCHOOL),
    ContextShell("На доске перед уроком написали: {text}", STYLE_SCHOOL),
    ContextShell("В {tech_item} видно сообщение: {text}", STYLE_TECH),
    ContextShell("На экране телефона появилось: {text}", STYLE_TECH),
    ContextShell("В приложении доставки сохранили строку: {text}", STYLE_TECH),
    ContextShell("В чате поддержки оставили сообщение: {text}", STYLE_TECH),
    ContextShell("В настройках профиля написали: {text}", STYLE_TECH),
    ContextShell("В журнале событий появилась строка: {text}", STYLE_TECH),
    ContextShell("На панели проверки показали фразу: {text}", STYLE_TECH),
    ContextShell("В форме обратной связи сохранили текст: {text}", STYLE_TECH),
    ContextShell("После обновления интерфейс вывел сообщение: {text}", STYLE_TECH),
    ContextShell("В {business_item} оставили строку: {text}", STYLE_BUSINESS),
    ContextShell("Перед созвоном в списке задач указали: {text}", STYLE_BUSINESS),
    ContextShell("В коротком письме клиенту написали: {text}", STYLE_BUSINESS),
    ContextShell("В рабочей заметке сохранили: {text}", STYLE_BUSINESS),
    ContextShell("В протоколе встречи зафиксировали: {text}", STYLE_BUSINESS),
    ContextShell("В реестре поручений появилась запись: {text}", STYLE_BUSINESS),
    ContextShell("Перед отправкой отчёта секретарь сверил строку: {text}", STYLE_BUSINESS),
    ContextShell("В плане проекта оставили формулировку: {text}", STYLE_BUSINESS),
    ContextShell("В {doc} {actor} оставил строку: {text}", STYLE_OFFICIAL),
    ContextShell("{actor_cap} сверил фразу в {doc}: {text}", STYLE_OFFICIAL),
    ContextShell("На полях {doc_gen} осталась запись: {text}", STYLE_OFFICIAL),
    ContextShell("После проверки {doc_gen} сохранилась строка: {text}", STYLE_OFFICIAL),
    ContextShell("В рабочей версии {doc_gen} указали фразу: {text}", STYLE_OFFICIAL),
    ContextShell("{actor_cap} {verb} такой фрагмент: {text}", STYLE_OFFICIAL),
    ContextShell("В приложении к {doc_gen} указали: {text}", STYLE_OFFICIAL),
    ContextShell("После сверки {doc_gen} {actor} записал: {text}", STYLE_OFFICIAL),
    ContextShell("В таблице замечаний {actor} оставил фразу: {text}", STYLE_OFFICIAL),
    ContextShell("{actor_cap} перенёс в {doc} такую строку: {text}", STYLE_OFFICIAL),
)
SHELLS: tuple[str, ...] = tuple(shell.template for shell in SHELL_SPECS)
SUFFIX_SPECS: tuple[ContextShell, ...] = (
    ContextShell("{text} Затем {person} перенёс запись в блокнот.", STYLE_EVERYDAY),
    ContextShell("{text} Позже {person} сверил запись ещё раз.", STYLE_EVERYDAY),
    ContextShell("{text} После этого дома сохранили такую же строку.", STYLE_EVERYDAY),
    ContextShell("{text} Вечером {person} отправил эту запись в чат.", STYLE_EVERYDAY),
    ContextShell("{text} Потом {person} переписал фразу на отдельный листок.", STYLE_EVERYDAY),
    ContextShell("{text} Утром эту запись сверили с заметкой.", STYLE_EVERYDAY),
    ContextShell("{text} В семейном чате строку повторили без пояснений.", STYLE_EVERYDAY),
    ContextShell("{text} После ужина {person} добавил рядом короткую пометку.", STYLE_EVERYDAY),
    ContextShell("{text} Учитель попросил переписать строку аккуратно.", STYLE_SCHOOL),
    ContextShell("{text} На следующем уроке ученик сверил эту запись.", STYLE_SCHOOL),
    ContextShell("{text} Методист добавил рядом короткое пояснение.", STYLE_SCHOOL),
    ContextShell("{text} После проверки тетради запись оставили без изменений.", STYLE_SCHOOL),
    ContextShell("{text} На уроке эту фразу разобрали отдельно.", STYLE_SCHOOL),
    ContextShell("{text} В проверочном листе ученик повторил строку ниже.", STYLE_SCHOOL),
    ContextShell("{text} После занятия преподаватель сверил эту запись.", STYLE_SCHOOL),
    ContextShell("{text} В конспекте рядом оставили грамматическую пометку.", STYLE_SCHOOL),
    ContextShell("{text} Система сохранила эту строку в журнале.", STYLE_TECH),
    ContextShell("{text} Затем интерфейс показал запись повторно.", STYLE_TECH),
    ContextShell("{text} После обновления форма сохранила тот же текст.", STYLE_TECH),
    ContextShell("{text} В чате поддержки эту строку процитировали отдельно.", STYLE_TECH),
    ContextShell("{text} Затем сервис добавил запись в историю проверки.", STYLE_TECH),
    ContextShell("{text} Интерфейс вывел эту фразу в отдельном уведомлении.", STYLE_TECH),
    ContextShell("{text} После синхронизации строка осталась в журнале событий.", STYLE_TECH),
    ContextShell("{text} В панели управления запись показали повторно.", STYLE_TECH),
    ContextShell("{text} После созвона секретарь перенёс строку в протокол.", STYLE_BUSINESS),
    ContextShell("{text} Затем менеджер добавил запись в таблицу задач.", STYLE_BUSINESS),
    ContextShell("{text} В письме клиенту эту фразу оставили без сокращений.", STYLE_BUSINESS),
    ContextShell("{text} После согласования куратор сохранил запись в плане.", STYLE_BUSINESS),
    ContextShell("{text} Затем руководитель внёс строку в список задач.", STYLE_BUSINESS),
    ContextShell("{text} В протоколе встречи фразу оставили отдельным пунктом.", STYLE_BUSINESS),
    ContextShell("{text} После обсуждения менеджер добавил пояснение в таблицу.", STYLE_BUSINESS),
    ContextShell("{text} В служебной записке эту формулировку повторили ниже.", STYLE_BUSINESS),
    ContextShell("{text} {actor_cap} сверил этот фрагмент с {doc}.", STYLE_OFFICIAL),
    ContextShell("{text} После проверки {doc_gen} строка осталась в архиве.", STYLE_OFFICIAL),
    ContextShell("{text} В рабочей версии {doc_gen} фрагмент повторили ниже.", STYLE_OFFICIAL),
    ContextShell("{text} Затем {actor} внёс пометку в журнал правок.", STYLE_OFFICIAL),
    ContextShell("{text} После сверки {doc_gen} {actor} оставил служебную отметку.", STYLE_OFFICIAL),
    ContextShell("{text} В приложении к {doc_gen} фрагмент процитировали полностью.", STYLE_OFFICIAL),
    ContextShell("{text} Затем {actor} перенёс формулировку в таблицу замечаний.", STYLE_OFFICIAL),
    ContextShell("{text} В карточке задания эту строку сохранили как образец.", STYLE_OFFICIAL),
)


def contextualize_case(case: LayerDirectCase, rng: RandomSource) -> LayerDirectCase:
    if not _should_contextualize(case):
        return case

    use_suffix = _can_use_suffix_for_case(case) and rng.chance(0.48)
    context = _sample_context(rng, suffix=use_suffix)
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
    if _truthy(example.metadata.get("disable_context_variation")) or _truthy(
        example.metadata.get("context_variation")
    ):
        return example

    for _ in range(8):
        candidate = _contextualize_example_once(example, rng, suffix=_can_use_suffix_for_example(example) and rng.chance(0.48))
        if candidate == example:
            return example
        if not _validation_reasons(candidate):
            return candidate
    return example


def _contextualize_example_once(example: GeneratedExample, rng: RandomSource, *, suffix: bool = False) -> GeneratedExample:
    if suffix:
        return _append_suffix(example, rng)
    return _prepend_prefix(example, rng)


def _prepend_prefix(example: GeneratedExample, rng: RandomSource) -> GeneratedExample:
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


def _append_suffix(example: GeneratedExample, rng: RandomSource) -> GeneratedExample:
    suffix_text, shell_id, style_bucket = _sample_suffix(rng)
    suffix_tokens = tokenize_runtime_words(suffix_text)
    if not suffix_tokens:
        return example

    shifted_suffix_tokens = [
        WordToken(
            text=token.text,
            start=token.start + len(example.source_text),
            end=token.end + len(example.source_text),
            lemma=token.lemma,
            pos=token.pos,
            feats=dict(token.feats),
        )
        for token in suffix_tokens
    ]
    metadata = dict(example.metadata)
    metadata["context_variation"] = True
    metadata["context_shell"] = shell_id
    metadata["context_style_bucket"] = style_bucket
    return GeneratedExample(
        source_text=f"{example.source_text}{suffix_text}",
        target_text=f"{example.target_text}{suffix_text}",
        source_tokens=[*example.source_tokens, *shifted_suffix_tokens],
        token_edit_labels=list(example.token_edit_labels) + ["KEEP"] * len(suffix_tokens),
        gap_labels=list(example.gap_labels) + gap_labels_from_text(suffix_text, suffix_tokens),
        boundary_before_labels=list(example.boundary_before_labels) + ["NONE"] * len(suffix_tokens),
        boundary_after_labels=list(example.boundary_after_labels) + ["NONE"] * len(suffix_tokens),
        rule_ids=list(example.rule_ids) + ["none"] * len(suffix_tokens),
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


def _sample_suffix(rng: RandomSource) -> tuple[str, str, str]:
    context = _sample_context(rng, suffix=True)
    suffix = _render_shell(context, text="")
    return suffix, context.shell.template.replace("{text}", "").strip(), context.shell.style_bucket


def _sample_context(rng: RandomSource, *, suffix: bool = False) -> RenderContext:
    doc_index = rng.randint(0, len(DOCUMENTS) - 1)
    actor = rng.choice(ACTORS)
    person = rng.choice(PEOPLE)
    shell = rng.choice(SUFFIX_SPECS if suffix else SHELL_SPECS)
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


def _can_use_suffix_for_case(case: LayerDirectCase) -> bool:
    if case.metadata.get("allowed_source_surface_failures"):
        return False
    return _surface_clean(case.source_text) and _surface_clean(case.target_text)


def _can_use_suffix_for_example(example: GeneratedExample) -> bool:
    from src.grammar_gen.safety import allowed_source_surface_failures

    if allowed_source_surface_failures(example):
        return False
    return _surface_clean(example.source_text) and _surface_clean(example.target_text)


def _surface_clean(text: str) -> bool:
    from src.grammar_gen.safety import validate_surface

    return not validate_surface(text)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


__all__ = ["contextualize_case", "contextualize_example"]
