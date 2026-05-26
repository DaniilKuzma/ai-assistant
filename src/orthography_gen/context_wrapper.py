from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

from src.runtime.tokenization import tokenize_runtime_words
from src.schema import WordToken


@dataclass(frozen=True)
class WrappedContext:
    source_text: str
    target_text: str
    source_tokens: list[WordToken]
    edited_token_index: int
    context: dict[str, Any]
    construction_id: str


class ContextWrapper:
    def wrap(
        self,
        *,
        source_word: str,
        target_word: str,
        pos: str,
        context: Mapping[str, Any] | None = None,
    ) -> WrappedContext:
        payload = dict(context or {})
        template = str(payload.get("template") or "").strip()
        construction_id = str(payload.get("construction_id") or "").strip()
        if not template:
            template, construction_id = self._default_template(pos, payload)
        if payload.get("vary_shell", True) is not False and payload.get("variant"):
            template, construction_id = _varied_template(pos, payload, template, construction_id)
        if not construction_id:
            construction_id = _construction_for_template(template, pos)

        format_args = {"word": source_word, "variant": payload.get("variant", "")}
        source_text = template.format(**format_args)
        format_args["word"] = target_word
        target_text = template.format(**format_args)
        tokens = tokenize_runtime_words(source_text)
        edited_index = _find_token(tokens, source_word)
        return WrappedContext(
            source_text=source_text,
            target_text=target_text,
            source_tokens=tokens,
            edited_token_index=edited_index,
            context=payload,
            construction_id=construction_id,
        )

    def _default_template(self, pos: str, context: Mapping[str, Any]) -> tuple[str, str]:
        if bool(context.get("quote", False)):
            return "В словаре указано слово «{word}».", "orthography_word_quote"
        if pos == "NOUN":
            return "Редактор проверил слово «{word}».", "orthography_word_quote"
        if pos in {"ADJ", "ADJF", "PRTF"}:
            noun = str(context.get("noun") or "пример")
            subject = str(context.get("subject") or "Редактор")
            return f"{subject} заметил {{word}} {noun}.", "orthography_adjective_safe_context"
        return "В словаре указано слово «{word}».", "orthography_word_quote"


QUOTE_DOCUMENTS: tuple[tuple[str, str], ...] = (
    ("черновике", "черновика"),
    ("диктанте", "диктанта"),
    ("учебном задании", "учебного задания"),
    ("редакторской правке", "редакторской правки"),
    ("словарной карточке", "словарной карточки"),
    ("рукописи", "рукописи"),
    ("корректорском отчёте", "корректорского отчёта"),
    ("листке самопроверки", "листка самопроверки"),
    ("письменной работе", "письменной работы"),
    ("текстовой заметке", "текстовой заметки"),
    ("таблице вариантов", "таблицы вариантов"),
    ("журнале правок", "журнала правок"),
)
QUOTE_ACTORS: tuple[str, ...] = (
    "корректор",
    "редактор",
    "преподаватель",
    "лингвист",
    "эксперт",
    "автор",
    "проверяющий",
    "методист",
)
QUOTE_VERBS: tuple[str, ...] = (
    "отметил",
    "заметил",
    "подчеркнул",
    "сверил",
    "нашёл",
    "исправил",
    "обсудил",
    "выделил",
)
NOUN_QUOTE_SHAPES: tuple[str, ...] = (
    "В {doc_loc} {actor} {verb} форму «{{word}}».",
    "{actor_cap} {verb} в {doc_loc} вариант «{{word}}».",
    "При проверке {doc_gen} {actor} {verb} написание «{{word}}».",
    "На полях {doc_gen} {actor} оставил пометку: «{{word}}».",
    "В карточке задания {actor} подчеркнул «{{word}}».",
    "После сверки {doc_gen} {actor} вынес в список форму «{{word}}».",
)
ADJECTIVE_QUOTE_SHAPES: tuple[str, ...] = (
    "В {doc_loc} {actor} {verb} сочетание «{{word}} {noun}».",
    "{actor_cap} {verb} в {doc_loc} фразу «{{word}} {noun}».",
    "При проверке {doc_gen} {actor} подчеркнул сочетание «{{word}} {noun}».",
    "На полях {doc_gen} {actor} оставил пометку: «{{word}} {noun}».",
)
DERIVATIONAL_NOUNS: dict[str, tuple[str, ...]] = {
    "буква": ("код", "индекс", "ряд", "разбор"),
    "листва": ("лес", "покров", "узор", "слой"),
    "лекарство": ("препарат", "состав", "сбор", "раствор"),
    "мысль": ("образ", "план", "вывод", "разбор"),
}
MATERIAL_NOUNS: dict[str, tuple[str, ...]] = {
    "кожа": ("ремень", "чехол", "портфель", "футляр"),
    "глина": ("кувшин", "горшок", "сосуд", "черепок"),
    "серебро": ("браслет", "поднос", "кубок", "значок"),
}
CONTEXT_NOUNS: dict[str, tuple[str, ...]] = {
    "oil": ("раствор", "состав", "след", "слой"),
    "food": ("блин", "пирог", "ломтик", "кусок"),
    "smeared": ("нож", "лист", "поднос", "край"),
    "participle": ("процесс", "порядок", "план", "раздел", "проект", "этап", "маршрут", "протокол"),
    "wind_power": ("двигатель", "механизм", "насос", "агрегат"),
}
MASC_PLACES: tuple[str, ...] = (
    "у ворот",
    "во дворе",
    "за мастерской",
    "у склада",
    "возле дома",
    "у входа",
)
FEMN_PLACES: tuple[str, ...] = (
    "на столе",
    "на тарелке",
    "на сковороде",
    "в кухне",
    "на подносе",
    "у плиты",
)


def _varied_template(
    pos: str,
    context: Mapping[str, Any],
    fallback_template: str,
    fallback_construction_id: str,
) -> tuple[str, str]:
    variant = context.get("variant")
    context_class = str(context.get("context_class") or "")
    generic = _generic_template(context_class, variant)
    if generic is not None:
        return generic
    if pos == "NOUN":
        return _noun_quote_template(variant), "orthography_word_quote"
    if pos in {"ADJ", "ADJF", "PRTF"}:
        return _adjective_template(context, variant, fallback_template, fallback_construction_id)
    return _noun_quote_template(variant), "orthography_word_quote"


def _noun_quote_template(variant: Any) -> str:
    doc_loc, doc_gen = _slot(QUOTE_DOCUMENTS, variant, "doc")
    actor = _slot(QUOTE_ACTORS, variant, "actor")
    verb = _slot(QUOTE_VERBS, variant, "verb")
    shape = _slot(NOUN_QUOTE_SHAPES, variant, "shape")
    return shape.format(
        doc_loc=doc_loc,
        doc_gen=doc_gen,
        actor=actor,
        actor_cap=actor.capitalize(),
        verb=verb,
    )


def _adjective_template(
    context: Mapping[str, Any],
    variant: Any,
    fallback_template: str,
    fallback_construction_id: str,
) -> tuple[str, str]:
    context_class = str(context.get("context_class") or "")
    gender = str(context.get("gender") or "")

    if context_class in {"dependent_word", "no_dependent_word"}:
        return _dependent_word_template(context_class, gender, variant), "orthography_adjective_safe_context"
    if context_class == "participle":
        return _simple_masc_template(variant, CONTEXT_NOUNS[context_class]), "orthography_adjective_safe_context"
    if context_class in {"exception", "weather_exception"}:
        return _weather_template(variant), "orthography_adjective_safe_context"
    if context_class == "wind_power":
        return _engine_template(variant), "orthography_adjective_safe_context"
    if context_class in {"oil", "food", "smeared"}:
        return _contextual_masc_template(context_class, variant), "orthography_adjective_safe_context"

    noun_options = _noun_options_for_card(context)
    if noun_options:
        return _quoted_adjective_template(variant, noun_options), "orthography_adjective_safe_context"
    return fallback_template, fallback_construction_id


def _noun_options_for_card(context: Mapping[str, Any]) -> tuple[str, ...]:
    base = str(context.get("derivational_base") or "").lower()
    context_class = str(context.get("context_class") or "")
    if context_class == "material" and base in MATERIAL_NOUNS:
        return MATERIAL_NOUNS[base]
    if context_class == "derivational" and base in DERIVATIONAL_NOUNS:
        return DERIVATIONAL_NOUNS[base]
    return ()


def _quoted_adjective_template(variant: Any, noun_options: tuple[str, ...]) -> str:
    doc_loc, doc_gen = _slot(QUOTE_DOCUMENTS, variant, "doc")
    actor = _slot(QUOTE_ACTORS, variant, "actor")
    verb = _slot(QUOTE_VERBS, variant, "verb")
    noun = _slot(noun_options, variant, "noun")
    shape = _slot(ADJECTIVE_QUOTE_SHAPES, variant, "shape")
    return shape.format(
        doc_loc=doc_loc,
        doc_gen=doc_gen,
        actor=actor,
        actor_cap=actor.capitalize(),
        verb=verb,
        noun=noun,
    )


def _contextual_masc_template(context_class: str, variant: Any) -> str:
    noun = _slot(CONTEXT_NOUNS[context_class], variant, "noun")
    if context_class == "food":
        shape = _slot(
            (
                "На тарелке лежал {{word}} {noun}.",
                "Повар подал {{word}} {noun}.",
                "В меню попал {{word}} {noun}.",
                "К завтраку приготовили {{word}} {noun}.",
            ),
            variant,
            "shape",
        )
        return shape.format(noun=noun)
    if context_class == "smeared":
        shape = _slot(
            (
                "На столе лежал {{word}} {noun}.",
                "Повар отложил {{word}} {noun}.",
                "У плиты остался {{word}} {noun}.",
                "В мойке лежал {{word}} {noun}.",
            ),
            variant,
            "shape",
        )
        return shape.format(noun=noun)

    shape = _slot(
        (
            "Лаборант осмотрел {{word}} {noun}.",
            "Технолог отметил {{word}} {noun} в журнале.",
            "На столе стоял {{word}} {noun}.",
            "В описании указали {{word}} {noun}.",
        ),
        variant,
        "shape",
    )
    return shape.format(noun=noun)


def _simple_masc_template(variant: Any, noun_options: tuple[str, ...]) -> str:
    noun = _slot(noun_options, variant, "noun")
    subject, verb = _slot(
        (
            ("комиссия", "утвердила"),
            ("комиссия", "обсудила"),
            ("совет", "утвердил"),
            ("отдел", "проверил"),
            ("редактор", "описал"),
            ("эксперт", "принял"),
        ),
        variant,
        "subject_verb",
    )
    phrase = f"{{word}} {noun}"
    doc_loc, doc_gen = _slot(QUOTE_DOCUMENTS, variant, "doc")
    actor = _slot(QUOTE_ACTORS, variant, "actor")
    quote_verb = _slot(QUOTE_VERBS, variant, "quote_verb")
    shape = _slot(
        (
            f"{subject.capitalize()} {verb} {{phrase}}.",
            "В {doc_loc} {actor} {quote_verb} сочетание «{phrase}».",
            "{actor_cap} подчеркнул в {doc_loc} фразу «{phrase}».",
            "При проверке {doc_gen} {actor} заметил сочетание «{phrase}».",
            "На полях {doc_gen} появилась пометка: «{phrase}».",
        ),
        variant,
        "shape",
    )
    return shape.format(
        doc_loc=doc_loc,
        doc_gen=doc_gen,
        actor=actor,
        actor_cap=actor.capitalize(),
        quote_verb=quote_verb,
        phrase=phrase,
    )


def _dependent_word_template(context_class: str, gender: str, variant: Any) -> str:
    has_dependent = context_class == "dependent_word"
    doc_loc, doc_gen = _slot(QUOTE_DOCUMENTS, variant, "doc")
    actor = _slot(QUOTE_ACTORS, variant, "actor")
    quote_verb = _slot(QUOTE_VERBS, variant, "quote_verb")
    if gender == "femn":
        place = _slot(FEMN_PLACES, variant, "place")
        noun = _slot(("картошка", "запеканка", "лепёшка", "рыба", "котлета", "курица"), variant, "noun")
        dependent = " на масле" if has_dependent else ""
        verb = _slot(("лежала", "была", "осталась"), variant, "verb")
        phrase = f"{{word}}{dependent} {noun}"
        shape = _slot(
            (
                f"{place.capitalize()} {verb} {{phrase}}.",
                "В {doc_loc} {actor} {quote_verb} сочетание «{phrase}».",
                "{actor_cap} подчеркнул в {doc_loc} фразу «{phrase}».",
                "При проверке {doc_gen} {actor} заметил сочетание «{phrase}».",
                "На полях {doc_gen} появилась пометка: «{phrase}».",
            ),
            variant,
            "shape",
        )
        return shape.format(
            doc_loc=doc_loc,
            doc_gen=doc_gen,
            actor=actor,
            actor_cap=actor.capitalize(),
            quote_verb=quote_verb,
            phrase=phrase,
        )

    place = _slot(MASC_PLACES, variant, "place")
    noun = _slot(
        ("забор", "щит", "столб", "шкаф", "фасад", "сарай", "мост", "стенд", "ящик", "подоконник"),
        variant,
        "noun",
    )
    dependent = " вчера" if has_dependent else ""
    verb = _slot(("стоял", "остался"), variant, "verb")
    phrase = f"{{word}}{dependent} {noun}"
    shape = _slot(
        (
            f"{place.capitalize()} {verb} {{phrase}}.",
            "В {doc_loc} {actor} {quote_verb} сочетание «{phrase}».",
            "{actor_cap} подчеркнул в {doc_loc} фразу «{phrase}».",
            "При проверке {doc_gen} {actor} заметил сочетание «{phrase}».",
            "На полях {doc_gen} появилась пометка: «{phrase}».",
        ),
        variant,
        "shape",
    )
    return shape.format(
        doc_loc=doc_loc,
        doc_gen=doc_gen,
        actor=actor,
        actor_cap=actor.capitalize(),
        quote_verb=quote_verb,
        phrase=phrase,
    )


def _weather_template(variant: Any) -> str:
    shape = _slot(
        (
            "Сегодня был {word} день.",
            "Утром выдался {word} день.",
            "К вечеру начался {word} день.",
            "После обеда стоял {word} день.",
        ),
        variant,
        "shape",
    )
    return shape


def _engine_template(variant: Any) -> str:
    noun = _slot(CONTEXT_NOUNS["wind_power"], variant, "noun")
    actor = _slot(("инженер", "техник", "мастер", "механик"), variant, "actor")
    verb = _slot(("проверил", "осмотрел", "запустил", "настроил"), variant, "verb")
    return f"{actor.capitalize()} {verb} {{word}} {noun}."


def _generic_template(context_class: str, variant: Any) -> tuple[str, str] | None:
    groups: dict[str, tuple[str, tuple[str, ...]]] = {
        "business": (
            "orthography_business_quote",
            (
            "В отчете отдела указали форму «{word}».",
            "Секретарь внес в протокол слово «{word}».",
            "В деловом письме оставили вариант «{word}».",
            "Куратор сверил в регламенте написание «{word}».",
            ),
        ),
        "educational": (
            "orthography_education_quote",
            (
            "В учебном задании встретилось слово «{word}».",
            "Учитель подчеркнул в диктанте форму «{word}».",
            "На уроке разобрали написание «{word}».",
            "Методист добавил в конспект вариант «{word}».",
            ),
        ),
        "everyday": (
            "orthography_everyday_quote",
            (
            "Дома на записке было написано «{word}».",
            "В разговорной заметке встретилась форма «{word}».",
            "В сообщении соседу осталось слово «{word}».",
            ),
        ),
        "technical": (
            "orthography_technical_quote",
            (
            "В техническом документе указали параметр «{word}».",
            "Инженер сверил в инструкции форму «{word}».",
            "В журнале настройки записали слово «{word}».",
            "Тестировщик внёс в лог вариант «{word}».",
            ),
        ),
        "action": (
            "orthography_action_quote",
            (
            "Редактор быстро исправил слово «{word}».",
            "Проверяющий утром отметил форму «{word}».",
            "Автор после сверки оставил вариант «{word}».",
            "Аналитик повторно проверил написание «{word}».",
            ),
        ),
        "introductory": (
            "orthography_introductory_quote",
            (
            "Кстати, в тексте встретилось слово «{word}».",
            "Разумеется, редактор заметил форму «{word}».",
            "Во-первых, проверили написание «{word}».",
            ),
        ),
        "time_place": (
            "orthography_time_place_quote",
            (
            "Утром в кабинете сверили слово «{word}».",
            "Вчера на совещании обсудили форму «{word}».",
            "На стенде у входа заметили вариант «{word}».",
            "В библиотеке после лекции записали форму «{word}».",
            ),
        ),
    }
    group = groups.get(context_class)
    if not group:
        return None
    construction_id, templates = group
    return _slot(templates, variant, "generic_shape"), construction_id


def _find_token(tokens: list[WordToken], source_word: str) -> int:
    lowered = source_word.lower()
    for index, token in enumerate(tokens):
        if token.text.lower() == lowered:
            return index
    raise ValueError(f"Wrapped source word {source_word!r} was not found as a token.")


def _construction_for_template(template: str, pos: str) -> str:
    if "«{word}»" in template:
        return "orthography_word_quote"
    if pos == "NOUN":
        return "orthography_noun_safe_context"
    return "orthography_adjective_safe_context"


def _slot(values: tuple[Any, ...], variant: Any, salt: str) -> Any:
    if not values:
        raise ValueError("Cannot choose from an empty context slot.")
    raw = f"{variant}:{salt}".encode("utf-8")
    index = int(hashlib.sha1(raw).hexdigest()[:8], 16) % len(values)
    return values[index]


__all__ = ["ContextWrapper", "WrappedContext"]
