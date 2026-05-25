from __future__ import annotations

import re
from typing import Any, Mapping

from src.grammar_gen.semantics import SemanticFrameLexicon, VerbFrame


WORD_RE = re.compile(r"[А-Яа-яЁё-]+")

_ACTION_VERBS = {
    "купил": "купить",
    "купила": "купить",
    "купили": "купить",
    "купило": "купить",
    "сказал": "сказать",
    "сказала": "сказать",
    "сказали": "сказать",
    "сказало": "сказать",
    "объяснил": "объяснить",
    "объяснила": "объяснить",
    "объяснили": "объяснить",
    "объяснило": "объяснить",
    "прочитал": "прочитать",
    "прочитала": "прочитать",
    "прочитали": "прочитать",
    "прочитало": "прочитать",
    "решил": "решить",
    "решила": "решить",
    "решили": "решить",
    "решило": "решить",
    "съел": "съесть",
    "съела": "съесть",
    "съели": "съесть",
    "съело": "съесть",
}
_LOCATION_VERBS = {
    "стоит",
    "стоял",
    "стояла",
    "стояло",
    "стояли",
    "лежит",
    "лежал",
    "лежала",
    "лежало",
    "лежали",
    "находится",
    "находился",
    "находилась",
    "находилось",
    "находились",
}
_CONTENT_VERBS = {
    "содержит",
    "содержал",
    "содержала",
    "содержало",
    "содержали",
    "включает",
    "включал",
    "включала",
    "включало",
    "включали",
    "регулирует",
    "регулировал",
    "регулировала",
    "регулировало",
    "регулировали",
    "показывает",
    "показывал",
    "показывала",
    "показывало",
    "показывали",
    "описывает",
    "описывал",
    "описывала",
    "описывало",
    "описывали",
}
_ADVERB_SKIP = {
    "быстро",
    "медленно",
    "внимательно",
    "точно",
    "правильно",
    "ошибочно",
    "вчера",
    "сегодня",
    "завтра",
    "утром",
    "вечером",
    "часто",
    "редко",
    "иногда",
    "обычно",
    "сразу",
    "потом",
    "снова",
    "заранее",
    "тихо",
    "громко",
    "спокойно",
    "уверенно",
    "подробно",
    "кратко",
    "вместе",
    "отдельно",
    "рядом",
    "далеко",
    "близко",
    "хорошо",
    "плохо",
}
_SOFTWARE_ALLOWED_VERBS = {
    "проверил",
    "проверила",
    "проверили",
    "обработал",
    "обработала",
    "обработали",
    "отправил",
    "отправила",
    "отправили",
    "сохранил",
    "сохранила",
    "сохранили",
    "исправил",
    "исправила",
    "исправили",
}

_SUBJECT_CLASS_BY_FORM = {
    "дом": "building",
    "здание": "building",
    "офис": "building",
    "магазин": "place",
    "город": "place",
    "парк": "place",
    "стол": "object",
    "окно": "object",
    "документ": "document",
    "отчёт": "report",
    "отчет": "report",
    "текст": "text",
    "письмо": "message",
    "продукты": "food",
    "система": "software_agent",
    "программа": "software_agent",
    "сервис": "software_agent",
}


def validate_clause_semantics(clause: Mapping[str, Any] | Any) -> list[str]:
    frame = _get_value(clause, "frame")
    subject = _get_value(clause, "subject")
    obj = _get_value(clause, "object_np")
    if obj is None:
        obj = _get_value(clause, "object")

    reasons: list[str] = []
    if not isinstance(frame, VerbFrame):
        reasons.append("missing_frame")
    if subject is None:
        reasons.append("missing_subject")
    if reasons:
        return reasons
    return validate_frame_fillers(frame, subject, obj)


def validate_frame_fillers(frame: VerbFrame, subject: Any, object_np: Any | None) -> list[str]:
    reasons: list[str] = []
    single_frame_lexicon = SemanticFrameLexicon((frame,))

    if not single_frame_lexicon.validate_subject(frame, subject):
        reasons.append("invalid_subject_semantics")

    if frame.object_classes:
        if object_np is None:
            reasons.append("missing_object")
        elif not single_frame_lexicon.validate_object(frame, object_np):
            reasons.append("invalid_object_semantics")
    elif object_np is not None:
        reasons.append("unexpected_object")

    return reasons


def reject_semantic_nonsense(text: str) -> list[str]:
    words = [word.lower().replace("ё", "е") for word in WORD_RE.findall(text)]
    original_words = [word.lower() for word in WORD_RE.findall(text)]
    if len(words) < 2:
        return []

    subject = original_words[0]
    normalized_subject = words[0]
    verb = _first_predicate_word(words[1:])
    subject_class = _SUBJECT_CLASS_BY_FORM.get(subject) or _SUBJECT_CLASS_BY_FORM.get(normalized_subject)
    if subject_class is None:
        return []

    reasons: list[str] = []
    action_lemma = _ACTION_VERBS.get(verb)
    if action_lemma is not None:
        if subject_class in {"building", "place", "object", "document", "report", "text"}:
            reasons.append(f"{subject_class}_cannot_{action_lemma}")
        if subject_class == "food":
            reasons.append("food_cannot_be_action_subject")
        if subject_class == "software_agent" and action_lemma in {"купить", "съесть"}:
            reasons.append("software_agent_cannot_buy_or_eat")

    if subject_class in {"document", "report", "text"} and verb not in _CONTENT_VERBS and verb not in _LOCATION_VERBS:
        reasons.append("content_source_used_as_agent")

    if subject_class == "building" and verb not in _LOCATION_VERBS and verb not in _CONTENT_VERBS:
        reasons.append("building_requires_location_frame")

    if subject_class == "software_agent" and verb not in _SOFTWARE_ALLOWED_VERBS and action_lemma in {"купить", "съесть"}:
        reasons.append("software_agent_action_not_allowed")

    return _dedupe(reasons)


def _get_value(value: Mapping[str, Any] | Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _first_predicate_word(words: list[str]) -> str:
    for word in words:
        if word not in _ADVERB_SKIP:
            return word
    return words[0] if words else ""


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
