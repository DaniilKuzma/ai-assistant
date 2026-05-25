from __future__ import annotations


EXPLANATIONS: dict[str, str] = {
    "spacing_normalization": "Лишние пробелы рядом со знаками препинания убраны.",
    "ne_verb": "Частица «не» с глаголами обычно пишется раздельно.",
    "takzhe_tak_zhe": "Слитное или раздельное написание «также»/«так же» зависит от значения.",
    "tozhe_to_zhe": "Слитное или раздельное написание «тоже»/«то же» зависит от значения.",
    "zato_za_to": "Слитное или раздельное написание «зато»/«за то» зависит от значения.",
    "hyphen_particles": "Частицы «то», «либо», «нибудь» пишутся через дефис.",
    "hyphen_koe": "«Кое-» в неопределённых местоимениях пишется через дефис.",
    "hyphen_po_adverb": "Наречия на «по-...-ски» пишутся через дефис.",
    "tsya_ttsya": "В инфинитиве пишется «-ться», в форме 3-го лица — «-тся».",
    "comma_subordinate": "Запятая отделяет придаточную часть предложения.",
    "comma_introductory": "Вводные слова обычно выделяются запятыми.",
    "comma_homogeneous": "Однородные члены предложения разделяются запятыми.",
    "comma_adversative": "Перед противительным союзом обычно ставится запятая.",
    "dash_subject_predicate": "Между подлежащим и именным сказуемым может ставиться тире.",
    "final_punctuation": "В конце законченного предложения ставится знак препинания.",
}


def explanation_for(rule_id: str) -> str:
    return EXPLANATIONS.get(rule_id, "")


def attach_explanations(edits):
    for edit in edits:
        if not getattr(edit, "explanation", ""):
            edit.explanation = explanation_for(getattr(edit, "rule_id", ""))
    return edits


__all__ = ["EXPLANATIONS", "attach_explanations", "explanation_for"]
