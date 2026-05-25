from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.grammar_gen.lexicon import NounEntry


LATIN_RE = re.compile(r"[A-Za-z]")
VO_RE = re.compile(r"(^|\s)во\s+", re.IGNORECASE)
BAD_K_OGOROD_RE = re.compile(r"(^|\s)к\s+огород", re.IGNORECASE)
RUSSIAN_WORD_RE = re.compile(r"^[А-Яа-яЁё-]+$")
VO_WHITELIST = ("во дворе",)


@dataclass(frozen=True)
class TokenAnalysis:
    token: str
    lemma: str
    pos: str | None
    tag: str | None
    is_known: bool


class MorphologyEngine:
    def __init__(self, use_pymorphy: bool = True) -> None:
        self._morph = None
        if use_pymorphy:
            try:
                from pymorphy3 import MorphAnalyzer
                self._morph = MorphAnalyzer()
            except Exception:
                self._morph = None

    def inflect_noun(self, lemma: str, case: str, number: str = "sing") -> str:
        fallback = _fallback_noun(lemma, case, number)
        result = self._inflect_with_pymorphy(lemma, "NOUN", {_case_tag(case), _number_tag(number)})
        return result or fallback or lemma

    def inflect_adjective(self, lemma: str, gender: str, case: str, number: str = "sing") -> str:
        tags = {_case_tag(case), _number_tag(number)}
        if number != "plur":
            tags.add(_gender_tag(gender))
        result = self._inflect_with_pymorphy(lemma, "ADJF", tags)
        return result or _fallback_adjective(lemma, gender, case, number) or lemma

    def inflect_verb_past(self, lemma: str, gender: str, number: str = "sing") -> str:
        tags = {"past", _number_tag(number)}
        if number != "plur":
            tags.add(_gender_tag(gender))
        result = self._inflect_with_pymorphy(lemma, "INFN", tags)
        return result or _fallback_verb_past(lemma, gender, number) or lemma

    def inflect_verb_present_3sg(self, lemma: str) -> str:
        result = self._inflect_with_pymorphy(lemma, "INFN", {"pres", "3per", "sing"})
        if result:
            return result
        result = self._inflect_with_pymorphy(lemma, "INFN", {"futr", "3per", "sing"})
        return result or _FALLBACK_PRESENT_3SG.get(lemma) or lemma

    def infinitive(self, lemma: str) -> str:
        if self._morph is not None and RUSSIAN_WORD_RE.fullmatch(lemma):
            parse = self._best_parse(lemma, {"INFN", "VERB"})
            normal = str(getattr(parse, "normal_form", "")) if parse is not None else ""
            if normal:
                return normal
        if lemma.endswith("тся"):
            return f"{lemma[:-3]}ться"
        return lemma

    def normalize_yo(self, text: str) -> str:
        return text.replace("ё", "е").replace("Ё", "Е")

    def analyze_token(self, token: str) -> TokenAnalysis:
        if self._morph is None or not RUSSIAN_WORD_RE.fullmatch(token):
            return TokenAnalysis(token=token, lemma=token.lower(), pos=None, tag=None, is_known=False)

        parse = self._best_parse(token, None)
        if parse is None:
            return TokenAnalysis(token=token, lemma=token.lower(), pos=None, tag=None, is_known=False)
        return TokenAnalysis(
            token=token,
            lemma=str(getattr(parse, "normal_form", token.lower())),
            pos=str(getattr(parse.tag, "POS", "")) or None,
            tag=str(parse.tag),
            is_known=bool(getattr(parse, "is_known", False)),
        )

    def _inflect_with_pymorphy(self, lemma: str, preferred_pos: str, tags: set[str]) -> str | None:
        if self._morph is None or not RUSSIAN_WORD_RE.fullmatch(lemma):
            return None
        parse = self._best_parse(lemma, {preferred_pos})
        if parse is None:
            return None
        inflected = parse.inflect(tags)
        word = str(getattr(inflected, "word", "")) if inflected is not None else ""
        return word or None

    def _best_parse(self, token: str, preferred_poses: set[str] | None) -> Any | None:
        parses = self._morph.parse(token) if self._morph is not None else ()
        if not parses:
            return None
        if preferred_poses:
            for parse in parses:
                if getattr(parse.tag, "POS", None) in preferred_poses:
                    return parse
        for parse in parses:
            if getattr(parse, "is_known", False):
                return parse
        return parses[0]


def is_valid_prepositional_phrase(preposition: str, noun_entry: NounEntry, case: str) -> bool:
    normalized_preposition = preposition.lower()
    normalized_case = _case_tag(case)

    if normalized_preposition == "во":
        return normalized_case == "loct" and noun_entry.lemma in {"двор"}

    allowed_cases = _PREPOSITION_CASES.get(normalized_preposition)
    if allowed_cases is not None and normalized_case not in allowed_cases:
        return False

    allowed_semantics = _PREPOSITION_SEMANTICS.get(normalized_preposition)
    if allowed_semantics is None:
        return True
    if "entity" in allowed_semantics and noun_entry.semantic_class in {"person", "organization"}:
        return True
    return noun_entry.semantic_class in allowed_semantics


def reject_bad_surface(text: str) -> list[str]:
    reasons: list[str] = []
    lowered = text.lower()

    if "  " in text:
        reasons.append("double_space")
    if LATIN_RE.search(text):
        reasons.append("latin_letters")
    if "{" in text or "}" in text:
        reasons.append("template_brace")
    if _has_bad_vo_phrase(lowered):
        reasons.append("bad_vo_phrase")
    if BAD_K_OGOROD_RE.search(lowered):
        reasons.append("bad_k_phrase")

    return reasons


def _has_bad_vo_phrase(text: str) -> bool:
    for match in VO_RE.finditer(text):
        prefix_length = len(match.group(1) or "")
        phrase_start = match.start() + prefix_length
        suffix = text[phrase_start:]
        if not any(_starts_with_whitelist_phrase(suffix, phrase) for phrase in VO_WHITELIST):
            return True
    return False


def _starts_with_whitelist_phrase(text: str, phrase: str) -> bool:
    if not text.startswith(phrase):
        return False
    if len(text) == len(phrase):
        return True
    return text[len(phrase)] in " \t\r\n,.!?;:"


def _case_tag(case: str) -> str:
    return {
        "nom": "nomn",
        "nomn": "nomn",
        "gen": "gent",
        "gent": "gent",
        "dat": "datv",
        "datv": "datv",
        "acc": "accs",
        "accs": "accs",
        "ins": "ablt",
        "ablt": "ablt",
        "prep": "loct",
        "loct": "loct",
    }.get(case, case)


def _number_tag(number: str) -> str:
    if number == "plur":
        return "plur"
    return "sing"


def _gender_tag(gender: str) -> str:
    return {
        "masc": "masc",
        "fem": "femn",
        "femn": "femn",
        "neut": "neut",
    }.get(gender, gender)


def _fallback_noun(lemma: str, case: str, number: str) -> str | None:
    forms = _FALLBACK_NOUNS.get(lemma, {})
    return forms.get((_case_tag(case), _number_tag(number)))


def _fallback_adjective(lemma: str, gender: str, case: str, number: str) -> str | None:
    key = (_gender_tag(gender), _case_tag(case), _number_tag(number))
    forms = _FALLBACK_ADJECTIVES.get(lemma, {})
    if key in forms:
        return forms[key]

    if number == "plur":
        return _fallback_adjective_plural(lemma, case)
    if lemma.endswith("ый"):
        stem = lemma[:-2]
        return _fallback_hard_adjective(stem, _gender_tag(gender), _case_tag(case))
    if lemma.endswith("ий"):
        stem = lemma[:-2]
        return _fallback_soft_adjective(stem, _gender_tag(gender), _case_tag(case))
    return None


def _fallback_hard_adjective(stem: str, gender: str, case: str) -> str | None:
    endings = {
        ("masc", "nomn"): "ый",
        ("femn", "nomn"): "ая",
        ("neut", "nomn"): "ое",
        ("masc", "gent"): "ого",
        ("femn", "gent"): "ой",
        ("neut", "gent"): "ого",
        ("masc", "datv"): "ому",
        ("femn", "datv"): "ой",
        ("neut", "datv"): "ому",
        ("masc", "accs"): "ый",
        ("femn", "accs"): "ую",
        ("neut", "accs"): "ое",
        ("masc", "ablt"): "ым",
        ("femn", "ablt"): "ой",
        ("neut", "ablt"): "ым",
        ("masc", "loct"): "ом",
        ("femn", "loct"): "ой",
        ("neut", "loct"): "ом",
    }
    ending = endings.get((gender, case))
    return f"{stem}{ending}" if ending else None


def _fallback_soft_adjective(stem: str, gender: str, case: str) -> str | None:
    endings = {
        ("masc", "nomn"): "ий",
        ("femn", "nomn"): "ая",
        ("neut", "nomn"): "ее",
        ("masc", "gent"): "его",
        ("femn", "gent"): "ей",
        ("neut", "gent"): "его",
        ("masc", "datv"): "ему",
        ("femn", "datv"): "ей",
        ("neut", "datv"): "ему",
        ("masc", "accs"): "ий",
        ("femn", "accs"): "юю",
        ("neut", "accs"): "ее",
        ("masc", "ablt"): "им",
        ("femn", "ablt"): "ей",
        ("neut", "ablt"): "им",
        ("masc", "loct"): "ем",
        ("femn", "loct"): "ей",
        ("neut", "loct"): "ем",
    }
    ending = endings.get((gender, case))
    return f"{stem}{ending}" if ending else None


def _fallback_adjective_plural(lemma: str, case: str) -> str | None:
    if not (lemma.endswith("ый") or lemma.endswith("ий")):
        return None
    stem = lemma[:-2]
    endings = {
        "nomn": "ые",
        "gent": "ых",
        "datv": "ым",
        "accs": "ые",
        "ablt": "ыми",
        "loct": "ых",
    }
    ending = endings.get(_case_tag(case))
    return f"{stem}{ending}" if ending else None


def _fallback_verb_past(lemma: str, gender: str, number: str) -> str | None:
    forms = _FALLBACK_VERB_PAST.get(lemma)
    number_tag = _number_tag(number)
    gender_tag = _gender_tag(gender)
    if forms:
        return forms["plur"] if number_tag == "plur" else forms.get(gender_tag)

    stem: str | None = None
    if lemma.endswith("ить"):
        stem = lemma[:-3] + "ил"
    elif lemma.endswith("ать"):
        stem = lemma[:-3] + "ал"
    elif lemma.endswith("ять"):
        stem = lemma[:-3] + "ял"
    elif lemma.endswith("еть"):
        stem = lemma[:-3] + "ел"
    elif lemma.endswith("уть"):
        stem = lemma[:-3] + "ул"

    if stem is None:
        return None
    if number_tag == "plur":
        return f"{stem}и"
    if gender_tag == "femn":
        return f"{stem}а"
    if gender_tag == "neut":
        return f"{stem}о"
    return stem


_PREPOSITION_CASES = {
    "в": {"loct", "accs"},
    "на": {"loct", "accs"},
    "к": {"datv"},
    "у": {"gent"},
    "после": {"gent"},
    "перед": {"ablt"},
    "о": {"loct"},
    "для": {"gent"},
    "без": {"gent"},
    "с": {"ablt"},
    "по": {"datv"},
    "из": {"gent"},
    "до": {"gent"},
    "через": {"accs"},
    "за": {"ablt", "accs"},
    "между": {"ablt"},
    "при": {"loct"},
    "внутри": {"gent"},
    "около": {"gent"},
}

_PREPOSITION_SEMANTICS = {
    "в": {"place", "organization", "event", "time"},
    "на": {"place", "event", "object", "document", "time"},
    "к": {"person", "organization"},
    "у": {"person", "organization"},
    "после": {"event", "time"},
    "перед": {"event", "person"},
    "о": {"abstract", "document", "event"},
    "для": {"person", "organization"},
    "без": {"object", "document", "abstract"},
    "с": {"person", "organization"},
    "по": {"place", "document", "abstract"},
    "из": {"place", "organization", "document"},
    "до": {"time", "event", "place"},
    "через": {"time", "place"},
    "за": {"object", "place"},
    "между": {"object", "place", "organization"},
    "при": {"organization", "event"},
    "внутри": {"place", "organization"},
    "около": {"place", "time"},
}

_FALLBACK_NOUNS = {
    "девочка": {
        ("nomn", "sing"): "девочка",
        ("gent", "sing"): "девочки",
        ("datv", "sing"): "девочке",
        ("accs", "sing"): "девочку",
        ("ablt", "sing"): "девочкой",
        ("loct", "sing"): "девочке",
    },
    "студент": {
        ("nomn", "sing"): "студент",
        ("gent", "sing"): "студента",
        ("datv", "sing"): "студенту",
        ("accs", "sing"): "студента",
        ("ablt", "sing"): "студентом",
        ("loct", "sing"): "студенте",
    },
    "комиссия": {
        ("nomn", "sing"): "комиссия",
        ("gent", "sing"): "комиссии",
        ("datv", "sing"): "комиссии",
        ("accs", "sing"): "комиссию",
        ("ablt", "sing"): "комиссией",
        ("loct", "sing"): "комиссии",
    },
    "здание": {
        ("nomn", "sing"): "здание",
        ("gent", "sing"): "здания",
        ("datv", "sing"): "зданию",
        ("accs", "sing"): "здание",
        ("ablt", "sing"): "зданием",
        ("loct", "sing"): "здании",
    },
    "огород": {
        ("nomn", "sing"): "огород",
        ("gent", "sing"): "огорода",
        ("datv", "sing"): "огороду",
        ("accs", "sing"): "огород",
        ("ablt", "sing"): "огородом",
        ("loct", "sing"): "огороде",
    },
    "двор": {
        ("nomn", "sing"): "двор",
        ("gent", "sing"): "двора",
        ("datv", "sing"): "двору",
        ("accs", "sing"): "двор",
        ("ablt", "sing"): "двором",
        ("loct", "sing"): "дворе",
    },
}

_FALLBACK_ADJECTIVES = {
    "умный": {
        ("masc", "nomn", "sing"): "умный",
        ("femn", "nomn", "sing"): "умная",
        ("neut", "nomn", "sing"): "умное",
        ("plur", "nomn", "plur"): "умные",
    }
}

_FALLBACK_VERB_PAST = {
    "пойти": {"masc": "пошёл", "femn": "пошла", "neut": "пошло", "plur": "пошли"},
    "идти": {"masc": "шёл", "femn": "шла", "neut": "шло", "plur": "шли"},
    "решить": {"masc": "решил", "femn": "решила", "neut": "решило", "plur": "решили"},
    "проверить": {"masc": "проверил", "femn": "проверила", "neut": "проверило", "plur": "проверили"},
    "сказать": {"masc": "сказал", "femn": "сказала", "neut": "сказало", "plur": "сказали"},
}

_FALLBACK_PRESENT_3SG = {
    "идти": "идёт",
    "говорить": "говорит",
    "писать": "пишет",
    "решать": "решает",
    "готовиться": "готовится",
}
