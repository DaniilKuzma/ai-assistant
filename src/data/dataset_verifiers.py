from __future__ import annotations

from dataclasses import dataclass
import difflib
import re
from typing import Iterable


SUBORDINATE_TRIGGERS = (
    "что",
    "чтобы",
    "если",
    "когда",
    "потому что",
    "так как",
    "поскольку",
    "где",
    "куда",
    "откуда",
)
CONJUNCTION_TRIGGERS = ("но", "а", "зато", "однако")
INTRODUCTORY_PHRASES = (
    "конечно",
    "возможно",
    "вероятно",
    "например",
    "по-видимому",
    "следовательно",
    "таким образом",
    "к счастью",
)
ADDRESS_HEADS = ("коллеги", "иван", "мария", "друзья", "уважаемые коллеги")
GERUNDS = (
    "проверив",
    "получив",
    "закончив",
    "обсудив",
    "изучив",
    "сравнив",
    "подготовив",
    "согласовав",
    "прочитав",
    "вернувшись",
    "учитывая",
    "используя",
    "рассмотрев",
    "оценив",
)
PARTICIPLE_RE = re.compile(
    r"\b[а-яё]+(?:нный|нная|нное|нные|нного|нной|нным|нными|енный|енная|енное|енные|"
    r"анный|анная|анное|анные|тый|тая|тое|тые|вший|вшая|вшее|вшие|ющий|ющая|ющее|ющие|"
    r"емый|емая|емое|емые|имый|имая|имое|имые)\b",
    re.IGNORECASE,
)
CONTEXT_PAIRS = {
    "context_tak_zhe": (("также", "так же"), ("так же", "также")),
    "context_to_zhe": (("тоже", "то же"), ("то же", "тоже")),
    "context_chto_by": (("чтобы", "что бы"), ("что бы", "чтобы")),
    "context_za_to": (("зато", "за то"), ("за то", "зато")),
    "context_vsledstvie": (("вследствие", "в следствие"), ("в следствие", "вследствие")),
    "context_nesmotrya": (("несмотря", "не смотря"), ("не смотря", "несмотря")),
}
TYPO_RULE_FAMILIES = {
    "double_consonant_candidate": "double_consonant",
    "swapped_letters_candidate": "swapped_letters",
    "missing_letter_candidate": "missing_letter",
    "extra_letter_candidate": "extra_letter",
    "keyboard_typo_candidate": "keyboard_typo",
    "dictionary_fuzzy": "dictionary_fuzzy",
}


@dataclass(frozen=True)
class SemanticVerification:
    semantic_alignment_pass: bool
    reason: str
    actual_error_family: str
    expected_error_family: str


def semantic_alignment_for_rule(rule_id: str, source: str, target: str) -> SemanticVerification:
    if rule_id in PUNCTUATION_RULE_FAMILIES:
        return verify_punctuation_rule(rule_id, source, target)
    if rule_id in CONTEXT_PAIRS:
        return verify_context_pair_rule(rule_id, source, target)
    if rule_id in TYPO_RULE_FAMILIES:
        return _verify_word_family(rule_id, source, target)
    if rule_id.startswith("ne_"):
        return verify_ne_rule(rule_id, source, target)
    if rule_id.startswith("n_nn_"):
        return verify_n_nn_rule(rule_id, source, target)
    if rule_id.startswith("tsya_"):
        return verify_tsya_rule(rule_id, source, target)
    if rule_id.startswith("hyphen_") or rule_id == "pol_polu_compounds":
        return verify_hyphen_rule(rule_id, source, target)
    if rule_id in {"capitalization_sentence_start", "abbreviation_case_protection"}:
        return verify_capitalization_rule(rule_id, source, target)
    if rule_id.startswith("pattern_") or rule_id in {
        "cy_exception",
        "missing_hard_sign",
        "soft_to_hard_sign",
        "sdelat_prefix",
        "prefix_pre_pri",
        "prefix_s_to_z",
        "prefix_z_to_s",
        "frequent_error_exact",
        "ni_stable_expression",
        "ni_particle_context",
    }:
        return _simple_non_identity(rule_id, source, target, "orthography")
    return _simple_non_identity(rule_id, source, target, rule_id)


PUNCTUATION_RULE_FAMILIES = {
    "comma_subordinate": "comma_subordinate",
    "comma_conjunction": "comma_conjunction",
    "introductory_comma": "introductory_comma",
    "address_comma": "address_comma",
    "homogeneous_comma": "homogeneous_comma",
    "detached_adverbial_comma": "detached_adverbial_comma",
    "detached_participial_comma": "detached_participial_comma",
    "apposition_comma": "apposition_comma",
    "clarification_comma": "clarification_comma",
    "comparative_turnover_comma": "comparative_turnover_comma",
    "subject_predicate_dash": "subject_predicate_dash",
    "asyndetic_dash": "asyndetic_dash",
    "consequence_dash": "consequence_dash",
    "explanation_colon": "explanation_colon",
    "enumeration_colon": "enumeration_colon",
    "enumeration_dash": "enumeration_dash",
    "semicolon": "semicolon",
    "direct_speech_quotes": "direct_speech_quotes",
    "direct_speech_colon": "direct_speech_colon",
    "direct_speech_dash": "direct_speech_dash",
    "quote_pair_balance": "quote_pair_balance",
    "bracket_pair_balance": "bracket_pair_balance",
    "punctuation_delete_replace": "punctuation_delete_replace",
    "final_punctuation_default": "final_punctuation",
}


def verify_punctuation_rule(rule_id: str, source: str, target: str) -> SemanticVerification:
    expected = PUNCTUATION_RULE_FAMILIES.get(rule_id, rule_id)
    if source == target:
        return SemanticVerification(False, "identity_pair", "identity", expected)
    if numeric_punctuation_mismatch(source, target):
        return SemanticVerification(False, "numeric_punctuation_mismatch", "numeric_punctuation", expected)
    actual = detect_punctuation_family(source, target)
    passed = actual == expected
    if rule_id in {"asyndetic_dash", "consequence_dash"} and actual in {"asyndetic_dash", "consequence_dash"}:
        passed = True
    return SemanticVerification(passed, "ok" if passed else "wrong_punctuation_family", actual, expected)


def detect_punctuation_family(source: str, target: str) -> str:
    lower = target.lower()
    if _final_punctuation_delta(source, target):
        return "final_punctuation"
    if _punctuation_noise_cleanup(source, target):
        return "punctuation_delete_replace"
    if _direct_speech_quotes(source, target):
        return "direct_speech_quotes"
    if _direct_speech_colon(source, target):
        return "direct_speech_colon"
    if _direct_speech_dash(source, target):
        return "direct_speech_dash"
    if _quote_pair_balance(source, target):
        return "quote_pair_balance"
    if _bracket_pair_balance(source, target):
        return "bracket_pair_balance"
    if _subject_predicate_dash(source, target):
        return "subject_predicate_dash"
    if _enumeration_dash(source, target):
        return "enumeration_dash"
    if _enumeration_colon(source, target):
        return "enumeration_colon"
    if _explanation_colon(source, target):
        return "explanation_colon"
    if _semicolon(source, target):
        return "semicolon"
    if _dash_delta(source, target):
        if re.search(r"\b(?:начался|истек|закончилось|опоздали)\b[^—]{1,80}\s—\s", lower):
            return "consequence_dash"
        return "asyndetic_dash"
    if _detached_adverbial(source, target):
        return "detached_adverbial_comma"
    if _detached_participial(source, target):
        return "detached_participial_comma"
    if _apposition(source, target):
        return "apposition_comma"
    if _clarification(source, target):
        return "clarification_comma"
    if _comparative(source, target):
        return "comparative_turnover_comma"
    if _homogeneous(source, target):
        return "homogeneous_comma"
    if _subordinate(source, target):
        return "comma_subordinate"
    if _conjunction(source, target):
        return "comma_conjunction"
    if _introductory(source, target):
        return "introductory_comma"
    if _address(source, target):
        return "address_comma"
    if _comma_delta(source, target):
        return "generic_comma"
    return "unknown"


def numeric_punctuation_mismatch(source: str, target: str) -> bool:
    if source == target:
        return False
    source_numeric = set(re.findall(r"\d+[,.]\d+", source))
    target_numeric = set(re.findall(r"\d+[,.]\d+", target))
    if target_numeric - source_numeric:
        return True
    return bool(re.search(r"\d\s+\d", source) and re.search(r"\d,\d", target))


def verify_context_pair_rule(rule_id: str, source: str, target: str) -> SemanticVerification:
    expected = "context_pair"
    for wrong, correct in CONTEXT_PAIRS.get(rule_id, ()):
        if wrong in source.lower() and correct in target.lower() and source.lower().replace(wrong, correct, 1) == target.lower():
            return SemanticVerification(True, "ok", expected, expected)
    return SemanticVerification(False, "wrong_context_pair", "unknown", expected)


def verify_dictionary_typo_family(rule_id: str, source_form: str, target_form: str) -> SemanticVerification:
    expected = TYPO_RULE_FAMILIES.get(rule_id, rule_id)
    actual = _typo_family(source_form.lower(), target_form.lower())
    if rule_id == "dictionary_fuzzy" and actual not in {"double_consonant", "swapped_letters", "missing_letter", "extra_letter", "keyboard_typo", "identity"}:
        actual = "dictionary_fuzzy"
    passed = actual == expected
    return SemanticVerification(passed, "ok" if passed else "wrong_dictionary_typo_family", actual, expected)


def verify_ne_rule(rule_id: str, source: str, target: str) -> SemanticVerification:
    expected = rule_id
    source_lower = source.lower()
    target_lower = target.lower()
    if "не" not in source_lower and "не" not in target_lower:
        return SemanticVerification(False, "missing_ne_delta", "unknown", expected)
    if source == target:
        return SemanticVerification(False, "identity_pair", "identity", expected)
    if rule_id == "ne_verb" and re.search(r"\bне\s+[а-яё]+(?:ть|тся|ться|ет|ют|ит|ат|ят|ал|ала|али)\b", target_lower):
        return SemanticVerification(True, "ok", "ne_verb", expected)
    if rule_id in {"ne_adjective", "ne_adverb", "ne_participle", "ne_short_form", "ne_predicative"}:
        if "несогласен с выводом" in target_lower or "ненужно комиссии" in target_lower:
            return SemanticVerification(False, "known_bad_ne_target", "bad_target", expected)
        return SemanticVerification(True, "ok", rule_id, expected)
    return SemanticVerification(False, "wrong_ne_family", "unknown", expected)


def verify_n_nn_rule(rule_id: str, source: str, target: str) -> SemanticVerification:
    if re.search(r"\bцены\b", source.lower()) and re.search(r"\bценны\b", target.lower()):
        return SemanticVerification(False, "known_bad_n_nn_noun_rewrite", "noun_rewrite", rule_id)
    if re.search(r"\bстраны\b", source.lower()) and re.search(r"\bстранны\b", target.lower()):
        return SemanticVerification(False, "known_bad_n_nn_noun_rewrite", "noun_rewrite", rule_id)
    if _only_n_nn_delta(source, target):
        return SemanticVerification(True, "ok", "n_nn", rule_id)
    return SemanticVerification(False, "not_n_nn_delta", "unknown", rule_id)


def verify_tsya_rule(rule_id: str, source: str, target: str) -> SemanticVerification:
    if source == target:
        return SemanticVerification(False, "identity_pair", "identity", rule_id)
    joined = source.lower() + "\n" + target.lower()
    passed = "тся" in joined and "ться" in joined
    return SemanticVerification(passed, "ok" if passed else "not_tsya_delta", "tsya" if passed else "unknown", rule_id)


def verify_hyphen_rule(rule_id: str, source: str, target: str) -> SemanticVerification:
    lower = target.lower()
    if any(phrase in lower for phrase in ("по-старому плану", "по-новому вариант", "по-новому договору", "по-старому адресу")):
        return SemanticVerification(False, "known_bad_hyphen_context", "bad_context", rule_id)
    passed = source != target and ("-" in source or "-" in target)
    return SemanticVerification(passed, "ok" if passed else "not_hyphen_delta", "hyphen" if passed else "unknown", rule_id)


def verify_capitalization_rule(rule_id: str, source: str, target: str) -> SemanticVerification:
    if source == target:
        return SemanticVerification(False, "identity_pair", "identity", rule_id)
    if source.lower() != target.lower():
        return SemanticVerification(False, "not_case_only", "unknown", rule_id)
    return SemanticVerification(True, "ok", "capitalization", rule_id)


def _verify_word_family(rule_id: str, source: str, target: str) -> SemanticVerification:
    source_word, target_word = _single_changed_word(source, target)
    if not source_word or not target_word:
        return SemanticVerification(False, "not_single_word_typo", "unknown", TYPO_RULE_FAMILIES[rule_id])
    return verify_dictionary_typo_family(rule_id, source_word, target_word)


def _simple_non_identity(rule_id: str, source: str, target: str, family: str) -> SemanticVerification:
    passed = source != target
    return SemanticVerification(passed, "ok" if passed else "identity_pair", family if passed else "identity", family or rule_id)


def _comma_delta(source: str, target: str) -> bool:
    return target.count(",") > source.count(",")


def _dash_delta(source: str, target: str) -> bool:
    return " — " in target and " — " not in source


def _subordinate(source: str, target: str) -> bool:
    if not _comma_delta(source, target):
        return False
    pattern = r",\s*(?:" + "|".join(re.escape(item) for item in SUBORDINATE_TRIGGERS) + r")\b"
    return bool(re.search(pattern, target.lower()))


def _conjunction(source: str, target: str) -> bool:
    if not _comma_delta(source, target) or _subordinate(source, target):
        return False
    return bool(re.search(r",\s*(?:но|а|зато|однако)\b", target.lower()))


def _introductory(source: str, target: str) -> bool:
    if not _comma_delta(source, target):
        return False
    lower = target.lower()
    return any(lower.startswith(phrase + ",") or lower.startswith(phrase + " ,") for phrase in INTRODUCTORY_PHRASES)


def _address(source: str, target: str) -> bool:
    if not _comma_delta(source, target):
        return False
    lower = target.lower()
    return any(lower.startswith(head + ",") for head in ADDRESS_HEADS)


def _homogeneous(source: str, target: str) -> bool:
    if not _comma_delta(source, target) or _subordinate(source, target) or _conjunction(source, target):
        return False
    lower = target.lower()
    return bool(re.search(r"\b(?:и|ни)\s+[а-яё]{3,}(?:\s+[а-яё]{3,}){0,2},\s*(?:и|ни)\s+[а-яё]{3,}", lower))


def _detached_adverbial(source: str, target: str) -> bool:
    if not _comma_delta(source, target):
        return False
    lower = target.lower()
    return bool(re.search(r"^\s*(?:" + "|".join(GERUNDS) + r")\b[^,]{1,120},", lower))


def _detached_participial(source: str, target: str) -> bool:
    return is_detached_participial_comma_pair(source, target)


def is_detached_participial_comma_pair(source: str, target: str) -> bool:
    if not _comma_delta(source, target):
        return False
    lower = target.lower()
    if numeric_punctuation_mismatch(source, target):
        return False
    if re.search(
        r"\b(?:что|чтобы|когда|если|потому\s+что|котор(?:ый|ая|ое|ые|ого|ому|ым|ой|ую|ых|ыми))\b",
        lower,
    ):
        return False
    inserted = _inserted_comma_positions(source, target)
    if not inserted:
        return False
    return all(_inserted_comma_is_participial_boundary(target, position) for position in inserted)


def _inserted_comma_positions(source: str, target: str) -> list[int]:
    if source == target or "," not in target:
        return []
    direct = [index for index, char in enumerate(target) if char == "," and target[:index] + target[index + 1 :] == source]
    if direct:
        return direct
    positions: list[int] = []
    matcher = difflib.SequenceMatcher(a=source, b=target, autojunk=False)
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag not in {"insert", "replace"}:
            continue
        positions.extend(j1 + offset for offset, char in enumerate(target[j1:j2]) if char == ",")
    return positions


def _inserted_comma_is_participial_boundary(target: str, comma_pos: int) -> bool:
    lower = target.lower()
    if comma_pos < 0 or comma_pos >= len(target) or target[comma_pos] != ",":
        return False
    if _is_source_attribution_comma(lower, comma_pos) or _is_date_clarification_comma(lower, comma_pos):
        return False
    if _comma_starts_participial_phrase(lower, comma_pos):
        return True
    return _comma_ends_participial_phrase(lower, comma_pos)


def _is_source_attribution_comma(lower: str, comma_pos: int) -> bool:
    prefix = lower[:comma_pos].strip()
    return bool(
        re.search(
            r"(?:^|[.!?]\s*)(?:по\s+(?:данным|словам|информации|сведениям)|как\s+сообща(?:ет|ют|л|ла|ли)|как\s+заяв(?:ил|ила|или|ляет|ляют))\b",
            prefix,
        )
    )


def _is_date_clarification_comma(lower: str, comma_pos: int) -> bool:
    before = lower[max(0, comma_pos - 24) : comma_pos].strip()
    after = lower[comma_pos + 1 : comma_pos + 24].strip()
    return bool(
        re.search(r"\b(?:понедельник|вторник|среду|четверг|пятницу|субботу|воскресенье|утром|вечером|днем|ночью)$", before)
        and re.match(r"\d{1,2}\s+[а-яё]+", after)
    )


def _comma_starts_participial_phrase(lower: str, comma_pos: int) -> bool:
    before_word = _previous_word(lower, comma_pos)
    if not _safe_participial_anchor(before_word):
        return False
    after = lower[comma_pos + 1 : comma_pos + 140].lstrip()
    match = PARTICIPLE_RE.match(after)
    if not match:
        return False
    phrase_tail = after[match.end() :]
    phrase_words = re.findall(r"[а-яё]+", after[:100])
    if len(phrase_words) < 2:
        return False
    return bool(re.search(r"^[^,]{2,90},", phrase_tail))


def _comma_ends_participial_phrase(lower: str, comma_pos: int) -> bool:
    previous_comma = lower.rfind(",", 0, comma_pos)
    if previous_comma < 0 or comma_pos - previous_comma > 120:
        return False
    if not _comma_starts_participial_phrase(lower, previous_comma):
        return False
    segment = lower[previous_comma + 1 : comma_pos].strip()
    return bool(PARTICIPLE_RE.match(segment))


def _previous_word(lower: str, position: int) -> str:
    matches = list(re.finditer(r"[а-яё]+", lower[:position]))
    return matches[-1].group(0) if matches else ""


def _safe_participial_anchor(word: str) -> bool:
    if len(word) < 3:
        return False
    return word not in {
        "данным",
        "словам",
        "информации",
        "сведениям",
        "который",
        "которая",
        "которые",
        "что",
        "если",
        "когда",
        "мая",
        "июня",
        "июля",
        "августа",
        "сентября",
        "октября",
        "ноября",
        "декабря",
        "января",
        "февраля",
        "марта",
        "апреля",
    }


def _apposition(source: str, target: str) -> bool:
    if target.count(",") - source.count(",") < 2:
        return False
    lower = target.lower()
    if _subordinate(source, target) or _conjunction(source, target):
        return False
    heads = "директор|руководитель|глава|редактор|эксперт|представитель|основатель|автор|инженер|аналитик"
    return bool(re.search(r",\s*(?:" + heads + r")\b[^,]{2,80},", lower))


def _clarification(source: str, target: str) -> bool:
    if not _comma_delta(source, target):
        return False
    lower = target.lower()
    if re.search(r",\s*(?:то\s+есть|а\s+именно|в\s+частности|именно|например)\b", lower):
        return True
    return bool(re.search(r"\b(?:в\s+понедельник|во\s+вторник|в\s+среду|в\s+четверг|в\s+пятницу|утром|вечером),\s*\d{1,2}\s+[а-яё]+", lower))


def _comparative(source: str, target: str) -> bool:
    if not _comma_delta(source, target):
        return False
    lower = target.lower()
    if re.search(r"\bкак\s+(?:известно|ожидают|отмечено|инженер)\b", lower):
        return False
    return bool(re.search(r",\s*(?:словно|будто|как\s+будто)\b", lower))


def _subject_predicate_dash(source: str, target: str) -> bool:
    if not _dash_delta(source, target):
        return False
    return bool(re.search(r"\b[а-яё]{3,}(?:\s+[а-яё]{3,}){0,3}\s+—\s+(?:это\s+)?[а-яё]{3,}", target.lower()))


def _enumeration_dash(source: str, target: str) -> bool:
    return _dash_delta(source, target) and bool(re.search(r",\s*[а-яё]+,\s*[а-яё]+\s+—\s+(?:все|всё)\b", target.lower()))


def _enumeration_colon(source: str, target: str) -> bool:
    return ":" in target and ":" not in source and bool(re.search(r"\b(?:следующее|следующие):", target.lower()))


def _explanation_colon(source: str, target: str) -> bool:
    return ":" in target and ":" not in source and bool(re.search(r"\b(?:одно|причина|вывод):", target.lower()))


def _semicolon(source: str, target: str) -> bool:
    return ";" in target and ";" not in source and len(target.split(";")) == 2


def _direct_speech_colon(source: str, target: str) -> bool:
    return ":" in target and ":" not in source and bool(re.search(r"\b(?:сказал|сказала|ответил|ответила|сообщил|сообщила):", target.lower()))


def _direct_speech_dash(source: str, target: str) -> bool:
    return ("» — " in target and "» — " not in source) or ('" — ' in target and '" — ' not in source)


def _direct_speech_quotes(source: str, target: str) -> bool:
    lower = target.lower()
    if not re.search(r"\b(?:сказал|сказала|ответил|ответила|сообщил|сообщила)\b", lower):
        return False
    return target.count("«") == target.count("»") and target.count("«") > source.count("«")


def _quote_pair_balance(source: str, target: str) -> bool:
    return target.count("«") == target.count("»") and source.count("«") != source.count("»")


def _bracket_pair_balance(source: str, target: str) -> bool:
    pairs = (("(", ")"), ("[", "]"), ("{", "}"))
    return any(target.count(left) == target.count(right) and source.count(left) != source.count(right) for left, right in pairs)


def _punctuation_noise_cleanup(source: str, target: str) -> bool:
    return source != target and bool(re.search(r"[,;:]{2,}|[.!?]{2,}", source)) and not re.search(r"[,;:]{2,}|[.!?]{2,}", target)


def _final_punctuation_delta(source: str, target: str) -> bool:
    return source.rstrip(".!?…") == target.rstrip(".!?…") and source.rstrip() != target.rstrip()


def _single_changed_word(source: str, target: str) -> tuple[str, str]:
    source_words = re.findall(r"[а-яёА-ЯЁ-]+", source)
    target_words = re.findall(r"[а-яёА-ЯЁ-]+", target)
    if len(source_words) != len(target_words):
        return "", ""
    changed = [(left, right) for left, right in zip(source_words, target_words, strict=False) if left.lower() != right.lower()]
    if len(changed) != 1:
        return "", ""
    return changed[0]


def _typo_family(source_word: str, target_word: str) -> str:
    if source_word == target_word:
        return "identity"
    if len(target_word) == len(source_word) + 1 and _one_deletion_restores(source_word, target_word):
        return "missing_letter"
    if len(source_word) == len(target_word) + 1 and _one_deletion_restores(target_word, source_word):
        return "extra_letter"
    if len(source_word) == len(target_word) and _adjacent_swap_restores(source_word, target_word):
        return "swapped_letters"
    if len(source_word) == len(target_word) and _single_substitution(source_word, target_word):
        return "keyboard_typo"
    if _double_consonant_delta(source_word, target_word):
        return "double_consonant"
    return "dictionary_fuzzy"


def _one_deletion_restores(shorter: str, longer: str) -> bool:
    return any(longer[:index] + longer[index + 1 :] == shorter for index in range(len(longer)))


def _adjacent_swap_restores(source_word: str, target_word: str) -> bool:
    for index in range(len(source_word) - 1):
        swapped = source_word[:index] + source_word[index + 1] + source_word[index] + source_word[index + 2 :]
        if swapped == target_word:
            return True
    return False


def _single_substitution(source_word: str, target_word: str) -> bool:
    return sum(left != right for left, right in zip(source_word, target_word, strict=False)) == 1


def _double_consonant_delta(source_word: str, target_word: str) -> bool:
    consonants = "бвгджзйклмнпрстфхцчшщ"
    for word in (source_word, target_word):
        if any(ch * 2 in word for ch in consonants):
            other = target_word if word == source_word else source_word
            return abs(len(source_word) - len(target_word)) == 1 and _one_deletion_restores(other, word)
    return False


def _only_n_nn_delta(source: str, target: str) -> bool:
    source_words = re.findall(r"[а-яёА-ЯЁ-]+", source)
    target_words = re.findall(r"[а-яёА-ЯЁ-]+", target)
    if len(source_words) != len(target_words):
        return False
    changed = [(left.lower(), right.lower()) for left, right in zip(source_words, target_words, strict=False) if left.lower() != right.lower()]
    if len(changed) != 1:
        return False
    left, right = changed[0]
    return left.replace("нн", "н") == right.replace("нн", "н") and ("нн" in left or "нн" in right)


def any_rule_semantic_failure(rows: Iterable[dict[str, str]]) -> bool:
    return any(not semantic_alignment_for_rule(str(row.get("rule_id", "")), str(row.get("source", "")), str(row.get("target", ""))).semantic_alignment_pass for row in rows)
