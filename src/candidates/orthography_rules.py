from __future__ import annotations

from dataclasses import dataclass

from src.candidates.morphology import has_pos, is_known_word, normal_forms


@dataclass(frozen=True)
class OrthographyCandidate:
    replacement: str
    edit_type: str = "spelling"
    confidence: float = 0.92
    requires_model: bool = False
    rule: str = ""


VERB_POSES = frozenset({"VERB", "INFN"})
TSYA_CONTEXT_POSES = frozenset({"VERB", "INFN"})
NE_JOINED_NORMAL_FORMS = frozenset(
    {
        "ненавидеть",
        "негодовать",
        "нездоровиться",
        "недоумевать",
        "недосмотреть",
        "недооценить",
        "недосыпать",
        "недополучить",
        "недоставать",
    }
)

SPELLING_PATTERNS = {
    "жы": "жи",
    "шы": "ши",
    "чя": "ча",
    "щя": "ща",
    "чю": "чу",
    "щю": "щу",
    "цы": "ци",
    "жо": "же",
    "шо": "ше",
    "чо": "че",
    "що": "ще",
}

CY_EXCEPTIONS = ("цыган", "цыпл", "цыц", "цык", "цып")
HARD_SIGN_PREFIXES = (
    "сверх",
    "меж",
    "контр",
    "пан",
    "двух",
    "трех",
    "пред",
    "под",
    "над",
    "раз",
    "без",
    "из",
    "об",
    "от",
    "в",
    "с",
    "ад",
    "ин",
    "кон",
    "суб",
)
HARD_SIGN_VOWELS = frozenset({"е", "ю", "я"})
VOICELESS = frozenset("кпстфхцчшщ")
VOICED_OR_SONORANT_OR_VOWEL = frozenset("бвгджзлмнраеёиоуыэюя")
PREFIX_Z_S_PAIRS = (
    ("без", "бес"),
    ("раз", "рас"),
    ("из", "ис"),
    ("воз", "вос"),
    ("низ", "нис"),
    ("вз", "вс"),
)


def generated_orthography_candidates(word: str) -> list[OrthographyCandidate]:
    lower = word.lower()
    candidates: list[OrthographyCandidate] = []

    _append(candidates, _ne_verb_candidate(lower))
    _append_many(candidates, _pattern_candidates(lower))
    _append_many(candidates, _hard_sign_candidates(lower))
    _append_many(candidates, _prefix_z_s_candidates(lower))
    _append_many(candidates, _tsya_candidates(lower))
    return _deduplicate(candidates, lower)


def _ne_verb_candidate(word: str) -> OrthographyCandidate | None:
    if not word.startswith("не") or len(word) <= 4:
        return None
    rest = word[2:]
    if word.startswith("недо") and is_known_word(word):
        return None
    if normal_forms(word) & NE_JOINED_NORMAL_FORMS:
        return None
    if not has_pos(rest, VERB_POSES):
        return None
    return OrthographyCandidate(
        f"не {rest}",
        edit_type="split_join",
        confidence=0.97,
        requires_model=is_known_word(word),
        rule="ne_verb",
    )


def _pattern_candidates(word: str) -> list[OrthographyCandidate]:
    candidates: list[OrthographyCandidate] = []
    for wrong, correct in SPELLING_PATTERNS.items():
        if wrong not in word:
            continue
        replacement = word.replace(wrong, correct, 1)
        candidate = _candidate_for_known_replacement(word, replacement, confidence=0.94, rule=f"pattern_{wrong}_{correct}")
        if candidate:
            candidates.append(candidate)
    candidates.extend(_cy_exception_candidates(word))
    return candidates


def _cy_exception_candidates(word: str) -> list[OrthographyCandidate]:
    if not word.startswith("ци"):
        return []
    replacement = "цы" + word[2:]
    if not replacement.startswith(CY_EXCEPTIONS):
        return []
    candidate = _candidate_for_known_replacement(word, replacement, confidence=0.94, rule="cy_exception")
    if candidate:
        return [candidate]
    return []


def _hard_sign_candidates(word: str) -> list[OrthographyCandidate]:
    candidates: list[OrthographyCandidate] = []

    if "ь" in word:
        replacement = word.replace("ь", "ъ", 1)
        candidate = _candidate_for_known_replacement(word, replacement, confidence=0.95, rule="soft_to_hard_sign")
        if _has_hard_sign_after_prefix(replacement) and candidate:
            candidates.append(candidate)

    for index, char in enumerate(word):
        if char not in HARD_SIGN_VOWELS:
            continue
        replacement = word[:index] + "ъ" + word[index:]
        candidate = _candidate_for_known_replacement(word, replacement, confidence=0.95, rule="missing_hard_sign")
        if _has_hard_sign_after_prefix(replacement) and candidate:
            candidates.append(candidate)

    return candidates


def _has_hard_sign_after_prefix(word: str) -> bool:
    if "ъ" not in word:
        return False
    index = word.find("ъ")
    before = word[:index]
    after = word[index + 1 : index + 2]
    return after in HARD_SIGN_VOWELS and any(before.endswith(prefix) for prefix in HARD_SIGN_PREFIXES)


def _prefix_z_s_candidates(word: str) -> list[OrthographyCandidate]:
    candidates: list[OrthographyCandidate] = []

    if word.startswith("зд"):
        replacement = "с" + word[1:]
        candidate = _candidate_for_known_replacement(word, replacement, confidence=0.95, rule="sdelat_prefix")
        if candidate:
            candidates.append(candidate)

    for z_prefix, s_prefix in PREFIX_Z_S_PAIRS:
        if word.startswith(z_prefix):
            rest = word[len(z_prefix) :]
            if rest[:1] in VOICELESS:
                replacement = s_prefix + rest
                candidate = _candidate_for_known_replacement(word, replacement, confidence=0.95, rule="prefix_z_to_s")
                if candidate:
                    candidates.append(candidate)
        if word.startswith(s_prefix):
            rest = word[len(s_prefix) :]
            if rest[:1] in VOICED_OR_SONORANT_OR_VOWEL:
                replacement = z_prefix + rest
                candidate = _candidate_for_known_replacement(word, replacement, confidence=0.95, rule="prefix_s_to_z")
                if candidate:
                    candidates.append(candidate)

    return candidates


def _tsya_candidates(word: str) -> list[OrthographyCandidate]:
    candidates: list[OrthographyCandidate] = []
    if word.endswith("ться"):
        replacement = word[: -len("ться")] + "тся"
        if is_known_word(replacement) and has_pos(replacement, TSYA_CONTEXT_POSES):
            candidates.append(
                OrthographyCandidate(
                    replacement,
                    confidence=0.9,
                    requires_model=is_known_word(word),
                    rule="tsya_soft_delete",
                )
            )
    if word.endswith("тся"):
        replacement = word[: -len("тся")] + "ться"
        if is_known_word(replacement) and has_pos(replacement, TSYA_CONTEXT_POSES):
            candidates.append(
                OrthographyCandidate(
                    replacement,
                    confidence=0.9,
                    requires_model=is_known_word(word),
                    rule="tsya_soft_insert",
                )
            )
    return candidates


def _candidate_for_known_replacement(
    source: str,
    replacement: str,
    *,
    confidence: float,
    rule: str,
    edit_type: str = "spelling",
) -> OrthographyCandidate | None:
    if source == replacement or not is_known_word(replacement):
        return None
    return OrthographyCandidate(
        replacement,
        edit_type=edit_type,
        confidence=confidence,
        requires_model=is_known_word(source),
        rule=rule,
    )


def _append(candidates: list[OrthographyCandidate], candidate: OrthographyCandidate | None) -> None:
    if candidate is not None:
        candidates.append(candidate)


def _append_many(candidates: list[OrthographyCandidate], generated: list[OrthographyCandidate]) -> None:
    candidates.extend(generated)


def _deduplicate(candidates: list[OrthographyCandidate], source: str) -> list[OrthographyCandidate]:
    seen: set[tuple[str, str]] = set()
    result: list[OrthographyCandidate] = []
    for candidate in candidates:
        key = (candidate.replacement, candidate.edit_type)
        if candidate.replacement == source or key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result
