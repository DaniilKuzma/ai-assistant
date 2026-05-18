from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from rapidfuzz import fuzz, process
from rapidfuzz.distance import DamerauLevenshtein, Levenshtein

from src.candidates.morphology import is_known_word
from src.preprocessing.protected_spans import find_protected_spans
from src.rules.base import RuleMode


RULE_ID = "dictionary_fuzzy"
DOUBLE_CONSONANT_RULE_ID = "double_consonant_candidate"
KEYBOARD_TYPO_RULE_ID = "keyboard_typo_candidate"
SWAPPED_LETTERS_RULE_ID = "swapped_letters_candidate"
MISSING_LETTER_RULE_ID = "missing_letter_candidate"
EXTRA_LETTER_RULE_ID = "extra_letter_candidate"
YO_E_RULE_ID = "yo_e_candidate"

MIN_WORD_LENGTH = 4
MIN_FUZZY_SCORE = 85.0
MIN_EXTRACT_SCORE = 65.0
MAX_EXTRACT_MULTIPLIER = 4
RUSSIAN_WORD_RE = re.compile(r"[а-яё]+", re.IGNORECASE)
PROTECTED_LITERAL_KINDS = frozenset({"url", "email", "number"})
RUSSIAN_CONSONANTS = frozenset("бвгджзйклмнпрстфхцчшщ")
RUSSIAN_KEYBOARD_ROWS = (
    "йцукенгшщзхъ",
    "фывапролджэ",
    "ячсмитьбю",
)
RUSSIAN_KEY_POSITIONS = {
    char: (row_index, column_index)
    for row_index, row in enumerate(RUSSIAN_KEYBOARD_ROWS)
    for column_index, char in enumerate(row)
}


@dataclass(frozen=True)
class DictionaryCandidateSpec:
    replacement: str
    edit_type: str = "spelling"
    confidence: float = 0.0
    requires_model: bool = True
    rule_id: str = RULE_ID
    mode: RuleMode = "model_required"


def dictionary_candidates(
    word: str,
    lexicon: Sequence[str],
    limit: int = 5,
    min_score: float = MIN_FUZZY_SCORE,
    *,
    yo_e_enabled: bool = False,
) -> list[str]:
    return [
        candidate.replacement
        for candidate in dictionary_candidate_specs(word, lexicon, limit, min_score, yo_e_enabled=yo_e_enabled)
    ]


def dictionary_candidate_specs(
    word: str,
    lexicon: Sequence[str],
    limit: int = 5,
    min_score: float = MIN_FUZZY_SCORE,
    *,
    yo_e_enabled: bool = False,
) -> list[DictionaryCandidateSpec]:
    if limit <= 0 or not _is_valid_source_literal(word):
        return []

    normalized = _normalize(word)
    lexicon_set = _normalized_lexicon_set(lexicon)
    choices = _candidate_choices(normalized, lexicon)
    results: list[DictionaryCandidateSpec] = []
    seen: set[str] = set()

    if yo_e_enabled:
        _extend_limited(results, seen, _yo_e_candidate_specs(normalized, lexicon_set), limit)
        if len(results) >= limit:
            return results[:limit]

    if is_known_word(normalized) or not choices:
        return results

    for generator in (
        _double_consonant_candidate_specs,
        _keyboard_typo_candidate_specs,
        _swapped_letters_candidate_specs,
        _missing_letter_candidate_specs,
        _extra_letter_candidate_specs,
    ):
        _extend_limited(results, seen, generator(normalized, choices), limit)
        if len(results) >= limit:
            return results[:limit]

    _extend_limited(results, seen, _fuzzy_candidate_specs(normalized, choices, limit, min_score), limit)
    return results[:limit]


def _fuzzy_candidate_specs(
    word: str,
    choices: Sequence[str],
    limit: int,
    min_score: float,
) -> list[DictionaryCandidateSpec]:
    matches = process.extract(
        word,
        choices,
        scorer=fuzz.WRatio,
        limit=max(limit * MAX_EXTRACT_MULTIPLIER, limit),
        score_cutoff=min(float(min_score), MIN_EXTRACT_SCORE),
    )

    results: list[DictionaryCandidateSpec] = []
    seen: set[str] = set()
    for replacement, score, _index in matches:
        if replacement in seen:
            continue
        if not _is_acceptable_match(word, replacement, float(score), float(min_score)):
            continue
        seen.add(replacement)
        results.append(DictionaryCandidateSpec(replacement=replacement, confidence=float(score) / 100.0))
        if len(results) >= limit:
            break
    return results


def _double_consonant_candidate_specs(word: str, choices: Sequence[str]) -> list[DictionaryCandidateSpec]:
    return [
        DictionaryCandidateSpec(replacement=replacement, rule_id=DOUBLE_CONSONANT_RULE_ID, confidence=0.93)
        for replacement in choices
        if _is_double_consonant_match(word, replacement)
    ]


def _keyboard_typo_candidate_specs(word: str, choices: Sequence[str]) -> list[DictionaryCandidateSpec]:
    return [
        DictionaryCandidateSpec(replacement=replacement, rule_id=KEYBOARD_TYPO_RULE_ID, confidence=0.9)
        for replacement in choices
        if _is_keyboard_typo(word, replacement)
    ]


def _swapped_letters_candidate_specs(word: str, choices: Sequence[str]) -> list[DictionaryCandidateSpec]:
    return [
        DictionaryCandidateSpec(replacement=replacement, rule_id=SWAPPED_LETTERS_RULE_ID, confidence=0.91)
        for replacement in choices
        if _is_adjacent_swap(word, replacement)
    ]


def _missing_letter_candidate_specs(word: str, choices: Sequence[str]) -> list[DictionaryCandidateSpec]:
    return [
        DictionaryCandidateSpec(replacement=replacement, rule_id=MISSING_LETTER_RULE_ID, confidence=0.9)
        for replacement in choices
        if len(replacement) == len(word) + 1
        and not _is_double_consonant_match(word, replacement)
        and _one_deletion_matches(replacement, word)
    ]


def _extra_letter_candidate_specs(word: str, choices: Sequence[str]) -> list[DictionaryCandidateSpec]:
    return [
        DictionaryCandidateSpec(replacement=replacement, rule_id=EXTRA_LETTER_RULE_ID, confidence=0.9)
        for replacement in choices
        if len(word) == len(replacement) + 1 and _one_deletion_matches(word, replacement)
    ]


def _yo_e_candidate_specs(word: str, lexicon: set[str]) -> list[DictionaryCandidateSpec]:
    if "е" not in word and "ё" not in word:
        return []
    if word not in lexicon:
        return []
    return [
        DictionaryCandidateSpec(replacement=replacement, rule_id=YO_E_RULE_ID, confidence=0.88)
        for replacement in _yo_e_variants(word)
        if replacement in lexicon and replacement != word
    ]


def _is_valid_source_literal(word: str) -> bool:
    normalized = _normalize(word)
    if len(normalized) < MIN_WORD_LENGTH:
        return False
    if not RUSSIAN_WORD_RE.fullmatch(normalized):
        return False
    if _is_protected_literal(word):
        return False
    return True


def _normalized_lexicon_set(lexicon: Sequence[str]) -> set[str]:
    result: set[str] = set()
    for item in lexicon:
        candidate = _normalize(str(item))
        if RUSSIAN_WORD_RE.fullmatch(candidate):
            result.add(candidate)
    return result


def _candidate_choices(word: str, lexicon: Sequence[str]) -> list[str]:
    choices: list[str] = []
    seen: set[str] = set()
    for item in lexicon:
        candidate = _normalize(str(item))
        if candidate in seen:
            continue
        if candidate == word:
            continue
        if not RUSSIAN_WORD_RE.fullmatch(candidate):
            continue
        if not _length_compatible(word, candidate):
            continue
        seen.add(candidate)
        choices.append(candidate)
    return choices


def _is_acceptable_match(word: str, replacement: str, score: float, min_score: float) -> bool:
    if not _length_compatible(word, replacement):
        return False
    if Levenshtein.distance(word, replacement) > _max_edit_distance(word):
        return False
    if score >= min_score:
        return True
    return _passes_edit_distance_override(word, replacement)


def _is_double_consonant_match(word: str, replacement: str) -> bool:
    if len(replacement) == len(word) + 1:
        return any(
            char == replacement[index - 1]
            and char in RUSSIAN_CONSONANTS
            and replacement[:index] + replacement[index + 1 :] == word
            for index, char in enumerate(replacement)
            if index > 0
        )
    if len(word) == len(replacement) + 1:
        return any(
            char == word[index - 1]
            and char in RUSSIAN_CONSONANTS
            and word[:index] + word[index + 1 :] == replacement
            for index, char in enumerate(word)
            if index > 0
        )
    return False


def _is_keyboard_typo(word: str, replacement: str) -> bool:
    if len(word) != len(replacement):
        return False
    differences = [(left, right) for left, right in zip(word, replacement) if left != right]
    return len(differences) == 1 and _keyboard_neighbors(*differences[0])


def _keyboard_neighbors(left: str, right: str) -> bool:
    left_position = RUSSIAN_KEY_POSITIONS.get(left)
    right_position = RUSSIAN_KEY_POSITIONS.get(right)
    if left_position is None or right_position is None:
        return False
    return abs(left_position[0] - right_position[0]) <= 1 and abs(left_position[1] - right_position[1]) <= 1


def _is_adjacent_swap(word: str, replacement: str) -> bool:
    if len(word) != len(replacement) or word == replacement:
        return False
    differences = [index for index, (left, right) in enumerate(zip(word, replacement)) if left != right]
    if len(differences) != 2:
        return False
    first, second = differences
    return second == first + 1 and word[first] == replacement[second] and word[second] == replacement[first]


def _one_deletion_matches(longer: str, shorter: str) -> bool:
    if len(longer) != len(shorter) + 1:
        return False
    return any(longer[:index] + longer[index + 1 :] == shorter for index in range(len(longer)))


def _yo_e_variants(word: str) -> list[str]:
    variants: list[str] = []
    seen: set[str] = set()
    for index, char in enumerate(word):
        replacement_char = "ё" if char == "е" else "е" if char == "ё" else ""
        if not replacement_char:
            continue
        variant = word[:index] + replacement_char + word[index + 1 :]
        if variant in seen:
            continue
        seen.add(variant)
        variants.append(variant)
    return variants


def _extend_limited(
    results: list[DictionaryCandidateSpec],
    seen: set[str],
    candidates: Sequence[DictionaryCandidateSpec],
    limit: int,
) -> None:
    for candidate in candidates:
        if candidate.replacement in seen:
            continue
        seen.add(candidate.replacement)
        results.append(candidate)
        if len(results) >= limit:
            return


def _passes_edit_distance_override(word: str, replacement: str) -> bool:
    if len(word) < 5 or len(replacement) < 5:
        return False
    if word[:1] != replacement[:1]:
        return False
    if abs(len(word) - len(replacement)) > 2:
        return False
    distance = DamerauLevenshtein.distance(word, replacement)
    if len(word) >= 8:
        return distance <= 2
    return distance <= 1


def _length_compatible(word: str, replacement: str) -> bool:
    return abs(len(word) - len(replacement)) <= max(2, len(word) // 3)


def _max_edit_distance(word: str) -> int:
    if len(word) <= 4:
        return 1
    if len(word) <= 7:
        return 2
    return 3


def _is_protected_literal(word: str) -> bool:
    stripped = word.strip()
    return any(
        span.start == 0 and span.end == len(stripped) and span.kind in PROTECTED_LITERAL_KINDS
        for span in find_protected_spans(stripped)
    )


def _normalize(word: str) -> str:
    return word.strip().lower()
