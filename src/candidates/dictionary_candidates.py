from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

from src.candidates.morphology import is_known_word
from src.preprocessing.protected_spans import find_protected_spans
from src.rules.base import RuleMode


RULE_ID = "dictionary_fuzzy"
TEST_LEXICON: tuple[str, ...] = ("корова", "молоко", "территория", "апелляция")

MIN_WORD_LENGTH = 4
MIN_FUZZY_SCORE = 65.0
MAX_EXTRACT_MULTIPLIER = 4
RUSSIAN_WORD_RE = re.compile(r"[а-яё]+", re.IGNORECASE)
PROTECTED_LITERAL_KINDS = frozenset({"url", "email", "number"})


@dataclass(frozen=True)
class DictionaryCandidateSpec:
    replacement: str
    edit_type: str = "spelling"
    confidence: float = 0.0
    requires_model: bool = True
    rule_id: str = RULE_ID
    mode: RuleMode = "model_required"


def dictionary_candidates(word: str, lexicon: Sequence[str], limit: int = 5) -> list[str]:
    return [candidate.replacement for candidate in dictionary_candidate_specs(word, lexicon, limit)]


def dictionary_candidate_specs(word: str, lexicon: Sequence[str], limit: int = 5) -> list[DictionaryCandidateSpec]:
    if limit <= 0 or not _is_eligible_source(word):
        return []

    normalized = _normalize(word)
    choices = _candidate_choices(normalized, lexicon)
    if not choices:
        return []

    matches = process.extract(
        normalized,
        choices,
        scorer=fuzz.WRatio,
        limit=max(limit * MAX_EXTRACT_MULTIPLIER, limit),
        score_cutoff=MIN_FUZZY_SCORE,
    )

    results: list[DictionaryCandidateSpec] = []
    seen: set[str] = set()
    for replacement, score, _index in matches:
        if replacement in seen:
            continue
        if not _is_acceptable_match(normalized, replacement):
            continue
        seen.add(replacement)
        results.append(DictionaryCandidateSpec(replacement=replacement, confidence=float(score) / 100.0))
        if len(results) >= limit:
            break
    return results


def _is_eligible_source(word: str) -> bool:
    normalized = _normalize(word)
    if len(normalized) < MIN_WORD_LENGTH:
        return False
    if not RUSSIAN_WORD_RE.fullmatch(normalized):
        return False
    if _is_protected_literal(word):
        return False
    return not is_known_word(normalized)


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


def _is_acceptable_match(word: str, replacement: str) -> bool:
    if not _length_compatible(word, replacement):
        return False
    if Levenshtein.distance(word, replacement) > _max_edit_distance(word):
        return False
    return is_known_word(replacement)


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
