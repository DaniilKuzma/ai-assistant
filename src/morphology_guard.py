"""Morphological safety checks for Russian dictionary replacements."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from text_utils import normalize_word


MIN_PARSE_SCORE = 0.20
INFLECTABLE_POS = {"NOUN", "ADJF", "ADJS", "PRTF", "PRTS", "VERB", "INFN", "NUMR"}


@lru_cache(maxsize=1)
def _morph() -> Any | None:
    try:
        import pymorphy2

        try:
            return pymorphy2.MorphAnalyzer()
        except ModuleNotFoundError as exc:
            if exc.name != "pkg_resources":
                raise
            import pymorphy2_dicts_ru

            return pymorphy2.MorphAnalyzer(path=pymorphy2_dicts_ru.get_path())
    except Exception:
        return None


@lru_cache(maxsize=50_000)
def _best_parse(word: str):
    morph = _morph()
    if morph is None:
        return None
    parses = morph.parse(normalize_word(word))
    if not parses:
        return None
    return parses[0]


def is_same_lemma_inflection(source: str, candidate: str) -> bool:
    """Return True when a replacement only changes a known grammatical form.

    The guard is intentionally narrow. It catches high-risk Russian inflection
    swaps while staying fail-open for typos, unknown words and missing pymorphy2.
    """
    source_norm = normalize_word(source)
    candidate_norm = normalize_word(candidate)
    if not source_norm or source_norm == candidate_norm:
        return False

    source_parse = _best_parse(source_norm)
    candidate_parse = _best_parse(candidate_norm)
    if source_parse is None or candidate_parse is None:
        return False
    if source_parse.score < MIN_PARSE_SCORE or candidate_parse.score < MIN_PARSE_SCORE:
        return False

    source_pos = source_parse.tag.POS
    candidate_pos = candidate_parse.tag.POS
    if source_pos != candidate_pos or source_pos not in INFLECTABLE_POS:
        return False

    return source_parse.normal_form == candidate_parse.normal_form


def is_morphological_dictionary_word(word: str) -> bool:
    """Return True for words that pymorphy2 found in the real dictionary.

    Guessed parses from FakeDictionary, UnknownPrefixAnalyzer and similar
    analyzers are deliberately not counted as dictionary words.
    """
    parse = _best_parse(normalize_word(word))
    if parse is None or parse.score < MIN_PARSE_SCORE:
        return False
    if not parse.tag.POS:
        return False

    stack = getattr(parse, "methods_stack", ())
    if len(stack) != 1:
        return False
    analyzer = stack[0][0] if stack and stack[0] else None
    return analyzer.__class__.__name__ == "DictionaryAnalyzer"
