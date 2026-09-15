from __future__ import annotations

from functools import lru_cache
import re
from typing import Any


RUSSIAN_WORD_RE = re.compile(r"[а-яё]+", re.IGNORECASE)


@lru_cache(maxsize=1)
def morph_analyzer() -> Any:
    try:
        from pymorphy3 import MorphAnalyzer
    except ImportError:  # pragma: no cover - compatibility for older local envs.
        from pymorphy2 import MorphAnalyzer  # type: ignore[no-redef]

    return MorphAnalyzer()


@lru_cache(maxsize=100_000)
def parses(word: str) -> tuple[Any, ...]:
    normalized = word.lower()
    if not RUSSIAN_WORD_RE.fullmatch(normalized):
        return ()
    return tuple(morph_analyzer().parse(normalized))


@lru_cache(maxsize=100_000)
def is_known_word(word: str) -> bool:
    return any(getattr(parse, "is_known", False) for parse in parses(word))


@lru_cache(maxsize=100_000)
def has_pos(word: str, poses: frozenset[str]) -> bool:
    return any(getattr(parse, "is_known", False) and getattr(parse.tag, "POS", None) in poses for parse in parses(word))


@lru_cache(maxsize=100_000)
def normal_forms(word: str) -> frozenset[str]:
    return frozenset(str(getattr(parse, "normal_form", "")) for parse in parses(word) if getattr(parse, "is_known", False))
