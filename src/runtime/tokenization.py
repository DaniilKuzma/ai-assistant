from __future__ import annotations

import re

from src.schema import WordToken


WORD_RE = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*", re.UNICODE)


def tokenize_runtime_words(text: str) -> list[WordToken]:
    """Return Russian word tokens with character offsets for runtime edits."""

    return [
        WordToken(text=match.group(0), start=match.start(), end=match.end())
        for match in WORD_RE.finditer(text)
    ]


__all__ = ["tokenize_runtime_words"]
