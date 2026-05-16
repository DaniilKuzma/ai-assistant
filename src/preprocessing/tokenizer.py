from __future__ import annotations

from dataclasses import dataclass
import re


TOKEN_RE = re.compile(r"[А-Яа-яЁёA-Za-z]+(?:-[А-Яа-яЁёA-Za-z]+)*|\d+(?:[.,:/-]\d+)*|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*", re.UNICODE)
RUSSIAN_RE = re.compile(r"[А-Яа-яЁё]", re.UNICODE)
PUNCTUATION = set(",.!?:;—…\"'()[]«»")


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int

    @property
    def is_word(self) -> bool:
        return bool(WORD_RE.fullmatch(self.text))

    @property
    def is_punctuation(self) -> bool:
        return self.text in PUNCTUATION


def tokenize(text: str) -> list[Token]:
    return [Token(match.group(0), match.start(), match.end()) for match in TOKEN_RE.finditer(text)]


def tokenize_words(text: str) -> list[Token]:
    return [token for token in tokenize(text) if token.is_word]


def has_russian(text: str) -> bool:
    return bool(RUSSIAN_RE.search(text))
