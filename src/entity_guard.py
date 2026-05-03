"""Runtime lexical/entity guards for conservative correction."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from text_utils import normalize_word, word_tokens


ENTITY_CONTEXT_PREV = {
    "город",
    "городе",
    "города",
    "городу",
    "деревня",
    "деревне",
    "житель",
    "жителя",
    "жительница",
    "жители",
    "имени",
    "крае",
    "области",
    "поселке",
    "поселок",
    "посёлке",
    "посёлок",
    "районе",
    "республике",
    "селе",
    "словам",
    "улица",
    "улице",
}

ENTITY_CONTEXT_BIGRAMS = {
    ("по", "словам"),
}


def _default_paths() -> list[Path]:
    return [
        Path("data/raw/texts.txt"),
        Path("data/processed/dataset.csv"),
        Path("data/processed/train.csv"),
        Path("data/processed/val.csv"),
        Path("data/processed/test.csv"),
    ]


def _looks_like_titlecase(word: str) -> bool:
    text = str(word or "")
    return bool(text[:1].isupper() and not text.isupper())


def _looks_synthetic_error(word: str) -> bool:
    norm = normalize_word(word)
    if len(norm) < 5:
        return False
    repeated = sum(1 for a, b in zip(norm, norm[1:]) if a == b)
    if repeated >= 2:
        return True
    return any(pair in norm for pair in ("ьь", "ъъ", "йй", "ёё"))


@dataclass
class ProtectedLexicon:
    """Small frequency lexicon loaded from trusted clean/raw project texts."""

    frequencies: Counter[str] = field(default_factory=Counter)
    paths: tuple[str, ...] = ()

    @classmethod
    def empty(cls) -> "ProtectedLexicon":
        return cls()

    @classmethod
    def from_default_paths(cls) -> "ProtectedLexicon":
        return cls.from_paths(_default_paths())

    @classmethod
    def from_paths(cls, paths: Sequence[str | Path] | None) -> "ProtectedLexicon":
        if paths is None:
            paths = _default_paths()
        counter: Counter[str] = Counter()
        used_paths: list[str] = []
        for raw_path in paths:
            path = Path(raw_path)
            if not path.exists():
                continue
            used_paths.append(str(path))
            try:
                if path.suffix.lower() == ".csv":
                    cls._add_csv(counter, path)
                else:
                    cls._add_texts(counter, path.read_text(encoding="utf-8").splitlines())
            except Exception:
                continue
        return cls(counter, tuple(used_paths))

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> "ProtectedLexicon":
        counter: Counter[str] = Counter()
        cls._add_texts(counter, texts)
        return cls(counter)

    @staticmethod
    def _add_texts(counter: Counter[str], texts: Iterable[str]) -> None:
        for text in texts:
            for word in word_tokens(str(text)):
                norm = normalize_word(word)
                if norm:
                    counter[norm] += 1

    @staticmethod
    def _add_csv(counter: Counter[str], path: Path) -> None:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            if "correct_text" not in fieldnames:
                return
            for row in reader:
                text = row.get("correct_text", "")
                ProtectedLexicon._add_texts(counter, [text])

    @property
    def size(self) -> int:
        return len(self.frequencies)

    def frequency(self, word: str) -> int:
        return int(self.frequencies.get(normalize_word(word), 0))

    def is_protected_source(self, word: str) -> bool:
        freq = self.frequency(word)
        if freq <= 0:
            return False
        if _looks_synthetic_error(word):
            return False
        if _looks_like_titlecase(word):
            return True
        norm = normalize_word(word)
        return len(norm) >= 6 or freq <= 3

    def is_entity_context(self, word: str, prev_word: str = "", next_word: str = "") -> bool:
        prev_norm = normalize_word(prev_word)
        next_norm = normalize_word(next_word)
        if (prev_norm, normalize_word(word)) in ENTITY_CONTEXT_BIGRAMS:
            return True
        if prev_norm in ENTITY_CONTEXT_PREV:
            return True
        if prev_norm == "словам":
            return True
        if _looks_like_titlecase(word) and (_looks_like_titlecase(prev_word) or _looks_like_titlecase(next_word)):
            return True
        if _looks_like_titlecase(word) and next_norm in {"области", "края", "района", "улицы"}:
            return True
        return False

    def metadata(self, word: str, prev_word: str = "", next_word: str = "") -> dict[str, object]:
        frequency = self.frequency(word)
        return {
            "protected_source": self.is_protected_source(word),
            "entity_context": self.is_entity_context(word, prev_word, next_word),
            "clean_lexicon_frequency": frequency,
        }
