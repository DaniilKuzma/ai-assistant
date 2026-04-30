"""Dictionary and rule based candidate generation for Russian correction.

This module intentionally does not implement keyboard-neighbor typos.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import pickle
from typing import Dict, Iterable, List, Sequence

from text_utils import apply_case_like, is_word, normalize_word, word_tokens


RUSSIAN_ALPHABET = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"


@dataclass(frozen=True)
class Candidate:
    text: str
    score: float
    source: str
    distance: int = 0


class CandidateGenerator:
    """Generate conservative spelling candidates from a corpus vocabulary."""

    def __init__(
        self,
        min_freq: int = 1,
        max_distance: int = 1,
        max_bucket_size: int = 5000,
        allow_split_candidates: bool = False,
        safe_split_only: bool = True,
        min_dictionary_word_length: int = 4,
    ):
        self.min_freq = min_freq
        self.max_distance = max_distance
        self.max_bucket_size = max_bucket_size
        self.allow_split_candidates = allow_split_candidates
        self.safe_split_only = safe_split_only
        self.min_dictionary_word_length = min_dictionary_word_length
        self.frequencies: Counter[str] = Counter()
        self.length_buckets: Dict[int, List[str]] = defaultdict(list)

        self.single_token_confusions: Dict[str, Sequence[str]] = {
            "тся": ("ться",),
            "ться": ("тся",),
            "ого": ("ово",),
            "его": ("ево",),
            "пре": ("при",),
            "при": ("пре",),
            "без": ("бес",),
            "бес": ("без",),
            "раз": ("рас",),
            "рас": ("раз",),
            "из": ("ис",),
            "ис": ("из",),
        }

        self.phrase_confusions: Dict[str, Sequence[str]] = {
            "также": ("так же",),
            "так же": ("также",),
            "чтобы": ("что бы",),
            "что бы": ("чтобы",),
            "зато": ("за то",),
            "за то": ("зато",),
            "причем": ("при чем",),
            "при чем": ("причем",),
            "притом": ("при том",),
            "при том": ("притом",),
            "вследствие": ("в следствие",),
            "в течение": ("в течении",),
            "в течении": ("в течение",),
        }

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        min_freq: int = 1,
        max_distance: int = 1,
        allow_split_candidates: bool = False,
        safe_split_only: bool = True,
    ) -> "CandidateGenerator":
        generator = cls(
            min_freq=min_freq,
            max_distance=max_distance,
            allow_split_candidates=allow_split_candidates,
            safe_split_only=safe_split_only,
        )
        generator.fit(texts)
        return generator

    def fit(self, texts: Iterable[str]) -> None:
        for text in texts:
            for word in word_tokens(str(text)):
                norm = normalize_word(word)
                if len(norm) > 1:
                    self.frequencies[norm] += 1
        self._build_buckets()

    def _build_buckets(self) -> None:
        self.length_buckets.clear()
        for word, freq in self.frequencies.items():
            if freq >= self.min_freq:
                self.length_buckets[len(word)].append(word)

        for length, words in list(self.length_buckets.items()):
            words.sort(key=lambda w: self.frequencies[w], reverse=True)
            self.length_buckets[length] = words[: self.max_bucket_size]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "CandidateGenerator":
        with open(path, "rb") as f:
            generator = pickle.load(f)
        if not hasattr(generator, "allow_split_candidates"):
            generator.allow_split_candidates = False
        if not hasattr(generator, "safe_split_only"):
            generator.safe_split_only = True
        if not hasattr(generator, "min_dictionary_word_length"):
            generator.min_dictionary_word_length = 4
        if getattr(generator, "max_distance", 2) > 1:
            generator.max_distance = 1
        return generator

    def is_known(self, word: str) -> bool:
        norm = normalize_word(word)
        return self.frequencies.get(norm, 0) >= self.min_freq

    def should_check(self, word: str) -> bool:
        norm = normalize_word(word)
        if len(norm) < 3:
            return False
        if any(ch.isdigit() for ch in norm):
            return False
        if not is_word(word):
            return False
        return not self.is_known(norm) or self._has_rule_candidate(norm)

    def _has_rule_candidate(self, norm: str) -> bool:
        if norm in self.phrase_confusions:
            return True
        return any(key in norm for key in self.single_token_confusions)

    def get_candidates(
        self,
        word: str,
        max_candidates: int = 8,
        include_known_dictionary: bool = False,
    ) -> List[Candidate]:
        norm = normalize_word(word)
        if len(norm) < 3 or any(ch.isdigit() for ch in norm):
            return []

        candidates: Dict[str, Candidate] = {}

        if self._has_rule_candidate(norm):
            for cand in self._rule_candidates(norm):
                self._add_candidate(candidates, cand, word, "rule", distance=1)

        if (include_known_dictionary or not self.is_known(norm)) and len(norm) >= self.min_dictionary_word_length:
            if self.max_distance <= 1:
                dictionary_candidates = self._edit_distance_one_candidates_fast(norm)
            else:
                dictionary_candidates = self._edit_distance_candidates(norm)
            for cand, distance in dictionary_candidates:
                self._add_candidate(candidates, cand, word, "dictionary", distance=distance)

        if self.allow_split_candidates and not self.is_known(norm):
            split_candidates = (
                self._safe_split_candidates(norm, word)
                if self.safe_split_only
                else self._split_candidates(norm)
            )
            for cand in split_candidates:
                self._add_candidate(candidates, cand, word, "split", distance=1)

        result = [c for key, c in candidates.items() if normalize_word(key) != norm]
        result.sort(key=lambda c: c.score, reverse=True)
        return result[:max_candidates]

    def best(
        self,
        word: str,
        min_score: float = 0.0,
        include_known_dictionary: bool = False,
    ) -> Candidate | None:
        candidates = self.get_candidates(
            word,
            max_candidates=1,
            include_known_dictionary=include_known_dictionary,
        )
        if not candidates:
            return None
        best = candidates[0]
        if best.score < min_score:
            return None
        return best

    def _add_candidate(
        self,
        target: Dict[str, Candidate],
        candidate: str,
        source_word: str,
        source: str,
        distance: int,
    ) -> None:
        candidate = candidate.strip()
        if not candidate:
            return

        parts = [normalize_word(part) for part in candidate.split()]
        if not parts:
            return
        if source != "rule" and not all(self.frequencies.get(part, 0) >= self.min_freq for part in parts):
            return

        freq_score = sum(math.log1p(self.frequencies.get(part, 0)) for part in parts)
        score = freq_score - (distance * 2.0)
        if source == "rule":
            score += 0.75
        if source == "split":
            score += 0.5

        text = apply_case_like(candidate, source_word)
        old = target.get(text)
        new = Candidate(text=text, score=score, source=source, distance=distance)
        if old is None or new.score > old.score:
            target[text] = new

    def _rule_candidates(self, norm: str) -> Iterable[str]:
        if norm in self.phrase_confusions:
            yield from self.phrase_confusions[norm]

        for key, replacements in self.single_token_confusions.items():
            start = 0
            while True:
                pos = norm.find(key, start)
                if pos == -1:
                    break
                for repl in replacements:
                    yield norm[:pos] + repl + norm[pos + len(key) :]
                start = pos + 1

    def _edit_distance_candidates(self, norm: str) -> Iterable[tuple[str, int]]:
        min_len = max(1, len(norm) - self.max_distance)
        max_len = len(norm) + self.max_distance
        for length in range(min_len, max_len + 1):
            for candidate in self.length_buckets.get(length, []):
                distance = bounded_damerau_levenshtein(norm, candidate, self.max_distance)
                if 0 < distance <= self.max_distance:
                    yield candidate, distance

    def _edit_distance_one_candidates_fast(self, norm: str) -> Iterable[tuple[str, int]]:
        """Generate Damerau-Levenshtein distance-1 neighbors by dictionary lookup.

        The old path scanned whole length buckets for every unknown token. That
        is acceptable for one-off runtime checks, but too slow for top-k
        training where every dirty token asks for candidates. Distance is fixed
        to 1 in the current project defaults, so generating all possible edits
        and checking the frequency table is much cheaper and exact enough for
        this mode.
        """
        seen: set[str] = set()

        def emit(candidate: str) -> tuple[str, int] | None:
            if candidate == norm or candidate in seen:
                return None
            if self.frequencies.get(candidate, 0) < self.min_freq:
                return None
            seen.add(candidate)
            return candidate, 1

        for i in range(len(norm)):
            item = emit(norm[:i] + norm[i + 1 :])
            if item is not None:
                yield item

        for i in range(len(norm) + 1):
            prefix = norm[:i]
            suffix = norm[i:]
            for ch in RUSSIAN_ALPHABET:
                item = emit(prefix + ch + suffix)
                if item is not None:
                    yield item

        for i, original in enumerate(norm):
            prefix = norm[:i]
            suffix = norm[i + 1 :]
            for ch in RUSSIAN_ALPHABET:
                if ch == original:
                    continue
                item = emit(prefix + ch + suffix)
                if item is not None:
                    yield item

        for i in range(len(norm) - 1):
            if norm[i] == norm[i + 1]:
                continue
            item = emit(norm[:i] + norm[i + 1] + norm[i] + norm[i + 2 :])
            if item is not None:
                yield item

    def _split_candidates(self, norm: str) -> Iterable[str]:
        for split_at in range(2, len(norm) - 1):
            left = norm[:split_at]
            right = norm[split_at:]
            if self.is_known(left) and self.is_known(right):
                yield f"{left} {right}"

    def _safe_split_candidates(self, norm: str, source_word: str) -> Iterable[str]:
        """Yield conservative two-word split candidates.

        This is intentionally narrower than the legacy split path. It targets
        glued whitespace errors while avoiding grammatical suffix artifacts such
        as ``говориться -> говорить ся``.
        """
        source = str(source_word or "")
        if len(norm) < 5:
            return
        if source != source.lower() or source.isupper():
            return
        if any("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in source):
            return
        if any(ch.isdigit() for ch in source):
            return
        if not is_word(source):
            return
        if norm.endswith(("ться", "тся", "ся")):
            return

        for split_at in range(2, len(norm) - 1):
            left = norm[:split_at]
            right = norm[split_at:]
            if len(right) < 2:
                continue
            if right in {"ся", "сь"}:
                continue
            if self.is_known(left) and self.is_known(right):
                yield f"{left} {right}"


def bounded_damerau_levenshtein(a: str, b: str, max_distance: int) -> int:
    """Damerau-Levenshtein distance with an early cutoff."""
    if abs(len(a) - len(b)) > max_distance:
        return max_distance + 1
    if a == b:
        return 0

    prev_prev = None
    prev = list(range(len(b) + 1))

    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        row_min = current[0]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            value = min(
                current[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + cost,
            )
            if (
                prev_prev is not None
                and i > 1
                and j > 1
                and ca == b[j - 2]
                and a[i - 2] == cb
            ):
                value = min(value, prev_prev[j - 2] + 1)
            current[j] = value
            row_min = min(row_min, value)

        if row_min > max_distance:
            return max_distance + 1
        prev_prev, prev = prev, current

    return prev[-1]
