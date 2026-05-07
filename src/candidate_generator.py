"""Dictionary and rule based candidate generation for Russian correction."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
import math
import pickle
from typing import Dict, Iterable, List, Sequence

from text_utils import apply_case_like, is_word, normalize_word, word_tokens


RUSSIAN_ALPHABET = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
RUSSIAN_CONSONANTS = "бвгджзйклмнпрстфхцчшщ"
RUSSIAN_KEYBOARD_ROWS = (
    "йцукенгшщзхъ",
    "фывапролджэ",
    "ячсмитьбю",
)
ORTHOGRAPHIC_CHAR_CONFUSIONS = {
    "а": ("о", "я"),
    "о": ("а",),
    "я": ("а",),
    "е": ("и",),
    "и": ("е",),
    "э": ("е",),
    "ю": ("у",),
    "у": ("ю",),
    "ы": ("и",),
}
SOURCE_SCORE_BOOSTS = {
    "rule": 6.0,
    "phrase_safe": 6.0,
    "mined": 5.0,
    "orthographic": 3.5,
    "keyboard": 3.0,
    "split": 0.5,
    "dictionary": 0.0,
}
SPLIT_PREFIX_BLOCKLIST = {
    "само",
    "сверх",
    "пост",
    "супер",
    "тур",
    "мат",
    "зам",
    "зампред",
    "трехсот",
    "много",
    "спец",
    "гос",
    "авто",
    "кино",
    "фото",
    "радио",
    "теле",
    "евро",
    "мини",
    "микро",
}
SPLIT_SERVICE_WHITELIST = {
    ("из", "них"),
    ("в", "них"),
    ("к", "ним"),
    ("с", "ним"),
    ("тот", "же"),
    ("так", "же"),
}
SPLIT_SHORT_LEFT_WHITELIST = {"в", "во", "к", "ко", "с", "со", "о", "об", "от", "из", "на", "по", "при", "за", "до"}
SPLIT_SHORT_RIGHT_WHITELIST = {"об", "от", "из", "их", "им", "ей", "её", "ее", "его", "ее", "же", "ли", "бы"}
SPLIT_SERVICE_LEFT_WHITELIST = {"не", "ни", "их"}
SPLIT_SERVICE_RIGHT_WHITELIST = {"в", "во", "на", "не", "ни", "о", "об", "обо", "по", "за", "до", "к", "ко", "с", "со", "из", "от"}
SHORT_SERVICE_SPLIT_LEFT = {"в", "во", "к", "ко", "с", "со", "о", "об", "у"}
VOWELS = set("аеёиоуыэюя")


def _keyboard_neighbors() -> Dict[str, tuple[str, ...]]:
    positions: dict[str, tuple[int, int]] = {}
    for row_index, row in enumerate(RUSSIAN_KEYBOARD_ROWS):
        for col_index, char in enumerate(row):
            positions[char] = (row_index, col_index)

    neighbors: dict[str, set[str]] = {char: set() for char in positions}
    for char, (row_index, col_index) in positions.items():
        for other, (other_row, other_col) in positions.items():
            if other == char:
                continue
            if abs(row_index - other_row) <= 1 and abs(col_index - other_col) <= 1:
                neighbors[char].add(other)
    return {char: tuple(sorted(values)) for char, values in neighbors.items()}


RUSSIAN_KEYBOARD_NEIGHBORS = _keyboard_neighbors()


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
        long_oov_max_distance: int = 2,
        long_oov_min_length: int = 8,
        long_oov_max_bucket_size: int = 2000,
        enable_keyboard_candidates: bool = True,
        enable_orthographic_candidates: bool = True,
        enable_mined_confusions: bool = True,
    ):
        self.min_freq = min_freq
        self.max_distance = max_distance
        self.max_bucket_size = max_bucket_size
        self.allow_split_candidates = allow_split_candidates
        self.safe_split_only = safe_split_only
        self.min_dictionary_word_length = min_dictionary_word_length
        self.long_oov_max_distance = max(1, int(long_oov_max_distance))
        self.long_oov_min_length = max(1, int(long_oov_min_length))
        self.long_oov_max_bucket_size = max(1, int(long_oov_max_bucket_size))
        self.enable_keyboard_candidates = bool(enable_keyboard_candidates)
        self.enable_orthographic_candidates = bool(enable_orthographic_candidates)
        self.enable_mined_confusions = bool(enable_mined_confusions)
        self.unsafe_split_blocked_count = 0
        self.frequencies: Counter[str] = Counter()
        self.length_buckets: Dict[int, List[str]] = defaultdict(list)
        self._candidate_cache: Dict[tuple[object, ...], tuple[Candidate, ...]] = {}
        self.mined_confusions: Dict[str, Sequence[str]] = {}
        self.mined_confusion_counts: Counter[tuple[str, str]] = Counter()

        self.exact_confusions: Dict[str, Sequence[str]] = {
            "фсе": ("все",),
            "щто": ("что",),
            "што": ("что",),
            "ыть": ("быть",),
            "эсть": ("есть",),
            "чтто": ("что",),
            "ччто": ("что",),
            "дэло": ("дело",),
            "ллет": ("лет",),
            "ууже": ("уже",),
            "паралон": ("поролон",),
            "координально": ("кардинально",),
            "расказать": ("рассказать",),
            "расчитать": ("рассчитать",),
            "железно-дорожный": ("железнодорожный",),
            "москва": ("Москва",),
            "президент": ("Президент",),
            "вуз": ("ВУЗ",),
        }

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
            "по новому": ("по-новому",),
            "в виду": ("ввиду",),
            "мини футбол": ("мини-футбол",),
            "вобщем": ("в общем",),
            "вообщем": ("в общем",),
            "какбудто": ("как будто",),
            "почемуто": ("почему-то",),
            "потомучто": ("потому что",),
            "таккак": ("так как",),
            "несмотряна": ("несмотря на",),
            "изза": ("из-за",),
        }

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        min_freq: int = 1,
        max_distance: int = 1,
        allow_split_candidates: bool = False,
        safe_split_only: bool = True,
        long_oov_max_distance: int = 2,
        long_oov_min_length: int = 8,
        enable_keyboard_candidates: bool = True,
        enable_orthographic_candidates: bool = True,
        enable_mined_confusions: bool = True,
    ) -> "CandidateGenerator":
        generator = cls(
            min_freq=min_freq,
            max_distance=max_distance,
            allow_split_candidates=allow_split_candidates,
            safe_split_only=safe_split_only,
            long_oov_max_distance=long_oov_max_distance,
            long_oov_min_length=long_oov_min_length,
            enable_keyboard_candidates=enable_keyboard_candidates,
            enable_orthographic_candidates=enable_orthographic_candidates,
            enable_mined_confusions=enable_mined_confusions,
        )
        generator.fit(texts)
        return generator

    def fit(self, texts: Iterable[str]) -> None:
        self._candidate_cache = {}
        for text in texts:
            for word in word_tokens(str(text)):
                norm = normalize_word(word)
                if len(norm) > 1:
                    self.frequencies[norm] += 1
        self._build_buckets()

    def fit_mined_confusions(
        self,
        error_texts: Iterable[str],
        correct_texts: Iterable[str],
        *,
        min_count: int = 2,
    ) -> dict[str, int]:
        """Mine conservative source->target pairs from the training split only."""
        self._candidate_cache = {}
        pair_counts: Counter[tuple[str, str]] = Counter()
        stats = {"pairs": 0, "replace_segments": 0, "accepted": 0}

        for error_text, correct_text in zip(error_texts, correct_texts):
            stats["pairs"] += 1
            src_words = word_tokens(str(error_text))
            tgt_words = word_tokens(str(correct_text))
            if not src_words or not tgt_words:
                continue
            src_norm = [normalize_word(word) for word in src_words]
            tgt_norm = [normalize_word(word) for word in tgt_words]
            matcher = SequenceMatcher(None, src_norm, tgt_norm)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag != "replace" or i2 - i1 != 1 or not (1 <= j2 - j1 <= 3):
                    continue
                stats["replace_segments"] += 1
                source_norm = src_norm[i1]
                target_text = " ".join(tgt_norm[j1:j2])
                if not self._is_mineable_confusion(source_norm, target_text):
                    continue
                pair_counts[(source_norm, target_text)] += 1

        min_count = max(1, int(min_count))
        mined: dict[str, list[str]] = defaultdict(list)
        for (source_norm, target_text), count in pair_counts.items():
            if count < min_count:
                continue
            mined[source_norm].append(target_text)
            stats["accepted"] += 1

        for source_norm, targets in mined.items():
            targets.sort(
                key=lambda item: (
                    -pair_counts[(source_norm, item)],
                    -sum(math.log1p(self.frequencies.get(part, 0)) for part in item.split()),
                    item,
                )
            )

        self.mined_confusions = {key: tuple(values[:8]) for key, values in mined.items()}
        self.mined_confusion_counts = pair_counts
        return stats

    def _is_mineable_confusion(self, source_norm: str, target_text: str) -> bool:
        if not source_norm or not target_text:
            return False
        if source_norm == normalize_word(target_text):
            return False
        if len(source_norm) < 3 or any(ch.isdigit() for ch in source_norm):
            return False
        if self.is_known(source_norm):
            return False
        parts = [normalize_word(part) for part in target_text.split()]
        if not parts:
            return False
        if len(parts) > 1:
            allowed = {normalize_word(item) for item in self.phrase_confusions.get(source_norm, ())}
            return normalize_word(target_text) in allowed
        target_norm = parts[0]
        if self.frequencies.get(target_norm, 0) < self.min_freq:
            return False
        if abs(len(source_norm) - len(target_norm)) > 2:
            return False
        if bounded_damerau_levenshtein(source_norm, target_norm, 2) <= 2:
            return True
        return source_norm[:2] == target_norm[:2] and source_norm[-2:] == target_norm[-2:]

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
        if not hasattr(generator, "long_oov_max_distance"):
            generator.long_oov_max_distance = 2
        if not hasattr(generator, "long_oov_min_length"):
            generator.long_oov_min_length = 8
        if not hasattr(generator, "long_oov_max_bucket_size"):
            generator.long_oov_max_bucket_size = 2000
        if not hasattr(generator, "enable_keyboard_candidates"):
            generator.enable_keyboard_candidates = True
        if not hasattr(generator, "enable_orthographic_candidates"):
            generator.enable_orthographic_candidates = True
        if not hasattr(generator, "enable_mined_confusions"):
            generator.enable_mined_confusions = True
        if not hasattr(generator, "unsafe_split_blocked_count"):
            generator.unsafe_split_blocked_count = 0
        if not hasattr(generator, "mined_confusions"):
            generator.mined_confusions = {}
        if not hasattr(generator, "mined_confusion_counts"):
            generator.mined_confusion_counts = Counter()
        if not hasattr(generator, "exact_confusions"):
            generator.exact_confusions = {}
        if not hasattr(generator, "_candidate_cache"):
            generator._candidate_cache = {}
        generator.exact_confusions.update(
            {
                "фсе": ("все",),
                "щто": ("что",),
                "што": ("что",),
                "ыть": ("быть",),
                "эсть": ("есть",),
                "чтто": ("что",),
                "ччто": ("что",),
                "дэло": ("дело",),
                "ллет": ("лет",),
                "ууже": ("уже",),
                "паралон": ("поролон",),
                "координально": ("кардинально",),
                "расказать": ("рассказать",),
                "расчитать": ("рассчитать",),
                "железно-дорожный": ("железнодорожный",),
                "москва": ("Москва",),
                "президент": ("Президент",),
                "вуз": ("ВУЗ",),
            }
        )
        if not hasattr(generator, "phrase_confusions"):
            generator.phrase_confusions = {}
        generator.phrase_confusions.update(
            {
                "по новому": ("по-новому",),
                "в виду": ("ввиду",),
                "мини футбол": ("мини-футбол",),
                "вобщем": ("в общем",),
                "вообщем": ("в общем",),
                "какбудто": ("как будто",),
                "почемуто": ("почему-то",),
                "потомучто": ("потому что",),
                "таккак": ("так как",),
                "несмотряна": ("несмотря на",),
                "изза": ("из-за",),
            }
        )
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
        if norm in self.exact_confusions:
            return True
        if norm in self.phrase_confusions:
            return True
        if self._short_repeated_first_candidate(norm) is not None:
            return True
        return any(key in norm for key in self.single_token_confusions)

    def get_candidates(
        self,
        word: str,
        max_candidates: int = 8,
        include_known_dictionary: bool = False,
        allow_long_oov: bool = True,
    ) -> List[Candidate]:
        norm = normalize_word(word)
        if len(norm) < 3 or any(ch.isdigit() for ch in norm):
            return []
        cache_key = (
            str(word),
            int(max_candidates),
            bool(include_known_dictionary),
            bool(allow_long_oov),
            bool(getattr(self, "enable_keyboard_candidates", True)),
            bool(getattr(self, "enable_orthographic_candidates", True)),
            bool(getattr(self, "enable_mined_confusions", True)),
            len(getattr(self, "mined_confusions", {})),
        )
        cache = getattr(self, "_candidate_cache", None)
        if cache is None:
            self._candidate_cache = {}
            cache = self._candidate_cache
        if cache_key in cache:
            return list(cache[cache_key])

        candidates: Dict[str, Candidate] = {}

        if self._has_rule_candidate(norm):
            for cand, source in self._rule_candidate_items(norm):
                self._add_candidate(candidates, cand, word, source, distance=1)

        if not self.is_known(norm):
            if getattr(self, "enable_mined_confusions", True):
                for cand in self.mined_confusions.get(norm, ()):
                    self._add_candidate(candidates, cand, word, "mined", distance=1)
            if getattr(self, "enable_orthographic_candidates", True):
                for cand in self._orthographic_candidates(norm):
                    self._add_candidate(candidates, cand, word, "orthographic", distance=1)
            if getattr(self, "enable_keyboard_candidates", True):
                for cand in self._keyboard_candidates(norm):
                    self._add_candidate(candidates, cand, word, "keyboard", distance=1)

        if (include_known_dictionary or not self.is_known(norm)) and len(norm) >= self.min_dictionary_word_length:
            if self.max_distance <= 1:
                dictionary_candidates = self._edit_distance_one_candidates_fast(norm)
            else:
                dictionary_candidates = self._edit_distance_candidates(norm)
            for cand, distance in dictionary_candidates:
                self._add_candidate(candidates, cand, word, "dictionary", distance=distance)
            if allow_long_oov and self.long_oov_max_distance > 1 and len(norm) >= self.long_oov_min_length:
                for cand, distance in self._long_oov_edit_candidates(norm):
                    self._add_candidate(candidates, cand, word, "dictionary", distance=distance)

        if self.allow_split_candidates and not self.is_known(norm):
            split_candidates = (
                self._safe_split_candidates(norm, word)
                if self.safe_split_only
                else self._split_candidates(norm)
            )
            for cand in split_candidates:
                self._add_candidate(candidates, cand, word, "split", distance=1)

        result = [
            c
            for key, c in candidates.items()
            if normalize_word(key) != norm or str(c.text) != str(word)
        ]
        result.sort(key=lambda c: c.score, reverse=True)
        result = result[:max_candidates]
        cache[cache_key] = tuple(result)
        return result

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
        is_allowed_split = source == "split" and self._is_allowed_service_split_pair(parts)
        if (
            source not in {"rule", "phrase_safe"}
            and not is_allowed_split
            and not all(self.frequencies.get(part, 0) >= self.min_freq for part in parts)
        ):
            return

        freq_score = sum(math.log1p(self.frequencies.get(part, 0)) for part in parts)
        score = freq_score - (distance * 2.0) + SOURCE_SCORE_BOOSTS.get(source, 0.0)
        if source == "mined":
            score += 0.5 * math.log1p(self.mined_confusion_counts.get((normalize_word(source_word), normalize_word(candidate)), 0))
        if source == "split":
            if is_allowed_split:
                score += 3.0

        text = apply_case_like(candidate, source_word)
        old = target.get(text)
        new = Candidate(text=text, score=score, source=source, distance=distance)
        if old is None or new.score > old.score:
            target[text] = new

    def _rule_candidate_items(self, norm: str) -> Iterable[tuple[str, str]]:
        if norm in self.exact_confusions:
            for candidate in self.exact_confusions[norm]:
                yield candidate, "rule"

        if norm in self.phrase_confusions:
            for candidate in self.phrase_confusions[norm]:
                yield candidate, "phrase_safe"

        repeated_first = self._short_repeated_first_candidate(norm)
        if repeated_first is not None:
            yield repeated_first, "rule"

        for key, replacements in self.single_token_confusions.items():
            start = 0
            while True:
                pos = norm.find(key, start)
                if pos == -1:
                    break
                for repl in replacements:
                    yield norm[:pos] + repl + norm[pos + len(key) :], "orthographic"
                start = pos + 1

    def _rule_candidates(self, norm: str) -> Iterable[str]:
        for candidate, _ in self._rule_candidate_items(norm):
            yield candidate

    def _short_repeated_first_candidate(self, norm: str) -> str | None:
        if len(norm) < 4 or len(norm) > 5:
            return None
        if norm[0] != norm[1]:
            return None
        candidate = norm[1:]
        if len(candidate) > 4:
            return None
        if self.frequencies.get(candidate, 0) < self.min_freq:
            return None
        return candidate

    def _orthographic_candidates(self, norm: str) -> Iterable[str]:
        seen: set[str] = set()

        def emit(candidate: str) -> str | None:
            if candidate == norm or candidate in seen:
                return None
            if self.frequencies.get(candidate, 0) < self.min_freq:
                return None
            seen.add(candidate)
            return candidate

        for i, char in enumerate(norm):
            for replacement in ORTHOGRAPHIC_CHAR_CONFUSIONS.get(char, ()):
                candidate = emit(norm[:i] + replacement + norm[i + 1 :])
                if candidate is not None:
                    yield candidate

        for i, char in enumerate(norm):
            if char not in RUSSIAN_CONSONANTS:
                continue
            candidate = emit(norm[:i] + char + norm[i:])
            if candidate is not None:
                yield candidate

        for i in range(len(norm) - 1):
            if norm[i] != norm[i + 1] or norm[i] not in RUSSIAN_CONSONANTS:
                continue
            candidate = emit(norm[:i] + norm[i + 1 :])
            if candidate is not None:
                yield candidate

    def _keyboard_candidates(self, norm: str) -> Iterable[str]:
        seen: set[str] = set()
        for i, char in enumerate(norm):
            for replacement in RUSSIAN_KEYBOARD_NEIGHBORS.get(char, ()):
                candidate = norm[:i] + replacement + norm[i + 1 :]
                if candidate == norm or candidate in seen:
                    continue
                if self.frequencies.get(candidate, 0) < self.min_freq:
                    continue
                seen.add(candidate)
                yield candidate

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

    def _long_oov_edit_candidates(self, norm: str) -> Iterable[tuple[str, int]]:
        """Scan a bounded frequency bucket for long OOV distance-2 candidates."""
        max_distance = max(2, int(self.long_oov_max_distance))
        min_len = max(1, len(norm) - max_distance)
        max_len = len(norm) + max_distance
        seen: set[str] = set()
        for length in range(min_len, max_len + 1):
            for candidate in self.length_buckets.get(length, [])[: self.long_oov_max_bucket_size]:
                if candidate in seen:
                    continue
                distance = bounded_damerau_levenshtein(norm, candidate, max_distance)
                if 1 < distance <= max_distance:
                    seen.add(candidate)
                    yield candidate, distance

    def _split_candidates(self, norm: str) -> Iterable[str]:
        for split_at in range(2, len(norm) - 1):
            left = norm[:split_at]
            right = norm[split_at:]
            if self.is_known(left) and self.is_known(right):
                yield f"{left} {right}"

    def _safe_split_candidates(self, norm: str, source_word: str) -> Iterable[str]:
        """Yield conservative two-word split candidates.

        This is intentionally narrower than the legacy split path. It targets
        curated lexical two-word splits while avoiding grammatical suffix
        artifacts such as ``говориться -> говорить ся``.
        """
        source = str(source_word or "")
        if len(norm) < 4:
            return
        source_is_titlecase = bool(source[:1].isupper() and source[1:] == source[1:].lower() and not source.isupper())
        if source.isupper() or (source != source.lower() and not source_is_titlecase):
            return
        if any("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in source):
            return
        if any(ch.isdigit() for ch in source):
            return
        if not is_word(source):
            return
        if norm.endswith(("ться", "тся", "ся")):
            return
        if len(norm) >= 4 and norm[0] == norm[1] and self.frequencies.get(norm[1:], 0) >= self.min_freq:
            self.unsafe_split_blocked_count += 1
            return

        for split_at in range(1, len(norm) - 1):
            left = norm[:split_at]
            right = norm[split_at:]
            pair = (left, right)
            if pair in SPLIT_SERVICE_WHITELIST:
                yield f"{left} {right}"
                continue
            if self._is_unsafe_short_service_split(left, right):
                self.unsafe_split_blocked_count += 1
                continue
            if self._is_allowed_service_split_pair([left, right]):
                yield f"{left} {right}"
                continue
            if len(left) < 3 or len(right) < 3:
                continue
            if left in SPLIT_PREFIX_BLOCKLIST or right in {"ся", "сь"}:
                continue
            if self.is_known(left) and self.is_known(right):
                yield f"{left} {right}"

    @staticmethod
    def _is_unsafe_short_service_split(left: str, right: str) -> bool:
        if left in SHORT_SERVICE_SPLIT_LEFT and (len(right) < 5 or right[:1] in VOWELS):
            return True
        if right in {"в", "к", "с", "о", "у"} and len(left) < 5:
            return True
        return False

    def _is_allowed_service_split_pair(self, parts: Sequence[str]) -> bool:
        if len(parts) != 2:
            return False
        left, right = parts
        pair = (left, right)
        if pair in SPLIT_SERVICE_WHITELIST:
            return True
        if left in SPLIT_PREFIX_BLOCKLIST:
            return False
        if left in SPLIT_SERVICE_LEFT_WHITELIST and len(right) >= 5 and self.is_known(right):
            return True
        if right in SPLIT_SERVICE_RIGHT_WHITELIST and len(left) >= 5 and self.is_known(left):
            return True
        if left in SPLIT_SHORT_LEFT_WHITELIST and len(right) >= 3 and self.is_known(right):
            return True
        if right in SPLIT_SHORT_RIGHT_WHITELIST and len(left) >= 3 and self.is_known(left):
            return True
        return False


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
