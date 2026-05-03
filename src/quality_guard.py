"""Safety checks that prevent low-confidence or destructive corrections."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
import re


PROTECTED_RE = re.compile(
    r"(https?://\S+|www\.\S+|[\w.+-]+@[\w-]+\.[\w.-]+|[A-Za-z][A-Za-z0-9._/-]*|\b\d[\d.,:/-]*\b|[A-ZА-ЯЁ]{2,})"
)
STRUCTURAL_PUNCT = set('"()[]{}—–«»№/\\')
SIMPLE_PUNCT = set(".,!?;:")


@dataclass
class GuardDecision:
    accepted: bool
    reason: str


class QualityGuard:
    def __init__(
        self,
        min_confidence: float = 0.45,
        max_change_ratio: float = 0.35,
        max_length_delta_ratio: float = 0.25,
        max_simple_punct_drop: int = 2,
        min_punct_similarity: float = 0.50,
    ):
        self.min_confidence = min_confidence
        self.max_change_ratio = max_change_ratio
        self.max_length_delta_ratio = max_length_delta_ratio
        self.max_simple_punct_drop = max_simple_punct_drop
        self.min_punct_similarity = min_punct_similarity

    def check(
        self,
        original: str,
        corrected: str,
        confidence: float,
        edit_count: int,
    ) -> GuardDecision:
        original = str(original)
        corrected = str(corrected)

        if not original.strip() or corrected.strip() == original.strip():
            return GuardDecision(True, "unchanged")

        if confidence < self.min_confidence:
            return GuardDecision(False, "low_confidence")

        if self._protected_values(original) != self._protected_values(corrected):
            return GuardDecision(False, "protected_value_changed")

        if self._structural_punctuation(original) != self._structural_punctuation(corrected):
            return GuardDecision(False, "structural_punctuation_changed")

        if self._simple_punctuation_drop_too_large(original, corrected):
            return GuardDecision(False, "punctuation_drop")

        base_len = max(len(original), 1)
        length_delta = abs(len(corrected) - len(original)) / base_len
        if length_delta > self.max_length_delta_ratio:
            return GuardDecision(False, "length_delta")

        if self._without_spacing(original) == self._without_spacing(corrected):
            return GuardDecision(True, "spacing_only")

        change_ratio = levenshtein_distance(original, corrected) / base_len
        if change_ratio > self.max_change_ratio:
            return GuardDecision(False, "too_many_changes")

        if edit_count <= 0:
            return GuardDecision(True, "no_edits")

        return GuardDecision(True, "accepted")

    @staticmethod
    def _protected_values(text: str) -> list[str]:
        return [m.group(0) for m in PROTECTED_RE.finditer(text)]

    @staticmethod
    def _structural_punctuation(text: str) -> str:
        return "".join(ch for ch in str(text) if ch in STRUCTURAL_PUNCT)

    @staticmethod
    def _without_spacing(text: str) -> str:
        return "".join(str(text).split())

    def _simple_punctuation_drop_too_large(self, original: str, corrected: str) -> bool:
        orig = "".join(ch for ch in str(original) if ch in SIMPLE_PUNCT)
        corr = "".join(ch for ch in str(corrected) if ch in SIMPLE_PUNCT)
        if not orig:
            return False

        drop = len(orig) - len(corr)
        allowed_drop = max(self.max_simple_punct_drop, int(len(orig) * 0.35))
        if drop > allowed_drop:
            return True

        if len(orig) >= 3:
            similarity = SequenceMatcher(None, orig, corr).ratio()
            if similarity < self.min_punct_similarity:
                return True

        orig_counts = Counter(orig)
        corr_counts = Counter(corr)
        for ch, count in orig_counts.items():
            if count - corr_counts.get(ch, 0) > allowed_drop:
                return True
        return False


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current
    return previous[-1]
