"""Safety checks that prevent low-confidence or destructive corrections."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
import re

from text_utils import (
    TRAINABLE_PUNCT_LABELS,
    canonical_punctuation_label,
    extract_word_slots,
    normalize_word,
)


PROTECTED_RE = re.compile(
    r"(https?://\S+|www\.\S+|[\w.+-]+@[\w-]+\.[\w.-]+|[A-Za-z][A-Za-z0-9._/-]*|\b\d[\d.,:/-]*\b|[A-ZА-ЯЁ]{2,})"
)
STRUCTURAL_PUNCT = set('"()[]{}—–«»№/\\')
SIMPLE_PUNCT = set(".,!?;:")
STRUCTURAL_PAIRS = (("(", ")"), ("[", "]"), ("{", "}"), ("«", "»"))


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

        if (
            self._protected_values(original) != self._protected_values(corrected)
            and not self._protected_change_is_trainable_case(original, corrected)
        ):
            return GuardDecision(False, "protected_value_changed")

        if (
            self._structural_punctuation(original) != self._structural_punctuation(corrected)
            and not self._structural_change_is_trainable(original, corrected)
        ):
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
    def _protected_change_is_trainable_case(original: str, corrected: str) -> bool:
        original_slots = extract_word_slots(original)
        corrected_slots = extract_word_slots(corrected)
        if len(original_slots) != len(corrected_slots):
            return False
        allowed = {("вуз", "ВУЗ")}
        changed = [
            (source.word, target.word)
            for source, target in zip(original_slots, corrected_slots)
            if source.word != target.word
        ]
        if not changed:
            return False
        return all((normalize_word(source), target) in allowed for source, target in changed)

    @staticmethod
    def _structural_punctuation(text: str) -> str:
        return "".join(ch for ch in str(text) if ch in STRUCTURAL_PUNCT)

    @staticmethod
    def _without_spacing(text: str) -> str:
        return "".join(str(text).split())

    @staticmethod
    def _structural_change_is_trainable(original: str, corrected: str) -> bool:
        if QualityGuard._quote_count(corrected) < QualityGuard._quote_count(original):
            return False
        if QualityGuard._structural_balance_error(corrected) > QualityGuard._structural_balance_error(original):
            return False
        if QualityGuard._lost_opening_structural_pair(original, corrected):
            return False
        original_slots = extract_word_slots(original)
        corrected_slots = extract_word_slots(corrected)
        if not original_slots or len(original_slots) != len(corrected_slots):
            return False

        original_words = [normalize_word(slot.word) for slot in original_slots]
        corrected_words = [normalize_word(slot.word) for slot in corrected_slots]
        if original_words != corrected_words:
            return False

        original_gaps = QualityGuard._slot_gap_compacts(original, original_slots)
        corrected_gaps = QualityGuard._slot_gap_compacts(corrected, corrected_slots)
        for original_gap, corrected_gap in zip(original_gaps, corrected_gaps):
            if original_gap == corrected_gap:
                continue
            if QualityGuard._has_untrainable_structural_gap(original_gap):
                return False
            if QualityGuard._has_untrainable_structural_gap(corrected_gap):
                return False
            if canonical_punctuation_label(original_gap) not in TRAINABLE_PUNCT_LABELS:
                return False
            if canonical_punctuation_label(corrected_gap) not in TRAINABLE_PUNCT_LABELS:
                return False
        return True

    @staticmethod
    def _quote_count(text: str) -> int:
        return sum(1 for ch in str(text) if ch in {'"', "«", "»"})

    @staticmethod
    def _structural_balance_error(text: str) -> int:
        value = sum(QualityGuard._pair_balance_error(text, left, right) for left, right in STRUCTURAL_PAIRS)
        value += str(text).count('"') % 2
        return value

    @staticmethod
    def _pair_balance_error(text: str, left: str, right: str) -> int:
        depth = 0
        unmatched_right = 0
        for ch in str(text):
            if ch == left:
                depth += 1
            elif ch == right:
                if depth:
                    depth -= 1
                else:
                    unmatched_right += 1
        return depth + unmatched_right

    @staticmethod
    def _lost_opening_structural_pair(original: str, corrected: str) -> bool:
        original = str(original)
        corrected = str(corrected)
        for left, right in STRUCTURAL_PAIRS:
            lost_left = original.count(left) - corrected.count(left)
            lost_right = original.count(right) - corrected.count(right)
            if lost_left > max(lost_right, 0):
                return True
        return False

    @staticmethod
    def _slot_gap_compacts(text: str, slots) -> list[str]:
        gaps: list[str] = []
        for i, slot in enumerate(slots):
            next_start = slots[i + 1].start if i + 1 < len(slots) else len(text)
            gaps.append("".join(str(text)[slot.end:next_start].split()).replace("–", "—"))
        return gaps

    @staticmethod
    def _has_untrainable_structural_gap(gap: str) -> bool:
        gap = str(gap or "")
        if any(ch in set("[]{}№/\\") for ch in gap):
            return True
        return bool(gap and canonical_punctuation_label(gap) == "" and any(ch in STRUCTURAL_PUNCT for ch in gap))

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
