from __future__ import annotations

from dataclasses import dataclass
import re

from src.candidates.candidate_generator import Candidate, CandidateGenerator
from src.inference.edit_realizer import apply_candidate, ensure_final_punctuation
from src.inference.postprocess import normalize_spacing
from src.validation.diff_analyzer import Edit
from src.validation.strict_validator import StrictValidator


@dataclass(frozen=True)
class CorrectionResult:
    source_text: str
    corrected_text: str
    edits: list[Edit]


class Corrector:
    """Safe inference facade.

    The lightweight path uses deterministic whitelist scoring. A trained model can
    be wired in later to choose among the same bounded candidates.
    """

    def __init__(self, max_passes: int = 3, mode: str = "balanced") -> None:
        self.max_passes = max_passes
        self.mode = mode
        self.candidates = CandidateGenerator()
        self.validator = StrictValidator()

    def correct(self, text: str) -> CorrectionResult:
        proposed, trusted_edits = self._decode_with_trusted_edits(text)
        validation = self.validator.validate(text, proposed, trusted_edits=trusted_edits)
        corrected = validation.apply_accepted()
        return CorrectionResult(text, corrected, validation.edits)

    def _decode_with_trusted_edits(self, text: str) -> tuple[str, list[Candidate]]:
        current = text
        trusted_edits: list[Candidate] = []
        for pass_index in range(self.max_passes):
            updated, selected = self._single_pass_with_candidates(current)
            if pass_index == 0:
                trusted_edits.extend(selected)
            if updated == current:
                break
            current = updated
        return current, trusted_edits

    def _single_pass(self, text: str) -> str:
        return self._single_pass_with_candidates(text)[0]

    def _single_pass_with_candidates(self, text: str) -> tuple[str, list[Candidate]]:
        proposed = text
        offset = 0
        applied: list[Candidate] = []
        for candidate in self.candidates.generate(text):
            if candidate.edit_type == "keep" or candidate.requires_model:
                continue
            shifted = candidate.__class__(
                source=candidate.source,
                replacement=candidate.replacement,
                edit_type=candidate.edit_type,
                start=candidate.start + offset,
                end=candidate.end + offset,
                confidence=candidate.confidence,
            )
            before = proposed
            proposed = apply_candidate(proposed, shifted)
            offset += len(proposed) - len(before)
            applied.append(candidate)

        proposed = self._punctuation_pass(proposed)
        return normalize_spacing(proposed), applied

    def _punctuation_pass(self, text: str) -> str:
        text = _remove_obvious_extra_punctuation(text)
        text = _normalize_simple_direct_speech_quotes(text)
        text = _add_simple_direct_speech_colon(text)
        text = _add_obvious_subject_predicate_dash(text)
        text = _add_simple_enumeration_colon(text)
        text = re.sub(r"\b(не знаю|думаю|считаю) что\b", r"\1, что", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(во-первых|во-вторых|в-третьих)\s+(?!,)", r"\1, ", text, flags=re.IGNORECASE)
        text = ensure_final_punctuation(text, ".")
        return text


def _remove_obvious_extra_punctuation(text: str) -> str:
    text = re.sub(r"([,;:])\s*\1+", r"\1", text)
    text = re.sub(r",\s*([.!?…])", r"\1", text)
    text = re.sub(r"([«(])\s*([,;:])\s*", r"\1", text)
    text = re.sub(r"\s*([,;:])\s*([»)])", r"\2", text)
    return text


def _normalize_simple_direct_speech_quotes(text: str) -> str:
    speech_verbs = r"сказал[аи]?|спросил[аи]?|ответил[аи]?|написал[аи]?"
    return re.sub(
        rf"\b({speech_verbs})\s*:?\s*\"([^\"\n]+)\"",
        r"\1: «\2»",
        text,
        flags=re.IGNORECASE,
    )


def _add_simple_direct_speech_colon(text: str) -> str:
    speech_verbs = r"сказал[аи]?|спросил[аи]?|ответил[аи]?|написал[аи]?"
    return re.sub(rf"\b({speech_verbs})\s+(«[^»]+»)", r"\1: \2", text, flags=re.IGNORECASE)


def _add_obvious_subject_predicate_dash(text: str) -> str:
    return re.sub(r"^([А-ЯЁ][а-яё]+)\s+это\s+", r"\1 — это ", text)


def _add_simple_enumeration_colon(text: str) -> str:
    return re.sub(r"\b(следующее)\s+(?=[а-яёА-ЯЁ])", r"\1: ", text, count=1)
