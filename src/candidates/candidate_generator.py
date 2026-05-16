from __future__ import annotations

from dataclasses import dataclass

from src.candidates.frequent_errors import CONTEXT_DEPENDENT_WHITELIST, HYPHEN_WHITELIST, SPLIT_JOIN_WHITELIST
from src.candidates.spelling_rules import spelling_candidates
from src.preprocessing.tokenizer import Token, tokenize_words


@dataclass(frozen=True)
class Candidate:
    source: str
    replacement: str
    edit_type: str
    start: int
    end: int
    confidence: float = 1.0
    requires_model: bool = False


class CandidateGenerator:
    """Generate bounded candidates. It never invents replacements outside local whitelists."""

    def generate(self, text: str) -> list[Candidate]:
        candidates: list[Candidate] = []
        words = tokenize_words(text)

        for index, token in enumerate(words):
            candidates.append(Candidate(token.text, token.text, "keep", token.start, token.end, 1.0))

            lower = token.text.lower()
            for replacement in spelling_candidates(token.text):
                edit_type = "split_join" if lower in SPLIT_JOIN_WHITELIST else "spelling"
                candidates.append(Candidate(token.text, _match_case(token.text, replacement), edit_type, token.start, token.end, 0.95))

            candidates.extend(_context_candidates_starting_at(text, words, index))

            if index == 0 and token.text[:1].islower():
                fixed = token.text[:1].upper() + token.text[1:]
                candidates.append(Candidate(token.text, fixed, "case", token.start, token.end, 0.9))

        lower_text = text.lower()
        for source, replacement in HYPHEN_WHITELIST.items():
            if source == replacement:
                continue
            start = lower_text.find(source)
            if start >= 0:
                end = start + len(source)
                candidates.append(Candidate(text[start:end], replacement, "hyphen", start, end, 0.95))

        return candidates


def _match_case(source: str, replacement: str) -> str:
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _context_candidates_starting_at(text: str, words: list[Token], index: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    for source, replacement in CONTEXT_DEPENDENT_WHITELIST.items():
        source_words = source.split()
        end_index = index + len(source_words)
        if end_index > len(words):
            continue
        span_words = words[index:end_index]
        if [word.text.lower() for word in span_words] != source_words:
            continue
        start = span_words[0].start
        end = span_words[-1].end
        if text[start:end].lower().split() != source_words:
            continue
        candidates.append(
            Candidate(
                source=text[start:end],
                replacement=_match_case(text[start:end], replacement),
                edit_type="split_join",
                start=start,
                end=end,
                confidence=0.0,
                requires_model=True,
            )
        )
    return candidates
