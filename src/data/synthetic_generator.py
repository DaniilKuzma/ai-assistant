from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import random

from src.candidates.frequent_errors import REVERSE_SYNTHETIC_ERRORS, REVERSE_SYNTHETIC_ERROR_TYPES


@dataclass(frozen=True)
class SyntheticExample:
    source: str
    target: str
    error_types: list[str]
    source_dataset: str = "synthetic"
    is_clean: bool = False
    is_synthetic: bool = True
    split: str = "train"
    domain: str = "general"


class SyntheticGenerator:
    def __init__(self, seed: int = 13, max_errors_per_sentence: int = 3) -> None:
        self.random = random.Random(seed)
        self.max_errors_per_sentence = max_errors_per_sentence

    def generate_from_clean(self, text: str) -> SyntheticExample:
        source = text
        error_types: list[str] = []

        transformations = self._available_transformations(text)
        self.random.shuffle(transformations)
        for clean, dirty, error_type in transformations[: self.max_errors_per_sentence]:
            if clean in source:
                source = source.replace(clean, dirty, 1)
                error_types.append(error_type)

        if not error_types:
            source = self._remove_final_punctuation(source)
            if source != text:
                error_types.append("final_punctuation")

        return SyntheticExample(source=source, target=text, error_types=error_types or ["punctuation"])

    def generate_variants_from_clean(self, text: str, max_variants: int = 8) -> list[SyntheticExample]:
        variants: list[SyntheticExample] = []
        seen_sources: set[str] = set()

        for source, error_types in _synthetic_variant_sources(text):
            if source == text or source in seen_sources:
                continue
            seen_sources.add(source)
            variants.append(SyntheticExample(source=source, target=text, error_types=error_types))
            if len(variants) >= max_variants:
                break

        return variants or [self.generate_from_clean(text)]

    def add_identity_examples(self, texts: list[str], source_dataset: str = "clean") -> list[SyntheticExample]:
        return [
            SyntheticExample(
                source=text,
                target=text,
                error_types=[],
                source_dataset=source_dataset,
                is_clean=True,
                is_synthetic=False,
            )
            for text in texts
        ]

    def _available_transformations(self, text: str) -> list[tuple[str, str, str]]:
        transformations: list[tuple[str, str, str]] = []
        lower = text.lower()
        for clean, dirty in REVERSE_SYNTHETIC_ERRORS.items():
            if clean in lower:
                error_type = REVERSE_SYNTHETIC_ERROR_TYPES.get(clean, "spelling")
                transformations.append((text[lower.find(clean) : lower.find(clean) + len(clean)], dirty, error_type))
        safe_commas = _safe_comma_positions(text)
        if safe_commas and safe_commas[0] == text.find(","):
            transformations.append((",", "", "punctuation"))
        if text.rstrip().endswith((".", "!", "?")):
            transformations.append((text.rstrip()[-1], "", "final_punctuation"))
        return transformations

    def _remove_final_punctuation(self, text: str) -> str:
        stripped = text.rstrip()
        if stripped and stripped[-1] in ".!?":
            return stripped[:-1]
        return text


def _synthetic_variant_sources(text: str) -> list[tuple[str, list[str]]]:
    variants: list[tuple[str, list[str]]] = []

    final_removed = _remove_final_punctuation_text(text)
    if final_removed != text:
        variants.append((final_removed, ["final_punctuation"]))

    comma_positions = _safe_comma_positions(text)
    for position in comma_positions[:4]:
        variants.append((_remove_positions(text, [position]), ["punctuation"]))
    if len(comma_positions) > 1:
        variants.append((_remove_positions(text, comma_positions), ["punctuation"]))
    for pair in combinations(comma_positions[:4], 2):
        variants.append((_remove_positions(text, pair), ["punctuation"]))

    lower = text.lower()
    replacements: list[tuple[str, str, str]] = []
    for clean, dirty in REVERSE_SYNTHETIC_ERRORS.items():
        start = lower.find(clean)
        if start < 0:
            continue
        error_type = REVERSE_SYNTHETIC_ERROR_TYPES.get(clean, "spelling")
        replacements.append((text[start : start + len(clean)], dirty, error_type))

    for clean, dirty, error_type in replacements:
        dirty_text = text.replace(clean, dirty, 1)
        variants.append((dirty_text, [error_type]))
        dirty_final = _remove_final_punctuation_text(dirty_text)
        if dirty_final != dirty_text:
            variants.append((dirty_final, [error_type, "final_punctuation"]))
        dirty_without_comma = _remove_first_safe_comma(dirty_text)
        if dirty_without_comma != dirty_text:
            variants.append((dirty_without_comma, [error_type, "punctuation"]))

    if final_removed != text and comma_positions:
        variants.append((_remove_positions(final_removed, [comma_positions[0]]), ["final_punctuation", "punctuation"]))

    return variants


def _remove_positions(text: str, positions: list[int] | tuple[int, ...]) -> str:
    remove = set(positions)
    return "".join(char for index, char in enumerate(text) if index not in remove)


def _remove_final_punctuation_text(text: str) -> str:
    stripped = text.rstrip()
    if stripped and stripped[-1] in ".!?":
        return stripped[:-1]
    return text


def _safe_comma_positions(text: str) -> list[int]:
    positions: list[int] = []
    for index, char in enumerate(text):
        if char != ",":
            continue
        previous_char = text[index - 1] if index > 0 else ""
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if previous_char.isdigit() and next_char.isdigit():
            continue
        positions.append(index)
    return positions


def _remove_first_safe_comma(text: str) -> str:
    positions = _safe_comma_positions(text)
    if not positions:
        return text
    return _remove_positions(text, [positions[0]])
