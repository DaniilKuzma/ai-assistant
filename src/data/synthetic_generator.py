from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import random
import re

from src.candidates.frequent_errors import REVERSE_SYNTHETIC_ERRORS, REVERSE_SYNTHETIC_ERROR_TYPES


@dataclass(frozen=True)
class SyntheticTransformation:
    start: int
    end: int
    replacement: str
    error_type: str

    def apply(self, text: str) -> str:
        return text[: self.start] + self.replacement + text[self.end :]


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
        transformations = self._available_transformations(text)
        self.random.shuffle(transformations)
        selected: list[SyntheticTransformation] = []
        for transformation in transformations:
            if len(selected) >= self.max_errors_per_sentence:
                break
            if _overlaps_any(transformation, selected):
                continue
            selected.append(transformation)

        source = _apply_transformations(text, selected)
        error_types = [transformation.error_type for transformation in selected]

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

    def _available_transformations(self, text: str) -> list[SyntheticTransformation]:
        return _synthetic_transformations(text)

    def _remove_final_punctuation(self, text: str) -> str:
        stripped = text.rstrip()
        if stripped and stripped[-1] in ".!?":
            return stripped[:-1]
        return text


def _synthetic_variant_sources(text: str) -> list[tuple[str, list[str]]]:
    variants: list[tuple[str, list[str]]] = []

    transformations = _synthetic_transformations(text)
    for transformation in transformations:
        variants.append((transformation.apply(text), [transformation.error_type]))

    for first, second in combinations(transformations[:18], 2):
        if _overlaps_any(first, [second]):
            continue
        variants.append((_apply_transformations(text, [first, second]), [first.error_type, second.error_type]))

    return variants


def _synthetic_transformations(text: str) -> list[SyntheticTransformation]:
    return [
        *_final_punctuation_transformations(text),
        *_punctuation_transformations(text),
        *_paired_punctuation_transformations(text),
        *_lexical_transformations(text),
    ]


def _lexical_transformations(text: str) -> list[SyntheticTransformation]:
    transformations: list[SyntheticTransformation] = []
    lower = text.lower()
    for clean, dirty in REVERSE_SYNTHETIC_ERRORS.items():
        start = lower.find(clean)
        if start < 0:
            continue
        source = text[start : start + len(clean)]
        error_type = REVERSE_SYNTHETIC_ERROR_TYPES.get(clean, "spelling")
        transformations.append(SyntheticTransformation(start, start + len(clean), _match_case(source, dirty), error_type))
    return transformations


def _punctuation_transformations(text: str) -> list[SyntheticTransformation]:
    transformations: list[SyntheticTransformation] = []
    for index, char in enumerate(text):
        if _is_final_punctuation_position(text, index) or _is_decimal_comma(text, index):
            continue
        if char in ",:;—":
            transformations.append(SyntheticTransformation(index, index + 1, "", "punctuation"))
            for replacement in _punctuation_replacements_for(char):
                transformations.append(SyntheticTransformation(index, index + 1, replacement, "punctuation"))
        if char in ",:;":
            transformations.append(SyntheticTransformation(index + 1, index + 1, char, "punctuation"))
    return transformations


def _paired_punctuation_transformations(text: str) -> list[SyntheticTransformation]:
    transformations: list[SyntheticTransformation] = []
    for match in re.finditer(r"«([^»\n]+)»", text):
        inner = match.group(1)
        transformations.append(SyntheticTransformation(match.start(), match.end(), inner, "punctuation"))
        transformations.append(SyntheticTransformation(match.start(), match.end(), f'"{inner}"', "punctuation"))
    for match in re.finditer(r"\(([^)\n]+)\)", text):
        transformations.append(SyntheticTransformation(match.start(), match.end(), match.group(1), "punctuation"))
    return transformations


def _final_punctuation_transformations(text: str) -> list[SyntheticTransformation]:
    stripped = text.rstrip()
    if not stripped or stripped[-1] not in ".!?…":
        return []
    start = len(stripped) - 1
    current = stripped[-1]
    replacements = {"?": [".", "!"], "!": [".", "?"], ".": ["?", "!"], "…": ["."]}.get(current, [])
    return [
        SyntheticTransformation(start, start + 1, "", "final_punctuation"),
        *[SyntheticTransformation(start, start + 1, replacement, "final_punctuation") for replacement in replacements],
    ]


def _punctuation_replacements_for(char: str) -> list[str]:
    return {
        ",": [":", ";"],
        ":": [",", ";"],
        ";": [","],
        "—": [","],
    }.get(char, [])


def _apply_transformations(text: str, transformations: list[SyntheticTransformation]) -> str:
    source = text
    for transformation in sorted(transformations, key=lambda item: (item.start, item.end), reverse=True):
        source = transformation.apply(source)
    return source


def _overlaps_any(transformation: SyntheticTransformation, selected: list[SyntheticTransformation]) -> bool:
    return any(_overlaps(transformation, item) for item in selected)


def _overlaps(left: SyntheticTransformation, right: SyntheticTransformation) -> bool:
    if left.start == left.end or right.start == right.end:
        return left.start == right.start
    return max(left.start, right.start) < min(left.end, right.end)


def _is_decimal_comma(text: str, index: int) -> bool:
    if text[index] != ",":
        return False
    previous_char = text[index - 1] if index > 0 else ""
    next_char = text[index + 1] if index + 1 < len(text) else ""
    return previous_char.isdigit() and next_char.isdigit()


def _is_final_punctuation_position(text: str, index: int) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and index == len(stripped) - 1 and stripped[-1] in ".!?…"


def _match_case(source: str, replacement: str) -> str:
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement
