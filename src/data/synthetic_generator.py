from __future__ import annotations

from dataclasses import dataclass
import random

from src.rules.synthetic import (
    SyntheticTransformation,
    apply_transformations,
    source_dataset_for_groups,
    synthetic_transformations,
    synthetic_variant_sources,
)


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
    rule_ids: list[str] | None = None


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

        source = apply_transformations(text, selected)
        error_types = [transformation.error_type for transformation in selected]
        rule_ids = [transformation.rule_id for transformation in selected if transformation.rule_id]

        if not error_types:
            source = self._remove_final_punctuation(source)
            if source != text:
                error_types.append("final_punctuation")
                rule_ids.append("final_punctuation_default")

        return SyntheticExample(
            source=source,
            target=text,
            error_types=error_types or ["punctuation"],
            rule_ids=rule_ids,
        )

    def generate_variants_from_clean(self, text: str, max_variants: int = 20) -> list[SyntheticExample]:
        variants: list[SyntheticExample] = []
        seen_sources: set[str] = set()

        for source, error_types, groups, rule_ids in synthetic_variant_sources(text):
            if source == text or source in seen_sources:
                continue
            seen_sources.add(source)
            variants.append(
                SyntheticExample(
                    source=source,
                    target=text,
                    error_types=error_types,
                    source_dataset=source_dataset_for_groups(groups),
                    rule_ids=[rule_id for rule_id in rule_ids if rule_id],
                )
            )
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
                rule_ids=[],
            )
            for text in texts
        ]

    def _available_transformations(self, text: str) -> list[SyntheticTransformation]:
        return synthetic_transformations(text)

    def _remove_final_punctuation(self, text: str) -> str:
        stripped = text.rstrip()
        if stripped and stripped[-1] in ".!?":
            return stripped[:-1]
        return text


def _overlaps_any(transformation: SyntheticTransformation, selected: list[SyntheticTransformation]) -> bool:
    return any(_overlaps(transformation, item) for item in selected)


def _overlaps(left: SyntheticTransformation, right: SyntheticTransformation) -> bool:
    if left.start == left.end or right.start == right.end:
        return left.start == right.start
    return max(left.start, right.start) < min(left.end, right.end)
