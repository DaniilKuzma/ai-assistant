from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import re

from src.candidates.frequent_errors import HYPHEN_WHITELIST, REVERSE_SYNTHETIC_ERRORS, REVERSE_SYNTHETIC_ERROR_TYPES
from src.preprocessing.protected_spans import find_protected_spans
from src.preprocessing.tokenizer import tokenize_words
from src.rules.orthography import orthography_rules


PUNCTUATION_BALANCE_GROUPS = (
    "comma_subordinate",
    "comma_conjunction",
    "introductory",
    "colon",
    "dash",
    "semicolon",
    "quotes_brackets",
    "final_punctuation",
    "delete_replace",
)
DEFAULT_PUNCTUATION_BALANCE = {
    "comma_subordinate": 20_000,
    "comma_conjunction": 15_000,
    "introductory": 10_000,
    "colon": 8_000,
    "dash": 8_000,
    "semicolon": 5_000,
    "quotes_brackets": 8_000,
    "final_punctuation": 20_000,
    "delete_replace": 15_000,
}
ORTHOGRAPHY_BALANCE_GROUPS = (
    "ne_verb",
    "tsya",
    "combo",
    "hard_sign",
    "prefix_z_s",
    "ci",
    "hissing_o_e",
)
DEFAULT_ORTHOGRAPHY_BALANCE = {
    "ne_verb": 8_000,
    "tsya": 6_000,
    "combo": 8_000,
    "hard_sign": 6_000,
    "prefix_z_s": 6_000,
    "ci": 4_000,
    "hissing_o_e": 5_000,
}


@dataclass(frozen=True)
class SyntheticTransformation:
    start: int
    end: int
    replacement: str
    error_type: str
    group: str = ""
    rule_id: str = ""

    def apply(self, text: str) -> str:
        return text[: self.start] + self.replacement + text[self.end :]


def synthetic_transformations(text: str) -> list[SyntheticTransformation]:
    protected = tuple((span.start, span.end) for span in find_protected_spans(text))
    return [
        *_final_punctuation_transformations(text),
        *_punctuation_transformations(text, protected),
        *_paired_punctuation_transformations(text, protected),
        *_orthography_transformations(text, protected),
        *_lexical_transformations(text, protected),
    ]


def synthetic_variant_sources(text: str) -> list[tuple[str, list[str], list[str], list[str]]]:
    variants: list[tuple[str, list[str], list[str], list[str]]] = []

    transformations = synthetic_transformations(text)
    for transformation in transformations:
        variants.append(
            (
                transformation.apply(text),
                [transformation.error_type],
                [transformation.group],
                [transformation.rule_id],
            )
        )

    for first, second in combinations(transformations[:18], 2):
        if _overlaps_any(first, [second]):
            continue
        variants.append(
            (
                apply_transformations(text, [first, second]),
                [first.error_type, second.error_type],
                [first.group, second.group],
                [first.rule_id, second.rule_id],
            )
        )

    return variants


def source_dataset_for_groups(groups: list[str]) -> str:
    unique_groups = sorted({group for group in groups if group})
    if len(unique_groups) == 1:
        return f"synthetic_open_corpus_{unique_groups[0]}"
    return "synthetic_open_corpus_mixed"


def apply_transformations(text: str, transformations: list[SyntheticTransformation]) -> str:
    source = text
    for transformation in sorted(transformations, key=lambda item: (item.start, item.end), reverse=True):
        source = transformation.apply(source)
    return source


def _lexical_transformations(text: str, protected: tuple[tuple[int, int], ...]) -> list[SyntheticTransformation]:
    transformations: list[SyntheticTransformation] = []
    lower = text.lower()
    for clean, dirty in REVERSE_SYNTHETIC_ERRORS.items():
        start = lower.find(clean)
        if start < 0:
            continue
        if _span_overlaps_protected(start, start + len(clean), protected):
            continue
        source = text[start : start + len(clean)]
        error_type = REVERSE_SYNTHETIC_ERROR_TYPES.get(clean, "spelling")
        transformations.append(
            SyntheticTransformation(
                start,
                start + len(clean),
                _match_case(source, dirty),
                error_type,
                _lexical_group(error_type),
                "hyphen_whitelist" if error_type == "hyphen" else "frequent_error_exact",
            )
        )
    return transformations


def _orthography_transformations(text: str, protected: tuple[tuple[int, int], ...]) -> list[SyntheticTransformation]:
    transformations: list[SyntheticTransformation] = []
    words = tuple(tokenize_words(text))

    for index, _word in enumerate(words):
        for rule in orthography_rules():
            if not hasattr(rule, "generate_corruptions"):
                continue
            for dirty in rule.generate_corruptions(text, words, index, protected):
                transformations.append(
                    SyntheticTransformation(
                        dirty.start,
                        dirty.end,
                        dirty.replacement,
                        dirty.error_type,
                        dirty.group,
                        dirty.rule_id,
                    )
                )
    return transformations


def _punctuation_transformations(text: str, protected: tuple[tuple[int, int], ...]) -> list[SyntheticTransformation]:
    transformations: list[SyntheticTransformation] = []
    for index, char in enumerate(text):
        if _span_overlaps_protected(index, index + 1, protected):
            continue
        if _is_final_punctuation_position(text, index) or _is_decimal_comma(text, index):
            continue
        if char in ",:;—":
            group = _punctuation_group(text, index, char)
            transformations.append(
                SyntheticTransformation(index, index + 1, "", "punctuation", group, _punctuation_rule_id(group))
            )
            for replacement in _punctuation_replacements_for(char):
                transformations.append(
                    SyntheticTransformation(
                        index,
                        index + 1,
                        replacement,
                        "punctuation",
                        "delete_replace",
                        "punctuation_delete_replace",
                    )
                )
        if char in ",:;":
            transformations.append(
                SyntheticTransformation(
                    index + 1,
                    index + 1,
                    char,
                    "punctuation",
                    "delete_replace",
                    "punctuation_delete_replace",
                )
            )
    return transformations


def _paired_punctuation_transformations(text: str, protected: tuple[tuple[int, int], ...]) -> list[SyntheticTransformation]:
    transformations: list[SyntheticTransformation] = []
    for match in re.finditer(r"«([^»\n]+)»", text):
        if _span_overlaps_protected(match.start(), match.end(), protected):
            continue
        inner = match.group(1)
        transformations.append(
            SyntheticTransformation(match.start(), match.end(), inner, "punctuation", "quotes_brackets", "quotes_brackets")
        )
        transformations.append(
            SyntheticTransformation(match.start(), match.end(), f'"{inner}"', "punctuation", "quotes_brackets", "quotes_brackets")
        )
    for match in re.finditer(r"\(([^)\n]+)\)", text):
        if _span_overlaps_protected(match.start(), match.end(), protected):
            continue
        transformations.append(
            SyntheticTransformation(match.start(), match.end(), match.group(1), "punctuation", "quotes_brackets", "quotes_brackets")
        )
    return transformations


def _final_punctuation_transformations(text: str) -> list[SyntheticTransformation]:
    stripped = text.rstrip()
    if not stripped or stripped[-1] not in ".!?…":
        return []
    start = len(stripped) - 1
    current = stripped[-1]
    replacements = {"?": [".", "!"], "!": [".", "?"], ".": ["?", "!"], "…": ["."]}.get(current, [])
    return [
        SyntheticTransformation(start, start + 1, "", "final_punctuation", "final_punctuation", "final_punctuation_default"),
        *[
            SyntheticTransformation(
                start,
                start + 1,
                replacement,
                "final_punctuation",
                "final_punctuation",
                "final_punctuation_default",
            )
            for replacement in replacements
        ],
    ]


def _punctuation_group(text: str, index: int, char: str) -> str:
    if char == ",":
        following = text[index + 1 : index + 24].lower()
        prefix = text[max(0, index - 20) : index + 1].lower()
        if re.match(r"\s*(что|чтобы|если|когда)\b", following):
            return "comma_subordinate"
        if re.match(r"\s*(но|а)\b", following):
            return "comma_conjunction"
        if re.search(r"\b(конечно|например|однако|во-первых),$", prefix):
            return "introductory"
        return "comma"
    if char == ":":
        return "colon"
    if char == "—":
        return "dash"
    if char == ";":
        return "semicolon"
    return "punctuation"


def _punctuation_rule_id(group: str) -> str:
    return {
        "comma_subordinate": "comma_subordinate",
        "comma_conjunction": "comma_conjunction",
        "introductory": "introductory_comma",
        "colon": "enumeration_colon",
        "dash": "subject_predicate_dash",
        "semicolon": "semicolon",
        "comma": "comma",
    }.get(group, "punctuation_delete_replace")


def _lexical_group(error_type: str) -> str:
    if error_type in {"split_join", "hyphen"}:
        return error_type
    return "frequent_spelling"


def _punctuation_replacements_for(char: str) -> list[str]:
    return {
        ",": [":", ";"],
        ":": [",", ";"],
        ";": [","],
        "—": [","],
    }.get(char, [])


def _overlaps_any(transformation: SyntheticTransformation, selected: list[SyntheticTransformation]) -> bool:
    return any(_overlaps(transformation, item) for item in selected)


def _overlaps(left: SyntheticTransformation, right: SyntheticTransformation) -> bool:
    if left.start == left.end or right.start == right.end:
        return left.start == right.start
    return max(left.start, right.start) < min(left.end, right.end)


def _span_overlaps_protected(start: int, end: int, protected: tuple[tuple[int, int], ...]) -> bool:
    return any(start < protected_end and protected_start < end for protected_start, protected_end in protected)


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
