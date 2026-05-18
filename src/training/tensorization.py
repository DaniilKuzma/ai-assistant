from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from src.alignment.punctuation_label_builder import (
    PUNCT_ACTION_LABELS,
    build_punctuation_gap_action_labels,
    build_punctuation_gap_labels,
)
from src.candidates.candidate_generator import Candidate, CandidateGenerator
from src.candidates.candidate_ranking import rank_candidates_for_budget
from src.candidates.matching import (
    candidate_edit_type_for_labels,
    candidate_matches_edit,
    candidate_overlaps_edit,
)
from src.preprocessing.punctuation_gaps import word_gap_context_token_indices
from src.preprocessing.tokenizer import Token, tokenize_words
from src.rules.base import RuleMode
from src.validation.diff_analyzer import DiffAnalyzer, Edit
from src.validation.edit_classifier import coarse_error_type


MAX_REPLACEMENT_TOKENS = 8
PUNCTUATION_EDIT_TYPES = {
    "punctuation_insert",
    "punctuation_delete",
    "punctuation_replace",
    "final_punctuation",
}


@dataclass(frozen=True)
class TrainingFeature:
    source: str
    target: str
    input_ids: list[int]
    attention_mask: list[int]
    offset_mapping: list[tuple[int, int]]
    candidate_spans: list[tuple[int, int]]
    candidate_mask: list[bool]
    candidate_labels: list[float]
    candidate_error_type_labels: list[int]
    confidence_labels: list[float]
    punctuation_labels: list[int]
    punctuation_action_labels: list[int]
    punctuation_mask: list[bool]
    punctuation_gap_indices: list[int]
    punctuation_right_gap_indices: list[int]
    punctuation_gap_mask: list[bool]
    punctuation_confidence_labels: list[float]
    punctuation_error_type_labels: list[int]
    candidate_replacements: list[str]
    candidate_rule_ids: list[str]
    candidate_modes: list[RuleMode]
    candidate_requires_model: list[bool]
    candidate_requires_scoring: list[bool]
    candidate_replacement_ids: list[list[int]]
    candidate_replacement_mask: list[list[bool]]


class DebugTokenizer:
    """Small tokenizer for tests and quick-debug mode."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {"[PAD]": 0, "[UNK]": 1}
        self.pattern = re.compile(r"[А-Яа-яЁёA-Za-z]+(?:-[А-Яа-яЁёA-Za-z]+)*|\d+|[^\w\s]", re.UNICODE)

    def __call__(
        self,
        text: str,
        *,
        return_offsets_mapping: bool = True,
        truncation: bool = True,
        padding: str = "max_length",
        max_length: int = 128,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        tokens = [(match.group(0), match.start(), match.end()) for match in self.pattern.finditer(text)]
        if truncation:
            tokens = tokens[:max_length]
        input_ids = [self._id(token) for token, _, _ in tokens]
        offsets = [(start, end) for _, start, end in tokens]
        attention = [1] * len(input_ids)
        if padding == "max_length":
            pad = max(0, max_length - len(input_ids))
            input_ids += [0] * pad
            attention += [0] * pad
            offsets += [(0, 0)] * pad
        result: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def _id(self, token: str) -> int:
        token = token.lower()
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)
        return self.vocab[token]


def build_training_feature(
    source: str,
    target: str,
    *,
    tokenizer: Any,
    punctuation_label_map: dict[str, int],
    error_type_label_map: dict[str, int],
    max_length: int,
    max_candidates: int,
    punctuation_action_label_map: dict[str, int] | None = None,
    candidate_generator: CandidateGenerator | None = None,
) -> TrainingFeature:
    encoded = _encode(tokenizer, source, max_length)
    offsets = encoded["offset_mapping"]
    candidates = (candidate_generator or CandidateGenerator()).generate(source)
    candidates = rank_candidates_for_budget(candidates, max_candidates)
    alignment_edits = DiffAnalyzer().analyze(source, target)
    spans = [_candidate_to_token_span(candidate, offsets) for candidate in candidates]
    labels = [_candidate_label(candidate, alignment_edits) for candidate in candidates]
    error_labels = [_candidate_error_label(candidate, label, error_type_label_map) for candidate, label in zip(candidates, labels, strict=False)]
    confidence_labels = labels[:]
    replacements = [candidate.replacement for candidate in candidates]
    rule_ids = [candidate.rule_id for candidate in candidates]
    modes = [candidate.mode for candidate in candidates]
    requires_model = [candidate.requires_model for candidate in candidates]
    requires_scoring = [candidate.requires_scoring for candidate in candidates]
    replacement_encodings = [_encode_replacement(tokenizer, replacement) for replacement in replacements]
    replacement_ids = [encoding["input_ids"] for encoding in replacement_encodings]
    replacement_masks = [[bool(value) for value in encoding["attention_mask"]] for encoding in replacement_encodings]

    pad_candidates = max_candidates - len(candidates)
    spans.extend([(0, 0)] * pad_candidates)
    labels.extend([0.0] * pad_candidates)
    error_labels.extend([-100] * pad_candidates)
    confidence_labels.extend([0.0] * pad_candidates)
    replacements.extend([""] * pad_candidates)
    rule_ids.extend([""] * pad_candidates)
    modes.extend(["deterministic"] * pad_candidates)
    requires_model.extend([False] * pad_candidates)
    requires_scoring.extend([False] * pad_candidates)
    replacement_ids.extend([[0] * MAX_REPLACEMENT_TOKENS for _ in range(pad_candidates)])
    replacement_masks.extend([[False] * MAX_REPLACEMENT_TOKENS for _ in range(pad_candidates)])
    candidate_mask = [True] * len(candidates) + [False] * pad_candidates

    punctuation_action_label_map = punctuation_action_label_map or PUNCT_ACTION_LABELS
    punctuation_gap_indices, punctuation_right_gap_indices, punctuation_gap_mask = word_gap_context_token_indices(
        source,
        offsets,
        max_length,
    )
    punctuation_mask = punctuation_gap_mask[:]
    punctuation_labels = _punctuation_labels(source, target, punctuation_label_map, max_length)
    punctuation_action_labels = _punctuation_action_labels(source, target, punctuation_action_label_map, max_length)
    punctuation_confidence_labels, punctuation_error_type_labels = _punctuation_confidence_and_error_labels(
        source,
        target,
        error_type_label_map,
        max_length,
        punctuation_gap_mask,
    )

    return TrainingFeature(
        source=source,
        target=target,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        offset_mapping=offsets,
        candidate_spans=spans,
        candidate_mask=candidate_mask,
        candidate_labels=labels,
        candidate_error_type_labels=error_labels,
        confidence_labels=confidence_labels,
        punctuation_labels=punctuation_labels,
        punctuation_action_labels=punctuation_action_labels,
        punctuation_mask=punctuation_mask,
        punctuation_gap_indices=punctuation_gap_indices,
        punctuation_right_gap_indices=punctuation_right_gap_indices,
        punctuation_gap_mask=punctuation_gap_mask,
        punctuation_confidence_labels=punctuation_confidence_labels,
        punctuation_error_type_labels=punctuation_error_type_labels,
        candidate_replacements=replacements,
        candidate_rule_ids=rule_ids,
        candidate_modes=modes,
        candidate_requires_model=requires_model,
        candidate_requires_scoring=requires_scoring,
        candidate_replacement_ids=replacement_ids,
        candidate_replacement_mask=replacement_masks,
    )


class EditBatchCollator:
    def collate(self, features: list[TrainingFeature]) -> dict[str, Any]:
        import torch

        return {
            "input_ids": torch.tensor([feature.input_ids for feature in features], dtype=torch.long),
            "attention_mask": torch.tensor([feature.attention_mask for feature in features], dtype=torch.long),
            "candidate_spans": torch.tensor([feature.candidate_spans for feature in features], dtype=torch.long),
            "candidate_mask": torch.tensor([feature.candidate_mask for feature in features], dtype=torch.bool),
            "candidate_rule_ids": [feature.candidate_rule_ids for feature in features],
            "candidate_modes": [feature.candidate_modes for feature in features],
            "candidate_requires_model": [feature.candidate_requires_model for feature in features],
            "candidate_requires_scoring": [feature.candidate_requires_scoring for feature in features],
            "candidate_replacement_ids": torch.tensor([feature.candidate_replacement_ids for feature in features], dtype=torch.long),
            "candidate_replacement_mask": torch.tensor(
                [feature.candidate_replacement_mask for feature in features], dtype=torch.bool
            ),
            "punctuation_gap_indices": torch.tensor(
                [feature.punctuation_gap_indices for feature in features], dtype=torch.long
            ),
            "punctuation_right_gap_indices": torch.tensor(
                [feature.punctuation_right_gap_indices for feature in features], dtype=torch.long
            ),
            "punctuation_gap_mask": torch.tensor([feature.punctuation_gap_mask for feature in features], dtype=torch.bool),
            "labels": {
                "candidate_labels": torch.tensor([feature.candidate_labels for feature in features], dtype=torch.float),
                "candidate_mask": torch.tensor([feature.candidate_mask for feature in features], dtype=torch.bool),
                "punctuation_labels": torch.tensor([feature.punctuation_labels for feature in features], dtype=torch.long),
                "punctuation_action_labels": torch.tensor(
                    [feature.punctuation_action_labels for feature in features], dtype=torch.long
                ),
                "punctuation_mask": torch.tensor([feature.punctuation_mask for feature in features], dtype=torch.bool),
                "confidence_labels": torch.tensor([feature.confidence_labels for feature in features], dtype=torch.float),
                "error_type_labels": torch.tensor([feature.candidate_error_type_labels for feature in features], dtype=torch.long),
                "punctuation_confidence_labels": torch.tensor(
                    [feature.punctuation_confidence_labels for feature in features], dtype=torch.float
                ),
                "punctuation_error_type_labels": torch.tensor(
                    [feature.punctuation_error_type_labels for feature in features], dtype=torch.long
                ),
            },
        }


def build_features_from_rows(
    rows: list[dict[str, Any]],
    *,
    tokenizer: Any,
    punctuation_label_map: dict[str, int],
    error_type_label_map: dict[str, int],
    max_length: int,
    max_candidates: int,
    punctuation_action_label_map: dict[str, int] | None = None,
    candidate_generator: CandidateGenerator | None = None,
    show_progress: bool = False,
) -> list[TrainingFeature]:
    row_iterable = _with_progress(rows, enabled=show_progress, description="Building training features")
    return [
        build_training_feature(
            row["source"],
            row["target"],
            tokenizer=tokenizer,
            punctuation_label_map=punctuation_label_map,
            punctuation_action_label_map=punctuation_action_label_map,
            error_type_label_map=error_type_label_map,
            max_length=max_length,
            max_candidates=max_candidates,
            candidate_generator=candidate_generator,
        )
        for row in row_iterable
    ]


def _with_progress(rows: list[dict[str, Any]], *, enabled: bool, description: str) -> Any:
    if not enabled:
        return rows
    try:
        from tqdm.auto import tqdm
    except Exception:
        return rows
    return tqdm(rows, total=len(rows), desc=description, dynamic_ncols=True)


def _encode(tokenizer: Any, text: str, max_length: int) -> dict[str, Any]:
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    input_ids = _to_list(encoded["input_ids"])
    attention_mask = _to_list(encoded["attention_mask"])
    offsets = [tuple(item) for item in _to_list(encoded["offset_mapping"])]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "offset_mapping": offsets}


def _encode_replacement(tokenizer: Any, text: str) -> dict[str, list[int]]:
    kwargs = {
        "truncation": True,
        "padding": "max_length",
        "max_length": MAX_REPLACEMENT_TOKENS,
        "add_special_tokens": False,
    }
    try:
        encoded = tokenizer(text, return_offsets_mapping=False, **kwargs)
    except TypeError:
        encoded = tokenizer(text, return_offsets_mapping=False, truncation=True, padding="max_length", max_length=MAX_REPLACEMENT_TOKENS)
    input_ids = _to_list(encoded["input_ids"])
    attention_mask = _to_list(encoded["attention_mask"])
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _to_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _candidate_to_token_span(candidate: Candidate, offsets: list[tuple[int, int]]) -> tuple[int, int]:
    token_indexes = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > start and start < candidate.end and candidate.start < end
    ]
    if not token_indexes:
        return (0, 1)
    return (min(token_indexes), max(token_indexes) + 1)


def _candidate_label(candidate: Candidate, alignment_edits: list[Edit]) -> float:
    if candidate.edit_type == "keep":
        return 0.0 if any(candidate_overlaps_edit(candidate, edit) for edit in alignment_edits) else 1.0
    return 1.0 if any(candidate_matches_edit(candidate, edit) for edit in alignment_edits) else 0.0


def _candidate_error_label(candidate: Candidate, label: float, error_type_label_map: dict[str, int]) -> int:
    if label <= 0:
        return error_type_label_map.get("keep", 0)
    return error_type_label_map.get(coarse_error_type(candidate_edit_type_for_labels(candidate.edit_type)), 0)


def _punctuation_labels(
    source: str,
    target: str,
    punctuation_label_map: dict[str, int],
    max_length: int,
) -> list[int]:
    labels = [punctuation_label_map.get("NONE", 0)] * max_length
    for gap in build_punctuation_gap_labels(source, target):
        if 0 <= gap.gap_index < max_length:
            labels[gap.gap_index] = punctuation_label_map.get(gap.label, punctuation_label_map.get("NONE", 0))
    return labels


def _punctuation_action_labels(
    source: str,
    target: str,
    punctuation_action_label_map: dict[str, int],
    max_length: int,
) -> list[int]:
    labels = [punctuation_action_label_map.get("KEEP_NONE", 0)] * max_length
    for gap in build_punctuation_gap_action_labels(source, target):
        if 0 <= gap.gap_index < max_length:
            labels[gap.gap_index] = punctuation_action_label_map.get(
                gap.action,
                punctuation_action_label_map.get("KEEP_NONE", 0),
            )
    return labels


def _punctuation_confidence_and_error_labels(
    source: str,
    target: str,
    error_type_label_map: dict[str, int],
    max_length: int,
    punctuation_gap_mask: list[bool],
) -> tuple[list[float], list[int]]:
    confidence_labels = [0.0] * max_length
    keep_label = error_type_label_map.get("keep", 0)
    error_type_labels = [
        keep_label if index < len(punctuation_gap_mask) and punctuation_gap_mask[index] else -100
        for index in range(max_length)
    ]
    words = tokenize_words(source)

    for edit in DiffAnalyzer().analyze(source, target):
        if edit.edit_type not in PUNCTUATION_EDIT_TYPES:
            continue
        gap_index = _punctuation_edit_gap_index(words, edit)
        if gap_index < 0 or gap_index >= max_length:
            continue
        if not punctuation_gap_mask[gap_index]:
            continue
        confidence_labels[gap_index] = 1.0
        error_type_labels[gap_index] = error_type_label_map.get(coarse_error_type(edit.edit_type), keep_label)

    return confidence_labels, error_type_labels


def _punctuation_edit_gap_index(words: list[Token], edit: Edit) -> int:
    if not words:
        return -1
    if edit.edit_type == "final_punctuation":
        return len(words) - 1
    return _gap_index_for_position(words, edit.start)


def _gap_index_for_position(words: list[Token], position: int) -> int:
    if position < 0:
        return 0
    gap = 0
    for index, word in enumerate(words):
        if word.end <= position:
            gap = index
        elif word.start > position:
            break
    return max(0, min(gap, len(words) - 1))
