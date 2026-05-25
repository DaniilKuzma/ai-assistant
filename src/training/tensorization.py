from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

from src.schema import GeneratedExample
from src.schema.labels import gap_label_to_id, rule_tag_to_id, token_label_to_id


IGNORE_INDEX = -100


@dataclass(frozen=True)
class DirectTrainingFeature:
    source_text: str
    target_text: str
    input_ids: list[int]
    attention_mask: list[int]
    offset_mapping: list[tuple[int, int]]
    word_token_indices: list[int]
    word_token_mask: list[bool]
    gap_left_indices: list[int]
    gap_right_indices: list[int]
    gap_mask: list[bool]
    token_edit_label_ids: list[int]
    gap_label_ids: list[int]
    rule_tag_ids: list[int]
    sample_weight: float = 1.0


class DebugTokenizer:
    """Small offset-aware tokenizer for tests and debug-model smoke runs."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[CLS]": 2,
            "[SEP]": 3,
        }
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.cls_token_id = 2
        self.sep_token_id = 3
        self.pattern = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*|\d+|[^\w\s]", re.UNICODE)

    def __call__(
        self,
        text: str,
        *,
        return_offsets_mapping: bool = True,
        truncation: bool = True,
        padding: str | bool = "max_length",
        max_length: int = 128,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
        **_: Any,
    ) -> dict[str, Any]:
        token_triples = [
            (match.group(0), match.start(), match.end())
            for match in self.pattern.finditer(text)
        ]
        if add_special_tokens:
            usable_length = max(0, max_length - 2) if truncation else len(token_triples)
            token_triples = token_triples[:usable_length]
            pieces = [("[CLS]", 0, 0), *token_triples, ("[SEP]", 0, 0)]
        else:
            pieces = token_triples[:max_length] if truncation else token_triples

        input_ids = [self._id(token) for token, _start, _end in pieces]
        offsets = [(start, end) for _token, start, end in pieces]
        attention_mask = [1] * len(input_ids)

        if padding == "max_length":
            pad_count = max(0, max_length - len(input_ids))
            input_ids.extend([self.pad_token_id] * pad_count)
            attention_mask.extend([0] * pad_count)
            offsets.extend([(0, 0)] * pad_count)

        if truncation:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            offsets = offsets[:max_length]

        result: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        if return_tensors is not None:
            raise ValueError("DebugTokenizer does not support return_tensors.")
        return result

    def _id(self, token: str) -> int:
        normalized = token.lower()
        if normalized not in self.vocab:
            self.vocab[normalized] = len(self.vocab)
        return self.vocab[normalized]


def build_direct_training_feature(
    example: GeneratedExample,
    tokenizer: Any,
    max_length: int,
) -> DirectTrainingFeature:
    encoded = _encode(tokenizer, example.source_text, max_length)
    offsets = encoded["offset_mapping"]
    max_words = max_length

    word_token_indices = [0] * max_words
    word_token_mask = [False] * max_words
    token_edit_label_ids = [IGNORE_INDEX] * max_words
    rule_tag_ids = [IGNORE_INDEX] * max_words

    source_tokens = example.source_tokens[:max_words]
    for index, word in enumerate(source_tokens):
        token_index = _first_overlapping_token_index(offsets, word.start, word.end)
        if token_index is None:
            continue
        word_token_indices[index] = token_index
        word_token_mask[index] = True
        token_edit_label_ids[index] = token_label_to_id(example.token_edit_labels[index])
        rule_tag_ids[index] = rule_tag_to_id(example.rule_ids[index])

    gap_left_indices = [0] * max_words
    gap_right_indices = [-1] * max_words
    gap_mask = [False] * max_words
    gap_label_ids = [IGNORE_INDEX] * max_words

    for index in range(len(source_tokens)):
        if not word_token_mask[index]:
            continue
        gap_left_indices[index] = word_token_indices[index]
        if index + 1 < len(source_tokens) and word_token_mask[index + 1]:
            gap_right_indices[index] = word_token_indices[index + 1]
        else:
            gap_right_indices[index] = -1
        gap_mask[index] = True
        gap_label_ids[index] = gap_label_to_id(example.gap_labels[index])

    return DirectTrainingFeature(
        source_text=example.source_text,
        target_text=example.target_text,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        offset_mapping=offsets,
        word_token_indices=word_token_indices,
        word_token_mask=word_token_mask,
        gap_left_indices=gap_left_indices,
        gap_right_indices=gap_right_indices,
        gap_mask=gap_mask,
        token_edit_label_ids=token_edit_label_ids,
        gap_label_ids=gap_label_ids,
        rule_tag_ids=rule_tag_ids,
        sample_weight=_sample_weight(example),
    )


class DirectBatchCollator:
    def collate(self, features: list[DirectTrainingFeature]) -> dict[str, Any]:
        import torch

        return {
            "input_ids": torch.tensor([feature.input_ids for feature in features], dtype=torch.long),
            "attention_mask": torch.tensor([feature.attention_mask for feature in features], dtype=torch.long),
            "word_token_indices": torch.tensor(
                [feature.word_token_indices for feature in features],
                dtype=torch.long,
            ),
            "word_token_mask": torch.tensor(
                [feature.word_token_mask for feature in features],
                dtype=torch.bool,
            ),
            "gap_left_indices": torch.tensor(
                [feature.gap_left_indices for feature in features],
                dtype=torch.long,
            ),
            "gap_right_indices": torch.tensor(
                [feature.gap_right_indices for feature in features],
                dtype=torch.long,
            ),
            "gap_mask": torch.tensor([feature.gap_mask for feature in features], dtype=torch.bool),
            "labels": {
                "token_edit_label_ids": torch.tensor(
                    [feature.token_edit_label_ids for feature in features],
                    dtype=torch.long,
                ),
                "gap_label_ids": torch.tensor(
                    [feature.gap_label_ids for feature in features],
                    dtype=torch.long,
                ),
                "rule_tag_ids": torch.tensor(
                    [feature.rule_tag_ids for feature in features],
                    dtype=torch.long,
                ),
                "sample_weight": torch.tensor(
                    [feature.sample_weight for feature in features],
                    dtype=torch.float,
                ),
            },
        }


def _encode(tokenizer: Any, text: str, max_length: int) -> dict[str, Any]:
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    input_ids = _to_flat_list(encoded["input_ids"])
    attention_mask = _to_flat_list(encoded["attention_mask"])
    offset_mapping = [tuple(item) for item in _to_flat_list(encoded["offset_mapping"])]

    if len(input_ids) < max_length:
        input_ids.extend([0] * (max_length - len(input_ids)))
    if len(attention_mask) < max_length:
        attention_mask.extend([0] * (max_length - len(attention_mask)))
    if len(offset_mapping) < max_length:
        offset_mapping.extend([(0, 0)] * (max_length - len(offset_mapping)))

    return {
        "input_ids": [int(value) for value in input_ids[:max_length]],
        "attention_mask": [int(value) for value in attention_mask[:max_length]],
        "offset_mapping": [
            (int(start), int(end))
            for start, end in offset_mapping[:max_length]
        ],
    }


def _to_flat_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        return list(value[0])
    return list(value)


def _first_overlapping_token_index(
    offsets: list[tuple[int, int]],
    start: int,
    end: int,
) -> int | None:
    for index, (token_start, token_end) in enumerate(offsets):
        if token_end <= token_start:
            continue
        if token_start < end and start < token_end:
            return index
    return None


def _sample_weight(example: GeneratedExample) -> float:
    raw_weight = example.metadata.get("loss_weight", 1.0)
    try:
        weight = float(raw_weight)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(weight) or weight <= 0:
        return 1.0
    return weight


__all__ = [
    "DirectBatchCollator",
    "DirectTrainingFeature",
    "DebugTokenizer",
    "IGNORE_INDEX",
    "build_direct_training_feature",
]

