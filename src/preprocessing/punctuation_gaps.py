from __future__ import annotations

from src.preprocessing.tokenizer import tokenize_words


def word_gap_token_indices(text: str, offsets: list[tuple[int, int]], max_gaps: int) -> tuple[list[int], list[bool]]:
    """Map each word-final punctuation gap to the encoder token that represents it."""

    indices: list[int] = []
    mask: list[bool] = []
    for word in tokenize_words(text)[:max_gaps]:
        token_indexes = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > start and start < word.end and word.start < end
        ]
        if token_indexes:
            indices.append(max(token_indexes))
            mask.append(True)
        else:
            indices.append(0)
            mask.append(False)

    pad_count = max_gaps - len(indices)
    if pad_count > 0:
        indices.extend([0] * pad_count)
        mask.extend([False] * pad_count)
    return indices, mask


def word_gap_context_token_indices(
    text: str,
    offsets: list[tuple[int, int]],
    max_gaps: int,
) -> tuple[list[int], list[int], list[bool]]:
    """Map word-final gaps to left and right encoder token indices."""

    words = tokenize_words(text)[:max_gaps]
    left_indices: list[int] = []
    right_indices: list[int] = []
    mask: list[bool] = []

    for index, word in enumerate(words):
        left_token_indexes = _token_indexes_for_span(offsets, word.start, word.end)
        next_word = words[index + 1] if index + 1 < len(words) else None
        right_token_indexes = _token_indexes_for_span(offsets, next_word.start, next_word.end) if next_word else []
        if left_token_indexes:
            left_index = max(left_token_indexes)
            right_index = min(right_token_indexes) if right_token_indexes else left_index
            left_indices.append(left_index)
            right_indices.append(right_index)
            mask.append(True)
        else:
            left_indices.append(0)
            right_indices.append(0)
            mask.append(False)

    pad_count = max_gaps - len(left_indices)
    if pad_count > 0:
        left_indices.extend([0] * pad_count)
        right_indices.extend([0] * pad_count)
        mask.extend([False] * pad_count)
    return left_indices, right_indices, mask


def _token_indexes_for_span(offsets: list[tuple[int, int]], start: int, end: int) -> list[int]:
    return [index for index, (token_start, token_end) in enumerate(offsets) if token_end > start and token_start < end]
