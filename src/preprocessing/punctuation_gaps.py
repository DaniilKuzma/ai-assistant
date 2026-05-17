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
