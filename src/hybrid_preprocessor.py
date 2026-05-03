"""Vocabulary and vectorization for the hybrid edit-based model."""

from __future__ import annotations

from collections import Counter
import pickle
from typing import Dict, Iterable, List, Sequence

import numpy as np

from edit_labels import (
    ACTION_DELETE,
    ACTION_KEEP,
    ACTION_REPLACE,
    HybridTrainingExample,
    candidate_list,
    replace_action,
)
from text_utils import normalize_word


class HybridPreprocessor:
    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(
        self,
        max_length: int = 128,
        min_freq: int = 1,
        max_vocab_size: int = 80_000,
        candidate_top_k: int = 8,
    ):
        self.max_length = max_length
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.candidate_top_k = max(1, int(candidate_top_k))
        self.word_to_id: Dict[str, int] = {self.PAD: 0, self.UNK: 1}
        self.id_to_word: Dict[int, str] = {0: self.PAD, 1: self.UNK}
        self.action_to_id = {
            ACTION_KEEP: 0,
            ACTION_DELETE: 1,
        }
        for rank in range(self.candidate_top_k):
            self.action_to_id[replace_action(rank)] = 2 + rank
        self.action_to_id[ACTION_REPLACE] = 2
        self.id_to_action = {0: ACTION_KEEP, 1: ACTION_DELETE}
        for rank in range(self.candidate_top_k):
            self.id_to_action[2 + rank] = replace_action(rank)
        self.punct_to_id = {
            "": 0,
            ",": 1,
            ".": 2,
            "?": 3,
            "!": 4,
            ":": 5,
            ";": 6,
            "—": 7,
        }
        self.id_to_punct = {v: k for k, v in self.punct_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.word_to_id)

    @property
    def action_classes(self) -> int:
        return 2 + self.candidate_top_k

    @property
    def punct_classes(self) -> int:
        return len(self.punct_to_id)

    def punct_id(self, punct: str) -> int:
        return self.punct_to_id.get(str(punct or ""), 0)

    def fit(self, examples: Iterable[HybridTrainingExample]) -> None:
        counts: Counter[str] = Counter()
        for example in examples:
            for word in example.source_words:
                counts[normalize_word(word)] += 1
            for candidates in example.candidate_words:
                for candidate in candidate_list(candidates):
                    for part in str(candidate).split():
                        counts[normalize_word(part)] += 1

        most_common = [
            word
            for word, count in counts.most_common(self.max_vocab_size - 2)
            if count >= self.min_freq and word not in self.word_to_id
        ]
        for word in most_common:
            idx = len(self.word_to_id)
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word

    def word_id(self, word: str) -> int:
        return self.word_to_id.get(normalize_word(word), self.word_to_id[self.UNK])

    def candidate_id(self, candidate: str) -> int:
        parts = str(candidate).split()
        if not parts:
            return self.word_to_id[self.UNK]
        if len(parts) == 1:
            return self.word_id(parts[0])
        known_ids = [self.word_id(part) for part in parts]
        known_ids = [idx for idx in known_ids if idx != self.word_to_id[self.UNK]]
        return known_ids[0] if known_ids else self.word_to_id[self.UNK]

    def vectorize_examples(
        self,
        examples: Sequence[HybridTrainingExample],
        clean_action_keep_weight: float = 2.0,
        dirty_action_keep_weight: float = 1.0,
        action_change_weight: float = 3.0,
        clean_punct_keep_weight: float = 2.0,
        dirty_punct_keep_weight: float = 1.0,
        punct_change_weight: float = 8.0,
        final_punct_weight: float = 8.0,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        n = len(examples)
        shape = (n, self.max_length)
        token_ids = np.zeros(shape, dtype=np.int32)
        candidate_ids = np.zeros((n, self.max_length, self.candidate_top_k), dtype=np.int32)
        source_punct_ids = np.zeros(shape, dtype=np.int32)
        action_ids = np.zeros(shape, dtype=np.int32)
        punct_ids = np.zeros(shape, dtype=np.int32)
        action_weights = np.zeros(shape, dtype=np.float32)
        punct_weights = np.zeros(shape, dtype=np.float32)

        for i, example in enumerate(examples):
            limit = min(len(example.source_words), self.max_length)
            for j in range(limit):
                token_ids[i, j] = self.word_id(example.source_words[j])
                candidates = self._candidate_list_for_vector(example.candidate_words[j])
                for rank, candidate in enumerate(candidates):
                    candidate_ids[i, j, rank] = self.candidate_id(candidate)
                source_punct_ids[i, j] = self.punct_id(example.source_punct_labels[j])
                action_ids[i, j] = self.action_to_id.get(example.action_labels[j], 0)
                punct_ids[i, j] = self.punct_id(example.target_punct_labels[j])
                if example.action_labels[j] != ACTION_KEEP:
                    action_weights[i, j] = action_change_weight
                elif example.is_clean:
                    action_weights[i, j] = clean_action_keep_weight
                else:
                    action_weights[i, j] = dirty_action_keep_weight
                is_final_token = j + 1 >= limit
                if example.source_punct_labels[j] != example.target_punct_labels[j]:
                    punct_weights[i, j] = final_punct_weight if is_final_token else punct_change_weight
                elif example.is_clean:
                    punct_weights[i, j] = clean_punct_keep_weight
                else:
                    punct_weights[i, j] = dirty_punct_keep_weight

        inputs = {
            "token_ids": token_ids,
            "candidate_ids": candidate_ids,
            "source_punct_ids": source_punct_ids,
        }
        outputs = {
            "action": action_ids,
            "punct": punct_ids,
        }
        sample_weight = {
            "action": action_weights,
            "punct": punct_weights,
        }
        return inputs, outputs, sample_weight

    def vectorize_inference(
        self,
        source_words: Sequence[str],
        candidate_words: Sequence[Sequence[str]] | Sequence[str],
        source_puncts: Sequence[str],
    ) -> dict[str, np.ndarray]:
        token_ids = np.zeros((1, self.max_length), dtype=np.int32)
        candidate_ids = np.zeros((1, self.max_length, self.candidate_top_k), dtype=np.int32)
        source_punct_ids = np.zeros((1, self.max_length), dtype=np.int32)
        limit = min(len(source_words), self.max_length)
        for j in range(limit):
            token_ids[0, j] = self.word_id(source_words[j])
            candidates = self._candidate_list_for_vector(candidate_words[j])
            for rank, candidate in enumerate(candidates):
                candidate_ids[0, j, rank] = self.candidate_id(candidate)
            source_punct_ids[0, j] = self.punct_id(source_puncts[j])
        return {
            "token_ids": token_ids,
            "candidate_ids": candidate_ids,
            "source_punct_ids": source_punct_ids,
        }

    def _candidate_list_for_vector(self, value: object) -> List[str]:
        candidates = candidate_list(value)
        if len(candidates) < self.candidate_top_k:
            candidates = candidates + [""] * (self.candidate_top_k - len(candidates))
        return candidates[: self.candidate_top_k]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "HybridPreprocessor":
        with open(path, "rb") as f:
            preprocessor = pickle.load(f)
        if not hasattr(preprocessor, "candidate_top_k"):
            preprocessor.candidate_top_k = 1
        return preprocessor
