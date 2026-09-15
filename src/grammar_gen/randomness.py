from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TypeVar


T = TypeVar("T")


class RandomSource:
    """Small deterministic wrapper around ``random.Random`` for generators."""

    def __init__(self, seed: int | None = None) -> None:
        self._random = random.Random(seed)

    def choice(self, items: Sequence[T]) -> T:
        if not items:
            raise ValueError("Cannot choose from an empty sequence.")
        return self._random.choice(items)

    def weighted_choice(self, items: Sequence[tuple[T, float]]) -> T:
        if not items:
            raise ValueError("Cannot choose from an empty weighted sequence.")

        total = 0.0
        for _, weight in items:
            if weight < 0:
                raise ValueError("Weights must be non-negative.")
            total += weight

        if total <= 0:
            raise ValueError("At least one weight must be positive.")

        threshold = self._random.uniform(0, total)
        cumulative = 0.0
        for item, weight in items:
            cumulative += weight
            if threshold <= cumulative:
                return item
        return items[-1][0]

    def chance(self, probability: float) -> bool:
        if probability < 0 or probability > 1:
            raise ValueError("Probability must be between 0 and 1.")
        return self._random.random() < probability

    def randint(self, left: int, right: int) -> int:
        return self._random.randint(left, right)
