from __future__ import annotations

from collections.abc import Callable


def run_until_stable(text: str, step: Callable[[str], str], max_passes: int = 3) -> str:
    current = text
    for _ in range(max_passes):
        updated = step(current)
        if updated == current:
            break
        current = updated
    return current
