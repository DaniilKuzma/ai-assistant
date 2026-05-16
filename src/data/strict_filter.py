from __future__ import annotations

from src.alignment.aligner import Aligner


def keep_supported_pair(source: str, target: str) -> bool:
    return Aligner().align(source, target).is_supported
