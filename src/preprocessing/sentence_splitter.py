from __future__ import annotations

import re


def split_sentences(text: str) -> list[str]:
    """A small dependency-free sentence splitter for pipeline smoke runs."""

    sentences = [part.strip() for part in re.split(r"(?<=[.!?…])\s+", text.strip()) if part.strip()]
    return sentences or ([text] if text else [])
