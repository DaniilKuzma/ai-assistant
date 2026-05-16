from __future__ import annotations

import re


def normalize_spacing(text: str) -> str:
    text = re.sub(r"\s+([,.!?:;])", r"\1", text)
    text = re.sub(r"([,;:])(?=\S)", r"\1 ", text)
    text = re.sub(r"\s*—\s*", " — ", text)
    text = re.sub(r"([«(])\s+", r"\1", text)
    text = re.sub(r"\s+([»)])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()
