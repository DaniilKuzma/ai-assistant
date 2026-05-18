from __future__ import annotations

import re

from src.preprocessing.protected_spans import find_protected_spans


def normalize_spacing(text: str) -> str:
    spans = find_protected_spans(text)
    if spans:
        normalized_parts: list[str] = []
        cursor = 0
        for span in spans:
            normalized_parts.append(_normalize_unprotected_spacing(text[cursor : span.start]))
            normalized_parts.append(text[span.start : span.end])
            cursor = span.end
        normalized_parts.append(_normalize_unprotected_spacing(text[cursor:]))
        text = "".join(normalized_parts)
        return re.sub(r"\s{2,}", " ", text).strip()
    return _normalize_unprotected_spacing(text).strip()


def _normalize_unprotected_spacing(text: str) -> str:
    text = re.sub(r"\s+([,.!?:;])", r"\1", text)
    text = re.sub(r"([,;:])(?=\S)", r"\1 ", text)
    text = re.sub(r"\s*—\s*", " — ", text)
    text = re.sub(r"([«(])\s+", r"\1", text)
    text = re.sub(r"\s+([»)])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text
