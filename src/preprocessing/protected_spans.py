from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ProtectedSpan:
    start: int
    end: int
    text: str
    kind: str


PROTECTED_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("url", re.compile(r"https?://[^\s]+|www\.[^\s]+", re.IGNORECASE)),
    ("email", re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")),
    ("date", re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")),
    ("number", re.compile(r"\b\d+(?:[.,:/-]\d+)*\b")),
    ("nickname", re.compile(r"(?<!\w)@[A-Za-zА-Яа-яЁё0-9_]+")),
    ("english", re.compile(r"\b[A-Za-z][A-Za-z0-9_+-]*\b")),
    ("code", re.compile(r"`[^`]+`|[A-Za-z_][A-Za-z0-9_]*\([^)]*\)")),
)


def find_protected_spans(text: str) -> list[ProtectedSpan]:
    spans: list[ProtectedSpan] = []
    occupied: list[range] = []

    for kind, pattern in PROTECTED_PATTERNS:
        for match in pattern.finditer(text):
            current = range(match.start(), match.end())
            if any(_overlaps(current, other) for other in occupied):
                continue
            occupied.append(current)
            spans.append(ProtectedSpan(match.start(), match.end(), match.group(0), kind))

    return sorted(spans, key=lambda span: span.start)


def _overlaps(left: range, right: range) -> bool:
    return left.start < right.stop and right.start < left.stop
