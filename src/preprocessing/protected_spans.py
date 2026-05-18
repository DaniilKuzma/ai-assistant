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
    ("abbreviation", re.compile(r"\b(?:США|РФ|НББ|ООО|АО|ИП)\b")),
    (
        "technical_id",
        re.compile(
            r"(?<!\w)(?=[A-Za-zА-Яа-яЁё0-9_]*\d)"
            r"(?=[A-Za-zА-Яа-яЁё0-9_]*[A-Za-zА-Яа-яЁё])"
            r"[A-Za-zА-Яа-яЁё]{1,}[A-Za-zА-Яа-яЁё0-9_]*\b"
        ),
    ),
    ("url", re.compile(r"https?://[^\s]+|www\.[^\s]+", re.IGNORECASE)),
    ("email", re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")),
    ("date", re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")),
    ("abbreviation", re.compile(r"\b\d{4}\s+[гГ]\.|\b(?:г|см|т\.д|т\.п|ул|стр|рис)\.|№\s*\d+", re.IGNORECASE)),
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
