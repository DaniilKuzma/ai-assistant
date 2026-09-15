from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeEdit:
    start: int
    end: int
    source: str
    replacement: str
    edit_type: str
    rule_id: str
    confidence: float = 1.0
    explanation: str = ""


@dataclass
class CorrectionResult:
    source_text: str
    corrected_text: str
    edits: list[RuntimeEdit]
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["CorrectionResult", "RuntimeEdit"]
