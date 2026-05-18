from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Protocol


RuleScope = Literal["token", "span", "punctuation_gap"]
RuleMode = Literal["deterministic", "candidate_only", "model_required"]


@dataclass(frozen=True)
class RuleSpec:
    id: str
    group: str
    scope: RuleScope
    edit_type: str
    mode: RuleMode
    confidence: float
    requires: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class RuleCandidate:
    replacement: str
    edit_type: str
    confidence: float
    requires_model: bool
    rule_id: str
    mode: RuleMode = "deterministic"


@dataclass(frozen=True)
class RuleCorruption:
    start: int
    end: int
    replacement: str
    error_type: str
    group: str = ""
    rule_id: str = ""

    def apply(self, text: str) -> str:
        return text[: self.start] + self.replacement + text[self.end :]


@dataclass(frozen=True)
class RuleContext:
    text: str = ""
    tokens: tuple[Any, ...] = ()
    token_index: int = -1
    protected_spans: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class RuleEdit:
    source: str
    replacement: str
    edit_type: str
    start: int
    end: int
    confidence: float
    requires_model: bool
    rule_id: str
    mode: RuleMode = "deterministic"


class Rule(Protocol):
    spec: RuleSpec


class TokenRule(Rule, Protocol):
    def generate_candidates(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        ...

    def generate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        ...


class SpanRule(Rule, Protocol):
    def generate_span(self, text: str, tokens: tuple[Any, ...], token_index: int) -> Iterable[RuleEdit]:
        ...


class PunctuationRule(Rule, Protocol):
    def apply(self, text: str) -> tuple[str, tuple[RuleEdit, ...]]:
        ...


class SyntheticRule(Rule, Protocol):
    def transformations(self, text: str, protected: tuple[tuple[int, int], ...]) -> Iterable[Any]:
        ...
