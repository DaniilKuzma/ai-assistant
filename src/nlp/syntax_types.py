from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ClauseKind = Literal["main", "subordinate", "coordinate", "asyndetic", "unknown"]
PhraseKind = Literal[
    "participial",
    "adverbial_participle",
    "apposition",
    "clarification",
    "comparative",
    "homogeneous_series",
    "subject_predicate",
    "direct_speech",
    "quote_span",
    "bracket_span",
]


@dataclass(frozen=True)
class SyntaxToken:
    id: int
    text: str
    lemma: str
    pos: str
    feats: dict[str, Any]
    start: int
    end: int
    head_id: int | None
    dep_rel: str
    sentence_id: int
    is_punctuation: bool
    is_conjunction: bool
    is_particle: bool
    is_verb: bool
    is_noun: bool
    is_adjective: bool
    is_participle: bool
    is_adverbial_participle: bool
    is_pronoun: bool
    is_named_entity: bool
    ner_type: str | None = None

    @property
    def rel(self) -> str:
        """Compatibility alias for the previous syntax layer."""

        return self.dep_rel

    @property
    def ner(self) -> str | None:
        """Compatibility alias for the previous syntax layer."""

        return self.ner_type


@dataclass(frozen=True)
class SyntaxSentence:
    text: str
    start: int
    end: int
    tokens: list[SyntaxToken] = field(default_factory=list)
    root_ids: list[int] = field(default_factory=list)
    has_finite_verb: bool = False
    predicate_token_ids: list[int] = field(default_factory=list)
    subject_token_ids: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class ClauseSpan:
    start: int
    end: int
    token_ids: list[int]
    kind: ClauseKind
    trigger_token_id: int | None = None
    trigger_text: str = ""
    confidence: float = 0.0


@dataclass(frozen=True)
class PhraseSpan:
    start: int
    end: int
    token_ids: list[int]
    kind: PhraseKind
    confidence: float = 0.0
    evidence: str = ""


@dataclass(frozen=True)
class SyntaxAnalysis:
    text: str
    tokens: list[SyntaxToken]
    sentences: list[SyntaxSentence]
    clauses: list[ClauseSpan]
    phrases: list[PhraseSpan]
    protected_spans: list[tuple[int, int]]
    errors: list[str]
    backend: str
    cache_key: str
