from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from src.candidates.dictionary_candidates import MIN_FUZZY_SCORE, dictionary_candidate_specs
from src.preprocessing.protected_spans import find_protected_spans
from src.preprocessing.tokenizer import tokenize_words
from src.rules.base import RuleContext, RuleMode
from src.rules.punctuation import generate_punctuation_candidates
from src.rules.registry import orthography_rules


class DictionaryProvider(Protocol):
    def get_lexicon(self) -> Sequence[str]:
        ...


@dataclass(frozen=True)
class Candidate:
    source: str
    replacement: str
    edit_type: str
    start: int
    end: int
    confidence: float = 1.0
    requires_model: bool = False
    rule_id: str = ""
    mode: RuleMode = "deterministic"
    action: str = ""
    label: str = ""
    gap_index: int | None = None
    requires: tuple[str, ...] = ()
    group: str = ""

    @property
    def requires_scoring(self) -> bool:
        return (
            self.requires_model
            or self.mode in {"candidate_only", "model_required"}
            or "model" in self.requires
            or "syntax" in self.requires
        )


class CandidateGenerator:
    """Generate bounded candidates from rules and an optional injected dictionary lexicon."""

    def __init__(
        self,
        dictionary_lexicon: Sequence[str] | None = None,
        dictionary_provider: DictionaryProvider | None = None,
        dictionary_limit: int = 2,
        dictionary_min_score: float = MIN_FUZZY_SCORE,
        dictionary_yo_e_enabled: bool = False,
        syntax_provider: Callable[[str], Sequence[Any]] | None = None,
    ) -> None:
        self.dictionary_lexicon = tuple(dictionary_lexicon) if dictionary_lexicon is not None else None
        self.dictionary_provider = dictionary_provider
        self.dictionary_limit = max(0, dictionary_limit)
        self.dictionary_min_score = float(dictionary_min_score)
        self.dictionary_yo_e_enabled = bool(dictionary_yo_e_enabled)
        self.syntax_provider = syntax_provider or _default_syntax_provider

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CandidateGenerator":
        dictionary_config = config.get("dictionary", {})
        yo_e_config = dictionary_config.get("yo_e", {})
        from src.config.dictionary import dictionary_provider_from_config

        return cls(
            dictionary_provider=dictionary_provider_from_config(config),
            dictionary_limit=int(dictionary_config.get("max_candidates", 2)),
            dictionary_min_score=float(dictionary_config.get("min_score", MIN_FUZZY_SCORE)),
            dictionary_yo_e_enabled=bool(yo_e_config.get("enabled", False)),
        )

    def generate(self, text: str) -> list[Candidate]:
        candidates: list[Candidate] = []
        seen: set[tuple[int, int, str, str]] = set()
        words = tokenize_words(text)
        word_tuple = tuple(words)
        rules = orthography_rules()
        rule_specs = {rule.spec.id: rule.spec for rule in rules}
        protected_spans = tuple((span.start, span.end) for span in find_protected_spans(text))
        syntax_tokens = tuple(self._syntax_tokens(text))

        for index, token in enumerate(words):
            _append_candidate(
                candidates,
                seen,
                Candidate(token.text, token.text, "keep", token.start, token.end, 1.0),
            )
            context = RuleContext(
                text=text,
                tokens=word_tuple,
                token_index=index,
                protected_spans=protected_spans,
                syntax_tokens=syntax_tokens,
            )

            lexicon = self._dictionary_lexicon()
            if lexicon and not _span_overlaps_protected(token.start, token.end, protected_spans):
                for spec in dictionary_candidate_specs(
                    token.text,
                    lexicon,
                    self.dictionary_limit,
                    self.dictionary_min_score,
                    yo_e_enabled=self.dictionary_yo_e_enabled,
                ):
                    rule_spec = rule_specs.get(spec.rule_id)
                    _append_candidate(
                        candidates,
                        seen,
                        Candidate(
                            token.text,
                            _match_case(token.text, spec.replacement),
                            spec.edit_type,
                            token.start,
                            token.end,
                            spec.confidence,
                            spec.requires_model,
                            spec.rule_id,
                            spec.mode,
                            requires=rule_spec.requires if rule_spec else ("dictionary", "model"),
                            group=rule_spec.group if rule_spec else "",
                        ),
                    )

            for rule in rules:
                generator = getattr(rule, "generate_candidates", getattr(rule, "generate", None))
                if rule.spec.scope != "token" or generator is None:
                    continue
                for spec in generator(token.text, context):
                    _append_candidate(
                        candidates,
                        seen,
                        Candidate(
                            token.text,
                            _match_case(token.text, spec.replacement),
                            spec.edit_type,
                            token.start,
                            token.end,
                            spec.confidence,
                            spec.requires_model,
                            spec.rule_id,
                            spec.mode,
                            requires=spec.requires,
                            group=spec.group,
                        ),
                    )

            for rule in rules:
                if rule.spec.scope != "span" or not hasattr(rule, "generate_span"):
                    continue
                for spec in rule.generate_span(text, word_tuple, index):
                    _append_candidate(
                        candidates,
                        seen,
                        Candidate(
                            spec.source,
                            spec.replacement,
                            spec.edit_type,
                            spec.start,
                            spec.end,
                            spec.confidence,
                            spec.requires_model,
                            spec.rule_id,
                            spec.mode,
                            requires=spec.requires,
                            group=spec.group,
                        ),
                    )

        for spec in generate_punctuation_candidates(text):
            _append_candidate(
                candidates,
                seen,
                Candidate(
                    spec.source,
                    spec.replacement,
                    spec.edit_type,
                    spec.start,
                    spec.end,
                    spec.confidence,
                    spec.requires_model,
                    spec.rule_id,
                    spec.mode,
                    spec.action,
                    spec.label,
                    spec.gap_index,
                    spec.requires,
                    spec.group,
                ),
            )

        return candidates

    def _dictionary_lexicon(self) -> Sequence[str]:
        if self.dictionary_lexicon is not None:
            return self.dictionary_lexicon
        if self.dictionary_provider is None:
            return ()
        return self.dictionary_provider.get_lexicon()

    def _syntax_tokens(self, text: str) -> Sequence[Any]:
        try:
            return self.syntax_provider(text)
        except Exception:
            return ()


def _match_case(source: str, replacement: str) -> str:
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _append_candidate(candidates: list[Candidate], seen: set[tuple[int, int, str, str]], candidate: Candidate) -> None:
    key = (candidate.start, candidate.end, candidate.replacement.lower(), candidate.edit_type)
    if key in seen:
        return
    seen.add(key)
    candidates.append(candidate)


def _span_overlaps_protected(start: int, end: int, protected_spans: tuple[tuple[int, int], ...]) -> bool:
    return any(start < protected_end and protected_start < end for protected_start, protected_end in protected_spans)


def _default_syntax_provider(text: str) -> Sequence[Any]:
    try:
        from src.nlp.syntax import parse_syntax
    except Exception:
        return ()
    return parse_syntax(text)
