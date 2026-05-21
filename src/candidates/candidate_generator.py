from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import time
from typing import Any, Callable, Protocol, Sequence

from src.candidates.dictionary_candidates import (
    MIN_FUZZY_SCORE,
    dictionary_candidate_specs,
    is_dictionary_candidate_token,
    normalize_dictionary_token,
)
from src.candidates.morphology import is_known_word
from src.preprocessing.protected_spans import find_protected_spans
from src.preprocessing.tokenizer import tokenize_words
from src.rules.base import RuleContext, RuleMode
from src.rules.punctuation import generate_punctuation_candidates
from src.rules.registry import orthography_rules


DEFAULT_MAX_DICTIONARY_CHOICES_PER_TOKEN = 50_000
RUSSIAN_ALPHABET = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"


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
        dictionary_token_cache_enabled: bool = True,
        dictionary_token_cache_max_size: int = 200_000,
        dictionary_max_choices_per_token: int = DEFAULT_MAX_DICTIONARY_CHOICES_PER_TOKEN,
        dictionary_use_second_letter_index: bool = False,
    ) -> None:
        self.dictionary_lexicon = tuple(dictionary_lexicon) if dictionary_lexicon is not None else None
        self.dictionary_provider = dictionary_provider
        self.dictionary_limit = max(0, dictionary_limit)
        self.dictionary_min_score = float(dictionary_min_score)
        self.dictionary_yo_e_enabled = bool(dictionary_yo_e_enabled)
        self.syntax_provider = syntax_provider or _default_syntax_provider
        self.dictionary_token_cache_enabled = bool(dictionary_token_cache_enabled)
        self.dictionary_token_cache_max_size = max(0, int(dictionary_token_cache_max_size))
        self.dictionary_max_choices_per_token = max(1, int(dictionary_max_choices_per_token))
        self.dictionary_use_second_letter_index = bool(dictionary_use_second_letter_index)
        self._dictionary_index_lexicon_id: int | None = None
        self._dictionary_index: dict[tuple[str, int], tuple[str, ...]] = {}
        self._dictionary_second_index: dict[tuple[str, str, int], tuple[str, ...]] = {}
        self._dictionary_lexicon_set: set[str] = set()
        self._dictionary_spec_cache: OrderedDict[tuple[Any, ...], tuple[Any, ...]] = OrderedDict()
        self._dictionary_cache_hits = 0
        self._dictionary_cache_misses = 0

    @classmethod
    def from_config(cls, config: dict[str, Any], *, purpose: str = "inference") -> "CandidateGenerator":
        dictionary_config = config.get("dictionary", {})
        yo_e_config = dictionary_config.get("yo_e", {})
        token_cache_config = dictionary_config.get("token_cache", {}) or {}
        syntax_provider = _syntax_provider_from_config(config, purpose)
        from src.config.dictionary import dictionary_provider_from_config

        return cls(
            dictionary_provider=dictionary_provider_from_config(config),
            dictionary_limit=int(dictionary_config.get("max_candidates", 2)),
            dictionary_min_score=float(dictionary_config.get("min_score", MIN_FUZZY_SCORE)),
            dictionary_yo_e_enabled=bool(yo_e_config.get("enabled", False)),
            syntax_provider=syntax_provider,
            dictionary_token_cache_enabled=bool(token_cache_config.get("enabled", True)),
            dictionary_token_cache_max_size=int(token_cache_config.get("max_size", 200_000)),
            dictionary_max_choices_per_token=int(
                dictionary_config.get("max_choices_per_token", DEFAULT_MAX_DICTIONARY_CHOICES_PER_TOKEN)
            ),
            dictionary_use_second_letter_index=bool(dictionary_config.get("use_second_letter_index", False)),
        )

    def generate(
        self,
        text: str,
        *,
        profile: dict[str, Any] | None = None,
        dictionary_policy: str = "all",
    ) -> list[Candidate]:
        candidates: list[Candidate] = []
        seen: set[tuple[int, int, str, str]] = set()
        words = tokenize_words(text)
        word_tuple = tuple(words)
        rules = orthography_rules()
        rule_specs = {rule.spec.id: rule.spec for rule in rules}
        protected_spans = tuple((span.start, span.end) for span in find_protected_spans(text))
        syntax_started = time.perf_counter()
        syntax_tokens = tuple(self._syntax_tokens(text))
        if profile is not None:
            profile["syntax_ms"] = profile.get("syntax_ms", 0.0) + _elapsed_ms(syntax_started)
            profile.setdefault("dictionary_candidate_ms", 0.0)
            profile.setdefault("dictionary_candidate_count", 0)
            profile.setdefault("punctuation_candidate_ms", 0.0)
            profile.setdefault("punctuation_candidate_count", 0)
            profile.setdefault("keep_candidate_count", 0)

        for index, token in enumerate(words):
            _append_candidate(
                candidates,
                seen,
                Candidate(token.text, token.text, "keep", token.start, token.end, 1.0),
            )
            if profile is not None:
                profile["keep_candidate_count"] = profile.get("keep_candidate_count", 0) + 1
            context = RuleContext(
                text=text,
                tokens=word_tuple,
                token_index=index,
                protected_spans=protected_spans,
                syntax_tokens=syntax_tokens,
            )

            if not _span_overlaps_protected(token.start, token.end, protected_spans):
                dictionary_started = time.perf_counter()
                specs = self._dictionary_specs_for_token(token.text, dictionary_policy=dictionary_policy)
                if profile is not None:
                    profile["dictionary_candidate_ms"] = profile.get("dictionary_candidate_ms", 0.0) + _elapsed_ms(
                        dictionary_started
                    )
                    profile["dictionary_candidate_count"] = profile.get("dictionary_candidate_count", 0) + len(specs)
                for spec in specs:
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

        punctuation_started = time.perf_counter()
        punctuation_specs = generate_punctuation_candidates(
            text,
            syntax_tokens=syntax_tokens,
            syntax_provider=self.syntax_provider,
        )
        if profile is not None:
            profile["punctuation_candidate_ms"] = profile.get("punctuation_candidate_ms", 0.0) + _elapsed_ms(
                punctuation_started
            )
            profile["punctuation_candidate_count"] = profile.get("punctuation_candidate_count", 0) + len(punctuation_specs)
        for spec in punctuation_specs:
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

    def _dictionary_specs_for_token(
        self,
        token: str,
        lexicon: Sequence[str] | None = None,
        *,
        dictionary_policy: str = "all",
    ) -> tuple[Any, ...]:
        if self.dictionary_limit <= 0:
            return ()
        normalized = normalize_dictionary_token(token)
        if not is_dictionary_candidate_token(normalized):
            return ()
        known_word = is_known_word(normalized)
        if known_word and (not self.dictionary_yo_e_enabled or dictionary_policy == "unknown_only"):
            return ()
        lexicon = lexicon if lexicon is not None else self._dictionary_lexicon()
        if not lexicon:
            return ()
        key = (
            id(lexicon),
            normalized,
            self.dictionary_limit,
            self.dictionary_min_score,
            self.dictionary_yo_e_enabled,
            self.dictionary_max_choices_per_token,
            self.dictionary_use_second_letter_index,
            dictionary_policy,
        )
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        dictionary_choices = lexicon if self.dictionary_yo_e_enabled else self._dictionary_choices_for_token(normalized, lexicon)
        specs = tuple(
            dictionary_candidate_specs(
                normalized,
                dictionary_choices,
                self.dictionary_limit,
                self.dictionary_min_score,
                yo_e_enabled=self.dictionary_yo_e_enabled,
            )
        )
        self._cache_set(key, specs)
        return specs

    def _dictionary_choices_for_token(self, token: str, lexicon: Sequence[str]) -> Sequence[str]:
        normalized = normalize_dictionary_token(token)
        if len(normalized) < 4:
            return ()
        first = normalized[:1]
        if not first:
            return ()
        choices: list[str] = list(self._direct_edit_choices(normalized, lexicon))
        if self.dictionary_use_second_letter_index and len(normalized) >= 5 and len(normalized) >= 2:
            index = self._dictionary_second_index_for(lexicon)
            second = normalized[1]
            max_delta = 1
            for length in range(max(1, len(normalized) - max_delta), len(normalized) + max_delta + 1):
                choices.extend(index.get((first, second, length), ()))
        else:
            index = self._dictionary_index_for(lexicon)
            max_delta = max(2, len(normalized) // 3)
            for length in range(max(1, len(normalized) - max_delta), len(normalized) + max_delta + 1):
                choices.extend(index.get((first, length), ()))
        deduped: list[str] = []
        seen: set[str] = set()
        for word in choices:
            if word == normalized or word in seen:
                continue
            seen.add(word)
            deduped.append(word)
            if len(deduped) >= self.dictionary_max_choices_per_token:
                break
        return tuple(deduped)

    def _dictionary_index_for(self, lexicon: Sequence[str]) -> dict[tuple[str, int], tuple[str, ...]]:
        self._ensure_dictionary_indexes(lexicon)
        return self._dictionary_index

    def _dictionary_second_index_for(self, lexicon: Sequence[str]) -> dict[tuple[str, str, int], tuple[str, ...]]:
        self._ensure_dictionary_indexes(lexicon)
        return self._dictionary_second_index

    def _dictionary_lexicon_set_for(self, lexicon: Sequence[str]) -> set[str]:
        self._ensure_dictionary_indexes(lexicon)
        return self._dictionary_lexicon_set

    def _ensure_dictionary_indexes(self, lexicon: Sequence[str]) -> None:
        lexicon_id = id(lexicon)
        if self._dictionary_index_lexicon_id == lexicon_id:
            return
        buckets: dict[tuple[str, int], list[str]] = {}
        second_buckets: dict[tuple[str, str, int], list[str]] = {}
        lexicon_set: set[str] = set()
        for item in lexicon:
            word = str(item).strip().lower()
            if not word:
                continue
            lexicon_set.add(word)
            buckets.setdefault((word[:1], len(word)), []).append(word)
            if len(word) >= 2:
                second_buckets.setdefault((word[:1], word[1], len(word)), []).append(word)
        self._dictionary_index = {key: tuple(values) for key, values in buckets.items()}
        self._dictionary_second_index = {key: tuple(values) for key, values in second_buckets.items()}
        self._dictionary_lexicon_set = lexicon_set
        self._dictionary_index_lexicon_id = lexicon_id

    def _direct_edit_choices(self, token: str, lexicon: Sequence[str]) -> tuple[str, ...]:
        lexicon_set = self._dictionary_lexicon_set_for(lexicon)
        variants: list[str] = []
        seen: set[str] = set()

        def add(candidate: str) -> None:
            if candidate != token and candidate in lexicon_set and candidate not in seen:
                seen.add(candidate)
                variants.append(candidate)

        for index in range(len(token)):
            add(token[:index] + token[index + 1 :])
        for index in range(len(token) - 1):
            add(token[:index] + token[index + 1] + token[index] + token[index + 2 :])
        for index in range(len(token)):
            for char in RUSSIAN_ALPHABET:
                if char != token[index]:
                    add(token[:index] + char + token[index + 1 :])
        for index in range(len(token) + 1):
            for char in RUSSIAN_ALPHABET:
                add(token[:index] + char + token[index:])
        return tuple(variants)

    def _cache_get(self, key: tuple[Any, ...]) -> tuple[Any, ...] | None:
        if not self.dictionary_token_cache_enabled or self.dictionary_token_cache_max_size <= 0:
            return None
        cached = self._dictionary_spec_cache.get(key)
        if cached is None:
            self._dictionary_cache_misses += 1
            return None
        self._dictionary_cache_hits += 1
        self._dictionary_spec_cache.move_to_end(key)
        return cached

    def _cache_set(self, key: tuple[Any, ...], value: tuple[Any, ...]) -> None:
        if not self.dictionary_token_cache_enabled or self.dictionary_token_cache_max_size <= 0:
            return
        self._dictionary_spec_cache[key] = value
        self._dictionary_spec_cache.move_to_end(key)
        while len(self._dictionary_spec_cache) > self.dictionary_token_cache_max_size:
            self._dictionary_spec_cache.popitem(last=False)

    def dictionary_cache_stats(self) -> dict[str, float | int]:
        total = self._dictionary_cache_hits + self._dictionary_cache_misses
        return {
            "hits": self._dictionary_cache_hits,
            "misses": self._dictionary_cache_misses,
            "size": len(self._dictionary_spec_cache),
            "max_size": self.dictionary_token_cache_max_size,
            "hit_rate": (self._dictionary_cache_hits / total) if total else 0.0,
        }

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


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000.0


def _default_syntax_provider(text: str) -> Sequence[Any]:
    try:
        from src.nlp.syntax import parse_syntax
    except Exception:
        return ()
    return parse_syntax(text)


def _syntax_provider_from_config(config: dict[str, Any], purpose: str) -> Callable[[str], Sequence[Any]] | None:
    feature_build_config = config.get("training", {}).get("feature_build", {}) or {}
    if purpose == "training_features" and not bool(feature_build_config.get("enable_syntax", False)):
        return lambda _text: ()

    syntax_config = (config.get("nlp", {}) or {}).get("syntax", {}) or {}
    if not bool(syntax_config.get("enabled", True)):
        return lambda _text: ()

    try:
        from src.nlp.syntax_analyzer import SyntaxAnalyzer
    except Exception:
        return None

    analyzer = SyntaxAnalyzer(config)
    return lambda text: analyzer.analyze(text).tokens
