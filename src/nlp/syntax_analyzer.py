from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import logging
from typing import Any

from src.nlp.syntax_types import SyntaxAnalysis, SyntaxSentence, SyntaxToken
from src.preprocessing.protected_spans import find_protected_spans


logger = logging.getLogger(__name__)

DEFAULT_SYNTAX_CONFIG: dict[str, Any] = {
    "enabled": True,
    "backend": "natasha",
    "cache_enabled": True,
    "cache_max_size": 50_000,
    "cache_max_text_length": 20_000,
    "fail_open": True,
    "debug_errors": False,
}
PUNCTUATION_CHARS = set(",.!?:;—…\"'()«»[]")
CONJUNCTION_POSES = {"CCONJ", "SCONJ"}
CONJUNCTION_WORDS = {"а", "но", "однако", "зато", "что", "чтобы", "если", "когда", "поскольку", "где", "куда", "откуда"}
PARTICIPLE_SUFFIXES = ("вший", "вшая", "вшее", "вшие", "енный", "енная", "енное", "енные", "ённый", "ённая", "ённое", "ённые", "анный", "анная", "анное", "анные")
ADVERBIAL_PARTICIPLE_SUFFIXES = ("вшись", "ившись", "авшись", "явшись", "вши", "ивши", "авши", "явши", "ив", "ав", "яв")


@dataclass(frozen=True)
class _SyntaxPipeline:
    segmenter: Any
    morph_vocab: Any
    morph_tagger: Any
    syntax_parser: Any
    ner_tagger: Any | None
    doc_factory: Any


@lru_cache(maxsize=1)
def syntax_pipeline() -> _SyntaxPipeline | None:
    try:
        from natasha import (
            Doc,
            MorphVocab,
            NewsEmbedding,
            NewsMorphTagger,
            NewsNERTagger,
            NewsSyntaxParser,
            Segmenter,
        )

        embedding = NewsEmbedding()
        return _SyntaxPipeline(
            segmenter=Segmenter(),
            morph_vocab=MorphVocab(),
            morph_tagger=NewsMorphTagger(embedding),
            syntax_parser=NewsSyntaxParser(embedding),
            ner_tagger=_optional_ner_tagger(NewsNERTagger, embedding),
            doc_factory=Doc,
        )
    except Exception as exc:  # pragma: no cover - depends on optional NLP stack.
        logger.warning("Natasha syntax pipeline unavailable: %s: %s", type(exc).__name__, exc)
        return None


class SyntaxAnalyzer:
    """Canonical fail-open syntax analyzer with lazy backend loading."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = _syntax_config(config)
        self._cache: OrderedDict[str, SyntaxAnalysis] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

    def analyze(self, text: str) -> SyntaxAnalysis:
        cache_key = _cache_key(text)
        if self._can_cache(text):
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache_hits += 1
                self._cache.move_to_end(cache_key)
                return cached

        self._cache_misses += 1
        analysis = self._analyze_uncached(text, cache_key)
        if self._can_cache(text):
            self._cache[cache_key] = analysis
            self._cache.move_to_end(cache_key)
            self._trim_cache()
        return analysis

    def cache_info(self) -> dict[str, int | bool]:
        return {
            "enabled": self._cache_enabled(),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._cache),
            "max_size": int(self.config["cache_max_size"]),
            "max_text_length": int(self.config["cache_max_text_length"]),
        }

    def clear_cache(self) -> None:
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def _analyze_uncached(self, text: str, cache_key: str) -> SyntaxAnalysis:
        protected_spans = [(span.start, span.end) for span in find_protected_spans(text)]
        if not text.strip():
            return SyntaxAnalysis(text, [], [], [], [], protected_spans, [], "fallback", cache_key)
        if not bool(self.config["enabled"]):
            return self._fallback_analysis(text, cache_key, ["syntax analysis disabled"], protected_spans)
        if str(self.config["backend"]) != "natasha":
            return self._fallback_analysis(text, cache_key, [f"unsupported syntax backend: {self.config['backend']}"], protected_spans)

        try:
            pipeline = syntax_pipeline()
            if pipeline is None:
                return self._fallback_analysis(text, cache_key, ["natasha syntax pipeline unavailable"], protected_spans)
            analysis = self._analyze_with_natasha(text, cache_key, pipeline, protected_spans)
            return analysis
        except Exception as exc:  # pragma: no cover - depends on optional NLP stack failures.
            if not bool(self.config["fail_open"]) or bool(self.config["debug_errors"]):
                raise
            logger.debug("Natasha syntax analysis failed", exc_info=True)
            return self._fallback_analysis(text, cache_key, [f"{type(exc).__name__}: {exc}"], protected_spans)

    def _analyze_with_natasha(
        self,
        text: str,
        cache_key: str,
        pipeline: _SyntaxPipeline,
        protected_spans: list[tuple[int, int]],
    ) -> SyntaxAnalysis:
        doc = pipeline.doc_factory(text)
        doc.segment(pipeline.segmenter)
        doc.tag_morph(pipeline.morph_tagger)
        _lemmatize_tokens(doc, pipeline.morph_vocab)
        doc.parse_syntax(pipeline.syntax_parser)
        ner_by_token_id = _tag_ner(doc, pipeline)
        tokens = _syntax_tokens(doc, ner_by_token_id)
        sentences = _syntax_sentences(text, doc, tokens)
        base = SyntaxAnalysis(text, tokens, sentences, [], [], protected_spans, [], "natasha", cache_key)
        from src.nlp.syntax_features import SyntaxFeatureExtractor

        extractor = SyntaxFeatureExtractor(base)
        clauses = extractor.find_clauses(base)
        phrases = extractor.collect_phrase_spans(base)
        return SyntaxAnalysis(text, tokens, sentences, clauses, phrases, protected_spans, [], "natasha", cache_key)

    def _fallback_analysis(
        self,
        text: str,
        cache_key: str,
        errors: list[str],
        protected_spans: list[tuple[int, int]] | None = None,
    ) -> SyntaxAnalysis:
        protected = protected_spans if protected_spans is not None else [(span.start, span.end) for span in find_protected_spans(text)]
        from src.nlp.syntax_features import SyntaxFeatureExtractor

        base = SyntaxAnalysis(text, [], _fallback_sentences(text), [], [], protected, errors, "fallback", cache_key)
        extractor = SyntaxFeatureExtractor(base)
        phrases = extractor.collect_phrase_spans(base)
        return SyntaxAnalysis(text, [], base.sentences, [], phrases, protected, errors, "fallback", cache_key)

    def _can_cache(self, text: str) -> bool:
        return self._cache_enabled() and len(text) <= int(self.config["cache_max_text_length"])

    def _cache_enabled(self) -> bool:
        return bool(self.config["cache_enabled"]) and int(self.config["cache_max_size"]) > 0

    def _trim_cache(self) -> None:
        while len(self._cache) > int(self.config["cache_max_size"]):
            self._cache.popitem(last=False)


def _syntax_config(config: dict[str, Any] | None) -> dict[str, Any]:
    syntax_config = config or {}
    if "nlp" in syntax_config:
        syntax_config = ((syntax_config.get("nlp") or {}).get("syntax") or {})
    merged = dict(DEFAULT_SYNTAX_CONFIG)
    merged.update(syntax_config)
    merged["cache_max_size"] = max(0, int(merged["cache_max_size"]))
    merged["cache_max_text_length"] = max(0, int(merged["cache_max_text_length"]))
    return merged


def _cache_key(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"sha256:{digest}:len:{len(text)}"


_DEFAULT_ANALYZER = SyntaxAnalyzer()


def analyze_syntax(text: str, config: dict[str, Any] | None = None) -> SyntaxAnalysis:
    if config is None:
        return _DEFAULT_ANALYZER.analyze(text)
    return SyntaxAnalyzer(config).analyze(text)


def clear_syntax_cache() -> None:
    _DEFAULT_ANALYZER.clear_cache()


def syntax_cache_info() -> dict[str, int | bool]:
    return _DEFAULT_ANALYZER.cache_info()


def _optional_ner_tagger(ner_tagger_factory: Any, embedding: Any) -> Any | None:
    try:
        return ner_tagger_factory(embedding)
    except Exception:  # pragma: no cover - NER resources are optional.
        logger.debug("Natasha NER tagger unavailable", exc_info=True)
        return None


def _lemmatize_tokens(doc: Any, morph_vocab: Any) -> None:
    for token in getattr(doc, "tokens", ()):
        token.lemmatize(morph_vocab)


def _tag_ner(doc: Any, pipeline: _SyntaxPipeline) -> dict[str, str]:
    if pipeline.ner_tagger is None:
        return {}
    try:
        doc.tag_ner(pipeline.ner_tagger)
        return _ner_by_token_id(doc)
    except Exception:  # pragma: no cover - NER is optional for the syntax layer.
        logger.debug("Natasha NER tagging failed", exc_info=True)
        return {}


def _ner_by_token_id(doc: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for span in getattr(doc, "spans", ()):
        span_type = getattr(span, "type", None)
        if not span_type:
            continue
        for token in _span_tokens(doc, span):
            token_id = getattr(token, "id", None)
            if token_id is not None:
                result[str(token_id)] = str(span_type)
    return result


def _span_tokens(doc: Any, span: Any) -> list[Any]:
    tokens = getattr(span, "tokens", None)
    if tokens is not None:
        return list(tokens)
    start = int(getattr(span, "start", -1))
    stop = int(getattr(span, "stop", getattr(span, "end", -1)))
    if start < 0 or stop < 0:
        return []
    return [
        token
        for token in getattr(doc, "tokens", ())
        if int(getattr(token, "start", -1)) < stop and start < int(getattr(token, "stop", getattr(token, "end", -1)))
    ]


def _syntax_tokens(doc: Any, ner_by_token_id: dict[str, str]) -> list[SyntaxToken]:
    raw_tokens = list(getattr(doc, "tokens", ()))
    token_index_by_id = {
        str(getattr(token, "id")): index
        for index, token in enumerate(raw_tokens)
        if getattr(token, "id", None) is not None
    }
    sentence_id_by_token_id = _sentence_id_by_token_id(doc)
    result: list[SyntaxToken] = []
    for index, token in enumerate(raw_tokens):
        token_id = str(getattr(token, "id", "") or "")
        ner_type = ner_by_token_id.get(token_id)
        pos = str(getattr(token, "pos", "") or "")
        feats = _token_feats(token)
        text = str(getattr(token, "text", ""))
        result.append(
            SyntaxToken(
                id=index,
                text=text,
                lemma=str(getattr(token, "lemma", "") or ""),
                pos=pos,
                feats=feats,
                start=int(getattr(token, "start", 0)),
                end=int(getattr(token, "stop", getattr(token, "end", 0))),
                head_id=_head_index(token, token_index_by_id),
                dep_rel=str(getattr(token, "rel", "") or ""),
                sentence_id=sentence_id_by_token_id.get(token_id, 0),
                is_punctuation=_is_punctuation(text, pos),
                is_conjunction=pos in CONJUNCTION_POSES or text.lower() in CONJUNCTION_WORDS,
                is_particle=pos == "PART",
                is_verb=pos in {"VERB", "AUX"},
                is_noun=pos in {"NOUN", "PROPN"},
                is_adjective=pos == "ADJ",
                is_participle=_is_participle(text, feats),
                is_adverbial_participle=_is_adverbial_participle(text, feats),
                is_pronoun=pos in {"PRON", "DET"},
                is_named_entity=ner_type is not None,
                ner_type=ner_type,
            )
        )
    return result


def _syntax_sentences(text: str, doc: Any, tokens: list[SyntaxToken]) -> list[SyntaxSentence]:
    raw_sents = list(getattr(doc, "sents", ()))
    if not raw_sents and tokens:
        return [_sentence_from_tokens(text, 0, len(text), tokens)]
    sentences: list[SyntaxSentence] = []
    for sentence_id, sent in enumerate(raw_sents):
        start = int(getattr(sent, "start", 0))
        end = int(getattr(sent, "stop", getattr(sent, "end", len(text))))
        sentence_tokens = [token for token in tokens if token.sentence_id == sentence_id]
        if not sentence_tokens:
            sentence_tokens = [token for token in tokens if start <= token.start and token.end <= end]
        sentences.append(_sentence_from_tokens(text, start, end, sentence_tokens))
    return sentences


def _fallback_sentences(text: str) -> list[SyntaxSentence]:
    if not text:
        return []
    from src.nlp.syntax_features import SyntaxFeatureExtractor

    boundaries = SyntaxFeatureExtractor().find_sentence_boundaries(text)
    return [SyntaxSentence(text[start:end], start, end, [], [], False, [], []) for start, end in boundaries]


def _sentence_from_tokens(text: str, start: int, end: int, tokens: list[SyntaxToken]) -> SyntaxSentence:
    roots = [token.id for token in tokens if token.head_id is None and not token.is_punctuation]
    predicate_ids = [
        token.id
        for token in tokens
        if token.dep_rel in {"root", "advcl", "conj", "xcomp", "ccomp"} or _has_finite_verb_form(token)
    ]
    subject_ids = [token.id for token in tokens if token.dep_rel in {"nsubj", "csubj"}]
    return SyntaxSentence(
        text=text[start:end],
        start=start,
        end=end,
        tokens=tokens,
        root_ids=roots,
        has_finite_verb=any(_has_finite_verb_form(token) for token in tokens),
        predicate_token_ids=predicate_ids,
        subject_token_ids=subject_ids,
    )


def _sentence_id_by_token_id(doc: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for sentence_id, sent in enumerate(getattr(doc, "sents", ())):
        for token in getattr(sent, "tokens", ()) or ():
            token_id = getattr(token, "id", None)
            if token_id is not None:
                result[str(token_id)] = sentence_id
    if result:
        return result
    for sentence_id, sent in enumerate(getattr(doc, "sents", ())):
        start = int(getattr(sent, "start", -1))
        stop = int(getattr(sent, "stop", getattr(sent, "end", -1)))
        for token in getattr(doc, "tokens", ()):
            token_id = getattr(token, "id", None)
            token_start = int(getattr(token, "start", -1))
            token_stop = int(getattr(token, "stop", getattr(token, "end", -1)))
            if token_id is not None and start <= token_start and token_stop <= stop:
                result[str(token_id)] = sentence_id
    return result


def _token_feats(token: Any) -> dict[str, Any]:
    feats = getattr(token, "feats", None)
    if isinstance(feats, dict):
        return dict(feats)
    if feats is None:
        return {}
    try:
        return dict(feats)
    except (TypeError, ValueError):
        return {"raw": str(feats)}


def _head_index(token: Any, token_index_by_id: dict[str, int]) -> int | None:
    token_id = str(getattr(token, "id", "") or "")
    head_id = str(getattr(token, "head_id", "") or "")
    if not head_id or head_id == token_id:
        return None
    return token_index_by_id.get(head_id)


def _is_punctuation(text: str, pos: str) -> bool:
    return pos == "PUNCT" or text in PUNCTUATION_CHARS


def _is_participle(text: str, feats: dict[str, Any]) -> bool:
    if str(feats.get("VerbForm", "")) == "Part":
        return True
    lowered = text.lower()
    return len(lowered) >= 7 and lowered.endswith(PARTICIPLE_SUFFIXES)


def _is_adverbial_participle(text: str, feats: dict[str, Any]) -> bool:
    if str(feats.get("VerbForm", "")) == "Conv":
        return True
    lowered = text.lower()
    return len(lowered) >= 6 and lowered.endswith(ADVERBIAL_PARTICIPLE_SUFFIXES)


def _has_finite_verb_form(token: SyntaxToken) -> bool:
    return token.is_verb and str(token.feats.get("VerbForm", "")) == "Fin"
