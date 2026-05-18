from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
from typing import Any


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyntaxToken:
    text: str
    lemma: str
    pos: str
    feats: dict[str, Any]
    head_id: int | None
    rel: str
    start: int
    end: int
    ner: str | None = None


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
        ner_tagger = _optional_ner_tagger(NewsNERTagger, embedding)
        return _SyntaxPipeline(
            segmenter=Segmenter(),
            morph_vocab=MorphVocab(),
            morph_tagger=NewsMorphTagger(embedding),
            syntax_parser=NewsSyntaxParser(embedding),
            ner_tagger=ner_tagger,
            doc_factory=Doc,
        )
    except Exception as exc:  # pragma: no cover - depends on optional NLP stack.
        logger.warning("Natasha syntax pipeline unavailable: %s: %s", type(exc).__name__, exc)
        return None


def parse_syntax(text: str) -> list[SyntaxToken]:
    if not text.strip():
        return []

    try:
        pipeline = syntax_pipeline()
        if pipeline is None:
            return []
        doc = pipeline.doc_factory(text)
        doc.segment(pipeline.segmenter)
        doc.tag_morph(pipeline.morph_tagger)
        _lemmatize_tokens(doc, pipeline.morph_vocab)
        doc.parse_syntax(pipeline.syntax_parser)
        ner_by_token_id = _tag_ner(doc, pipeline)
        return _syntax_tokens(doc, ner_by_token_id)
    except Exception:  # pragma: no cover - exercised only when optional NLP stack fails.
        logger.debug("Natasha syntax parsing failed", exc_info=True)
        return []


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

    return [
        SyntaxToken(
            text=str(getattr(token, "text", "")),
            lemma=str(getattr(token, "lemma", "") or ""),
            pos=str(getattr(token, "pos", "") or ""),
            feats=_token_feats(token),
            head_id=_head_index(token, token_index_by_id),
            rel=str(getattr(token, "rel", "") or ""),
            start=int(getattr(token, "start", 0)),
            end=int(getattr(token, "stop", getattr(token, "end", 0))),
            ner=ner_by_token_id.get(str(getattr(token, "id", ""))),
        )
        for token in raw_tokens
    ]


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
