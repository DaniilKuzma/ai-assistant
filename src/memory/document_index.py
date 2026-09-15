from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any

from src.preprocessing.sentence_splitter import split_sentences


_WHITESPACE_RE = re.compile(r"\s+")
_FALLBACK_SEGMENT_RE = re.compile(r"\S.*?(?:[.!?…]+(?=\s|$)|(?=\n\s*\n)|$)", re.DOTALL)
_GAPS_BY_SEGMENTS_ID: dict[int, list[str]] = {}


@dataclass(frozen=True)
class DocumentSegment:
    segment_id: str
    index: int
    text: str
    normalized_text: str
    text_hash: str
    start: int
    end: int


@dataclass(frozen=True)
class SegmentCorrectionCache:
    segment_hash: str
    source_text: str
    corrected_text: str
    edit_count: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class IncrementalPlan:
    unchanged: list[DocumentSegment]
    changed: list[DocumentSegment]
    added: list[DocumentSegment]
    removed_hashes: list[str]
    reused_corrections: dict[str, SegmentCorrectionCache]


def normalize_segment_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", str(text).lower().replace("ё", "е")).strip()


def segment_text(text: str) -> list[DocumentSegment]:
    if not text or not text.strip():
        return []

    spans = _spans_from_sentence_splitter(text)
    if spans is None:
        spans = _fallback_segment_spans(text)

    segments = [_build_segment(text, index, start, end) for index, (start, end) in enumerate(spans)]
    _remember_segment_gaps(text, segments)
    return segments


def build_segment_cache(
    source_text: str,
    corrected_text: str,
    edit_count: int = 0,
    metadata: dict[str, Any] | None = None,
) -> SegmentCorrectionCache:
    normalized_text = normalize_segment_text(source_text)
    return SegmentCorrectionCache(
        segment_hash=_hash_normalized_text(normalized_text),
        source_text=source_text,
        corrected_text=corrected_text,
        edit_count=int(edit_count),
        metadata=dict(metadata or {}),
    )


def build_incremental_plan(previous_cache: dict[str, SegmentCorrectionCache], new_text: str) -> IncrementalPlan:
    segments = segment_text(new_text)
    current_hashes = {segment.text_hash for segment in segments}
    unchanged: list[DocumentSegment] = []
    changed: list[DocumentSegment] = []
    added: list[DocumentSegment] = []
    reused_corrections: dict[str, SegmentCorrectionCache] = {}

    # Current incremental planning uses hash reuse and approximate index-based classification because
    # previous segment order is not part of the cache contract. More precise insertion/reordering
    # detection can be added later by storing segment order metadata or using sequence alignment.
    for segment in segments:
        cached = previous_cache.get(segment.text_hash)
        if cached is not None:
            unchanged.append(segment)
            reused_corrections[segment.text_hash] = cached
        elif segment.index < len(previous_cache):
            changed.append(segment)
        else:
            added.append(segment)

    removed_hashes = [segment_hash for segment_hash in previous_cache if segment_hash not in current_hashes]
    return IncrementalPlan(
        unchanged=unchanged,
        changed=changed,
        added=added,
        removed_hashes=removed_hashes,
        reused_corrections=reused_corrections,
    )


def merge_corrected_segments(segments: list[DocumentSegment], corrected_by_hash: dict[str, str]) -> str:
    if not segments:
        return ""

    ordered = sorted(segments, key=lambda segment: (segment.start, segment.index))
    gaps = _GAPS_BY_SEGMENTS_ID.get(id(segments), [])
    merged: list[str] = []

    for index, segment in enumerate(ordered):
        if index > 0:
            gap = gaps[index - 1] if index - 1 < len(gaps) else ""
            merged.append("\n" if "\n" in gap else " ")
        merged.append(corrected_by_hash.get(segment.text_hash, segment.text))

    return "".join(merged)


def _spans_from_sentence_splitter(text: str) -> list[tuple[int, int]] | None:
    cursor = 0
    spans: list[tuple[int, int]] = []
    for sentence in split_sentences(text):
        segment_text_value = sentence.strip()
        if not segment_text_value:
            continue
        start = text.find(segment_text_value, cursor)
        if start < 0:
            return None
        end = start + len(segment_text_value)
        spans.append((start, end))
        cursor = end
    return spans


def _fallback_segment_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in _FALLBACK_SEGMENT_RE.finditer(text):
        start, end = _trim_span(text, match.start(), match.end())
        if start < end:
            spans.append((start, end))
    return spans


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _build_segment(text: str, index: int, start: int, end: int) -> DocumentSegment:
    segment_source = text[start:end]
    normalized_text = normalize_segment_text(segment_source)
    return DocumentSegment(
        segment_id=f"segment-{index}",
        index=index,
        text=segment_source,
        normalized_text=normalized_text,
        text_hash=_hash_normalized_text(normalized_text),
        start=start,
        end=end,
    )


def _hash_normalized_text(normalized_text: str) -> str:
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def _remember_segment_gaps(text: str, segments: list[DocumentSegment]) -> None:
    if not segments:
        return
    ordered = sorted(segments, key=lambda segment: (segment.start, segment.index))
    gaps = [text[current.end : next_segment.start] for current, next_segment in zip(ordered, ordered[1:])]
    _GAPS_BY_SEGMENTS_ID[id(segments)] = gaps
