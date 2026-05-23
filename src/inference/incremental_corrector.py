from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from src.memory.document_index import (
    DocumentSegment,
    SegmentCorrectionCache,
    build_incremental_plan,
    build_segment_cache,
    merge_corrected_segments,
    segment_text,
)
from src.validation.diff_analyzer import Edit


@dataclass(frozen=True)
class IncrementalCorrectionResult:
    source_text: str
    corrected_text: str
    edits: list[Edit]
    checked_segments: int
    reused_segments: int
    changed_segments: int
    segment_cache: dict[str, SegmentCorrectionCache]


class IncrementalCorrector:
    def __init__(self, corrector: Any) -> None:
        self.corrector = corrector

    def correct_incremental(
        self,
        text: str,
        previous_cache: dict[str, SegmentCorrectionCache] | None = None,
    ) -> IncrementalCorrectionResult:
        if not text or not text.strip():
            return IncrementalCorrectionResult(
                source_text=text,
                corrected_text=text,
                edits=[],
                checked_segments=0,
                reused_segments=0,
                changed_segments=0,
                segment_cache={},
            )

        segments = segment_text(text)
        reusable_cache = _reusable_cache(previous_cache, text)
        corrected_by_hash: dict[str, str] = {}
        new_cache: dict[str, SegmentCorrectionCache] = {}
        edits: list[Edit] = []
        checked_segments = 0
        reused_segments = 0

        for segment in sorted(segments, key=lambda item: (item.start, item.index)):
            cached = reusable_cache.get(segment.text_hash)
            if cached is not None:
                local_edits = _cached_edits(cached)
                corrected_by_hash[segment.text_hash] = cached.corrected_text
                new_cache[segment.text_hash] = _segment_cache(segment, cached.corrected_text, local_edits)
                edits.extend(_global_edits(local_edits, segment))
                reused_segments += 1
                continue

            result = self.corrector.correct(segment.text)
            local_edits = list(result.edits)
            corrected_by_hash[segment.text_hash] = result.corrected_text
            new_cache[segment.text_hash] = _segment_cache(segment, result.corrected_text, local_edits)
            edits.extend(_global_edits(local_edits, segment))
            checked_segments += 1

        return IncrementalCorrectionResult(
            source_text=text,
            corrected_text=merge_corrected_segments(segments, corrected_by_hash),
            edits=edits,
            checked_segments=checked_segments,
            reused_segments=reused_segments,
            changed_segments=checked_segments,
            segment_cache=new_cache,
        )


def _reusable_cache(
    previous_cache: dict[str, SegmentCorrectionCache] | None,
    text: str,
) -> dict[str, SegmentCorrectionCache]:
    if not previous_cache:
        return {}
    return build_incremental_plan(previous_cache, text).reused_corrections


def _segment_cache(
    segment: DocumentSegment,
    corrected_text: str,
    local_edits: list[Edit],
) -> SegmentCorrectionCache:
    return build_segment_cache(
        segment.text,
        corrected_text,
        edit_count=len(local_edits),
        metadata={
            "segment_id": segment.segment_id,
            "segment_index": segment.index,
            "segment_start": segment.start,
            "segment_end": segment.end,
            "edits": list(local_edits),
        },
    )


def _cached_edits(cache: SegmentCorrectionCache) -> list[Edit]:
    edits = cache.metadata.get("edits", [])
    return [edit for edit in edits if isinstance(edit, Edit)]


def _global_edits(local_edits: list[Edit], segment: DocumentSegment) -> list[Edit]:
    return [_global_edit(edit, segment) for edit in local_edits]


def _global_edit(edit: Edit, segment: DocumentSegment) -> Edit:
    if edit.start < 0 or edit.end < 0:
        return edit
    return replace(edit, start=segment.start + edit.start, end=segment.start + edit.end)
