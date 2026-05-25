from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.docx.docx_reader import read_paragraphs
from src.docx.docx_writer import write_paragraphs_like
from src.memory.document_index import SegmentCorrectionCache, build_segment_cache
from src.runtime.corrector import Corrector
from src.schema.edits import CorrectionResult, RuntimeEdit


class DocxCorrector(Protocol):
    def correct(self, text: str) -> CorrectionResult:
        ...


@dataclass(frozen=True)
class DocxIncrementalResult:
    edits: list[RuntimeEdit]
    checked_paragraphs: int
    reused_paragraphs: int
    paragraph_cache: dict[str, SegmentCorrectionCache]


def correct_docx(input_path: str | Path, output_path: str | Path, corrector: DocxCorrector | None = None) -> list[RuntimeEdit]:
    if corrector is None:
        corrector = Corrector()
    paragraphs = read_paragraphs(input_path)
    corrected: list[str] = []
    edits: list[RuntimeEdit] = []

    for paragraph in paragraphs:
        result = corrector.correct(paragraph) if paragraph.strip() else None
        if result is None:
            corrected.append(paragraph)
            continue
        corrected.append(result.corrected_text)
        edits.extend(result.edits)

    write_paragraphs_like(input_path, output_path, corrected)
    return edits


def correct_docx_incremental(
    input_path: str | Path,
    output_path: str | Path,
    corrector: DocxCorrector | None = None,
    previous_cache: dict[str, SegmentCorrectionCache] | None = None,
) -> DocxIncrementalResult:
    if corrector is None:
        corrector = Corrector()
    paragraphs = read_paragraphs(input_path)
    corrected: list[str] = []
    edits: list[RuntimeEdit] = []
    paragraph_cache: dict[str, SegmentCorrectionCache] = {}
    checked_paragraphs = 0
    reused_paragraphs = 0

    for paragraph_index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            corrected.append(paragraph)
            continue

        paragraph_hash = build_segment_cache(paragraph, paragraph).segment_hash
        cached = previous_cache.get(paragraph_hash) if previous_cache else None
        if cached is not None:
            local_edits = _cached_edits(cached)
            corrected.append(cached.corrected_text)
            edits.extend(local_edits)
            paragraph_cache[paragraph_hash] = _paragraph_cache(
                paragraph,
                cached.corrected_text,
                local_edits,
                paragraph_index,
            )
            reused_paragraphs += 1
            continue

        result = corrector.correct(paragraph)
        local_edits = list(result.edits)
        corrected.append(result.corrected_text)
        edits.extend(local_edits)
        paragraph_cache[paragraph_hash] = _paragraph_cache(
            paragraph,
            result.corrected_text,
            local_edits,
            paragraph_index,
        )
        checked_paragraphs += 1

    write_paragraphs_like(input_path, output_path, corrected)
    return DocxIncrementalResult(
        edits=edits,
        checked_paragraphs=checked_paragraphs,
        reused_paragraphs=reused_paragraphs,
        paragraph_cache=paragraph_cache,
    )


def _paragraph_cache(
    source_text: str,
    corrected_text: str,
    edits: list[RuntimeEdit],
    paragraph_index: int,
) -> SegmentCorrectionCache:
    return build_segment_cache(
        source_text,
        corrected_text,
        edit_count=len(edits),
        metadata={
            "paragraph_index": paragraph_index,
            "edits": list(edits),
        },
    )


def _cached_edits(cache: SegmentCorrectionCache) -> list[RuntimeEdit]:
    edits = cache.metadata.get("edits", [])
    return [edit for edit in edits if isinstance(edit, RuntimeEdit)]
