from __future__ import annotations

import difflib
from pathlib import Path

from docx import Document


def write_paragraphs_like(source_path: str | Path, output_path: str | Path, paragraphs: list[str]) -> None:
    document = Document(source_path)
    for paragraph, text in zip(document.paragraphs, paragraphs, strict=False):
        if paragraph.runs:
            _write_text_preserving_runs(paragraph, text)
        else:
            paragraph.text = text
    document.save(output_path)


def _write_text_preserving_runs(paragraph, corrected_text: str) -> None:  # type: ignore[no-untyped-def]
    original_text = paragraph.text
    run_spans = _run_spans(paragraph)
    pieces = [""] * len(paragraph.runs)
    matcher = difflib.SequenceMatcher(a=original_text, b=corrected_text)

    for tag, source_start, source_end, corrected_start, corrected_end in matcher.get_opcodes():
        if tag == "equal":
            _append_equal_segment(pieces, run_spans, source_start, source_end, corrected_start, original_text, corrected_text)
        elif tag == "delete":
            continue
        else:
            owner = _run_index_for_position(run_spans, source_start, len(original_text))
            pieces[owner] += corrected_text[corrected_start:corrected_end]

    if "".join(pieces) != corrected_text:
        pieces = [corrected_text, *([""] * (len(paragraph.runs) - 1))]

    for run, text in zip(paragraph.runs, pieces, strict=True):
        run.text = text


def _run_spans(paragraph) -> list[tuple[int, int]]:  # type: ignore[no-untyped-def]
    spans: list[tuple[int, int]] = []
    position = 0
    for run in paragraph.runs:
        start = position
        position += len(run.text)
        spans.append((start, position))
    return spans


def _append_equal_segment(
    pieces: list[str],
    run_spans: list[tuple[int, int]],
    source_start: int,
    source_end: int,
    corrected_start: int,
    original_text: str,
    corrected_text: str,
) -> None:
    for run_index, (run_start, run_end) in enumerate(run_spans):
        overlap_start = max(source_start, run_start)
        overlap_end = min(source_end, run_end)
        if overlap_start >= overlap_end:
            continue
        source_fragment = original_text[overlap_start:overlap_end]
        target_fragment_start = corrected_start + (overlap_start - source_start)
        target_fragment_end = target_fragment_start + len(source_fragment)
        pieces[run_index] += corrected_text[target_fragment_start:target_fragment_end]


def _run_index_for_position(run_spans: list[tuple[int, int]], position: int, original_length: int) -> int:
    if not run_spans:
        return 0
    if position >= original_length:
        return len(run_spans) - 1
    for index, (start, end) in enumerate(run_spans):
        if start <= position < end:
            return index
        if position == start:
            return index
    return len(run_spans) - 1
