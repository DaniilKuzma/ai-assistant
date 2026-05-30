from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Protocol

from docx import Document

from src.docx.docx_writer import write_paragraph_text
from src.memory.document_index import SegmentCorrectionCache, build_segment_cache
from src.runtime.corrector import Corrector
from src.runtime.edit_realizer import apply_runtime_edits
from src.runtime.explanations import attach_explanations
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


def count_docx_text_blocks(path: str | Path) -> int:
    document = Document(path)
    return sum(1 for paragraph, _context in _iter_docx_text_blocks(document) if paragraph.text.strip())


@dataclass(frozen=True)
class DocxTextContext:
    block_index: int
    block_kind: str
    style_name: str = ""


DOCX_PUNCTUATION_RULE_IDS = frozenset(
    {
        "final_punctuation",
        "spacing_normalization",
        "comma_subordinate",
        "comma_introductory",
        "comma_adversative",
        "comma_homogeneous",
        "dash_subject_predicate",
        "punctuation",
    }
)
DOCX_CONTEXT_DEPENDENT_RULE_IDS = frozenset(
    {
        "compound_service_words",
        "compound_prepositions",
        "takzhe_tak_zhe",
        "tozhe_to_zhe",
        "zato_za_to",
    }
)
CONTROLLED_DOCX_CASING = (
    (
        "\u0420\u043e\u0441\u0441\u0438\u0439\u0441\u043a\u043e\u0439 \u0444\u0435\u0434\u0435\u0440\u0430\u0446\u0438\u0438",
        "\u0444\u0435\u0434\u0435\u0440\u0430\u0446\u0438\u0438",
        "\u0424\u0435\u0434\u0435\u0440\u0430\u0446\u0438\u0438",
        "casing_geo_names",
    ),
)
QUOTE_PAIRS = (("\u00ab", "\u00bb"), ("\u201c", "\u201d"), ("\u201e", "\u201c"), ('"', '"'))
ARROW_EXAMPLE_RE = re.compile(r"(?:->|\u2192)")


def correct_docx(input_path: str | Path, output_path: str | Path, corrector: DocxCorrector | None = None) -> list[RuntimeEdit]:
    if corrector is None:
        corrector = Corrector()
    document = Document(input_path)
    edits: list[RuntimeEdit] = []

    for paragraph, context in _iter_docx_text_blocks(document):
        text = paragraph.text
        if not text.strip():
            continue
        result = _correct_docx_text(text, corrector, context)
        if result.corrected_text != text:
            write_paragraph_text(paragraph, result.corrected_text)
        edits.extend(result.edits)

    document.save(output_path)
    return edits


def correct_docx_incremental(
    input_path: str | Path,
    output_path: str | Path,
    corrector: DocxCorrector | None = None,
    previous_cache: dict[str, SegmentCorrectionCache] | None = None,
) -> DocxIncrementalResult:
    if corrector is None:
        corrector = Corrector()
    document = Document(input_path)
    edits: list[RuntimeEdit] = []
    paragraph_cache: dict[str, SegmentCorrectionCache] = {}
    checked_paragraphs = 0
    reused_paragraphs = 0

    for paragraph, context in _iter_docx_text_blocks(document):
        text = paragraph.text
        if not text.strip():
            continue

        paragraph_hash = build_segment_cache(text, text).segment_hash
        cached = previous_cache.get(paragraph_hash) if previous_cache else None
        if cached is not None:
            local_edits = _safe_cached_edits(text, cached, context)
            corrected_text = apply_runtime_edits(text, local_edits)
            if corrected_text != text:
                write_paragraph_text(paragraph, corrected_text)
            edits.extend(local_edits)
            paragraph_cache[paragraph_hash] = _paragraph_cache(
                text,
                corrected_text,
                local_edits,
                context,
            )
            reused_paragraphs += 1
            continue

        result = _correct_docx_text(text, corrector, context)
        local_edits = list(result.edits)
        if result.corrected_text != text:
            write_paragraph_text(paragraph, result.corrected_text)
        edits.extend(local_edits)
        paragraph_cache[paragraph_hash] = _paragraph_cache(
            text,
            result.corrected_text,
            local_edits,
            context,
        )
        checked_paragraphs += 1

    document.save(output_path)
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
    context: DocxTextContext,
) -> SegmentCorrectionCache:
    return build_segment_cache(
        source_text,
        corrected_text,
        edit_count=len(edits),
        metadata={
            "paragraph_index": context.block_index,
            "block_kind": context.block_kind,
            "style_name": context.style_name,
            "edits": list(edits),
        },
    )


def _cached_edits(cache: SegmentCorrectionCache) -> list[RuntimeEdit]:
    edits = cache.metadata.get("edits", [])
    return [edit for edit in edits if isinstance(edit, RuntimeEdit)]


def _iter_docx_text_blocks(document):  # type: ignore[no-untyped-def]
    block_index = 0
    seen_paragraphs: set[int] = set()
    for paragraph in document.paragraphs:
        paragraph_id = id(paragraph._p)
        if paragraph_id in seen_paragraphs:
            continue
        seen_paragraphs.add(paragraph_id)
        yield paragraph, DocxTextContext(block_index, "body", _style_name(paragraph))
        block_index += 1

    seen_cells: set[int] = set()
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                cell_id = id(cell._tc)
                if cell_id in seen_cells:
                    continue
                seen_cells.add(cell_id)
                for paragraph in cell.paragraphs:
                    paragraph_id = id(paragraph._p)
                    if paragraph_id in seen_paragraphs:
                        continue
                    seen_paragraphs.add(paragraph_id)
                    yield paragraph, DocxTextContext(block_index, "table", _style_name(paragraph))
                    block_index += 1


def _correct_docx_text(text: str, corrector: DocxCorrector, context: DocxTextContext) -> CorrectionResult:
    result = corrector.correct(text)
    edits = [
        *_filter_docx_edits(text, list(result.edits), context),
        *_controlled_docx_casing_edits(text),
    ]
    edits = _without_overlapping_edits(edits)
    corrected_text = apply_runtime_edits(text, edits)
    attach_explanations(edits)
    return CorrectionResult(text, corrected_text, edits, metadata=dict(result.metadata))


def _safe_cached_edits(text: str, cache: SegmentCorrectionCache, context: DocxTextContext) -> list[RuntimeEdit]:
    edits = [
        *_filter_docx_edits(text, _cached_edits(cache), context),
        *_controlled_docx_casing_edits(text),
    ]
    edits = _without_overlapping_edits(edits)
    attach_explanations(edits)
    return edits


def _filter_docx_edits(text: str, edits: list[RuntimeEdit], context: DocxTextContext) -> list[RuntimeEdit]:
    return [edit for edit in edits if _docx_edit_allowed(text, edit, context)]


def _docx_edit_allowed(text: str, edit: RuntimeEdit, context: DocxTextContext) -> bool:
    del context
    rule_id = str(edit.rule_id or "")
    edit_type = str(edit.edit_type or "")
    if edit_type == "punctuation" or rule_id in DOCX_PUNCTUATION_RULE_IDS:
        return False
    if rule_id in DOCX_CONTEXT_DEPENDENT_RULE_IDS or rule_id.startswith("semantic_"):
        return False
    if not _edit_matches_source(text, edit):
        return False
    if _edit_is_inside_quoted_span(text, edit):
        return False
    if _edit_is_near_arrow_example(text, edit):
        return False
    return True


def _controlled_docx_casing_edits(text: str) -> list[RuntimeEdit]:
    edits: list[RuntimeEdit] = []
    for phrase, source, replacement, rule_id in CONTROLLED_DOCX_CASING:
        offset = 0
        while True:
            phrase_start = text.find(phrase, offset)
            if phrase_start < 0:
                break
            source_start = text.find(source, phrase_start, phrase_start + len(phrase))
            if source_start >= 0:
                candidate = RuntimeEdit(
                    start=source_start,
                    end=source_start + len(source),
                    source=source,
                    replacement=replacement,
                    edit_type="casing",
                    rule_id=rule_id,
                    confidence=1.0,
                )
                if _edit_matches_source(text, candidate) and not _edit_is_inside_quoted_span(text, candidate):
                    edits.append(candidate)
            offset = phrase_start + len(phrase)
    return edits


def _without_overlapping_edits(edits: list[RuntimeEdit]) -> list[RuntimeEdit]:
    accepted: list[RuntimeEdit] = []
    occupied: list[tuple[int, int]] = []
    seen: set[tuple[int, int, str, str, str]] = set()
    for edit in sorted(edits, key=lambda item: (item.start, item.end, item.rule_id, item.replacement)):
        key = (edit.start, edit.end, edit.source, edit.replacement, edit.rule_id)
        if key in seen:
            continue
        seen.add(key)
        if edit.source == edit.replacement:
            continue
        if edit.start != edit.end and any(not (edit.end <= start or edit.start >= end) for start, end in occupied):
            continue
        accepted.append(edit)
        if edit.start != edit.end:
            occupied.append((edit.start, edit.end))
    return accepted


def _edit_matches_source(text: str, edit: RuntimeEdit) -> bool:
    if edit.start < 0 or edit.end < edit.start or edit.end > len(text):
        return False
    return text[edit.start : edit.end] == edit.source


def _edit_is_inside_quoted_span(text: str, edit: RuntimeEdit) -> bool:
    start = max(0, min(edit.start, len(text)))
    end = max(start, min(edit.end, len(text)))
    for open_quote, close_quote in QUOTE_PAIRS:
        if open_quote == close_quote:
            if text[:start].count(open_quote) % 2 == 1 and text[end:].count(close_quote) > 0:
                return True
            continue
        open_position = text.rfind(open_quote, 0, start + 1)
        close_before = text.rfind(close_quote, 0, start + 1)
        if open_position > close_before and text.find(close_quote, end) != -1:
            return True
    return False


def _edit_is_near_arrow_example(text: str, edit: RuntimeEdit) -> bool:
    window_start = max(0, edit.start - 80)
    window_end = min(len(text), edit.end + 80)
    return ARROW_EXAMPLE_RE.search(text[window_start:window_end]) is not None


def _style_name(paragraph) -> str:  # type: ignore[no-untyped-def]
    try:
        return str(paragraph.style.name or "")
    except Exception:
        return ""
