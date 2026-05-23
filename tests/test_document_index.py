from __future__ import annotations

from src.memory.document_index import (
    build_incremental_plan,
    build_segment_cache,
    merge_corrected_segments,
    segment_text,
)


def _cache_for_text(text: str):
    return {segment.text_hash: build_segment_cache(segment.text, segment.text) for segment in segment_text(text)}


def test_segment_text_splits_text_into_ordered_segments_with_offsets() -> None:
    text = "Первое предложение. Второе предложение!"

    segments = segment_text(text)

    assert [segment.text for segment in segments] == ["Первое предложение.", "Второе предложение!"]
    assert [segment.index for segment in segments] == [0, 1]
    assert text[segments[0].start : segments[0].end] == "Первое предложение."
    assert text[segments[1].start : segments[1].end] == "Второе предложение!"
    assert segments[0].end < segments[1].start


def test_equal_normalized_text_produces_equal_hash() -> None:
    first = segment_text("Ёлка   стоит.")[0]
    second = segment_text("елка стоит.")[0]

    assert first.normalized_text == second.normalized_text
    assert first.text_hash == second.text_hash


def test_one_sentence_change_marks_only_that_segment_changed() -> None:
    previous_cache = _cache_for_text("Первое предложение. Второе предложение.")

    plan = build_incremental_plan(previous_cache, "Первое предложение. Новое предложение.")

    assert [segment.text for segment in plan.unchanged] == ["Первое предложение."]
    assert [segment.text for segment in plan.changed] == ["Новое предложение."]
    assert plan.added == []


def test_unchanged_sentence_reuses_correction_cache() -> None:
    previous_text = "Первое предложение. Второе предложение."
    previous_segments = segment_text(previous_text)
    cache = build_segment_cache(previous_segments[0].text, "Первое исправленное.", edit_count=1)
    previous_cache = {
        cache.segment_hash: cache,
        previous_segments[1].text_hash: build_segment_cache(previous_segments[1].text, previous_segments[1].text),
    }

    plan = build_incremental_plan(previous_cache, "Первое предложение. Новое предложение.")

    assert plan.reused_corrections == {cache.segment_hash: cache}


def test_appended_sentence_is_added() -> None:
    previous_cache = _cache_for_text("Первое предложение.")

    plan = build_incremental_plan(previous_cache, "Первое предложение. Второе предложение.")

    assert plan.changed == []
    assert [segment.text for segment in plan.added] == ["Второе предложение."]


def test_removed_sentence_hash_is_reported() -> None:
    previous_text = "Первое предложение. Второе предложение."
    previous_segments = segment_text(previous_text)
    previous_cache = _cache_for_text(previous_text)

    plan = build_incremental_plan(previous_cache, "Первое предложение.")

    assert plan.removed_hashes == [previous_segments[1].text_hash]


def test_merge_corrected_segments_joins_simple_sentences() -> None:
    segments = segment_text("Первое предложение. Второе предложение.")

    merged = merge_corrected_segments(
        segments,
        {
            segments[0].text_hash: "Первое исправленное.",
            segments[1].text_hash: "Второе исправленное.",
        },
    )

    assert merged == "Первое исправленное. Второе исправленное."


def test_merge_corrected_segments_preserves_simple_newline_between_segments() -> None:
    segments = segment_text("Первое предложение.\nВторое предложение.")

    merged = merge_corrected_segments(
        segments,
        {
            segments[0].text_hash: "Первое исправленное.",
            segments[1].text_hash: "Второе исправленное.",
        },
    )

    assert merged == "Первое исправленное.\nВторое исправленное."
