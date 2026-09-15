from __future__ import annotations

from src.inference.incremental_corrector import IncrementalCorrector
from src.schema.edits import CorrectionResult, RuntimeEdit


class FakeCorrector:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def correct(self, text: str) -> CorrectionResult:
        self.calls.append(text)
        source = "незнаю"
        replacement = "не знаю"
        start = text.find(source)
        if start < 0:
            return CorrectionResult(text, text, [])
        edit = RuntimeEdit(
            start=start,
            end=start + len(source),
            source=text[start : start + len(source)],
            replacement=replacement,
            edit_type="split_join",
            rule_id="fake_split_word",
            confidence=0.99,
            explanation="Раздельное написание с не.",
        )
        return CorrectionResult(text, text.replace(source, replacement, 1), [edit])


def test_first_run_checks_all_segments_and_builds_cache() -> None:
    corrector = FakeCorrector()
    incremental = IncrementalCorrector(corrector)

    result = incremental.correct_incremental("Я незнаю. Ты знаешь.")

    assert corrector.calls == ["Я незнаю.", "Ты знаешь."]
    assert result.source_text == "Я незнаю. Ты знаешь."
    assert result.corrected_text == "Я не знаю. Ты знаешь."
    assert result.checked_segments == 2
    assert result.reused_segments == 0
    assert result.changed_segments == 2
    assert len(result.segment_cache) == 2
    assert result.edits[0].start == 2
    assert result.edits[0].end == 8
    assert result.edits[0].replacement == "не знаю"


def test_second_run_with_same_text_reuses_all_segments() -> None:
    corrector = FakeCorrector()
    incremental = IncrementalCorrector(corrector)
    first = incremental.correct_incremental("Я незнаю. Ты знаешь.")
    corrector.calls.clear()

    result = incremental.correct_incremental("Я незнаю. Ты знаешь.", first.segment_cache)

    assert corrector.calls == []
    assert result.corrected_text == "Я не знаю. Ты знаешь."
    assert result.checked_segments == 0
    assert result.reused_segments == 2
    assert result.changed_segments == 0
    assert len(result.segment_cache) == 2


def test_changed_sentence_checks_only_that_segment() -> None:
    corrector = FakeCorrector()
    incremental = IncrementalCorrector(corrector)
    first = incremental.correct_incremental("Я незнаю. Ты знаешь.")
    corrector.calls.clear()

    result = incremental.correct_incremental("Я незнаю. Мы незнаю.", first.segment_cache)

    assert corrector.calls == ["Мы незнаю."]
    assert result.corrected_text == "Я не знаю. Мы не знаю."
    assert result.checked_segments == 1
    assert result.reused_segments == 1
    assert result.changed_segments == 1


def test_corrected_text_combines_cached_and_new_segments() -> None:
    corrector = FakeCorrector()
    incremental = IncrementalCorrector(corrector)
    first = incremental.correct_incremental("Первое незнаю. Второе верно.")
    corrector.calls.clear()

    result = incremental.correct_incremental("Первое незнаю. Третье незнаю.", first.segment_cache)

    assert corrector.calls == ["Третье незнаю."]
    assert result.corrected_text == "Первое не знаю. Третье не знаю."


def test_reused_cached_edits_use_current_global_offsets() -> None:
    corrector = FakeCorrector()
    incremental = IncrementalCorrector(corrector)
    first = incremental.correct_incremental("Я незнаю. Ты знаешь.")
    corrector.calls.clear()

    result = incremental.correct_incremental("Ты знаешь. Я незнаю.", first.segment_cache)

    assert corrector.calls == []
    assert result.corrected_text == "Ты знаешь. Я не знаю."
    assert result.checked_segments == 0
    assert result.reused_segments == 2
    assert result.changed_segments == 0
    assert result.edits[0].start == 13
    assert result.edits[0].end == 19


def test_whitespace_only_text_returns_empty_cache_without_checking() -> None:
    corrector = FakeCorrector()
    incremental = IncrementalCorrector(corrector)

    result = incremental.correct_incremental(" \n ")

    assert corrector.calls == []
    assert result.source_text == " \n "
    assert result.corrected_text == " \n "
    assert result.edits == []
    assert result.checked_segments == 0
    assert result.reused_segments == 0
    assert result.changed_segments == 0
    assert result.segment_cache == {}
