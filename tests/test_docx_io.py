from pathlib import Path

from docx import Document

from src.docx.docx_corrector import correct_docx, correct_docx_incremental
from src.inference.corrector import CorrectionResult
from src.validation.diff_analyzer import Edit


FAKE_EDITS = [
    Edit("незнаю", "не знаю", "split_word", 2, 8, status="accepted", rule_id="fake_split"),
    Edit("", ",", "punctuation_insert", 15, 15, status="accepted", rule_id="fake_comma"),
]


class FakeCorrector:
    def __init__(self, corrections: dict[str, str], *, falsey: bool = False) -> None:
        self.corrections = corrections
        self.falsey = falsey
        self.calls: list[str] = []

    def __bool__(self) -> bool:
        return not self.falsey

    def correct(self, text: str) -> CorrectionResult:
        self.calls.append(text)
        corrected_text = self.corrections.get(text, text)
        edits = list(FAKE_EDITS) if corrected_text != text else []
        return CorrectionResult(text, corrected_text, edits)


def test_docx_correction_preserves_paragraph_count(tmp_path: Path):
    input_path = tmp_path / "input.docx"
    output_path = tmp_path / "output.docx"

    document = Document()
    document.add_paragraph("Я незнаю что делать")
    document.add_paragraph("Чистый текст.")
    document.save(input_path)

    fake_corrector = FakeCorrector({"Я незнаю что делать": "Я не знаю, что делать."})
    edits = correct_docx(input_path, output_path, corrector=fake_corrector)

    corrected = Document(output_path)
    paragraphs = [paragraph.text for paragraph in corrected.paragraphs]

    assert paragraphs == ["Я не знаю, что делать.", "Чистый текст."]
    assert fake_corrector.calls == ["Я незнаю что делать", "Чистый текст."]
    assert edits == FAKE_EDITS


def test_docx_correction_preserves_first_run_bold_formatting(tmp_path: Path):
    input_path = tmp_path / "styled.docx"
    output_path = tmp_path / "styled_output.docx"

    document = Document()
    run = document.add_paragraph().add_run("Я незнаю что делать")
    run.bold = True
    document.save(input_path)

    fake_corrector = FakeCorrector({"Я незнаю что делать": "Я не знаю, что делать."})
    edits = correct_docx(input_path, output_path, corrector=fake_corrector)

    corrected = Document(output_path)
    assert corrected.paragraphs[0].runs[0].text == "Я не знаю, что делать."
    assert corrected.paragraphs[0].runs[0].bold is True
    assert fake_corrector.calls == ["Я незнаю что делать"]
    assert edits == FAKE_EDITS


def test_docx_correction_preserves_inline_run_formatting_boundaries(tmp_path: Path):
    input_path = tmp_path / "inline_styled.docx"
    output_path = tmp_path / "inline_styled_output.docx"

    document = Document()
    paragraph = document.add_paragraph()
    paragraph.add_run("Я ")
    bold_run = paragraph.add_run("незнаю")
    bold_run.bold = True
    italic_run = paragraph.add_run(" что делать")
    italic_run.italic = True
    document.save(input_path)

    fake_corrector = FakeCorrector({"Я незнаю что делать": "Я не знаю, что делать."})
    edits = correct_docx(input_path, output_path, corrector=fake_corrector)

    corrected = Document(output_path)
    paragraph = corrected.paragraphs[0]
    assert paragraph.text == "Я не знаю, что делать."
    assert paragraph.runs[0].text == "Я "
    assert paragraph.runs[1].text == "не знаю"
    assert paragraph.runs[1].bold is True
    assert paragraph.runs[2].text == ", что делать."
    assert paragraph.runs[2].italic is True
    assert fake_corrector.calls == ["Я незнаю что делать"]
    assert edits == FAKE_EDITS


def test_docx_correction_uses_injected_corrector_without_constructing_default(tmp_path: Path, monkeypatch):
    input_path = tmp_path / "injected.docx"
    output_path = tmp_path / "injected_output.docx"

    document = Document()
    document.add_paragraph("Я незнаю что делать")
    document.save(input_path)

    def fail_if_default_corrector_is_constructed():
        raise AssertionError("DOCX correction must use the injected corrector")

    monkeypatch.setattr("src.docx.docx_corrector.Corrector", fail_if_default_corrector_is_constructed)
    fake_corrector = FakeCorrector({"Я незнаю что делать": "Я не знаю, что делать."}, falsey=True)

    edits = correct_docx(input_path, output_path, corrector=fake_corrector)

    corrected = Document(output_path)
    assert corrected.paragraphs[0].text == "Я не знаю, что делать."
    assert fake_corrector.calls == ["Я незнаю что делать"]
    assert edits == FAKE_EDITS


def test_incremental_docx_first_run_checks_non_empty_paragraphs_and_writes_output(tmp_path: Path):
    input_path = tmp_path / "incremental_first.docx"
    output_path = tmp_path / "incremental_first_output.docx"

    document = Document()
    document.add_paragraph("Я незнаю что делать")
    document.add_paragraph("")
    document.add_paragraph("   ")
    document.add_paragraph("Чистый текст.")
    document.save(input_path)

    fake_corrector = FakeCorrector({"Я незнаю что делать": "Я не знаю, что делать."})
    result = correct_docx_incremental(input_path, output_path, corrector=fake_corrector)

    corrected = Document(output_path)
    paragraphs = [paragraph.text for paragraph in corrected.paragraphs]

    assert output_path.exists()
    assert paragraphs == ["Я не знаю, что делать.", "", "   ", "Чистый текст."]
    assert fake_corrector.calls == ["Я незнаю что делать", "Чистый текст."]
    assert result.checked_paragraphs == 2
    assert result.reused_paragraphs == 0
    assert len(result.paragraph_cache) == 2
    assert result.edits == FAKE_EDITS


def test_incremental_docx_second_run_reuses_same_document_paragraphs(tmp_path: Path):
    input_path = tmp_path / "incremental_reuse.docx"
    first_output_path = tmp_path / "incremental_reuse_first_output.docx"
    second_output_path = tmp_path / "incremental_reuse_second_output.docx"

    document = Document()
    document.add_paragraph("Я незнаю что делать")
    document.add_paragraph("")
    document.add_paragraph("Чистый текст.")
    document.save(input_path)

    first_corrector = FakeCorrector({"Я незнаю что делать": "Я не знаю, что делать."})
    first = correct_docx_incremental(input_path, first_output_path, corrector=first_corrector)

    second_corrector = FakeCorrector({})
    second = correct_docx_incremental(
        input_path,
        second_output_path,
        corrector=second_corrector,
        previous_cache=first.paragraph_cache,
    )

    corrected = Document(second_output_path)
    paragraphs = [paragraph.text for paragraph in corrected.paragraphs]

    assert second_output_path.exists()
    assert paragraphs == ["Я не знаю, что делать.", "", "Чистый текст."]
    assert second_corrector.calls == []
    assert second.checked_paragraphs == 0
    assert second.reused_paragraphs == 2
    assert len(second.paragraph_cache) == 2
    assert second.edits == FAKE_EDITS


def test_incremental_docx_changed_paragraph_checks_only_changed_paragraph(tmp_path: Path):
    first_input_path = tmp_path / "incremental_changed_first.docx"
    second_input_path = tmp_path / "incremental_changed_second.docx"
    first_output_path = tmp_path / "incremental_changed_first_output.docx"
    second_output_path = tmp_path / "incremental_changed_second_output.docx"

    first_document = Document()
    first_document.add_paragraph("Первый незнаю")
    first_document.add_paragraph("Второй текст.")
    first_document.save(first_input_path)

    second_document = Document()
    second_document.add_paragraph("Первый незнаю")
    second_document.add_paragraph("Второй незнаю")
    second_document.save(second_input_path)

    first_corrector = FakeCorrector(
        {
            "Первый незнаю": "Первый не знаю",
            "Второй текст.": "Второй текст.",
        }
    )
    first = correct_docx_incremental(first_input_path, first_output_path, corrector=first_corrector)

    second_corrector = FakeCorrector({"Второй незнаю": "Второй не знаю"})
    second = correct_docx_incremental(
        second_input_path,
        second_output_path,
        corrector=second_corrector,
        previous_cache=first.paragraph_cache,
    )

    corrected = Document(second_output_path)
    paragraphs = [paragraph.text for paragraph in corrected.paragraphs]

    assert second_output_path.exists()
    assert paragraphs == ["Первый не знаю", "Второй не знаю"]
    assert second_corrector.calls == ["Второй незнаю"]
    assert second.checked_paragraphs == 1
    assert second.reused_paragraphs == 1
    assert len(second.paragraph_cache) == 2
