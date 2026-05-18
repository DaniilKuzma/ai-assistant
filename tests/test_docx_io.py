from pathlib import Path

from docx import Document

from src.docx.docx_corrector import correct_docx
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
