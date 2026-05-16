from pathlib import Path

from docx import Document

from src.docx.docx_corrector import correct_docx


def test_docx_correction_preserves_paragraph_count(tmp_path: Path):
    input_path = tmp_path / "input.docx"
    output_path = tmp_path / "output.docx"

    document = Document()
    document.add_paragraph("Я незнаю что делать")
    document.add_paragraph("Чистый текст.")
    document.save(input_path)

    edits = correct_docx(input_path, output_path)

    corrected = Document(output_path)
    paragraphs = [paragraph.text for paragraph in corrected.paragraphs]

    assert paragraphs == ["Я не знаю, что делать.", "Чистый текст."]
    assert edits


def test_docx_correction_preserves_first_run_bold_formatting(tmp_path: Path):
    input_path = tmp_path / "styled.docx"
    output_path = tmp_path / "styled_output.docx"

    document = Document()
    run = document.add_paragraph().add_run("Я незнаю что делать")
    run.bold = True
    document.save(input_path)

    correct_docx(input_path, output_path)

    corrected = Document(output_path)
    assert corrected.paragraphs[0].runs[0].text == "Я не знаю, что делать."
    assert corrected.paragraphs[0].runs[0].bold is True
