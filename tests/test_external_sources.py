from pathlib import Path

from src.data.external_sources import load_local_jsonl_pairs, load_local_m2_pairs, load_local_table_pairs


def test_external_source_loader_filters_unsupported_pairs(tmp_path: Path):
    path = tmp_path / "external.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"source": "Я незнаю что делать", "correction": "Я не знаю, что делать.", "domain": "unit"}',
                '{"source": "Я люблю дом", "correction": "Я обожаю дом", "domain": "unit"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_local_jsonl_pairs(path, source_dataset="unit_external")

    assert len(rows) == 1
    assert rows[0]["source_dataset"] == "unit_external"
    assert rows[0]["is_synthetic"] is False
    assert rows[0]["is_clean"] is False


def test_external_source_loader_filters_context_dependent_pairs(tmp_path: Path):
    path = tmp_path / "external.jsonl"
    path.write_text(
        '{"source": "Он пришел чтобы помочь.", "correction": "Он пришел что бы помочь.", "domain": "unit"}\n',
        encoding="utf-8",
    )

    rows = load_local_jsonl_pairs(path, source_dataset="unit_external")

    assert rows == []


def test_external_source_loader_keeps_supported_edits_from_mixed_pairs(tmp_path: Path):
    path = tmp_path / "external.jsonl"
    path.write_text(
        '{"source": "а так хочеться что-то менять", "correction": "А так хочется что-то менять.", "domain": "unit"}\n',
        encoding="utf-8",
    )

    rows = load_local_jsonl_pairs(path, source_dataset="unit_external")

    assert len(rows) == 1
    assert rows[0]["source"] == "а так хочеться что-то менять"
    assert rows[0]["target"] == "А так хочется что-то менять."
    assert '"spelling"' in rows[0]["error_types"]
    assert '"case"' in rows[0]["error_types"]
    assert '"final_punctuation"' in rows[0]["error_types"]


def test_external_table_loader_accepts_trusted_word_pairs(tmp_path: Path):
    path = tmp_path / "pairs.csv"
    path.write_text("incorrect,correct\nпредпологаю,предполагаю\nword,слово\n", encoding="utf-8")

    rows = load_local_table_pairs(path, source_dataset="unit_word_pairs", trusted_word_pairs=True)

    assert len(rows) == 1
    assert rows[0]["source"] == "предпологаю"
    assert rows[0]["target"] == "предполагаю"
    assert rows[0]["error_types"] == '["spelling"]'


def test_external_m2_loader_extracts_punctuation_pairs(tmp_path: Path):
    path = tmp_path / "sample.m2"
    path.write_text(
        "S Они поговорили о том , о сём и разошлись по домам .\n"
        "A 4 5|||None||||||REQUIRED|||-NONE-|||0\n\n",
        encoding="utf-8",
    )

    rows = load_local_m2_pairs(path, source_dataset="unit_m2")

    assert len(rows) == 1
    assert rows[0]["source"] == "Они поговорили о том, о сём и разошлись по домам."
    assert rows[0]["target"] == "Они поговорили о том о сём и разошлись по домам."
    assert rows[0]["error_types"] == '["punctuation"]'
