from pathlib import Path

from src.data.external_sources import load_local_jsonl_pairs


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
