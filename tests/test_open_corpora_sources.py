from pathlib import Path

from src.data.clean_sentence_pool import build_clean_sentence_pool, is_clean_sentence_acceptable
from src.data.open_corpora_sources import load_open_corpora_sentences


def test_local_text_clean_source_loads_sentences_with_metadata(tmp_path: Path):
    source_path = tmp_path / "news.txt"
    source_path.write_text(
        "Эксперты сообщили, что новый индекс вырос после публикации отчета.\n"
        "Коротко.\n",
        encoding="utf-8",
    )

    result = load_open_corpora_sentences(
        {
            "clean_sources": {
                "unit_news": {
                    "enabled": True,
                    "type": "local_text",
                    "local_path": str(source_path),
                    "source_subcorpus": "news",
                    "domain": "news",
                    "style": "neutral",
                    "license_status": "unit",
                    "max_sentences": 10,
                }
            }
        }
    )

    assert [record.text for record in result.records] == [
        "Эксперты сообщили, что новый индекс вырос после публикации отчета."
    ]
    assert result.records[0].source_name == "unit_news"
    assert result.records[0].source_subcorpus == "news"
    assert result.source_reports[0]["status"] == "loaded"


def test_missing_clean_source_is_skipped_with_reason(tmp_path: Path):
    result = load_open_corpora_sentences(
        {
            "clean_sources": {
                "missing_lenta": {
                    "enabled": True,
                    "type": "corus_lenta",
                    "local_path": str(tmp_path / "missing.csv.bz2"),
                    "domain": "news",
                    "style": "neutral",
                }
            }
        }
    )

    assert result.records == []
    assert result.source_reports[0]["status"] == "skipped"
    assert result.source_reports[0]["reason"] == "missing_local_path"


def test_clean_sentence_filter_rejects_meta_language_and_social_noise():
    metadata = {"source_name": "unit_news", "domain": "news", "style": "neutral"}

    assert is_clean_sentence_acceptable(
        "Эксперты сообщили, что новый индекс вырос после публикации отчета.",
        metadata,
    )
    assert not is_clean_sentence_acceptable(
        "В проверочном примере форма «зато» проверяет семейство context-pairs в серии 12.",
        metadata,
    )
    assert not is_clean_sentence_acceptable(
        "Ну что, чувак, это #тест от @user 😂",
        metadata,
    )


def test_clean_sentence_pool_writes_filter_report_and_deduplicates(tmp_path: Path):
    source_path = tmp_path / "news.txt"
    source_path.write_text(
        "\n".join(
            [
                "Эксперты сообщили, что новый индекс вырос после публикации отчета.",
                "Эксперты сообщили, что новый индекс вырос после публикации отчета.",
                "В проверочном примере форма «зато» проверяет семейство context-pairs в серии 12.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    reports_dir = tmp_path / "reports"
    output_path = tmp_path / "clean_sentence_pool.csv.gz"

    result = build_clean_sentence_pool(
        {
            "clean_sources": {
                "unit_news": {
                    "enabled": True,
                    "type": "local_text",
                    "local_path": str(source_path),
                    "source_subcorpus": "news",
                    "domain": "news",
                    "style": "neutral",
                    "license_status": "unit",
                    "max_sentences": 20,
                }
            },
            "pool": {"min_clean_sentences": 1, "max_source_share": 1.0, "max_subcorpus_share": 1.0},
        },
        output_path=output_path,
        reports_dir=reports_dir,
    )

    assert result.accepted_count == 1
    assert output_path.exists()
    assert (reports_dir / "clean_source_filter_report.csv").exists()
    assert result.rejection_reason_counts["duplicate_normalized_text"] == 1
    assert result.rejection_reason_counts["synthetic_meta_language"] == 1
