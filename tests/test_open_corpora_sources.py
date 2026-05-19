from pathlib import Path

import pandas as pd
import yaml

from src.data.clean_sentence_pool import build_clean_sentence_pool, is_clean_sentence_acceptable
from src.data.open_corpora_sources import load_open_corpora_sentences
from src.data.source_downloads import DownloadBudget, download_if_allowed


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


def test_download_if_allowed_respects_env_flag_and_local_first(tmp_path: Path, monkeypatch):
    local_path = tmp_path / "source.txt"
    local_path.write_text("Текст уже лежит локально.\n", encoding="utf-8")
    policy = {
        "allow_downloads_env": "UNIT_ALLOW_DOWNLOADS",
        "cache_dir": str(tmp_path),
        "max_source_download_mb": 1,
        "max_total_download_mb": 1,
    }

    local = download_if_allowed("unit", {"local_path": str(local_path), "url": "https://example.test/source.txt"}, policy, DownloadBudget())
    missing = download_if_allowed("missing", {"local_path": str(tmp_path / "missing.txt"), "url": "https://example.test/source.txt"}, policy, DownloadBudget())
    monkeypatch.setenv("UNIT_ALLOW_DOWNLOADS", "1")
    oversize = download_if_allowed(
        "oversize",
        {"local_path": str(tmp_path / "oversize.txt"), "url": "https://example.test/source.txt", "max_download_mb": 2},
        policy,
        DownloadBudget(),
    )

    assert local.mode == "local"
    assert local.used is True
    assert missing.mode == "skipped_downloads_disabled"
    assert "UNIT_ALLOW_DOWNLOADS=1" in missing.required_commands[0]
    assert oversize.mode == "skipped_size_limit"


def test_open_corpora_config_uses_controlled_download_schema():
    config = yaml.safe_load(Path("configs/open_corpora_sources.yaml").read_text(encoding="utf-8"))

    policy = config["sources"]["download_policy"]
    assert policy["mode"] == "local_first_with_controlled_downloads"
    assert policy["allow_downloads_env"] == "RUSSIAN_CORRECTOR_ALLOW_SOURCE_DOWNLOADS"
    assert policy["fail_if_insufficient_sources"] is True
    assert config["clean_sources"]["lenta_news"]["type"] == "corus_lenta2"
    assert config["clean_sources"]["nerus_news"]["type"] == "nerus_conllu"
    assert config["clean_sources"]["opencorpora"]["archive_path"].endswith(".zip")
    assert config["clean_sources"]["taiga_news_wiki"]["enabled"] is False


def test_nerus_conllu_extractor_does_not_require_nerus_package(tmp_path: Path):
    source_path = tmp_path / "nerus.conllu"
    source_path.write_text(
        "# sent_id = 1\n"
        "# text = Эксперты сообщили, что новый индекс вырос после публикации отчета.\n"
        "1\tЭксперты\t_\t_\t_\t_\t_\t_\t_\t_\n"
        "\n",
        encoding="utf-8",
    )

    result = load_open_corpora_sentences(
        {
            "clean_sources": {
                "unit_nerus": {
                    "enabled": True,
                    "type": "nerus_conllu",
                    "local_path": str(source_path),
                    "source_subcorpus": "nerus_lenta",
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
    assert result.source_reports[0]["status"] == "loaded"


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


def test_clean_sentence_pool_uses_license_status_column_and_reports_download_mode(tmp_path: Path):
    source_path = tmp_path / "news.txt"
    source_path.write_text(
        "Эксперты сообщили, что новый индекс вырос после публикации отчета.\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "clean_sentence_pool.csv.gz"
    reports_dir = tmp_path / "reports"

    result = build_clean_sentence_pool(
        {
            "sources": {"download_policy": {"mode": "local_first_with_controlled_downloads"}},
            "clean_sources": {
                "unit_news": {
                    "enabled": True,
                    "type": "local_text",
                    "local_path": str(source_path),
                    "source_subcorpus": "news",
                    "domain": "news",
                    "style": "neutral",
                    "license_status": "unit-license",
                    "max_sentences": 10,
                }
            },
            "pool": {"min_clean_sentences": 1, "max_source_share": 1.0, "max_subcorpus_share": 1.0},
        },
        output_path=output_path,
        reports_dir=reports_dir,
    )

    frame = pd.read_csv(output_path)
    report = (reports_dir / "source_ingestion_report.md").read_text(encoding="utf-8")
    assert result.accepted_count == 1
    assert "license_status" in frame.columns
    assert "license/status" not in frame.columns
    assert "| unit_news | loaded | local |" in report


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
