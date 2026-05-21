from pathlib import Path

import pandas as pd

from src.data.clean_sentence_pool import (
    CleanSentenceDecision,
    build_clean_sentence_pool,
    is_clean_sentence_acceptable,
)


def test_clean_sentence_filter_returns_decision_with_reasons():
    metadata = {"source_name": "unit_news", "domain": "news", "style": "neutral"}

    accepted = is_clean_sentence_acceptable(
        "Эксперты сообщили, что новый индекс вырос после публикации отчета.",
        metadata,
    )
    rejected = is_clean_sentence_acceptable(
        "Вот код `foo/bar.py`, смотри #тест и пиши @user прямо сейчас 😂.",
        metadata,
    )

    assert isinstance(accepted, CleanSentenceDecision)
    assert accepted.accepted is True
    assert accepted.reasons == []
    assert bool(accepted) is True
    assert rejected.accepted is False
    assert {"markup", "social_marker", "emoji"} & set(rejected.reasons)
    assert bool(rejected) is False


def test_clean_sentence_filter_rejects_forbidden_styles_and_meta_language():
    news = {"source_name": "unit_news", "domain": "news", "style": "neutral"}
    fiction = {"source_name": "unit_fiction", "domain": "fiction", "style": "literary"}

    assert not is_clean_sentence_acceptable(
        "В проверочном примере форма «зато» проверяет семейство context-pairs в серии 12.",
        news,
    )
    assert not is_clean_sentence_acceptable(
        "— Я обязательно вернусь завтра, — сказал он и посмотрел на темное окно.",
        fiction,
    )


def test_clean_sentence_pool_writes_canonical_columns_and_markdown_report(tmp_path: Path):
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
    output_path = tmp_path / "clean_sentence_pool.csv.gz"
    reports_dir = tmp_path / "reports"

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

    frame = pd.read_csv(output_path)
    assert result.accepted_count == 1
    assert {
        "token_count",
        "char_count",
        "normalized_text_hash",
        "accepted_reason",
        "rejected_reason",
        "raw_source_path",
    } <= set(frame.columns)
    assert "tokens_count" not in frame.columns
    assert (reports_dir / "clean_source_filter_report.csv").exists()
    assert (reports_dir / "clean_source_filter_report.md").exists()
    assert result.rejection_reason_counts["duplicate_normalized_text"] == 1
    assert result.rejection_reason_counts["synthetic_meta_language"] == 1
