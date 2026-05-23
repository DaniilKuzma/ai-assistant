from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.data.clean_sentence_pool import (
    CleanSentenceDecision,
    build_clean_sentence_pool,
    clean_sentence_rejection_reasons,
    has_latin_confusable_inside_cyrillic_word,
    has_mixed_script_token,
    is_clean_sentence_acceptable,
    mixed_script_tokens,
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


def test_mixed_script_helpers_detect_contiguous_runs_and_ignore_separated_latin_names():
    text = "OpenAI модель и AI-модель, но обeзврежено и cистема."

    assert mixed_script_tokens(text) == ["обeзврежено", "cистема"]
    assert has_mixed_script_token(text) is True
    assert has_latin_confusable_inside_cyrillic_word(text) is True
    assert mixed_script_tokens("Компания OpenAI представила новую AI-модель.") == []
    assert has_mixed_script_token("IT-компания открыла новую PR-службу.") is False
    assert has_latin_confusable_inside_cyrillic_word("Новая система обработки данных заработала утром.") is False


def test_clean_sentence_filter_rejects_mixed_script_tokens_by_default_and_allows_opt_out():
    metadata = {"source_name": "unit_news", "domain": "news", "style": "neutral"}
    mixed = "В аэропорту обeзврежено взрывное устройство после проверки."
    confusable = "Новая cистема обработки данных заработала утром."
    clean = "Новая система обработки данных заработала утром."
    latin_name = "Компания OpenAI представила новый сервис для русских пользователей."
    hyphen_compound = "Новая AI-модель обработки данных заработала утром после теста."

    assert "mixed_script_token" in clean_sentence_rejection_reasons(mixed, metadata)
    assert "latin_confusable_inside_cyrillic_word" in clean_sentence_rejection_reasons(mixed, metadata)
    assert "mixed_script_token" in clean_sentence_rejection_reasons(confusable, metadata)
    assert "latin_confusable_inside_cyrillic_word" in clean_sentence_rejection_reasons(confusable, metadata)
    assert clean_sentence_rejection_reasons(clean, metadata) == []
    assert clean_sentence_rejection_reasons(latin_name, metadata) == []
    assert clean_sentence_rejection_reasons(hyphen_compound, metadata) == []
    assert clean_sentence_rejection_reasons(
        mixed,
        metadata,
        pool_config={
            "reject_mixed_script_tokens": False,
            "reject_latin_confusable_inside_cyrillic_word": False,
        },
    ) == []


def test_clean_sentence_filter_opt_in_high_confidence_candidate_rejection():
    metadata = {"source_name": "unit_news", "domain": "news", "style": "neutral"}
    text = "Эксперты сообщили, что жизнь в городе стала спокойнее после реформы."
    start = text.index("жизнь")
    candidate = SimpleNamespace(
        source="жизнь",
        replacement="жызнь",
        edit_type="spelling",
        start=start,
        end=start + len("жизнь"),
        confidence=0.96,
        requires_scoring=False,
    )

    class Generator:
        def generate(self, value: str):
            assert value == text
            return [candidate]

    assert clean_sentence_rejection_reasons(
        text,
        metadata,
        pool_config={"reject_if_candidate_generator_finds_high_confidence_fix": False},
        candidate_generator=Generator(),
    ) == []
    assert clean_sentence_rejection_reasons(
        text,
        metadata,
        pool_config={
            "reject_if_candidate_generator_finds_high_confidence_fix": True,
            "high_confidence_candidate_threshold": 0.95,
        },
        candidate_generator=Generator(),
    ) == ["high_confidence_autocorrection_candidate"]


def test_high_confidence_candidate_filter_skips_primary_rejections_and_protected_spans():
    metadata = {"source_name": "unit_news", "domain": "news", "style": "neutral"}
    mixed = "В аэропорту обeзврежено взрывное устройство после проверки."
    protected = "Компания OpenAI представила новый сервис для русских пользователей."

    class CountingGenerator:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, _value: str):
            self.calls += 1
            return []

    class ProtectedCandidate:
        source = "OpenAI"
        replacement = "OpenAO"
        edit_type = "spelling"
        start = protected.index("OpenAI")
        end = start + len("OpenAI")
        confidence = 1.0
        requires_scoring = False

    class ProtectedGenerator:
        def generate(self, _value: str):
            return [ProtectedCandidate()]

    counting = CountingGenerator()
    assert "mixed_script_token" in clean_sentence_rejection_reasons(
        mixed,
        metadata,
        pool_config={"reject_if_candidate_generator_finds_high_confidence_fix": True},
        candidate_generator=counting,
    )
    assert counting.calls == 0
    assert clean_sentence_rejection_reasons(
        protected,
        metadata,
        pool_config={"reject_if_candidate_generator_finds_high_confidence_fix": True},
        candidate_generator=ProtectedGenerator(),
    ) == []


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
                "Новая cистема обработки данных заработала утром.",
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
    assert result.rejection_reason_counts["mixed_script_token"] == 1
    assert result.rejection_reason_counts["synthetic_meta_language"] == 1
