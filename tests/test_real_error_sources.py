from pathlib import Path

import yaml

from src.candidates.candidate_generator import CandidateGenerator
from src.data.real_error_sources import load_real_error_pairs, validate_real_error_pair


def test_real_pair_validation_requires_minimal_candidate_backed_edit():
    accepted = validate_real_error_pair(
        "Жызнь в городе стала заметно спокойнее.",
        "Жизнь в городе стала заметно спокойнее.",
        source_dataset="unit",
        candidate_generator=CandidateGenerator(),
    )
    rejected = validate_real_error_pair(
        "Я люблю этот старый дом возле широкой реки.",
        "Мы полностью изменили смысл предложения без сохранения исходной мысли.",
        source_dataset="unit",
        candidate_generator=CandidateGenerator(),
    )

    assert accepted.accepted is True
    assert accepted.row is not None
    assert accepted.row["source_type"] == "real_error_pair"
    assert accepted.row["is_real_pair"] is True
    assert rejected.accepted is False
    assert rejected.reason in {"char_edit_distance_too_high", "token_edit_distance_too_high", "unsupported_edit_type"}


def test_real_pair_validation_requires_sentence_length_and_rejects_social_noise():
    too_short = validate_real_error_pair(
        "Жызнь прекрасна.",
        "Жизнь прекрасна.",
        source_dataset="unit",
        candidate_generator=CandidateGenerator(),
    )
    social = validate_real_error_pair(
        "Я думал, что жызнь прекрасна, чувак, и написал это вечером.",
        "Я думал, что жизнь прекрасна, чувак, и написал это вечером.",
        source_dataset="unit",
        candidate_generator=CandidateGenerator(),
    )

    assert too_short.accepted is False
    assert too_short.reason == "too_few_tokens"
    assert social.accepted is False
    assert social.reason == "forbidden_domain_noise"


def test_real_error_config_uses_controlled_download_schema():
    config = yaml.safe_load(Path("configs/real_error_sources.yaml").read_text(encoding="utf-8"))

    policy = config["sources"]["download_policy"]
    assert policy["mode"] == "local_first_with_controlled_downloads"
    assert policy["allow_downloads_env"] == "RUSSIAN_CORRECTOR_ALLOW_SOURCE_DOWNLOADS"
    assert policy["fail_if_insufficient_sources"] is False
    assert config["real_sources"]["spellcheck_benchmark"]["type"] == "huggingface_dataset"
    assert config["real_sources"]["spellcheck_benchmark"]["hf_id"] == "ai-forever/spellcheck_benchmark"
    assert config["real_sources"]["spellcheck_punctuation_benchmark"]["cap_share"] == 0.25


def test_real_pair_loader_reports_rejections(tmp_path: Path):
    source_path = tmp_path / "pairs.jsonl"
    source_path.write_text(
        '{"source": "Жызнь прекрасна.", "correction": "Жизнь прекрасна.", "domain": "unit"}\n'
        '{"source": "Я люблю дом.", "correction": "Мы полностью изменили смысл предложения.", "domain": "unit"}\n',
        encoding="utf-8",
    )
    reports_dir = tmp_path / "reports"
    output_path = tmp_path / "real_error_pairs_validated.csv.gz"

    result = load_real_error_pairs(
        {
            "real_sources": {
                "unit_pairs": {
                    "enabled": True,
                    "type": "local_jsonl",
                    "local_path": str(source_path),
                    "max_examples": 10,
                }
            },
            "validation": {"min_tokens": 2},
        },
        candidate_generator=CandidateGenerator(),
        output_path=output_path,
        reports_dir=reports_dir,
    )

    assert result.accepted_count == 1
    assert result.rejected_count == 1
    assert output_path.exists()
    assert (reports_dir / "real_pair_filter_report.csv").exists()
    assert (reports_dir / "rejected_real_pairs.csv").exists()
    assert result.source_reports[0]["status"] == "loaded"
