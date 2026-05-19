from pathlib import Path

from src.candidates.candidate_generator import CandidateGenerator
from src.data.real_error_sources import load_real_error_pairs, validate_real_error_pair


def test_real_pair_validation_requires_minimal_candidate_backed_edit():
    accepted = validate_real_error_pair(
        "Жызнь прекрасна.",
        "Жизнь прекрасна.",
        source_dataset="unit",
        candidate_generator=CandidateGenerator(),
    )
    rejected = validate_real_error_pair(
        "Я люблю дом.",
        "Мы полностью изменили смысл предложения.",
        source_dataset="unit",
        candidate_generator=CandidateGenerator(),
    )

    assert accepted.accepted is True
    assert accepted.row is not None
    assert accepted.row["source_type"] == "real_error_pair"
    assert accepted.row["is_real_pair"] is True
    assert rejected.accepted is False
    assert rejected.reason in {"char_edit_distance_too_high", "token_edit_distance_too_high", "unsupported_edit_type"}


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
            }
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
