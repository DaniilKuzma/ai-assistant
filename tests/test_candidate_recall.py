import json
from pathlib import Path

import pandas as pd

from src.evaluation.candidate_recall import build_candidate_recall_reports
from src.evaluation.candidate_recall import write_candidate_recall_reports


class EmptyCandidateGenerator:
    def generate(self, text: str):
        del text
        return []


def test_candidate_recall_is_one_when_gold_edit_is_generated(tmp_path: Path):
    rows = [
        {
            "source": "Жызнь прекрасна.",
            "target": "Жизнь прекрасна.",
            "edit_operations": json.dumps(
                [
                    {
                        "source": "Жызнь",
                        "replacement": "Жизнь",
                        "edit_type": "spelling_replace",
                        "start": 0,
                        "end": 5,
                        "rule_id": "frequent_error_exact",
                    }
                ],
                ensure_ascii=False,
            ),
        }
    ]

    reports = build_candidate_recall_reports(rows, rules_config_path=_rules_config(tmp_path))

    summary = reports["candidate_recall_by_rule"].set_index("rule_id")
    assert summary.loc["frequent_error_exact", "group"] == "typical_dictionary_words"
    assert summary.loc["frequent_error_exact", "gold_count"] == 1
    assert summary.loc["frequent_error_exact", "candidate_present_count"] == 1
    assert summary.loc["frequent_error_exact", "candidate_recall"] == 1.0
    assert summary.loc["frequent_error_exact", "missing_count"] == 0


def test_candidate_recall_marks_missing_when_gold_edit_is_not_generated(tmp_path: Path):
    rows = [
        {
            "source": "Молко свежее.",
            "target": "Молоко свежее.",
            "edit_operations": json.dumps(
                [
                    {
                        "source": "Молко",
                        "replacement": "Молоко",
                        "edit_type": "spelling_replace",
                        "start": 0,
                        "end": 5,
                        "rule_id": "dictionary_fuzzy",
                    }
                ],
                ensure_ascii=False,
            ),
        }
    ]

    reports = build_candidate_recall_reports(rows, rules_config_path=_rules_config(tmp_path))

    summary = reports["candidate_recall_by_rule"].set_index("rule_id")
    assert summary.loc["dictionary_fuzzy", "gold_count"] == 1
    assert summary.loc["dictionary_fuzzy", "candidate_present_count"] == 0
    assert summary.loc["dictionary_fuzzy", "candidate_recall"] == 0.0
    assert summary.loc["dictionary_fuzzy", "missing_count"] == 1
    assert "Молко" in summary.loc["dictionary_fuzzy", "missing_examples"]


def test_candidate_recall_keeps_unknown_rule_id_without_crashing(tmp_path: Path):
    rows = [
        {
            "source": "Жызнь прекрасна.",
            "target": "Жизнь прекрасна.",
            "edit_operations": json.dumps(
                [
                    {
                        "source": "Жызнь",
                        "replacement": "Жизнь",
                        "edit_type": "spelling_replace",
                        "start": 0,
                        "end": 5,
                    }
                ],
                ensure_ascii=False,
            ),
        }
    ]

    reports = build_candidate_recall_reports(rows, rules_config_path=_rules_config(tmp_path))

    summary = reports["candidate_recall_by_rule"].set_index("rule_id")
    assert summary.loc["unknown", "group"] == "unknown"
    assert summary.loc["unknown", "gold_count"] == 1
    assert summary.loc["unknown", "candidate_present_count"] == 1


def test_candidate_recall_writer_creates_csv_files(tmp_path: Path):
    rows = [
        {
            "source": "Жызнь прекрасна",
            "target": "Жизнь прекрасна.",
            "edit_operations": json.dumps(
                [
                    {
                        "source": "Жызнь",
                        "replacement": "Жизнь",
                        "edit_type": "spelling_replace",
                        "start": 0,
                        "end": 5,
                        "rule_id": "frequent_error_exact",
                    },
                    {
                        "source": "",
                        "replacement": ".",
                        "edit_type": "final_punctuation",
                        "start": 15,
                        "end": 15,
                        "rule_id": "final_punctuation_default",
                    },
                ],
                ensure_ascii=False,
            ),
        }
    ]

    write_candidate_recall_reports(rows, tmp_path, rules_config_path=_rules_config(tmp_path))

    candidate_report = pd.read_csv(tmp_path / "candidate_recall_by_rule.csv")
    gap_report = pd.read_csv(tmp_path / "gap_label_coverage_by_rule.csv")
    assert list(candidate_report.columns) == [
        "rule_id",
        "group",
        "gold_count",
        "candidate_present_count",
        "candidate_recall",
        "missing_count",
        "missing_examples",
    ]
    assert list(gap_report.columns) == [
        "rule_id",
        "group",
        "gold_gap_count",
        "candidate_gap_present_count",
        "gap_candidate_recall",
        "missing_examples",
    ]


def test_gap_label_coverage_counts_punctuation_candidates_by_rule(tmp_path: Path):
    rows = [
        {
            "source": "Я думаю что готово",
            "target": "Я думаю, что готово.",
            "edit_operations": json.dumps(
                [
                    {
                        "source": "",
                        "replacement": ",",
                        "edit_type": "punctuation_insert",
                        "start": 7,
                        "end": 7,
                        "rule_id": "comma_subordinate",
                    },
                    {
                        "source": "",
                        "replacement": ".",
                        "edit_type": "final_punctuation",
                        "start": 18,
                        "end": 18,
                        "rule_id": "final_punctuation_default",
                    },
                ],
                ensure_ascii=False,
            ),
        }
    ]

    reports = build_candidate_recall_reports(rows, rules_config_path=_rules_config(tmp_path))

    gap_summary = reports["gap_label_coverage_by_rule"].set_index("rule_id")
    assert gap_summary.loc["comma_subordinate", "gold_gap_count"] == 1
    assert gap_summary.loc["comma_subordinate", "candidate_gap_present_count"] == 1
    assert gap_summary.loc["comma_subordinate", "gap_candidate_recall"] == 1.0
    assert gap_summary.loc["final_punctuation_default", "gold_gap_count"] == 1
    assert gap_summary.loc["final_punctuation_default", "candidate_gap_present_count"] == 1
    assert gap_summary.loc["final_punctuation_default", "gap_candidate_recall"] == 1.0


def test_candidate_recall_does_not_trust_pre_dataset_probe_metadata_by_default(tmp_path: Path):
    rows = [
        {
            "source": "Документ готов",
            "target": "Документ готов.",
            "rule_id": "final_punctuation_default",
            "edit_operations": json.dumps(
                [
                    {
                        "source": "",
                        "replacement": ".",
                        "edit_type": "final_punctuation",
                        "start": 14,
                        "end": 14,
                        "rule_id": "final_punctuation_default",
                    }
                ],
                ensure_ascii=False,
            ),
            "metadata": json.dumps(
                {"candidate_present": True, "target_family": "final_punctuation_default"},
                ensure_ascii=False,
            ),
        }
    ]

    reports = build_candidate_recall_reports(
        rows,
        candidate_generator=EmptyCandidateGenerator(),
        rules_config_path=_rules_config(tmp_path),
    )

    summary = reports["candidate_recall_by_rule"].set_index("rule_id")
    assert summary.loc["final_punctuation_default", "candidate_present_count"] == 0
    assert summary.loc["final_punctuation_default", "candidate_recall"] == 0.0


def _rules_config(tmp_path: Path) -> Path:
    path = tmp_path / "rules.yaml"
    path.write_text(
        """
orthography:
  typical_dictionary_words:
    rules: [frequent_error_exact]
  typos_letter_operations:
    rules: [dictionary_fuzzy]
punctuation:
  complex_sentence_subordinate:
    rules: [comma_subordinate]
  sentence_final_default_dot:
    rules: [final_punctuation_default]
""",
        encoding="utf-8",
    )
    return path
