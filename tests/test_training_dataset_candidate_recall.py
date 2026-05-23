import json

import pandas as pd

import src.data.operator_dataset_builder as operator_builder
from src.evaluation.candidate_recall import build_candidate_recall_reports


class EmptyCandidateGenerator:
    def generate(self, text: str):
        del text
        return []


def test_candidate_recall_does_not_trust_candidate_backed_target_family_metadata():
    rows = [
        {
            "source": "Если отчет готов мы отправим его утром.",
            "target": "Если отчет готов, мы отправим его утром.",
            "rule_id": "comma_subordinate",
            "rule_ids": json.dumps(["comma_subordinate"], ensure_ascii=False),
            "edit_operations": json.dumps(
                [
                    {
                        "source": "",
                        "replacement": ",",
                        "edit_type": "punctuation_insert",
                        "start": 16,
                        "end": 16,
                        "rule_id": "comma_subordinate",
                    }
                ],
                ensure_ascii=False,
            ),
            "metadata": json.dumps(
                {"candidate_present": True, "target_family": "comma_subordinate"},
                ensure_ascii=False,
            ),
        }
    ]

    reports = build_candidate_recall_reports(rows, candidate_generator=EmptyCandidateGenerator())
    candidate = reports["candidate_recall_by_rule"].set_index("rule_id")
    gap = reports["gap_label_coverage_by_rule"].set_index("rule_id")

    assert candidate.loc["comma_subordinate", "candidate_recall"] == 0.0
    assert candidate.loc["comma_subordinate", "missing_count"] == 1
    assert gap.loc["comma_subordinate", "gap_candidate_recall"] == 0.0


def test_operator_candidate_reports_use_real_generator_not_manifest_counts(tmp_path, monkeypatch):
    frame = pd.DataFrame(
        [
            {
                "source": "Если отчет готов мы отправим его утром.",
                "target": "Если отчет готов, мы отправим его утром.",
                "source_type": "synthetic_augmented_from_open_clean",
                "rule_id": "comma_subordinate",
                "rule_ids": json.dumps(["comma_subordinate"], ensure_ascii=False),
                "edit_operations": json.dumps(
                    [
                        {
                            "source": "",
                            "replacement": ",",
                            "edit_type": "punctuation_insert",
                            "start": 16,
                            "end": 16,
                            "rule_id": "comma_subordinate",
                        }
                    ],
                    ensure_ascii=False,
                ),
                "metadata": json.dumps(
                    {"candidate_present": True, "target_family": "comma_subordinate"},
                    ensure_ascii=False,
                ),
            }
        ]
    )
    manifest = {
        "operator_acceptance_counts": {"comma_subordinate": 1},
        "active_rule_ids": ["comma_subordinate"],
    }
    config = {"model": {"max_candidates": 16}, "data": {"training_dataset_core": {"audit": {"candidate_recall_min": 0.95}}}}
    monkeypatch.setattr(operator_builder.CandidateGenerator, "from_config", lambda _config: EmptyCandidateGenerator())

    reports = operator_builder._candidate_reports(frame, manifest, tmp_path, config=config)

    report = pd.read_csv(tmp_path / "candidate_recall_by_rule.csv").set_index("rule_id")
    gate = pd.read_csv(tmp_path / "candidate_recall_gate_report.csv").set_index("rule_id")
    assert reports["candidate_recall_by_rule"].set_index("rule_id").loc["comma_subordinate", "candidate_recall"] == 0.0
    assert report.loc["comma_subordinate", "candidate_recall"] == 0.0
    assert gate.loc["comma_subordinate", "status"] == "fail"
    assert gate.loc["comma_subordinate", "min_required_recall"] == 0.95
