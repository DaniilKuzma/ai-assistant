import json

from src.evaluation.candidate_recall import build_candidate_recall_reports


def test_candidate_recall_trusts_candidate_backed_target_family_metadata_for_broad_synthetic_rows():
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

    reports = build_candidate_recall_reports(rows)
    candidate = reports["candidate_recall_by_rule"].set_index("rule_id")
    gap = reports["gap_label_coverage_by_rule"].set_index("rule_id")

    assert candidate.loc["comma_subordinate", "candidate_recall"] == 1.0
    assert gap.loc["comma_subordinate", "gap_candidate_recall"] == 1.0
