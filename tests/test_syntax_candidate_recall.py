from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.evaluation.candidate_recall import build_candidate_recall_reports
from src.rules.syntax_synthetic import (
    SUPPORTED_SYNTAX_RULE_IDS,
    build_syntax_candidate_recall,
    build_syntax_eval_examples,
    build_syntax_hard_negatives,
)


def test_training_eligible_syntax_rules_have_candidate_backed_eval_examples(tmp_path: Path):
    config = yaml.safe_load(Path("configs/rules.yaml").read_text(encoding="utf-8"))
    examples = build_syntax_eval_examples(
        min_examples_per_rule=50,
        clean_pool_path=tmp_path / "missing_clean_pool.csv.gz",
    )
    recall = build_syntax_candidate_recall(examples).set_index("rule_id")
    counts = examples[~examples["hard_negative"].astype(bool)].groupby("rule_id").size().to_dict()

    eligible_rule_ids = {
        rule_id
        for section in ("orthography", "punctuation")
        for entry in config[section].values()
        if entry["dataset"].get("training_eligible_now")
        for rule_id in entry["implementation"].get("rule_ids", [])
        if rule_id in SUPPORTED_SYNTAX_RULE_IDS
    }

    assert eligible_rule_ids
    assert eligible_rule_ids <= set(counts)
    assert all(counts[rule_id] >= 50 for rule_id in eligible_rule_ids)
    assert all(float(recall.loc[rule_id, "candidate_recall"]) >= 0.85 for rule_id in eligible_rule_ids)


def test_hard_negatives_do_not_enter_positive_candidate_recall_denominator(tmp_path: Path):
    positives = build_syntax_eval_examples(
        selected_rule_ids=["comma_subordinate"],
        min_examples_per_rule=3,
        clean_pool_path=tmp_path / "missing_clean_pool.csv.gz",
    )
    hard_negatives = build_syntax_hard_negatives(
        selected_families=["subordinate_clause_comma"],
        min_examples_per_family=3,
        clean_pool_path=tmp_path / "missing_clean_pool.csv.gz",
    )

    combined = pd.concat([positives, hard_negatives], ignore_index=True)
    recall = build_syntax_candidate_recall(combined).set_index("rule_id")

    assert int(recall.loc["comma_subordinate", "gold_count"]) == 3
    assert int(recall.loc["comma_subordinate", "candidate_present_count"]) == 3
    assert float(recall.loc["comma_subordinate", "candidate_recall"]) == 1.0


def test_candidate_recall_can_recompute_syntax_rows_without_trusting_metadata(tmp_path: Path):
    rows = [
        {
            "source": "Документ готов.",
            "target": "Документ, готов.",
            "edit_operations": json.dumps(
                [
                    {
                        "source": "",
                        "replacement": ",",
                        "edit_type": "punctuation_insert",
                        "start": len("Документ"),
                        "end": len("Документ"),
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

    reports = build_candidate_recall_reports(
        rows,
        rules_config_path=_rules_config(tmp_path),
        trust_candidate_backed_metadata=False,
    )

    summary = reports["candidate_recall_by_rule"].set_index("rule_id")
    assert int(summary.loc["comma_subordinate", "candidate_present_count"]) == 0
    assert float(summary.loc["comma_subordinate", "candidate_recall"]) == 0.0


def _rules_config(tmp_path: Path) -> Path:
    path = tmp_path / "rules.yaml"
    path.write_text(
        """
punctuation:
  complex_sentence_subordinate:
    rules: [comma_subordinate]
orthography: {}
""",
        encoding="utf-8",
    )
    return path
