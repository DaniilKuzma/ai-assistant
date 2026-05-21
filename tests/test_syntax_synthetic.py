from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.rules.syntax_synthetic import (
    EXAMPLE_COLUMNS,
    HARD_NEGATIVE_COLUMNS,
    SUPPORTED_SYNTAX_RULE_IDS,
    build_syntax_candidate_recall,
    build_syntax_eval_examples,
    build_syntax_hard_negative_report,
    build_syntax_hard_negatives,
    write_syntax_eval_support_reports,
)


def test_syntax_eval_examples_cover_supported_rules_without_meta_language(tmp_path: Path):
    examples = build_syntax_eval_examples(
        min_examples_per_rule=50,
        clean_pool_path=tmp_path / "missing_clean_pool.csv.gz",
    )

    assert list(examples.columns) == EXAMPLE_COLUMNS
    positives = examples[~examples["hard_negative"].astype(bool)]
    counts = positives.groupby("rule_id").size().to_dict()

    assert set(SUPPORTED_SYNTAX_RULE_IDS) <= set(counts)
    assert all(counts[rule_id] >= 50 for rule_id in SUPPORTED_SYNTAX_RULE_IDS)
    assert positives["candidate_present"].astype(bool).all()
    for row in positives.to_dict("records"):
        assert row["rule_id"] in str(row["candidate_rule_ids"])
        assert "правило" not in row["source"].lower()
        assert "семейство" not in row["source"].lower()
        assert "серия" not in row["source"].lower()

    duplicate_keys = ["source", "target", "rule_id", "hard_negative"]
    assert not examples.duplicated(duplicate_keys).any()


def test_syntax_hard_negatives_are_mixed_and_have_no_accepted_bad_edits(tmp_path: Path):
    hard_negatives = build_syntax_hard_negatives(
        min_examples_per_family=50,
        clean_pool_path=tmp_path / "missing_clean_pool.csv.gz",
    )

    assert list(hard_negatives.columns) == HARD_NEGATIVE_COLUMNS
    assert {"trap_candidate", "guard_only"} <= set(hard_negatives["hard_negative_kind"])
    assert (hard_negatives["source"] == hard_negatives["target"]).all()
    assert hard_negatives["expected_accept"].eq(False).all()
    assert hard_negatives["expected_accepted_edits"].eq(0).all()
    assert hard_negatives[hard_negatives["hard_negative_kind"] == "trap_candidate"]["candidate_present"].astype(bool).all()
    assert not hard_negatives[hard_negatives["hard_negative_kind"] == "guard_only"]["candidate_present"].astype(bool).any()
    assert not hard_negatives.duplicated(["source", "target", "rule_id", "hard_negative_kind"]).any()

    report = build_syntax_hard_negative_report(hard_negatives)
    assert int(report["unsafe_candidate_accepted_count"].sum()) == 0
    assert float(report["hard_negative_overcorrection_rate"].max()) == 0.0


def test_write_syntax_eval_support_reports_creates_fixed_artifacts(tmp_path: Path):
    paths = write_syntax_eval_support_reports(
        reports_dir=tmp_path,
        min_examples_per_rule=2,
        min_hard_negatives_per_family=2,
        clean_pool_path=tmp_path / "missing_clean_pool.csv.gz",
        update_rules_yaml=False,
    )

    expected = {
        "syntax_eval_examples",
        "syntax_hard_negatives",
        "syntax_candidate_recall_by_rule",
        "syntax_hard_negative_report",
        "syntax_rules_dataset_readiness",
    }
    assert expected <= set(paths)
    assert (tmp_path / "syntax_eval_examples.csv.gz").exists()
    assert (tmp_path / "syntax_hard_negatives.csv.gz").exists()
    assert (tmp_path / "syntax_candidate_recall_by_rule.csv").exists()
    assert (tmp_path / "syntax_hard_negative_report.csv").exists()
    assert (tmp_path / "syntax_rules_dataset_readiness.md").exists()

    examples = pd.read_csv(tmp_path / "syntax_eval_examples.csv.gz")
    recall = build_syntax_candidate_recall(examples)
    assert not recall.empty
    assert recall["candidate_recall"].min() >= 0.85
