from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.rules.syntax_synthetic import SUPPORTED_SYNTAX_RULE_IDS


REPORT_DIR = Path("reports/syntax_module")
SUMMARY_PATH = REPORT_DIR / "final_syntax_capability_summary.md"
CAPABILITY_PATH = REPORT_DIR / "final_syntax_capability.csv"
ELIGIBLE_PATH = REPORT_DIR / "syntax_training_eligible_rules.csv"
BLOCKED_PATH = REPORT_DIR / "syntax_blocked_rules.csv"
RULES_PATH = Path("configs/rules.yaml")
REQUIRED_COLUMNS = {
    "matrix_key",
    "title",
    "rule_id",
    "syntax_family",
    "candidate_path",
    "synthetic_support",
    "hard_negative_support",
    "candidate_recall",
    "training_eligible_now",
    "decision",
    "reason",
}


def test_final_syntax_reports_exist_and_have_required_columns():
    assert SUMMARY_PATH.exists()
    assert CAPABILITY_PATH.exists()
    assert ELIGIBLE_PATH.exists()
    assert BLOCKED_PATH.exists()

    for path in (CAPABILITY_PATH, ELIGIBLE_PATH, BLOCKED_PATH):
        frame = pd.read_csv(path)
        assert REQUIRED_COLUMNS <= set(frame.columns), path


def test_final_syntax_eligible_rows_have_all_required_support():
    capability = pd.read_csv(CAPABILITY_PATH)
    eligible = capability[capability["training_eligible_now"].astype(bool)]

    assert not eligible.empty
    assert eligible["candidate_path"].astype(bool).all()
    assert eligible["synthetic_support"].astype(bool).all()
    assert eligible["hard_negative_support"].astype(bool).all()
    assert eligible["candidate_recall"].astype(float).ge(0.85).all()


def test_final_syntax_summary_reports_both_coverage_denominators():
    text = SUMMARY_PATH.read_text(encoding="utf-8")

    assert "syntax_surface_total:" in text
    assert "syntax_surface_covered:" in text
    assert "syntax_surface_coverage_percent:" in text
    assert "current_syntax_required_total:" in text
    assert "current_syntax_required_coverage_percent:" in text
    assert "NER-required entries excluded from syntax coverage denominator" in text
    assert "Verdict: SYNTAX_CAPABILITY_READY_FOR_DATASET" in text


def test_final_syntax_reports_and_rules_yaml_agree_on_supported_eligible_rule_ids():
    config = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))
    expected_rule_ids = {
        rule_id
        for section in ("orthography", "punctuation")
        for entry in config[section].values()
        if entry["dataset"].get("training_eligible_now")
        for rule_id in entry["implementation"].get("rule_ids", [])
        if rule_id in SUPPORTED_SYNTAX_RULE_IDS
    }

    eligible = pd.read_csv(ELIGIBLE_PATH)
    actual_rule_ids = {str(rule_id) for rule_id in eligible["rule_id"].dropna() if str(rule_id)}

    assert expected_rule_ids
    assert expected_rule_ids <= actual_rule_ids


def test_final_syntax_blocked_rules_use_allowed_reasons():
    blocked = pd.read_csv(BLOCKED_PATH)
    allowed_reasons = {
        "no_candidate_path",
        "needs_semantic_model",
        "needs_dictionary",
        "needs_NER",
        "needs_decoder_constraint",
        "unsafe",
    }

    assert not blocked.empty
    assert set(blocked["reason"].dropna()) <= allowed_reasons
