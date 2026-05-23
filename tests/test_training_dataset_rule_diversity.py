import json
from pathlib import Path

import pandas as pd


MANIFEST_PATH = Path("data/processed/dataset_manifest.json")
DIVERSITY_REPORT_PATH = Path("reports/dataset_build/rule_diversity_report.csv")
GENERATION_STRATEGY_REPORT_PATH = Path("reports/dataset_build/generation_strategy_report.csv")
EXTENDED_QUALITY_AUDIT_PATH = Path("reports/dataset_build/extended_quality_audit.csv")


def _manifest() -> dict:
    assert MANIFEST_PATH.exists(), MANIFEST_PATH
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_manifest_corpus_first_strategy_gates_pass():
    manifest = _manifest()

    assert manifest["corpus_opportunity_share"] >= 0.70
    assert manifest["fallback_template_share"] <= 0.20
    assert manifest["error_bearing_sentence_source_counts"]["corpus"] > 0
    assert manifest["rule_diversity_summary"]["failed_rule_count"] == 0
    assert manifest["extended_quality_audit_summary"]["blocking_issue_count"] == 0


def test_generation_strategy_report_enforces_per_rule_fallback_share():
    assert GENERATION_STRATEGY_REPORT_PATH.exists(), GENERATION_STRATEGY_REPORT_PATH
    report = pd.read_csv(GENERATION_STRATEGY_REPORT_PATH)
    active = report[report["is_active_rule"].astype(bool)]

    assert not active.empty
    assert "corpus_error_bearing_count" in report.columns
    assert "fallback_error_bearing_count" in report.columns
    assert (active["fallback_template_share"] <= 0.25).all()
    assert (active["corpus_opportunity_share"] >= 0.70).all()


def test_rule_diversity_report_passes_required_gates():
    assert DIVERSITY_REPORT_PATH.exists(), DIVERSITY_REPORT_PATH
    report = pd.read_csv(DIVERSITY_REPORT_PATH)
    active = report[report["is_active_rule"].astype(bool)]

    assert not active.empty
    assert (active["count"] >= 1000).all()
    assert (active["unique_carrier_sentences"] >= active[["count"]].assign(limit=500).min(axis=1)).all()
    assert (active["top_template_share"] <= 0.10).all()
    assert (active["normalized_pair_duplicate_rate"] <= 0.15).all()
    assert active["passes_required_gates"].astype(bool).all()

    applicable = active[active["forms_applicable"].astype(bool)]
    assert (applicable["unique_error_forms"] >= 30).all()
    assert (applicable["unique_target_forms"] >= 30).all()
    assert (applicable["top_error_form_share"] <= 0.15).all()
    assert (applicable["top_target_form_share"] <= 0.15).all()


def test_manifest_verdict_matches_extended_quality_blockers():
    manifest = _manifest()
    assert EXTENDED_QUALITY_AUDIT_PATH.exists(), EXTENDED_QUALITY_AUDIT_PATH
    audit = pd.read_csv(EXTENDED_QUALITY_AUDIT_PATH)
    blocking_count = int(audit["severity"].astype(str).str.lower().eq("blocking").sum()) if not audit.empty else 0

    assert manifest["extended_quality_audit_summary"]["blocking_issue_count"] == blocking_count
    if blocking_count:
        assert manifest["verdict"] != "READY_FOR_TRAINING_DATASET"


def test_hyphen_particles_passes_diversity_or_is_excluded():
    manifest = _manifest()
    report = pd.read_csv(DIVERSITY_REPORT_PATH)
    active_ids = set(manifest["active_rule_ids"])

    if "hyphen_particles" in active_ids:
        row = report[report["rule_id"].astype(str).eq("hyphen_particles")]
        assert not row.empty
        assert bool(row.iloc[0]["passes_required_gates"])
    else:
        excluded = {
            str(row.get("rule_id")): str(row.get("reason"))
            for row in manifest.get("excluded_rule_ids", [])
            if isinstance(row, dict)
        }
        assert excluded.get("hyphen_particles") in {
            "diversity_failed",
            "insufficient_verified_examples_after_backfill",
            "insufficient_verified_examples",
        }
