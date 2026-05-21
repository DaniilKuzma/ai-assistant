from pathlib import Path

import pandas as pd

from src.evaluation.wave1_expansion import normalize_phase3_decision, write_phase3_final_report


def test_phase3_decision_normalization_uses_required_labels():
    assert normalize_phase3_decision("READY_NEXT_DATASET", blockers=()) == "READY_NEXT_DATASET"
    assert normalize_phase3_decision("NEEDS_THRESHOLD", blockers=()) == "NEEDS_THRESHOLD"
    assert normalize_phase3_decision("NEEDS_MORE_TRAINING", blockers=()) == "NEEDS_MORE_TRAINING"
    assert normalize_phase3_decision("NEEDS_RULE_IMPLEMENTATION", blockers=()) == "NEEDS_CANDIDATE_GENERATOR"
    assert normalize_phase3_decision("KEEP_INACTIVE", blockers=("NEEDS_SYNTAX_FEATURES",)) == "NEEDS_SYNTAX"
    assert normalize_phase3_decision("KEEP_INACTIVE", blockers=("NEEDS_DICTIONARY_SUPPORT",)) == "NEEDS_DICTIONARY"
    assert normalize_phase3_decision("KEEP_INACTIVE", blockers=("NEEDS_NER_SUPPORT",)) == "NEEDS_NER"
    assert normalize_phase3_decision("UNSAFE", blockers=()) == "UNSAFE"


def test_phase3_final_report_writes_complete_verdict_for_full_artifacts(tmp_path: Path):
    reports = tmp_path / "reports"
    data = tmp_path / "data"
    reports.mkdir()
    data.mkdir()
    pd.DataFrame(
        [
            {"rule_id": "address_comma", "candidate_recall": 1.0, "phase3_decision": "READY_NEXT_DATASET"},
            {"rule_id": "quote_pair_balance", "candidate_recall": 1.0, "phase3_decision": "NEEDS_THRESHOLD"},
        ]
    ).to_csv(reports / "matrix_rule_eval_summary.csv", index=False)
    pd.DataFrame([{"source": "a", "target": "b"}]).to_csv(data / "matrix_eval.csv.gz", index=False)
    (reports / "matrix_eval_manifest.json").write_text('{"model_training_ran": false}', encoding="utf-8")

    verdict = write_phase3_final_report(reports_dir=reports, data_dir=data)

    assert verdict == "MATRIX_EVAL_PHASE3_COMPLETE"
    assert "MATRIX_EVAL_PHASE3_COMPLETE" in (reports / "phase3_final_report.md").read_text(encoding="utf-8")
