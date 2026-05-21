import json
from pathlib import Path

import pandas as pd

from src.data.matrix_eval_dataset import MATRIX_EVAL_COLUMNS
from src.evaluation.matrix_eval_reports import (
    REQUIRED_MATRIX_EVAL_OUTPUTS,
    assert_matrix_eval_audit,
    write_expansion_backlog,
    write_hard_negative_matrix_coverage,
    write_matrix_rule_eval_summary,
    write_next_dataset_activation_plan,
)


def test_matrix_rule_eval_summary_classifies_ready_and_validator_rules(tmp_path: Path):
    dataset_path = _write_dataset(tmp_path)
    inventory_path = _write_inventory(tmp_path)
    eval_dir = tmp_path / "working_v1_eval"
    eval_dir.mkdir()
    pd.DataFrame(
        [
            {
                "rule_id": "final_punctuation_default",
                "group": "sentence_final_default_dot",
                "gold_count": 2,
                "predicted_count": 2,
                "true_positive": 2,
                "false_positive": 0,
                "false_negative": 0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
            },
            {
                "rule_id": "dictionary_fuzzy",
                "group": "typos_letter_operations",
                "gold_count": 1,
                "predicted_count": 2,
                "true_positive": 1,
                "false_positive": 1,
                "false_negative": 0,
                "precision": 0.5,
                "recall": 1.0,
                "f1": 0.666,
            },
        ]
    ).to_csv(eval_dir / "rule_precision_recall.csv", index=False)
    pd.DataFrame(
        [
            {
                "rule_id": "final_punctuation_default",
                "group": "sentence_final_default_dot",
                "gold_count": 2,
                "candidate_present_count": 2,
                "candidate_recall": 1.0,
                "missing_count": 0,
                "missing_examples": "[]",
            },
            {
                "rule_id": "dictionary_fuzzy",
                "group": "typos_letter_operations",
                "gold_count": 1,
                "candidate_present_count": 1,
                "candidate_recall": 1.0,
                "missing_count": 0,
                "missing_examples": "[]",
            },
        ]
    ).to_csv(eval_dir / "candidate_recall_by_rule.csv", index=False)
    pd.DataFrame(
        [
            {
                "rule_id": "final_punctuation_default",
                "group": "sentence_final_default_dot",
                "gold_gap_count": 2,
                "candidate_gap_present_count": 2,
                "gap_candidate_recall": 1.0,
                "missing_examples": "[]",
            }
        ]
    ).to_csv(eval_dir / "gap_label_coverage_by_rule.csv", index=False)
    pd.DataFrame(
        [
            {"row_id": 0, "source": "", "replacement": ".", "edit_type": "final_punctuation", "rule_id": "final_punctuation_default", "status": "accepted", "reason": "trusted", "confidence": 0.99},
            {"row_id": 3, "source": "карова", "replacement": "корова", "edit_type": "spelling_replace", "rule_id": "dictionary_fuzzy", "status": "accepted", "reason": "trusted", "confidence": 0.99},
        ]
    ).to_csv(eval_dir / "accepted_edits.csv", index=False)
    pd.DataFrame(columns=["row_id", "source", "replacement", "edit_type", "rule_id", "status", "reason", "confidence"]).to_csv(eval_dir / "rejected_edits.csv", index=False)

    summary = write_matrix_rule_eval_summary(
        dataset_path=dataset_path,
        inventory_path=inventory_path,
        eval_dir=eval_dir,
        output_path=eval_dir / "matrix_rule_eval_summary.csv",
    )

    by_rule = summary.set_index("rule_id")
    assert by_rule.loc["final_punctuation_default", "decision"] == "READY_NEXT_DATASET"
    assert by_rule.loc["dictionary_fuzzy", "decision"] == "NEEDS_VALIDATOR"


def test_backlog_activation_plan_and_audit_outputs(tmp_path: Path):
    dataset_path = _write_dataset(tmp_path)
    inventory_path = _write_inventory(tmp_path)
    eval_dir = tmp_path / "working_v1_eval"
    eval_dir.mkdir()
    summary_path = eval_dir / "matrix_rule_eval_summary.csv"
    pd.DataFrame(
        [
            {"rule_id": "final_punctuation_default", "matrix_group": "sentence_final_default_dot", "eval_examples": 2, "candidate_recall": 1.0, "gap_coverage": 1.0, "predicted_count": 2, "true_positive": 2, "false_positive": 0, "false_negative": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "clean_overcorrection_count": 0, "dirty_worse_count": 0, "accepted_count": 2, "rejected_count": 0, "top_rejection_reasons": "", "decision": "READY_NEXT_DATASET", "reason": "passes gates"}
        ]
    ).to_csv(summary_path, index=False)

    backlog_path = tmp_path / "rule_expansion_backlog.csv"
    plan_path = tmp_path / "next_dataset_activation_plan.md"
    hard_negative_path = tmp_path / "hard_negative_matrix_coverage.csv"
    write_expansion_backlog(inventory_path=inventory_path, summary_path=summary_path, output_path=backlog_path)
    write_next_dataset_activation_plan(backlog_path=backlog_path, summary_path=summary_path, output_path=plan_path)
    write_hard_negative_matrix_coverage(dataset_path=dataset_path, output_path=hard_negative_path)

    for name in REQUIRED_MATRIX_EVAL_OUTPUTS:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text("placeholder\n", encoding="utf-8")
    (tmp_path / "working_v1_eval").mkdir(exist_ok=True)
    for name in ["evaluation_summary.csv", "rule_precision_recall.csv", "error_by_rule.csv", "accepted_edits.csv", "rejected_edits.csv", "clean_overcorrection_examples.csv", "dirty_worse_examples.csv", "matrix_rule_eval_summary.csv"]:
        path = tmp_path / "working_v1_eval" / name
        if not path.exists():
            pd.DataFrame([{"ok": 1}]).to_csv(path, index=False)
    pd.DataFrame([{"decision": "EVALUATE_NOW"}]).to_csv(tmp_path / "rule_matrix_inventory.csv", index=False)
    pd.DataFrame([{"source_type": "matrix_synthetic_eval", "candidate_present": True}]).to_csv(tmp_path / "matrix_eval.csv.gz", index=False)
    (tmp_path / "matrix_eval_manifest.json").write_text(json.dumps({"verdict": "MATRIX_EVAL_DATASET_READY"}), encoding="utf-8")

    audit = assert_matrix_eval_audit(root_dir=tmp_path)
    assert audit["verdict"] in {"MATRIX_EVAL_COMPLETE", "MATRIX_EVAL_PARTIAL"}


def _write_dataset(tmp_path: Path) -> Path:
    rows = [
        _row("Я готов", "Я готов.", "final_punctuation_default", "matrix_synthetic_eval", True),
        _row("Отчет готов", "Отчет готов.", "final_punctuation_default", "matrix_synthetic_eval", True),
        _row("Чистый текст.", "Чистый текст.", "clean_identity_hard_negative", "matrix_hard_negative", False, is_clean=True),
        _row("Карова пришла.", "Корова пришла.", "dictionary_fuzzy", "matrix_synthetic_eval", True),
    ]
    path = tmp_path / "matrix_eval.csv.gz"
    pd.DataFrame(rows, columns=MATRIX_EVAL_COLUMNS).to_csv(path, index=False)
    return path


def _row(source: str, target: str, rule_id: str, source_type: str, candidate_present: bool, is_clean: bool = False) -> dict:
    edit_operations = [] if is_clean else [{"source": "", "replacement": ".", "edit_type": "final_punctuation", "start": len(source), "end": len(source), "rule_id": rule_id}]
    return {
        "source": source,
        "target": target,
        "rule_id": rule_id,
        "rule_ids": json.dumps([rule_id], ensure_ascii=False),
        "error_type": "clean_identity" if is_clean else "final_punctuation",
        "error_types": json.dumps([] if is_clean else ["final_punctuation"], ensure_ascii=False),
        "matrix_group": "sentence_final_default_dot",
        "source_type": source_type,
        "candidate_present": candidate_present,
        "candidate_rule_ids": json.dumps([rule_id] if candidate_present else [], ensure_ascii=False),
        "gap_label_present": candidate_present,
        "split": "matrix_eval",
        "metadata": json.dumps({"candidate_present": candidate_present}, ensure_ascii=False),
        "template_id": "",
        "normalized_pair_hash": source,
        "edit_operations": json.dumps(edit_operations, ensure_ascii=False),
        "edits": json.dumps(edit_operations, ensure_ascii=False),
        "source_dataset": source_type,
        "is_clean": is_clean,
        "is_synthetic": not is_clean,
        "is_hard_negative": is_clean,
        "domain": "matrix_eval",
    }


def _write_inventory(tmp_path: Path) -> Path:
    path = tmp_path / "rule_matrix_inventory.csv"
    pd.DataFrame(
        [
            {"matrix_id": "sentence_final_default_dot", "group": "sentence_final_default_dot", "listed_rule_ids": json.dumps(["final_punctuation_default"]), "decision": "EVALUATE_NOW", "decision_reason": "candidate-backed", "candidate_generator_support": True},
            {"matrix_id": "typos_letter_operations", "group": "typos_letter_operations", "listed_rule_ids": json.dumps(["dictionary_fuzzy"]), "decision": "EVALUATE_NOW", "decision_reason": "candidate-backed", "candidate_generator_support": True},
        ]
    ).to_csv(path, index=False)
    return path
