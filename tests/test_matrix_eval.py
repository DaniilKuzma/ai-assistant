import json
from pathlib import Path

import pandas as pd

from src.data.matrix_eval_dataset import MATRIX_EVAL_COLUMNS
from src.evaluation.matrix_eval import (
    BACKLOG_CORE_COLUMNS,
    UNDER_QUOTA_RULE_AUDIT_COLUMNS,
    ACTIVATION_ACTIVATION_COLUMNS,
    assert_matrix_eval_audit,
    write_rule_expansion_backlog,
    write_under_quota_rule_audit,
    write_next_dataset_activation_plan,
)


def test_under_quota_audit_classifies_backfilled_and_no_candidate_path(tmp_path: Path):
    baseline_manifest = tmp_path / "baseline_manifest.json"
    baseline_manifest.write_text(
        json.dumps(
            {
                "underfilled_rule_ids": ["pattern_чо_че", "yo_e_candidate"],
                "no_candidate_path_rule_ids": ["yo_e_candidate"],
                "rule_id_counts": {"pattern_чо_че": 22},
                "quota": {"min_eval_examples_per_executable_rule": 50},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    inventory = tmp_path / "inventory.csv"
    pd.DataFrame(
        [
            _inventory_row("pattern_чо_че", "letter_basic_hissing_vowel_patterns", candidate=True, synthetic=True),
            _inventory_row("yo_e_candidate", "letter_vowels_not_after_hissing_c", candidate=False, synthetic=False),
        ]
    ).to_csv(inventory, index=False)
    summary = tmp_path / "summary.csv"
    pd.DataFrame(
        [
            _summary_row("pattern_чо_че", "letter_basic_hissing_vowel_patterns", 50, 1.0),
            _summary_row("yo_e_candidate", "letter_vowels_not_after_hissing_c", 0, 0.0),
        ]
    ).to_csv(summary, index=False)
    dataset = tmp_path / "matrix_eval.csv.gz"
    pd.DataFrame(
        [_matrix_row(f"Чорный пример {index}.", f"Черный пример {index}.", "pattern_чо_че") for index in range(50)],
        columns=MATRIX_EVAL_COLUMNS,
    ).to_csv(dataset, index=False)

    audit = write_under_quota_rule_audit(
        baseline_manifest_path=baseline_manifest,
        inventory_path=inventory,
        matrix_summary_path=summary,
        matrix_dataset_path=dataset,
        output_path=tmp_path / "under_quota_rule_audit.csv",
    )

    assert list(audit.columns) == UNDER_QUOTA_RULE_AUDIT_COLUMNS
    by_rule = audit.set_index("rule_id")
    assert by_rule.loc["pattern_чо_че", "decision"] == "BACKFILLED"
    assert by_rule.loc["pattern_чо_че", "backfill_success_count"] == 28
    assert by_rule.loc["yo_e_candidate", "decision"] == "NO_CANDIDATE_PATH"


def test_matrix_activation_plan_and_backlog_apply_safety_gates(tmp_path: Path):
    summary_path = tmp_path / "summary.csv"
    pd.DataFrame(
        [
            _summary_row("address_comma", "addresses", 100, 1.0, precision=1.0, recall=1.0, decision="READY_NEXT_DATASET"),
            _summary_row("pattern_чо_че", "letters", 50, 1.0, precision=0.0, recall=0.0, decision="NEEDS_THRESHOLD"),
            _summary_row("yo_e_candidate", "yo_e", 0, 0.0, decision="KEEP_INACTIVE"),
        ]
    ).to_csv(summary_path, index=False)
    under_quota_path = tmp_path / "under_quota.csv"
    pd.DataFrame(
        [
            {
                "rule_id": "pattern_чо_че",
                "matrix_group": "letters",
                "current_eval_count": 22,
                "target_eval_count": 50,
                "candidate_path_exists": True,
                "synthetic_generator_exists": True,
                "hard_negative_exists": False,
                "can_backfill_with_existing_generator": True,
                "backfill_attempted": True,
                "backfill_success_count": 28,
                "candidate_recall_after_backfill": 1.0,
                "decision": "BACKFILLED",
                "reason": "quota met",
            }
        ],
        columns=UNDER_QUOTA_RULE_AUDIT_COLUMNS,
    ).to_csv(under_quota_path, index=False)
    backlog = tmp_path / "backlog.csv"
    pd.DataFrame(
        [
            {"priority": 4, "rule_id": "yo_e_candidate", "matrix_id": "yo_e", "group": "yo_e", "status": "KEEP_INACTIVE", "blocker": "NEEDS_CANDIDATE_GENERATOR", "required_work": "Implement bounded candidate support.", "estimated_risk": "medium", "recommended_next_prompt": "Implement bounded candidate support for yo_e_candidate.", "examples_needed": "candidate-backed positives", "notes": ""},
            {"priority": 3, "rule_id": "pattern_чо_че", "matrix_id": "letters", "group": "letters", "status": "NEEDS_THRESHOLD", "blocker": "NEEDS_THRESHOLD_CALIBRATION", "required_work": "Calibrate thresholds on validation data.", "estimated_risk": "medium", "recommended_next_prompt": "Calibrate pattern_чо_че thresholds without test tuning.", "examples_needed": "validation positives", "notes": ""},
        ],
        columns=BACKLOG_CORE_COLUMNS,
    ).to_csv(backlog, index=False)

    activation = write_next_dataset_activation_plan(
        summary_path=summary_path,
        under_quota_audit_path=under_quota_path,
        backlog_core_path=backlog,
        output_csv_path=tmp_path / "activation_activation_plan.csv",
        output_md_path=tmp_path / "activation_activation_plan.md",
    )

    assert list(activation.columns) == ACTIVATION_ACTIVATION_COLUMNS
    by_rule = activation.set_index("rule_id")
    assert by_rule.loc["address_comma", "activation_decision"] == "INCLUDE"
    assert by_rule.loc["pattern_чо_че", "activation_decision"] == "EXCLUDE"
    assert "NEEDS_THRESHOLD_CALIBRATION" in by_rule.loc["pattern_чо_че", "blockers"]


def test_matrix_backlog_uses_required_blocker_taxonomy(tmp_path: Path):
    inventory_path = tmp_path / "inventory.csv"
    pd.DataFrame(
        [
            _inventory_row("capitalization_ner", "capitalization", requires=["ner", "syntax", "dictionary", "model"], candidate=False),
            _inventory_row("hyphen_po_adverbs", "hyphen", candidate=True, synthetic=True),
        ]
    ).to_csv(inventory_path, index=False)
    summary_path = tmp_path / "summary.csv"
    pd.DataFrame(
        [
            _summary_row("capitalization_ner", "capitalization", 0, 0.0, decision="KEEP_INACTIVE"),
            _summary_row("hyphen_po_adverbs", "hyphen", 100, 1.0, precision=0.0, recall=0.0, false_positive=1, decision="NEEDS_VALIDATOR"),
        ]
    ).to_csv(summary_path, index=False)

    backlog = write_rule_expansion_backlog(
        inventory_path=inventory_path,
        summary_path=summary_path,
        output_csv_path=tmp_path / "rule_expansion_backlog_core.csv",
        output_md_path=tmp_path / "rule_expansion_backlog_core.md",
    )

    assert list(backlog.columns) == BACKLOG_CORE_COLUMNS
    blockers = set(backlog["blocker"])
    assert "NEEDS_NER_SUPPORT" in blockers
    assert "NEEDS_VALIDATOR" in blockers


def test_matrix_audit_rejects_positive_rows_without_candidates(tmp_path: Path):
    root = tmp_path / "reports"
    data = tmp_path / "data"
    eval_dir = root / "working_eval"
    root.mkdir()
    data.mkdir()
    eval_dir.mkdir()
    pd.DataFrame(
        [_matrix_row("Ошибка", "Ошибка", "address_comma", candidate_present=False)],
        columns=MATRIX_EVAL_COLUMNS,
    ).to_csv(data / "matrix_eval.csv.gz", index=False)
    (data / "matrix_eval_manifest.json").write_text(json.dumps({"verdict": "MATRIX_EVAL_DATASET_READY"}), encoding="utf-8")
    for name in ("under_quota_rule_audit.csv", "validator_fix_report.md", "activation_activation_plan.csv", "rule_expansion_backlog_core.csv"):
        path = root / name
        if path.suffix == ".csv":
            pd.DataFrame([{"ok": 1}]).to_csv(path, index=False)
        else:
            path.write_text("ok\n", encoding="utf-8")
    for name in (
        "evaluation_summary.csv",
        "rule_precision_recall.csv",
        "error_by_rule.csv",
        "accepted_edits.csv",
        "rejected_edits.csv",
        "matrix_rule_eval_summary.csv",
    ):
        pd.DataFrame([{"ok": 1}]).to_csv(eval_dir / name, index=False)

    audit = assert_matrix_eval_audit(root_dir=root, data_dir=data, eval_dir=eval_dir)

    assert audit["verdict"] == "MATRIX_EVAL_MATRIX_BLOCKED"
    assert "positive matrix eval rows without candidate_present=true" in audit["errors"]


def _inventory_row(rule_id: str, group: str, *, requires=None, candidate=True, synthetic=False):
    return {
        "matrix_id": group,
        "group": group,
        "requires": json.dumps(requires or [], ensure_ascii=False),
        "listed_rule_ids": json.dumps([rule_id], ensure_ascii=False),
        "candidate_generator_support": str(candidate).lower(),
        "synthetic_corruption_support": str(synthetic).lower(),
        "validator_guard_support": "unknown",
        "executable_status": "EXECUTABLE_ACTIVE" if candidate else "NO_CANDIDATE_PATH",
        "decision": "EVALUATE_NOW" if candidate else "BACKLOG_IMPLEMENTATION",
        "decision_reason": "",
    }


def _summary_row(
    rule_id: str,
    group: str,
    eval_examples: int,
    candidate_recall: float,
    *,
    precision=0.0,
    recall=0.0,
    false_positive=0,
    decision="KEEP_INACTIVE",
):
    return {
        "rule_id": rule_id,
        "matrix_group": group,
        "eval_examples": eval_examples,
        "candidate_recall": candidate_recall,
        "gap_coverage": 0.0,
        "predicted_count": 0,
        "true_positive": 0,
        "false_positive": false_positive,
        "false_negative": eval_examples,
        "precision": precision,
        "recall": recall,
        "f1": 0.0,
        "clean_overcorrection_count": 0,
        "dirty_worse_count": 0,
        "accepted_count": 0,
        "rejected_count": 0,
        "top_rejection_reasons": "",
        "decision": decision,
        "reason": "",
    }


def _matrix_row(source: str, target: str, rule_id: str, candidate_present=True):
    return {
        "source": source,
        "target": target,
        "rule_id": rule_id,
        "rule_ids": json.dumps([rule_id], ensure_ascii=False),
        "error_type": "spelling",
        "error_types": json.dumps(["spelling"], ensure_ascii=False),
        "matrix_group": "group",
        "source_type": "matrix_synthetic_eval",
        "candidate_present": candidate_present,
        "candidate_rule_ids": json.dumps([rule_id] if candidate_present else [], ensure_ascii=False),
        "gap_label_present": False,
        "split": "matrix_eval",
        "metadata": json.dumps({"candidate_present": candidate_present}, ensure_ascii=False),
        "template_id": "",
        "normalized_pair_hash": source,
        "edit_operations": json.dumps([], ensure_ascii=False),
        "edits": json.dumps([], ensure_ascii=False),
        "source_dataset": "test",
        "is_clean": False,
        "is_synthetic": True,
        "is_hard_negative": False,
        "domain": "matrix_eval",
    }
