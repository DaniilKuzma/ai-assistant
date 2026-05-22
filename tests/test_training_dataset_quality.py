import json
from pathlib import Path

import pandas as pd

from src.data.training_dataset import training_dataset_quality_errors
from src.data.training_quality_audit import audit_training_dataset, known_quality_bug_counts


METKA = "\u043c\u0435\u0442\u043a\u0430"
LATER_EDITOR_CHECKED_RECORD = (
    "\u043f\u043e\u0437\u0436\u0435 "
    "\u0440\u0435\u0434\u0430\u043a\u0442\u043e\u0440 "
    "\u043f\u0440\u043e\u0432\u0435\u0440\u0438\u043b "
    "\u0437\u0430\u043f\u0438\u0441\u044c"
)


def test_training_dataset_quality_gate_accepts_ready_manifest_contract():
    manifest = {
        "total": 250000,
        "split_sizes": {"train": 200000, "val": 25000, "test": 25000},
        "composition": {
            "synthetic_augmented_from_open_clean": 160000,
            "real_error_pair": 1236,
            "clean_identity_from_open_clean": 25000,
            "hard_negative_from_open_clean": 25000,
            "multi_error_stress": 10000,
        },
        "active_rule_ids": ["comma_subordinate", "ne_verb"],
        "rule_id_counts": {"comma_subordinate": 2500, "ne_verb": 2500},
        "candidate_recall_summary": {"active_min_excluding_unknown": 1.0},
        "gap_label_coverage_summary": {"active_min_excluding_unknown": 1.0},
        "template_leakage_summary": {"val_overlap_with_train_rate": 0.0, "test_overlap_with_train_rate": 0.0},
        "synthetic_normalized_pair_duplicate_rate": 0.01,
        "top_normalized_pair_count": 2,
        "meta_language_counts": {"правило": 0},
        "suspicious_template_counts": {"проверяет семейство": 0},
        "underfilled_rule_ids": [],
        "source_counts": {"lenta_news": 150000, "nerus_news": 80000, "opencorpora": 20000},
        "hard_negative_accepted_bad_edits": 0,
        "corpus_opportunity_share": 0.80,
        "fallback_template_share": 0.10,
        "known_quality_bugs": {"synthetic_positive_identity": 0},
        "artificial_marker_counts": {"metka": 0, "later_editor_checked_record": 0, "random_filler_tokens": 0},
        "error_bearing_sentence_source_counts": {"corpus": 140000, "fallback_template": 20000},
        "exact_clean_hard_duplicate_count": 0,
        "rule_diversity_summary": {"failed_rule_count": 0},
        "extended_quality_audit_summary": {"blocking_issue_count": 0},
    }

    assert training_dataset_quality_errors(manifest) == []


def test_training_dataset_quality_gate_rejects_underfilled_and_duplicate_manifest():
    manifest = {
        "total": 199999,
        "split_sizes": {"train": 160000, "val": 20000, "test": 19999},
        "composition": {
            "synthetic_augmented_from_open_clean": 120000,
            "real_error_pair": 0,
            "clean_identity_from_open_clean": 10000,
            "hard_negative_from_open_clean": 10000,
            "multi_error_stress": 2000,
        },
        "active_rule_ids": ["comma_subordinate"],
        "rule_id_counts": {"comma_subordinate": 999},
        "candidate_recall_summary": {"active_min_excluding_unknown": 0.8},
        "gap_label_coverage_summary": {"active_min_excluding_unknown": 0.8},
        "template_leakage_summary": {"val_overlap_with_train_rate": 0.04, "test_overlap_with_train_rate": 0.0},
        "synthetic_normalized_pair_duplicate_rate": 0.26,
        "top_normalized_pair_count": 21,
        "meta_language_counts": {"правило": 1},
        "suspicious_template_counts": {"проверяет семейство": 1},
        "underfilled_rule_ids": ["comma_subordinate"],
        "source_counts": {"lenta_news": 199999},
        "hard_negative_accepted_bad_edits": 1,
        "corpus_opportunity_share": 0.69,
        "fallback_template_share": 0.21,
        "known_quality_bugs": {"synthetic_positive_identity": 1},
        "artificial_marker_counts": {"metka": 2, "later_editor_checked_record": 1, "random_filler_tokens": 1},
        "error_bearing_sentence_source_counts": {},
        "exact_clean_hard_duplicate_count": 12,
        "rule_diversity_summary": {"failed_rule_count": 1},
        "extended_quality_audit_summary": {"blocking_issue_count": 1},
    }

    errors = training_dataset_quality_errors(manifest)

    assert "total_below_200000" in errors
    assert "missing_real_pairs" in errors
    assert "active_rule_under_min:comma_subordinate" in errors
    assert "candidate_recall_active_min_below_threshold" in errors
    assert "synthetic_normalized_duplicate_rate_above_threshold" in errors
    assert "unsafe_source_dominance:lenta_news" in errors
    assert "corpus_opportunity_share_below_threshold" in errors
    assert "fallback_template_share_above_threshold" in errors
    assert "known_quality_bugs_present:synthetic_positive_identity" in errors
    assert "artificial_marker_present:metka" in errors
    assert "artificial_marker_present:later_editor_checked_record" in errors
    assert "artificial_marker_present:random_filler_tokens" in errors
    assert "rule_diversity_gates_failed" in errors
    assert "extended_quality_audit_blocking_issues" in errors


def test_quality_audit_counts_artificial_marker_suffixes_as_blocking():
    frame = pd.DataFrame(
        [
            {
                "source": f"\u041e\u0442\u0447\u0435\u0442 \u0433\u043e\u0442\u043e\u0432. {LATER_EDITOR_CHECKED_RECORD} \u0430\u0431.",
                "target": f"\u041e\u0442\u0447\u0435\u0442 \u0433\u043e\u0442\u043e\u0432. {LATER_EDITOR_CHECKED_RECORD} \u0430\u0431.",
                "source_type": "synthetic_augmented_from_open_clean",
                "rule_ids": '["final_punctuation_default"]',
                "metadata": (
                    '{"generation_strategy":"corpus_opportunity",'
                    '"candidate_present":true,'
                    '"error_bearing_sentence_source":"corpus"}'
                ),
                "template_id": "x",
                "normalized_pair_hash": "x",
            },
            {
                "source": f"\u0421\u043b\u043e\u0432\u043e {METKA} \u0432 \u0442\u0435\u043a\u0441\u0442\u0435.",
                "target": f"\u0421\u043b\u043e\u0432\u043e {METKA} \u0432 \u0442\u0435\u043a\u0441\u0442\u0435.",
                "source_type": "clean_identity_from_open_clean",
                "rule_ids": '["clean_identity"]',
                "metadata": "{}",
                "template_id": "y",
                "normalized_pair_hash": "y",
            },
        ]
    )

    known = known_quality_bug_counts(frame)
    audit = audit_training_dataset(frame, ["final_punctuation_default"])

    assert known["artificial_marker_metka"] == 1
    assert known["artificial_marker_later_editor_checked_record"] == 1
    assert known["artificial_marker_random_filler_tokens"] == 1
    assert audit["artificial_marker_counts"] == {
        "metka": 1,
        "later_editor_checked_record": 1,
        "random_filler_tokens": 1,
    }
    assert audit["extended_quality_audit_summary"]["blocking_issue_count"] >= 1
