import json
from pathlib import Path

import pandas as pd

from src.data.training_dataset import training_dataset_quality_errors
from src.data._training_dataset_builder import _hyphen_po_bad_positive
from src.data.training_quality_audit import (
    artificial_marker_counts,
    atomic_purity_audit_frame,
    audit_training_dataset,
    extra_edit_audit_frame,
    known_quality_bug_counts,
    mixed_script_clean_audit_frame,
    real_pair_atomization_audit_frame,
    report_manifest_errors,
    unknown_rule_train_audit_frame,
    write_report_manifest,
)
from src.data.operator_dataset_builder import _contains_artificial_marker


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
        "quote_bracket_balance_bugs": {
            "unbalanced_target_guillemets": 0,
            "unbalanced_target_ascii_quotes": 0,
            "unbalanced_target_parentheses": 0,
            "unbalanced_target_square_brackets": 0,
            "unbalanced_target_curly_brackets": 0,
        },
        "clean_hard_balance_bugs": {
            "unbalanced_guillemets": 0,
            "unbalanced_ascii_quotes": 0,
            "unbalanced_parentheses": 0,
            "unbalanced_square_brackets": 0,
            "unbalanced_curly_brackets": 0,
        },
        "rule_semantic_alignment": {"failed_rows": 0, "failed_by_rule": {}},
        "error_bearing_sentence_source_counts": {"corpus": 140000, "fallback_template": 20000},
        "exact_clean_hard_duplicate_count": 0,
        "rule_diversity_summary": {"failed_rule_count": 0},
        "extended_quality_audit_summary": {"blocking_issue_count": 0},
    }

    assert training_dataset_quality_errors(manifest) == []


def test_training_dataset_quality_gate_rejects_blocking_quality_manifest():
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
        "quote_bracket_balance_bugs": {
            "unbalanced_target_guillemets": 1,
            "unbalanced_target_ascii_quotes": 1,
            "unbalanced_target_parentheses": 1,
            "unbalanced_target_square_brackets": 1,
            "unbalanced_target_curly_brackets": 1,
        },
        "clean_hard_balance_bugs": {
            "unbalanced_guillemets": 1,
            "unbalanced_ascii_quotes": 1,
            "unbalanced_parentheses": 1,
            "unbalanced_square_brackets": 1,
            "unbalanced_curly_brackets": 1,
        },
        "rule_semantic_alignment": {
            "failed_rows": 2,
            "failed_by_rule": {"apposition_comma": 1, "clarification_comma": 1},
        },
        "error_bearing_sentence_source_counts": {},
        "exact_clean_hard_duplicate_count": 12,
        "rule_diversity_summary": {"failed_rule_count": 1},
        "extended_quality_audit_summary": {"blocking_issue_count": 1},
    }

    errors = training_dataset_quality_errors(manifest)

    assert "total_below_200000" in errors
    assert "missing_real_pairs" in errors
    assert "candidate_recall_active_min_below_threshold" in errors
    assert "synthetic_normalized_duplicate_rate_above_threshold" in errors
    assert "unsafe_source_dominance:lenta_news" in errors
    assert "corpus_opportunity_share_below_threshold" in errors
    assert "fallback_template_share_above_threshold" in errors
    assert "known_quality_bugs_present:synthetic_positive_identity" in errors
    assert "artificial_marker_present:metka" in errors
    assert "artificial_marker_present:later_editor_checked_record" in errors
    assert "artificial_marker_present:random_filler_tokens" in errors
    assert "quote_bracket_balance_present:unbalanced_target_guillemets" in errors
    assert "quote_bracket_balance_present:unbalanced_target_ascii_quotes" in errors
    assert "quote_bracket_balance_present:unbalanced_target_parentheses" in errors
    assert "quote_bracket_balance_present:unbalanced_target_square_brackets" in errors
    assert "quote_bracket_balance_present:unbalanced_target_curly_brackets" in errors
    assert "clean_hard_balance_present:unbalanced_guillemets" in errors
    assert "clean_hard_balance_present:unbalanced_ascii_quotes" in errors
    assert "clean_hard_balance_present:unbalanced_parentheses" in errors
    assert "clean_hard_balance_present:unbalanced_square_brackets" in errors
    assert "clean_hard_balance_present:unbalanced_curly_brackets" in errors
    assert "rule_semantic_alignment_failed" in errors
    assert "active_rule_quota_underfilled" in errors
    assert "active_rule_under_min:comma_subordinate" in errors
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


def test_artificial_marker_detection_uses_cyrillic_token_boundaries():
    assert _contains_artificial_marker("метка", "метка") is True
    assert _contains_artificial_marker("[МЕТКА]", "[МЕТКА]") is True
    assert _contains_artificial_marker("__метка__", "__метка__") is True
    assert _contains_artificial_marker("позже редактор проверил запись", "") is True
    assert _contains_artificial_marker("", "позже редактор проверил материал") is True
    assert _contains_artificial_marker("заметка", "заметка") is False
    assert _contains_artificial_marker("заметки редактора", "заметки редактора") is False
    assert _contains_artificial_marker("короткая заметка", "короткая заметка") is False


def test_artificial_marker_counts_do_not_count_zametka_as_metka():
    frame = pd.DataFrame(
        [
            {"source": "Короткая заметка готова.", "target": "Короткая заметка готова.", "source_type": "clean_identity_from_open_clean"},
            {"source": "Заметки редактора лежат рядом.", "target": "Заметки редактора лежат рядом.", "source_type": "clean_identity_from_open_clean"},
            {"source": "Слово метка осталось.", "target": "Слово метка осталось.", "source_type": "clean_identity_from_open_clean"},
        ]
    )

    assert artificial_marker_counts(frame)["metka"] == 1


def test_quality_audit_counts_malformed_dash_spacing_outside_dash_rules():
    frame = pd.DataFrame(
        [
            {
                "source": "\u0415\u0433\u043e\u0440 \u041b\u0435\u0442\u043e\u0432 \u2014\u0440\u043e\u043a-\u043c\u0443\u0437\u044b\u043a\u0430\u043d\u0442.",
                "target": "\u0415\u0433\u043e\u0440 \u041b\u0435\u0442\u043e\u0432 \u2014\u0440\u043e\u043a-\u043c\u0443\u0437\u044b\u043a\u0430\u043d\u0442.",
                "source_type": "synthetic_augmented_from_open_clean",
                "rule_ids": '["pattern_\u0447\u043e_\u0447\u0435"]',
                "metadata": (
                    '{"generation_strategy":"corpus_opportunity",'
                    '"candidate_present":true,'
                    '"error_bearing_sentence_source":"corpus"}'
                ),
                "template_id": "x",
                "normalized_pair_hash": "x",
            }
        ]
    )

    known = known_quality_bug_counts(frame)
    audit = audit_training_dataset(frame, ["pattern_\u0447\u043e_\u0447\u0435"])

    assert known["bad_dash_spacing"] == 1
    assert audit["extended_quality_audit_summary"]["blocking_issue_count"] >= 1


def test_hyphen_po_adverbs_bad_positive_guard_returns_true_for_adjective_contexts():
    assert _hyphen_po_bad_positive("", "Команда работала по-старому плану.")
    assert _hyphen_po_bad_positive("", "Юрист проверил по-новому договору.")
    assert not _hyphen_po_bad_positive("", "Он ответил по-дружески.")


def test_training_quality_audits_find_atomic_unknown_mixed_and_real_pair_failures():
    frame = pd.DataFrame(
        [
            {
                "source": "Сегодня я незнаю что делать.",
                "target": "Сегодня я не знаю, что делать.",
                "split": "train",
                "source_type": "synthetic_augmented_from_open_clean",
                "dataset_layer": "atomic_positive",
                "rule_id": "ne_verb",
                "rule_ids": json.dumps(["ne_verb"], ensure_ascii=False),
                "gold_edit_count": 2,
                "metadata": "{}",
            },
            {
                "source": "Отчет гтов.",
                "target": "Отчет готов.",
                "split": "train",
                "source_type": "synthetic_augmented_from_open_clean",
                "dataset_layer": "atomic_positive",
                "rule_id": "missing_letter_candidate",
                "rule_ids": json.dumps(["missing_letter_candidate"], ensure_ascii=False),
                "gold_edit_count": 1,
                "extra_edit_count": 1,
                "metadata": json.dumps({"operator_verification": {"missing_letter_candidate": {"extra_edit_count": 1}}}),
            },
            {
                "source": "Неизвестная правка.",
                "target": "Неизвестная правка!",
                "split": "train",
                "source_type": "real_error_pair",
                "dataset_layer": "real_atomic",
                "rule_id": "unknown",
                "rule_ids": json.dumps(["unknown"], ensure_ascii=False),
                "gold_edit_count": 1,
                "metadata": "{}",
            },
            {
                "source": "Новая cистема обработки данных заработала утром.",
                "target": "Новая cистема обработки данных заработала утром.",
                "split": "train",
                "source_type": "clean_identity_from_open_clean",
                "dataset_layer": "clean_identity",
                "rule_id": "clean_identity",
                "rule_ids": json.dumps(["clean_identity"], ensure_ascii=False),
                "gold_edit_count": 0,
                "metadata": "{}",
            },
            {
                "source": "Жызнь спокойней.",
                "target": "Жизнь спокойнее.",
                "split": "train",
                "source_type": "real_error_pair",
                "dataset_layer": "real_atomic",
                "rule_id": "frequent_error_exact",
                "rule_ids": json.dumps(["frequent_error_exact"], ensure_ascii=False),
                "gold_edit_count": 2,
                "metadata": "{}",
            },
        ]
    )

    assert atomic_purity_audit_frame(frame)["reason"].tolist() == ["atomic_gold_edit_count_not_one"]
    assert extra_edit_audit_frame(frame)["extra_edit_count"].tolist() == [1]
    assert unknown_rule_train_audit_frame(frame)["reason"].tolist() == ["unknown_rule_in_train"]
    assert mixed_script_clean_audit_frame(frame)["reason"].tolist() == ["mixed_script_token"]
    assert real_pair_atomization_audit_frame(frame)["reason"].tolist() == ["real_atomic_gold_edit_count_not_one"]


def test_training_dataset_quality_errors_include_new_hard_gate_summaries():
    manifest = {
        "total": 250000,
        "split_sizes": {"train": 200000, "val": 25000, "test": 25000},
        "composition": {
            "synthetic_augmented_from_open_clean": 160000,
            "real_error_pair": 1236,
            "clean_identity": 25000,
            "hard_negative": 25000,
            "multi_error_stress": 10000,
        },
        "active_rule_ids": ["comma_subordinate"],
        "rule_id_counts": {"comma_subordinate": 2500},
        "candidate_recall_summary": {"active_min_excluding_unknown": 1.0},
        "gap_label_coverage_summary": {"active_min_excluding_unknown": 1.0},
        "hard_negative_accepted_bad_edits": 0,
        "corpus_opportunity_share": 0.80,
        "fallback_template_share": 0.10,
        "known_quality_bugs": {},
        "artificial_marker_counts": {},
        "quote_bracket_balance_bugs": {},
        "clean_hard_balance_bugs": {},
        "rule_semantic_alignment": {"failed_rows": 0},
        "error_bearing_sentence_source_counts": {"corpus": 10},
        "rule_diversity_summary": {"failed_rule_count": 0},
        "extended_quality_audit_summary": {"blocking_issue_count": 0},
        "atomic_purity_summary": {"failed_rows": 1},
        "extra_edit_summary": {"failed_rows": 1},
        "unknown_rule_train_summary": {"failed_rows": 1},
        "mixed_script_clean_summary": {"failed_rows": 1},
        "real_pair_atomization_summary": {"failed_rows": 1},
        "report_freshness": {"status": "stale", "errors": ["stale_reports_hash_mismatch:candidate_recall_by_rule.csv"]},
    }

    errors = training_dataset_quality_errors(manifest)

    assert "atomic_positive_gold_edit_count_not_one" in errors
    assert "atomic_positive_extra_edits_present" in errors
    assert "unknown_rule_in_train" in errors
    assert "mixed_script_clean_or_hard_present" in errors
    assert "real_pair_atomization_failed" in errors
    assert "stale_reports_hash_mismatch:candidate_recall_by_rule.csv" in errors


def test_report_manifest_detects_stale_report_hash(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report_path = reports_dir / "candidate_recall_by_rule.csv"
    report_path.write_text("rule_id,candidate_recall\ncomma_subordinate,1.0\n", encoding="utf-8")

    manifest = write_report_manifest(reports_dir, dataset_hash="d" * 64, config_hash="c" * 64)
    assert manifest["dataset_hash"] == "d" * 64
    assert report_manifest_errors(reports_dir, dataset_hash="d" * 64, config_hash="c" * 64) == []

    report_path.write_text("rule_id,candidate_recall\ncomma_subordinate,0.0\n", encoding="utf-8")

    assert "stale_reports_hash_mismatch:candidate_recall_by_rule.csv" in report_manifest_errors(
        reports_dir,
        dataset_hash="d" * 64,
        config_hash="c" * 64,
    )
