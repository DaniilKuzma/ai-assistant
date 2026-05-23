from pathlib import Path

import pandas as pd

from src.data.training_quality_audit import (
    rule_semantic_alignment_audit_frame,
    rule_semantic_alignment_summary,
)


DATASET_PATH = Path("data/processed/correction_dataset.csv.gz")


def _synthetic_row(rule_id: str, source: str, target: str) -> dict[str, object]:
    return {
        "source": source,
        "target": target,
        "split": "train",
        "source_type": "synthetic_augmented_from_open_clean",
        "rule_id": rule_id,
        "rule_ids": f'["{rule_id}"]',
        "edits": f'[{{"rule_id":"{rule_id}","edit_type":"punctuation_insert","source":"","replacement":","}}]',
        "metadata": '{"generation_strategy":"corpus_opportunity","candidate_present":true,"error_bearing_sentence_source":"corpus"}',
    }


def test_rule_semantic_alignment_rejects_generic_comma_under_wrong_family():
    frame = pd.DataFrame(
        [
            _synthetic_row(
                "apposition_comma",
                "Он сказал что проект готов.",
                "Он сказал, что проект готов.",
            ),
            _synthetic_row(
                "clarification_comma",
                "Позднее стало известно что встречу перенесли.",
                "Позднее стало известно, что встречу перенесли.",
            ),
            _synthetic_row(
                "detached_participial_comma",
                "По словам редактора документ готов.",
                "По словам редактора, документ готов.",
            ),
            _synthetic_row(
                "homogeneous_comma",
                "Как и ожидалось проект приняли.",
                "Как и ожидалось, проект приняли.",
            ),
        ]
    )

    audit = rule_semantic_alignment_audit_frame(
        frame,
        active_rule_ids=[
            "apposition_comma",
            "clarification_comma",
            "detached_participial_comma",
            "homogeneous_comma",
        ],
    )
    summary = rule_semantic_alignment_summary(audit)

    failed = audit[~audit["alignment_pass"].astype(bool)]
    assert len(failed) == 4
    assert summary["failed_rows"] == 4
    assert summary["failed_by_rule"] == {
        "apposition_comma": 1,
        "clarification_comma": 1,
        "detached_participial_comma": 1,
        "homogeneous_comma": 1,
    }


def test_rule_semantic_alignment_accepts_rule_specific_examples():
    frame = pd.DataFrame(
        [
            _synthetic_row(
                "apposition_comma",
                "Иванов директор компании выступил утром.",
                "Иванов, директор компании, выступил утром.",
            ),
            _synthetic_row(
                "clarification_comma",
                "В пятницу 12 мая комиссия соберется.",
                "В пятницу, 12 мая, комиссия соберется.",
            ),
            _synthetic_row(
                "detached_participial_comma",
                "Документ подготовленный комиссией направили в отдел.",
                "Документ, подготовленный комиссией, направили в отдел.",
            ),
            _synthetic_row(
                "detached_adverbial_comma",
                "Проверив отчет редактор отправил письмо.",
                "Проверив отчет, редактор отправил письмо.",
            ),
            _synthetic_row(
                "homogeneous_comma",
                "Команда проверила отчеты письма и заявки.",
                "Команда проверила отчеты, письма и заявки.",
            ),
            _synthetic_row(
                "comparative_turnover_comma",
                "Он замер как будто услышал шум.",
                "Он замер, как будто услышал шум.",
            ),
        ]
    )

    audit = rule_semantic_alignment_audit_frame(
        frame,
        active_rule_ids=[
            "apposition_comma",
            "clarification_comma",
            "detached_participial_comma",
            "detached_adverbial_comma",
            "homogeneous_comma",
            "comparative_turnover_comma",
        ],
    )

    assert audit["alignment_pass"].astype(bool).all(), audit.to_dict("records")


def test_canonical_active_syntax_punctuation_rows_are_rule_aligned():
    assert DATASET_PATH.exists(), DATASET_PATH
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    active = [
        "apposition_comma",
        "clarification_comma",
        "detached_participial_comma",
        "detached_adverbial_comma",
        "homogeneous_comma",
        "comparative_turnover_comma",
        "comma_subordinate",
        "comma_conjunction",
        "direct_speech_quotes",
        "subject_predicate_dash",
        "asyndetic_dash",
    ]

    audit = rule_semantic_alignment_audit_frame(df, active_rule_ids=active)
    summary = rule_semantic_alignment_summary(audit)

    assert summary["failed_rows"] == 0, audit[~audit["alignment_pass"].astype(bool)].head(20).to_dict("records")


def test_detached_participial_rejects_source_attribution_with_participle_elsewhere():
    frame = pd.DataFrame(
        [
            _synthetic_row(
                "detached_participial_comma",
                "По данным штаба отчет, подготовленный комиссией, направили в отдел.",
                "По данным штаба, отчет, подготовленный комиссией, направили в отдел.",
            )
        ]
    )

    audit = rule_semantic_alignment_audit_frame(
        frame,
        active_rule_ids=["detached_participial_comma"],
    )
    summary = rule_semantic_alignment_summary(audit)

    assert summary["failed_rows"] == 1
    assert summary["failed_by_rule"] == {"detached_participial_comma": 1}
