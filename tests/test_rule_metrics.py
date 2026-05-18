import json
from pathlib import Path

import pandas as pd

from src.evaluation.rule_metrics import build_rule_reports, write_rule_reports


def test_write_rule_reports_creates_all_csv_files(tmp_path: Path):
    rules_path = _rules_config(tmp_path)
    rows = [_row_with_ne_verb()]
    accepted = [_accepted_ne_verb()]

    write_rule_reports(rows, accepted, [], tmp_path, rules_config_path=rules_path)

    for name in [
        "rule_precision_recall.csv",
        "error_by_rule.csv",
        "rule_worse_examples.csv",
    ]:
        assert (tmp_path / name).exists()


def test_rule_precision_recall_includes_ne_verb_from_gold_and_prediction(tmp_path: Path):
    reports = build_rule_reports(
        [_row_with_ne_verb()],
        [_accepted_ne_verb()],
        [],
        rules_config_path=_rules_config(tmp_path),
    )

    summary = reports["rule_precision_recall"]
    ne_verb = summary.loc[summary["rule_id"] == "ne_verb"].iloc[0]

    assert ne_verb["group"] == "ne_ni_particles"
    assert ne_verb["gold_count"] == 1
    assert ne_verb["predicted_count"] == 1
    assert ne_verb["true_positive"] == 1
    assert ne_verb["precision"] == 1.0
    assert ne_verb["recall"] == 1.0
    assert ne_verb["f1"] == 1.0


def test_missing_rule_id_is_reported_as_unknown(tmp_path: Path):
    rows = [
        {
            "source": "Я незнаю",
            "target": "Я не знаю",
            "prediction": "Я не знаю",
            "edit_operations": json.dumps(
                [
                    {
                        "source": "незнаю",
                        "replacement": "не знаю",
                        "edit_type": "split_word",
                        "start": 2,
                        "end": 8,
                    }
                ],
                ensure_ascii=False,
            ),
        }
    ]
    accepted = [
        {
            "row_id": 0,
            "source": "незнаю",
            "replacement": "не знаю",
            "edit_type": "split_word",
            "start": 2,
            "end": 8,
            "status": "accepted",
            "confidence": 0.92,
        }
    ]

    reports = build_rule_reports(rows, accepted, [], rules_config_path=_rules_config(tmp_path))

    summary = reports["rule_precision_recall"]
    unknown = summary.loc[summary["rule_id"] == "unknown"].iloc[0]
    assert unknown["group"] == "unknown"
    assert unknown["true_positive"] == 1


def test_rule_precision_recall_has_zero_safe_scores(tmp_path: Path):
    rows = [
        {
            "source": "Чистый текст.",
            "target": "Чистый текст.",
            "prediction": "Чистый, текст.",
            "edit_operations": "[]",
        }
    ]
    accepted = [
        {
            "row_id": 0,
            "source": "",
            "replacement": ",",
            "edit_type": "punctuation_insert",
            "start": 6,
            "end": 6,
            "rule_id": "",
            "status": "accepted",
            "reason": "unit false positive",
            "confidence": 0.4,
        }
    ]

    reports = build_rule_reports(rows, accepted, [], rules_config_path=_rules_config(tmp_path))

    summary = reports["rule_precision_recall"]
    unknown = summary.loc[summary["rule_id"] == "unknown"].iloc[0]
    assert unknown["gold_count"] == 0
    assert unknown["predicted_count"] == 1
    assert unknown["true_positive"] == 0
    assert unknown["false_positive"] == 1
    assert unknown["false_negative"] == 0
    assert unknown["precision"] == 0.0
    assert unknown["recall"] == 0.0
    assert unknown["f1"] == 0.0
    assert not pd.isna(unknown["precision"])
    assert not pd.isna(unknown["recall"])
    assert not pd.isna(unknown["f1"])


def test_real_rules_config_maps_rule_ids_to_report_groups():
    reports = build_rule_reports(
        [
            _row(
                source="Я недумаю",
                target="Я не думаю",
                rule_id="ne_verb",
                edit_type="split_word",
                edit_source="недумаю",
                replacement="не думаю",
                start=2,
                end=9,
            ),
            _row(
                source="Я думаю что готово",
                target="Я думаю, что готово",
                rule_id="comma_subordinate",
                edit_type="punctuation_insert",
                edit_source="",
                replacement=",",
                start=7,
                end=7,
            ),
            _row(
                source="Проект готов",
                target="Проект готов.",
                rule_id="final_punctuation_default",
                edit_type="final_punctuation",
                edit_source="",
                replacement=".",
                start=12,
                end=12,
            ),
        ],
        [
            _accepted("ne_verb", "split_word", "недумаю", "не думаю", 2, 9, row_id=0),
            _accepted("comma_subordinate", "punctuation_insert", "", ",", 7, 7, row_id=1),
            _accepted("final_punctuation_default", "final_punctuation", "", ".", 12, 12, row_id=2),
        ],
        [],
        rules_config_path=Path("configs/rules.yaml"),
    )

    summary = reports["rule_precision_recall"].set_index("rule_id")

    assert summary.loc["ne_verb", "group"] == "ne_ni_particles"
    assert summary.loc["comma_subordinate", "group"] == "complex_sentence_subordinate"
    assert summary.loc["final_punctuation_default", "group"] == "sentence_final_default_dot"


def _row_with_ne_verb() -> dict:
    return {
        "source": "Я недумаю",
        "target": "Я не думаю",
        "prediction": "Я не думаю",
        "edit_operations": json.dumps(
            [
                {
                    "source": "недумаю",
                    "replacement": "не думаю",
                    "edit_type": "split_word",
                    "start": 2,
                    "end": 9,
                    "rule_id": "ne_verb",
                    "confidence": 0.95,
                }
            ],
            ensure_ascii=False,
        ),
    }


def _row(
    *,
    source: str,
    target: str,
    rule_id: str,
    edit_type: str,
    edit_source: str,
    replacement: str,
    start: int,
    end: int,
) -> dict:
    return {
        "source": source,
        "target": target,
        "prediction": target,
        "edit_operations": json.dumps(
            [
                {
                    "source": edit_source,
                    "replacement": replacement,
                    "edit_type": edit_type,
                    "start": start,
                    "end": end,
                    "rule_id": rule_id,
                    "confidence": 0.95,
                }
            ],
            ensure_ascii=False,
        ),
    }


def _accepted(
    rule_id: str,
    edit_type: str,
    source: str,
    replacement: str,
    start: int,
    end: int,
    *,
    row_id: int,
) -> dict:
    return {
        "row_id": row_id,
        "source": source,
        "replacement": replacement,
        "edit_type": edit_type,
        "start": start,
        "end": end,
        "rule_id": rule_id,
        "status": "accepted",
        "reason": "trusted bounded candidate",
        "confidence": 0.97,
    }


def _accepted_ne_verb() -> dict:
    return {
        "row_id": 0,
        "source": "недумаю",
        "replacement": "не думаю",
        "edit_type": "split_word",
        "start": 2,
        "end": 9,
        "rule_id": "ne_verb",
        "status": "accepted",
        "reason": "trusted bounded candidate",
        "confidence": 0.97,
    }


def _rules_config(tmp_path: Path) -> Path:
    path = tmp_path / "rules.yaml"
    path.write_text(
        """
orthography:
  ne_ni_particles:
    title: "Частицы не и ни"
    status: candidate_only
    requires: [morphology, model]
    rules: [ne_verb]
    notes: "unit"
punctuation: {}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path
