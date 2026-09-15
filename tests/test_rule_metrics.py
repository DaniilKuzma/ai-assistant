import json
from pathlib import Path

import pandas as pd

from src.evaluation.rule_metrics import build_rule_reports, write_rule_reports
from src.rules.rule_ids import normalize_rule_id


def test_legacy_frequent_errors_rule_id_normalizes_to_canonical_id():
    assert normalize_rule_id("frequent_errors") == "frequent_error_exact"
    assert normalize_rule_id("frequent_error_exact") == "frequent_error_exact"


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
        path = tmp_path / name
        assert path.exists()
        assert path.stat().st_size > 0


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
            _row(
                source="Проект готов но требует проверки.",
                target="Проект готов, но требует проверки.",
                rule_id="comma_conjunction",
                edit_type="punctuation_insert",
                edit_source="",
                replacement=",",
                start=12,
                end=12,
            ),
            _row(
                source="Документ сохранен, отчет открыт.",
                target="Документ сохранен; отчет открыт.",
                rule_id="semicolon",
                edit_type="punctuation_replace",
                edit_source=",",
                replacement=";",
                start=17,
                end=18,
            ),
            _row(
                source="Автор назвал это проект.",
                target="Автор назвал это «проект».",
                rule_id="quotes_brackets",
                edit_type="punctuation_insert",
                edit_source="",
                replacement="«",
                start=17,
                end=17,
            ),
            _row(
                source="Библеотека открыта.",
                target="Библиотека открыта.",
                rule_id="dictionary_fuzzy",
                edit_type="spelling_replace",
                edit_source="Библеотека",
                replacement="Библиотека",
                start=0,
                end=10,
            ),
        ],
        [
            _accepted("ne_verb", "split_word", "недумаю", "не думаю", 2, 9, row_id=0),
            _accepted("comma_subordinate", "punctuation_insert", "", ",", 7, 7, row_id=1),
            _accepted("final_punctuation_default", "final_punctuation", "", ".", 12, 12, row_id=2),
            _accepted("comma_conjunction", "punctuation_insert", "", ",", 12, 12, row_id=3),
            _accepted("semicolon", "punctuation_replace", ",", ";", 17, 18, row_id=4),
            _accepted("quotes_brackets", "punctuation_insert", "", "«", 17, 17, row_id=5),
            _accepted("dictionary_fuzzy", "spelling_replace", "Библеотека", "Библиотека", 0, 10, row_id=6),
        ],
        [],
        rules_config_path=Path("configs/rules.yaml"),
    )

    summary = reports["rule_precision_recall"].set_index("rule_id")

    assert summary.loc["ne_verb", "group"] == "orthography_3_7_2_1"
    assert summary.loc["comma_subordinate", "group"] == "punctuation_7_2_1"
    assert summary.loc["final_punctuation_default", "group"] == "punctuation_1_1"
    assert summary.loc["comma_conjunction", "group"] == "punctuation_4_3"
    assert summary.loc["semicolon", "group"] == "punctuation_7_1_2"
    assert summary.loc["quotes_brackets", "group"] == "unknown"
    assert summary.loc["dictionary_fuzzy", "group"] == "orthography_8_1"


def test_legacy_alias_matches_canonical_rule_id_as_per_rule_true_positive(tmp_path: Path):
    reports = build_rule_reports(
        [
            _row(
                source="Жызнь прекрасна.",
                target="Жизнь прекрасна.",
                rule_id="frequent_errors",
                edit_type="spelling_replace",
                edit_source="Жызнь",
                replacement="Жизнь",
                start=0,
                end=5,
            )
        ],
        [
            _accepted(
                "frequent_error_exact",
                "spelling_replace",
                "Жызнь",
                "Жизнь",
                0,
                5,
                row_id=0,
            )
        ],
        [],
        rules_config_path=_rules_config(tmp_path),
    )

    summary = reports["rule_precision_recall"].set_index("rule_id")
    frequent_error = summary.loc["frequent_error_exact"]

    assert "frequent_errors" not in summary.index
    assert frequent_error["gold_count"] == 1
    assert frequent_error["predicted_count"] == 1
    assert frequent_error["true_positive"] == 1
    assert frequent_error["false_positive"] == 0
    assert frequent_error["false_negative"] == 0


def test_mismatched_non_alias_rule_id_is_not_per_rule_true_positive(tmp_path: Path):
    reports = build_rule_reports(
        [
            _row(
                source="Я думаю что готово",
                target="Я думаю, что готово",
                rule_id="comma_subordinate",
                edit_type="punctuation_insert",
                edit_source="",
                replacement=",",
                start=7,
                end=7,
            )
        ],
        [
            _accepted(
                "introductory_comma",
                "punctuation_insert",
                "",
                ",",
                7,
                7,
                row_id=0,
            )
        ],
        [],
        rules_config_path=_rules_config(tmp_path),
    )

    summary = reports["rule_precision_recall"].set_index("rule_id")

    assert summary.loc["comma_subordinate", "true_positive"] == 0
    assert summary.loc["comma_subordinate", "false_negative"] == 1
    assert summary.loc["introductory_comma", "true_positive"] == 0
    assert summary.loc["introductory_comma", "false_positive"] == 1


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
  typical_dictionary_words:
    title: "Типичные ошибки: словарные слова"
    status: partial
    requires: [dictionary, model]
    rules: [frequent_error_exact]
    aliases: [frequent_errors]
    notes: "unit"
punctuation:
  complex_sentence_subordinate:
    title: "Сложноподчиненное предложение"
    status: model_required
    requires: [syntax, model]
    rules: [comma_subordinate]
    notes: "unit"
  introductory_words:
    title: "Вводные слова"
    status: model_required
    requires: [syntax, model]
    rules: [introductory_comma]
    notes: "unit"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path
