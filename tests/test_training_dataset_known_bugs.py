import json
import re
from pathlib import Path

import pandas as pd

from src.data.training_quality_audit import quote_bracket_balance_audit_frame


DATASET_PATH = Path("data/processed/correction_dataset.csv.gz")
METKA = "\u043c\u0435\u0442\u043a\u0430"
LATER_EDITOR_CHECKED_RECORD = (
    "\u043f\u043e\u0437\u0436\u0435 "
    "\u0440\u0435\u0434\u0430\u043a\u0442\u043e\u0440 "
    "\u043f\u0440\u043e\u0432\u0435\u0440\u0438\u043b "
    "\u0437\u0430\u043f\u0438\u0441\u044c"
)
LATER_EDITOR_CHECKED_MATERIAL = (
    "\u043f\u043e\u0437\u0436\u0435 "
    "\u0440\u0435\u0434\u0430\u043a\u0442\u043e\u0440 "
    "\u043f\u0440\u043e\u0432\u0435\u0440\u0438\u043b "
    "\u043c\u0430\u0442\u0435\u0440\u0438\u0430\u043b"
)


def _dataset() -> pd.DataFrame:
    assert DATASET_PATH.exists(), DATASET_PATH
    return pd.read_csv(DATASET_PATH)


def _metadata(value: object) -> dict:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return value if isinstance(value, dict) else {}


def _has_rule(frame: pd.DataFrame, rule_id: str) -> pd.Series:
    return frame["rule_ids"].astype(str).str.contains(f'"{rule_id}"', regex=False, na=False)


def test_synthetic_positive_rows_are_never_identity_pairs():
    df = _dataset()
    source_equals_target = df["source"].astype(str) == df["target"].astype(str)
    allowed_identity = df["source_type"].isin({"clean_identity_from_open_clean", "hard_negative_from_open_clean"})

    assert not bool((source_equals_target & ~allowed_identity).any())


def test_final_punctuation_default_positive_rows_have_real_edit():
    df = _dataset()
    final_positive = df[
        df["source_type"].eq("synthetic_augmented_from_open_clean")
        & _has_rule(df, "final_punctuation_default")
    ]

    assert not final_positive.empty
    assert not bool((final_positive["source"].astype(str) == final_positive["target"].astype(str)).any())


def test_dash_rule_targets_have_spaces_around_em_dash():
    df = _dataset()
    dash_rules = ("asyndetic_dash", "consequence_dash", "enumeration_dash")
    bad_spacing = re.compile(r"\S—\s|\s—\S")

    for rule_id in dash_rules:
        rows = df[_has_rule(df, rule_id)]
        bad = rows[rows["target"].astype(str).str.contains(bad_spacing, na=False)]
        assert bad.empty, (rule_id, bad[["source", "target"]].head(5).to_dict("records"))


def test_known_unsafe_positive_phrases_are_absent():
    df = _dataset()
    combined = (df["source"].astype(str) + "\n" + df["target"].astype(str)).str.lower()
    forbidden_phrases = (
        "несогласен с выводом",
        "ненужно комиссии",
        "по-старому плану",
        "по-новому вариант",
        "по-новому договору",
        "по-старому адресу",
        "сохранил территория",
        "записал житель",
        "читал свежая сводка",
        "в закрытая заявка",
    )

    for phrase in forbidden_phrases:
        assert not bool(combined.str.contains(phrase, regex=False, na=False).any()), phrase


def test_n_nn_short_form_does_not_train_noun_to_short_adjective_rewrites():
    df = _dataset()
    rows = df[_has_rule(df, "n_nn_short_form")]
    bad = rows[
        rows["source"].astype(str).str.contains(r"\b(?:цены|страны)\b", regex=True, case=False, na=False)
        & rows["target"].astype(str).str.contains(r"\b(?:ценны|странны)\b", regex=True, case=False, na=False)
    ]

    assert bad.empty, bad[["source", "target"]].head(5).to_dict("records")


def test_positive_synthetic_rows_have_generation_strategy_and_candidate_metadata():
    df = _dataset()
    synthetic = df[df["source_type"].eq("synthetic_augmented_from_open_clean")]
    assert not synthetic.empty

    metadata = synthetic["metadata"].map(_metadata)
    strategies = metadata.map(lambda item: item.get("generation_strategy"))
    bearing_sources = metadata.map(lambda item: item.get("error_bearing_sentence_source"))

    assert not bool(strategies.isna().any())
    assert set(strategies.unique()) <= {"corpus_opportunity", "fallback_natural_template", "multi_error_stress"}
    assert bool(metadata.map(lambda item: item.get("candidate_present") is True).all())
    assert not bool(bearing_sources.isna().any())
    assert set(bearing_sources.unique()) <= {"corpus", "fallback_template"}


def test_artificial_uniqueness_markers_are_absent_from_dataset():
    df = _dataset()
    combined = (df["source"].astype(str) + "\n" + df["target"].astype(str)).str.lower()

    assert int(combined.str.contains(METKA, regex=False, na=False).sum()) == 0
    assert int(combined.str.contains(LATER_EDITOR_CHECKED_RECORD, regex=False, na=False).sum()) == 0
    assert int(combined.str.contains(LATER_EDITOR_CHECKED_MATERIAL, regex=False, na=False).sum()) == 0


def test_corpus_opportunity_rows_do_not_use_artificial_suffixes():
    df = _dataset()
    synthetic = df[df["source_type"].eq("synthetic_augmented_from_open_clean")]
    metadata = synthetic["metadata"].map(_metadata)
    corpus_rows = synthetic[metadata.map(lambda item: item.get("error_bearing_sentence_source") == "corpus")]
    combined = (corpus_rows["source"].astype(str) + "\n" + corpus_rows["target"].astype(str)).str.lower()

    assert not bool(combined.str.contains(METKA, regex=False, na=False).any())
    assert not bool(combined.str.contains(LATER_EDITOR_CHECKED_RECORD, regex=False, na=False).any())
    assert not bool(combined.str.contains(LATER_EDITOR_CHECKED_MATERIAL, regex=False, na=False).any())


def test_positive_targets_have_balanced_quotes_and_brackets():
    df = _dataset()
    positive = df[df["source_type"].isin({"synthetic_augmented_from_open_clean", "real_error_pair"})]
    audit = quote_bracket_balance_audit_frame(positive)

    assert audit.empty, audit.head(20).to_dict("records")
