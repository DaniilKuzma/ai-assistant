import json

import pandas as pd

from src.data.dataset_contract import (
    CLEAN_IDENTITY_OPEN,
    CONTRACT_OPTIONAL_COLUMNS,
    HARD_NEGATIVE_OPEN,
    REAL_ERROR_PAIR,
    SYNTHETIC_OPEN_CLEAN,
    ensure_contract_columns,
    row_dataset_layer,
    row_gold_edit_count,
    row_rule_ids,
    stable_dataset_hash,
)


def test_row_rule_ids_parses_json_string_comma_string_list_and_fallback():
    assert row_rule_ids({"rule_ids": json.dumps(["comma_subordinate", "hyphen_particles"])}) == [
        "comma_subordinate",
        "hyphen_particles",
    ]
    assert row_rule_ids({"rule_ids": "comma_subordinate, hyphen_particles"}) == [
        "comma_subordinate",
        "hyphen_particles",
    ]
    assert row_rule_ids({"rule_ids": ["comma_subordinate", "hyphen_particles"]}) == [
        "comma_subordinate",
        "hyphen_particles",
    ]
    assert row_rule_ids({"rule_id": "comma_subordinate"}) == ["comma_subordinate"]


def test_row_gold_edit_count_uses_edits_and_falls_back_to_edit_operations():
    edits = [
        {"source": "превет", "replacement": "привет", "start": 0, "end": 6, "rule_id": "dictionary_fuzzy"},
        {"source": "", "replacement": ".", "start": 6, "end": 6, "rule_id": "final_punctuation_default"},
    ]

    assert row_gold_edit_count({"edits": json.dumps(edits, ensure_ascii=False)}) == 2
    assert row_gold_edit_count({"edits": "", "edit_operations": edits}) == 2
    assert row_gold_edit_count({"gold_edit_count": "3", "edits": []}) == 3


def test_ensure_contract_columns_preserves_old_dataframe_and_adds_defaults():
    frame = pd.DataFrame(
        [
            {
                "source": "превет",
                "target": "привет",
                "source_type": SYNTHETIC_OPEN_CLEAN,
                "rule_ids": json.dumps(["dictionary_fuzzy"], ensure_ascii=False),
                "edits": json.dumps(
                    [{"source": "превет", "replacement": "привет", "start": 0, "end": 6, "rule_id": "dictionary_fuzzy"}],
                    ensure_ascii=False,
                ),
            },
            {"source": "чисто", "target": "чисто", "source_type": CLEAN_IDENTITY_OPEN, "rule_ids": "[]", "edits": "[]"},
            {"source": "ловушка", "target": "ловушка", "source_type": HARD_NEGATIVE_OPEN, "rule_ids": "[]", "edits": "[]"},
            {"source": "реал", "target": "реал.", "source_type": REAL_ERROR_PAIR, "rule_ids": "final_punctuation_default"},
        ]
    )

    upgraded = ensure_contract_columns(frame)

    assert upgraded.loc[0, "source"] == "превет"
    assert upgraded.loc[0, "dataset_contract"] == "candidate_opportunity"
    assert upgraded.loc[0, "dataset_layer"] == "atomic_positive"
    assert bool(upgraded.loc[0, "is_atomic"]) is True
    assert bool(upgraded.loc[0, "count_toward_rule_quota"]) is True
    assert upgraded.loc[0, "loss_weight"] == 1.0
    assert upgraded.loc[0, "gold_edit_count"] == 1
    assert upgraded.loc[0, "candidate_source"] == "превет"
    assert upgraded.loc[0, "candidate_replacement"] == "привет"
    assert upgraded.loc[0, "candidate_start"] == 0
    assert upgraded.loc[0, "candidate_end"] == 6
    assert row_dataset_layer(upgraded.loc[1].to_dict()) == "clean_identity"
    assert row_dataset_layer(upgraded.loc[2].to_dict()) == "atomic_hard_negative"
    assert row_dataset_layer(upgraded.loc[3].to_dict()) == "real_atomic"
    assert set(CONTRACT_OPTIONAL_COLUMNS) <= set(upgraded.columns)


def test_stable_dataset_hash_tracks_semantic_fields_and_ignores_irrelevant_column_order():
    rows = [
        {"source": "а", "target": "а.", "rule_ids": json.dumps(["final_punctuation_default"]), "noise": "one"},
        {"target": "б", "source": "б", "rule_ids": ["clean_identity"], "noise": "two"},
    ]
    reordered = pd.DataFrame(rows, columns=["noise", "rule_ids", "target", "source"])

    assert stable_dataset_hash(rows) == stable_dataset_hash(reordered)
    assert stable_dataset_hash(rows) != stable_dataset_hash([{**rows[0], "source": "в"}, rows[1]])
    assert stable_dataset_hash(rows) != stable_dataset_hash([{**rows[0], "target": "а!"}, rows[1]])
    assert stable_dataset_hash(rows) != stable_dataset_hash([{**rows[0], "rule_ids": ["comma_subordinate"]}, rows[1]])
