import json
from pathlib import Path

import pandas as pd

from src.data._training_dataset_builder import should_refresh_real_pair_cache
import src.data.operator_dataset_builder as operator_builder


def test_real_pair_cache_refreshes_when_cached_count_is_below_minimum(tmp_path: Path):
    cache_path = tmp_path / "real_error_pairs_validated.csv.gz"
    cache_path.write_text("source,target\n", encoding="utf-8")

    assert should_refresh_real_pair_cache(cache_path, cached_count=1236, minimum=5000, preferred=30000) is True
    assert should_refresh_real_pair_cache(cache_path, cached_count=6000, minimum=5000, preferred=30000) is False
    assert should_refresh_real_pair_cache(cache_path.with_name("missing.csv.gz"), cached_count=0, minimum=5000, preferred=30000) is True


def test_operator_builder_loads_only_atomic_known_rule_real_train_rows(tmp_path: Path):
    path = tmp_path / "real_error_pairs_atomic.csv.gz"
    pd.DataFrame(
        [
            _real_row("Жызнь стала спокойнее.", "Жизнь стала спокойнее.", "frequent_error_exact", edit_count=1),
            _real_row("Жызнь стала спокойней.", "Жизнь стала спокойнее.", "unknown", edit_count=1),
            _real_row("Жызнь спокойней.", "Жизнь спокойнее.", "frequent_error_exact", edit_count=2),
            _real_row(
                "Жызнь стала лучше.",
                "Жизнь стала лучше.",
                "frequent_error_exact",
                edit_count=1,
                candidate_present=False,
            ),
        ]
    ).to_csv(path, index=False)

    assert hasattr(operator_builder, "_load_real_atomic_rows_from_cache")
    rows = operator_builder._load_real_atomic_rows_from_cache(path, target_count=10, seed=17)

    assert [row["source"] for row in rows] == ["Жызнь стала спокойнее."]
    assert rows[0]["dataset_layer"] == "real_atomic"
    assert rows[0]["source_type"] == "real_error_pair"
    assert rows[0]["count_toward_rule_quota"] is False
    assert rows[0]["gold_edit_count"] == 1


def test_operator_builder_routes_real_multi_edit_rows_to_stress_only(tmp_path: Path):
    path = tmp_path / "real_error_pairs_stress.csv.gz"
    pd.DataFrame(
        [
            _real_row("Жызнь спокойней.", "Жизнь спокойнее.", "frequent_error_exact", edit_count=2, layer="stress_multi_error"),
            _real_row("Жызнь стала спокойнее.", "Жизнь стала спокойнее.", "frequent_error_exact", edit_count=1),
        ]
    ).to_csv(path, index=False)

    assert hasattr(operator_builder, "_load_real_stress_rows_from_cache")
    rows = operator_builder._load_real_stress_rows_from_cache(path, target_count=10, seed=17)

    assert [row["source"] for row in rows] == ["Жызнь спокойней."]
    assert rows[0]["dataset_layer"] == "stress_multi_error"
    assert rows[0]["is_stress"] is True
    assert rows[0]["count_toward_rule_quota"] is False
    assert rows[0]["gold_edit_count"] == 2


def _real_row(
    source: str,
    target: str,
    rule_id: str,
    *,
    edit_count: int,
    layer: str = "real_atomic",
    candidate_present: bool = True,
) -> dict:
    edits = [
        {
            "source": "Жызнь" if index == 0 else "спокойней",
            "replacement": "Жизнь" if index == 0 else "спокойнее",
            "edit_type": "spelling_replace",
            "start": index * 6,
            "end": index * 6 + 5,
            "rule_id": rule_id,
        }
        for index in range(edit_count)
    ]
    metadata = {
        "candidate_present": candidate_present,
        "strict_validator_passed": True,
        "routing_category": "stress" if layer == "stress_multi_error" else "atomic_train",
    }
    return {
        "source": source,
        "target": target,
        "source_dataset": "unit_real",
        "source_subdataset": "",
        "domain": "real_error_pair",
        "detected_error_types": json.dumps(["spelling"], ensure_ascii=False),
        "candidate_present": candidate_present,
        "candidate_rule_ids": json.dumps([rule_id], ensure_ascii=False),
        "edit_count": edit_count,
        "char_edit_ratio": 0.05,
        "token_edit_ratio": 0.05,
        "is_real_pair": True,
        "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        "raw_id": "",
        "raw_source_path": "",
        "error_types": json.dumps(["spelling"], ensure_ascii=False),
        "error_type": "spelling",
        "source_type": "real_error_pair",
        "is_clean": False,
        "is_hard_negative": False,
        "is_synthetic": False,
        "split": "train",
        "rule_id": rule_id,
        "rule_ids": json.dumps([rule_id], ensure_ascii=False),
        "edit_operations": json.dumps(edits, ensure_ascii=False),
        "edits": json.dumps(edits, ensure_ascii=False),
        "dataset_layer": layer,
        "is_stress": layer == "stress_multi_error",
        "count_toward_rule_quota": layer == "real_atomic",
        "loss_weight": 1.0,
        "routing_category": metadata["routing_category"],
        "routing_reason": "unit",
        "strict_validator_passed": True,
    }
