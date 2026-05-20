from pathlib import Path

import pandas as pd

from src.candidates.candidate_generator import CandidateGenerator
from src.training.feature_cache import build_or_load_features
from src.training.tensorization import DebugTokenizer, build_training_feature


def test_feature_cache_miss_builds_and_saves_cache(tmp_path: Path):
    config, rows = _cache_config_and_rows(tmp_path)
    calls = {"count": 0}

    def builder():
        calls["count"] += 1
        return [_feature(rows[0], max_candidates=4)]

    result = build_or_load_features(config, rows, split="train", builder=builder)

    assert result.hit is False
    assert calls["count"] == 1
    assert result.path.exists()
    assert result.features[0].source == rows[0]["source"]


def test_feature_cache_hit_loads_without_rebuilding(tmp_path: Path):
    config, rows = _cache_config_and_rows(tmp_path)

    first = build_or_load_features(config, rows, split="train", builder=lambda: [_feature(rows[0], max_candidates=4)])
    second = build_or_load_features(
        config,
        rows,
        split="train",
        builder=lambda: (_ for _ in ()).throw(AssertionError("builder should not run on cache hit")),
    )

    assert first.hit is False
    assert second.hit is True
    assert second.path == first.path
    assert second.features[0].candidate_rule_ids == first.features[0].candidate_rule_ids


def test_feature_cache_invalidates_when_max_candidates_changes(tmp_path: Path):
    config, rows = _cache_config_and_rows(tmp_path)
    first = build_or_load_features(config, rows, split="train", builder=lambda: [_feature(rows[0], max_candidates=4)])

    changed = dict(config)
    changed["model"] = {**config["model"], "max_candidates": 6}
    second = build_or_load_features(changed, rows, split="train", builder=lambda: [_feature(rows[0], max_candidates=6)])

    assert second.hit is False
    assert second.path != first.path
    assert len(second.features[0].candidate_rule_ids) == 6


def test_feature_cache_invalidates_when_max_sequence_length_changes(tmp_path: Path):
    config, rows = _cache_config_and_rows(tmp_path)
    first = build_or_load_features(config, rows, split="train", builder=lambda: [_feature(rows[0], max_candidates=4)])

    changed = dict(config)
    changed["model"] = {**config["model"], "max_sequence_length": 20}
    second = build_or_load_features(changed, rows, split="train", builder=lambda: [_feature(rows[0], max_candidates=4)])

    assert second.hit is False
    assert second.path != first.path


def test_feature_cache_corrupt_file_rebuilds(tmp_path: Path):
    config, rows = _cache_config_and_rows(tmp_path)
    first = build_or_load_features(config, rows, split="train", builder=lambda: [_feature(rows[0], max_candidates=4)])
    first.path.write_bytes(b"not a pickle")
    calls = {"count": 0}

    def builder():
        calls["count"] += 1
        return [_feature(rows[0], max_candidates=4)]

    rebuilt = build_or_load_features(config, rows, split="train", builder=builder)

    assert rebuilt.hit is False
    assert rebuilt.rebuilt_from_corrupt is True
    assert calls["count"] == 1
    assert rebuilt.path == first.path


def test_feature_cache_preserves_candidate_metadata(tmp_path: Path):
    config, rows = _cache_config_and_rows(tmp_path)
    build_or_load_features(config, rows, split="train", builder=lambda: [_feature(rows[0], max_candidates=4)])

    loaded = build_or_load_features(
        config,
        rows,
        split="train",
        builder=lambda: (_ for _ in ()).throw(AssertionError("builder should not run on cache hit")),
    )
    feature = loaded.features[0]

    assert len(feature.candidate_rule_ids) == 4
    assert len(feature.candidate_modes) == 4
    assert len(feature.candidate_requires_model) == 4
    assert len(feature.candidate_requires_scoring) == 4


def _cache_config_and_rows(tmp_path: Path):
    dataset_path = tmp_path / "dataset.csv.gz"
    rows = [
        {
            "source": "Я незнаю что делать",
            "target": "Я не знаю, что делать.",
            "split": "train",
            "source_type": "synthetic_augmented_from_open_clean",
        }
    ]
    pd.DataFrame(rows).to_csv(dataset_path, index=False)
    config = {
        "model": {
            "primary_encoder": "debug-tokenizer",
            "max_sequence_length": 16,
            "max_candidates": 4,
        },
        "training": {
            "max_train_examples": 1,
            "feature_cache": {
                "enabled": True,
                "cache_dir": str(tmp_path / "features_cache"),
                "include_dataset_hash": True,
                "include_config_hash": True,
                "compression": False,
            },
            "feature_build": {"enable_syntax": False},
        },
        "data": {"processed_train_path": str(dataset_path)},
        "labels": {
            "punctuation": {"NONE": 0, "COMMA": 1, "DOT": 2},
            "punctuation_actions": {"KEEP_NONE": 0, "KEEP_EXISTING": 1, "INSERT": 2},
            "error_types": {"keep": 0, "split_join": 1, "punctuation": 2, "final_punctuation": 3},
        },
        "dictionary": {"enabled": False},
    }
    return config, rows


def _feature(row: dict[str, str], *, max_candidates: int):
    return build_training_feature(
        row["source"],
        row["target"],
        tokenizer=DebugTokenizer(),
        punctuation_label_map={"NONE": 0, "COMMA": 1, "DOT": 2},
        punctuation_action_label_map={"KEEP_NONE": 0, "KEEP_EXISTING": 1, "INSERT": 2},
        error_type_label_map={"keep": 0, "split_join": 1, "punctuation": 2, "final_punctuation": 3},
        max_length=16,
        max_candidates=max_candidates,
        candidate_generator=CandidateGenerator(syntax_provider=lambda _text: ()),
    )
