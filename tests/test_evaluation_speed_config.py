from pathlib import Path

import pandas as pd
import torch

from src.evaluation.evaluate import evaluate_rows_detailed
from src.evaluation.fast_eval import (
    EvaluationMode,
    evaluate_features_detailed,
    evaluation_batch_size,
    evaluation_runtime_config,
    limit_rows_for_evaluation,
)
from src.inference.corrector import Corrector
from src.inference.model_corrector import TorchCandidateModelBackend, TrainedModelCorrector
from src.training.feature_cache import build_or_load_features, feature_cache_path
from src.training.tensorization import DebugTokenizer, build_training_feature


def test_evaluation_batch_size_reads_evaluation_config():
    config = {"training": {"batch_size": 4}, "evaluation": {"batch_size": 256}}

    assert evaluation_batch_size(config) == 256


def test_evaluation_batch_size_falls_back_to_training_batch_size():
    config = {"training": {"batch_size": 8}}

    assert evaluation_batch_size(config) == 8


def test_evaluation_runtime_defaults_match_fast_eval_plan():
    settings = evaluation_runtime_config({})

    assert settings.fast_during_training is True
    assert settings.max_eval_examples_per_epoch == 1000
    assert settings.full_eval_after_training is True
    assert settings.full_eval_examples == 5000
    assert settings.write_detailed_reports_during_training is False


def test_fast_and_full_eval_limits_are_applied():
    rows = [{"source": str(index), "target": str(index)} for index in range(10)]
    config = {
        "evaluation": {
            "max_eval_examples_per_epoch": 3,
            "full_eval_examples": 7,
        }
    }

    assert len(limit_rows_for_evaluation(rows, config, mode=EvaluationMode.FAST_DURING_TRAINING)) == 3
    assert len(limit_rows_for_evaluation(rows, config, mode=EvaluationMode.FULL_AFTER_TRAINING)) == 7


def test_feature_cache_uses_separate_files_for_eval_splits(tmp_path: Path):
    config, rows = _cache_config_and_rows(tmp_path)

    train_path = feature_cache_path(config, rows, split="train", limit=1)
    val_path = feature_cache_path(config, rows, split="val", limit=1)
    test_path = feature_cache_path(config, rows, split="test", limit=1)

    assert train_path.name.startswith("train_")
    assert val_path.name.startswith("val_")
    assert test_path.name.startswith("test_")
    assert len({train_path, val_path, test_path}) == 3


def test_eval_feature_cache_hit_skips_feature_rebuild(tmp_path: Path):
    config, rows = _cache_config_and_rows(tmp_path)

    first = build_or_load_features(
        config,
        rows,
        split="val",
        limit=1,
        builder=lambda: [_feature(rows[0])],
    )
    second = build_or_load_features(
        config,
        rows,
        split="val",
        limit=1,
        builder=lambda: (_ for _ in ()).throw(AssertionError("eval features should load from cache")),
    )

    assert first.hit is False
    assert second.hit is True
    assert second.path == first.path
    assert second.report_metrics()["feature_build_skipped"] is True


def test_batched_eval_returns_same_metrics_shape_as_row_eval():
    rows = [
        {"source": "Чистый текст.", "target": "Чистый текст.", "is_clean": True, "is_synthetic": False},
        {"source": "Еще один текст.", "target": "Еще один текст.", "is_clean": True, "is_synthetic": False},
    ]
    features = [_feature(row) for row in rows]
    backend = TorchCandidateModelBackend(
        tokenizer=DebugTokenizer(),
        module=ZeroBatchModule(max_candidates=4, punctuation_label_count=3),
        device=torch.device("cpu"),
        punctuation_labels={"NONE": 0, "COMMA": 1, "DOT": 2},
        punctuation_action_labels={"KEEP_NONE": 0, "KEEP_EXISTING": 1, "INSERT": 2},
        max_length=16,
        max_candidates=4,
    )
    corrector = TrainedModelCorrector(backend, thresholds={"default_threshold": 0.99})

    batched = evaluate_features_detailed(
        rows,
        features,
        corrector=corrector,
        batch_size=2,
        mixed_precision=False,
    ).evaluation
    row_eval = evaluate_rows_detailed(rows, corrector=Corrector())

    assert set(batched.metrics) == set(row_eval.metrics)


def _cache_config_and_rows(tmp_path: Path):
    dataset_path = tmp_path / "dataset.csv.gz"
    rows = [
        {
            "source": "Я незнаю что делать",
            "target": "Я не знаю, что делать.",
            "split": "val",
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
            "max_val_examples": 1,
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


def _feature(row: dict[str, str]):
    return build_training_feature(
        row["source"],
        row["target"],
        tokenizer=DebugTokenizer(),
        punctuation_label_map={"NONE": 0, "COMMA": 1, "DOT": 2},
        punctuation_action_label_map={"KEEP_NONE": 0, "KEEP_EXISTING": 1, "INSERT": 2},
        error_type_label_map={"keep": 0, "split_join": 1, "punctuation": 2, "final_punctuation": 3},
        max_length=16,
        max_candidates=4,
    )


class ZeroBatchModule:
    def __init__(self, max_candidates: int, punctuation_label_count: int):
        self.max_candidates = max_candidates
        self.punctuation_label_count = punctuation_label_count

    def eval(self):
        return self

    def __call__(self, **kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        gap_count = kwargs["punctuation_gap_indices"].shape[1]
        return {
            "candidate_scores": torch.zeros(batch_size, self.max_candidates),
            "confidence_logits": torch.zeros(batch_size, self.max_candidates),
            "punctuation_logits": torch.zeros(batch_size, gap_count, self.punctuation_label_count),
            "punctuation_action_logits": torch.zeros(batch_size, gap_count, 3),
            "punctuation_confidence_logits": torch.zeros(batch_size, gap_count),
        }
