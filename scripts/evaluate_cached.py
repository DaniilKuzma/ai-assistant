from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys
from typing import Any

from src.config.load_config import load_config
from src.evaluation.evaluate import EvaluationReportOptions
from src.evaluation.fast_eval import (
    evaluate_features_detailed,
    evaluation_batch_size,
    write_evaluation_profile_reports,
)
from src.inference.corrector import Corrector
from src.inference.model_corrector import TrainedModelCorrector
from src.training.train import _build_features_with_metadata, _load_rows_for_split


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a cached split with batched model scoring.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config.")
    parser.add_argument("--split", default="val", help="Dataset split to evaluate.")
    parser.add_argument("--limit", type=int, help="Maximum examples to evaluate.")
    parser.add_argument("--profile", action="store_true", help="Write evaluation profile reports.")
    args = parser.parse_args()

    config = load_config(args.config)
    rows = _load_rows_for_split(_config_with_limit(config, args.split, args.limit), split=args.split, fallback_rows=[])
    if args.limit is not None:
        rows = rows[: args.limit]

    reports_dir = Path(config.get("paths", {}).get("reports_dir", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    try:
        corrector: Any = TrainedModelCorrector.from_config(config)
        backend_name = "existing_checkpoint"
    except Exception as exc:
        corrector = Corrector.from_config(config)
        backend_name = f"no_model:{exc.__class__.__name__}"

    if not isinstance(corrector, TrainedModelCorrector) or not hasattr(corrector.backend, "score_features_batched"):
        raise SystemExit(f"Cached batched evaluation requires a trained model backend, got {backend_name}")

    feature_config = _model_eval_feature_config(config)
    feature_run = _build_features_with_metadata(
        feature_config,
        rows,
        split=args.split,
        limit=args.limit,
        tokenizer=getattr(corrector.backend, "tokenizer", None),
    )
    result = evaluate_features_detailed(
        rows,
        feature_run.features,
        corrector=corrector,
        output_dir=reports_dir,
        metric_weights=config.get("metrics", {}).get("combined_score_weights"),
        report_metadata={"evaluation_backend": backend_name},
        report_options=EvaluationReportOptions(),
        batch_size=evaluation_batch_size(config),
        mixed_precision=bool(config.get("training", {}).get("mixed_precision", True)),
        feature_load_time_sec=float(feature_run.cache_result.build_time_sec),
        feature_cache_hit=bool(feature_run.cache_result.hit),
        show_progress=bool(config.get("training", {}).get("show_progress", False)),
    )
    if args.profile:
        write_evaluation_profile_reports(result.profile, reports_dir)

    row = result.profile.as_row()
    print(f"split={args.split}")
    print(f"examples={len(rows)}")
    print(f"eval_time_sec={row['total_eval_time_sec']}")
    print(f"rows_per_sec={row['rows_per_sec']}")
    print(f"eval_batch_size={row['eval_batch_size']}")
    print(f"feature_cache_hit={row['feature_cache_hit']}")
    print(f"feature_cache_path={feature_run.cache_result.path}")
    print(f"model_forward_time_sec={row['model_forward_time_sec']}")
    print(f"validation_time_sec={row['validation_time_sec']}")
    print(f"detailed_reports_time_sec={row['detailed_reports_time_sec']}")
    print(f"bottleneck={row['bottleneck']}")
    print(f"combined_score={result.evaluation.metrics.get('combined_score', 0.0)}")
    return 0


def _config_with_limit(config: dict[str, Any], split: str, limit: int | None) -> dict[str, Any]:
    if limit is None:
        return config
    cloned = copy.deepcopy(config)
    cloned.setdefault("training", {})[_limit_key_for_split(split)] = int(limit)
    return cloned


def _model_eval_feature_config(config: dict[str, Any]) -> dict[str, Any]:
    cloned = copy.deepcopy(config)
    cloned.setdefault("evaluation", {})["feature_tokenizer"] = "model"
    return cloned


def _limit_key_for_split(split: str) -> str:
    return {
        "train": "max_train_examples",
        "val": "max_val_examples",
        "test": "max_test_examples",
    }.get(split, f"max_{split}_examples")


if __name__ == "__main__":
    status = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Large CUDA/Transformers objects can spend a long time in interpreter teardown
    # after reports are already written. Exit the benchmark CLI once output is flushed.
    os._exit(status)
