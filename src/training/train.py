from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any

import pandas as pd

from src.candidates.candidate_generator import CandidateGenerator
from src.config.load_config import load_config
from src.data.dataset_builder import build_synthetic_dataset
from src.data.dataset_stats import dataset_stats
from src.evaluation.reports import write_loss_curve, write_threshold_precision_recall_plot, write_training_report
from src.evaluation.evaluate import EvaluationReportOptions, evaluate_rows_detailed
from src.evaluation.fast_eval import (
    EvaluationMode,
    EvaluationProfile,
    core_training_metrics,
    evaluate_features_detailed,
    evaluation_batch_size,
    evaluation_profile_metadata,
    evaluation_runtime_config,
    limit_rows_for_evaluation,
    training_fast_report_options,
    write_evaluation_profile_reports,
)
from src.evaluation.reports import write_dataset_report
from src.evaluation.threshold_sweep import threshold_sweep
from src.inference.corrector import Corrector
from src.inference.model_corrector import TorchCandidateModelBackend, TrainedModelCorrector
from src.model.edit_model import CandidateAwareEditModel, EditModelConfig
from src.model.encoder import EncoderLoadConfig, load_tokenizer
from src.training.save_load import save_training_artifacts
from src.training.feature_cache import FeatureCacheResult, build_or_load_features
from src.training.feature_profile import FeatureBuildProfiler
from src.training.diagnostics import training_positive_counts, write_training_label_distribution_by_rule
from src.training.tensorization import DebugTokenizer, EditBatchCollator, build_features_from_rows
from src.training.callbacks import BestMetricTracker
from src.training.trainer import EditModelTrainer, TrainLoopConfig


DISABLE_MODEL_TRAINING_ENV = "RUSSIAN_CORRECTOR_DISABLE_MODEL_TRAINING"


@dataclass(frozen=True)
class FeatureBuildRun:
    features: list[Any]
    cache_result: FeatureCacheResult
    metadata: dict[str, Any]


def evaluate_trained_model(
    config_path: str | Path = "configs/config.yaml",
    *,
    split: str = "test",
    corrector: Any | None = None,
) -> dict[str, Any]:
    """Run final evaluation for a saved/fine-tuned corrector on a dataset split."""

    config = load_config(config_path)
    reports_dir = Path(config.get("paths", {}).get("reports_dir", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows_for_split(config, split=split, fallback_rows=[])
    evaluation_corrector = corrector or _build_evaluation_corrector(config, model_training_ran=True)
    print(f"Evaluating {len(rows)} {split} examples with {evaluation_corrector.__class__.__name__}...")
    report_metadata = {
        "evaluation_backend": "stub_scorer" if corrector is not None else "existing_checkpoint",
        "model_training_disabled": False,
        "model_training_disabled_source": "",
    }
    evaluation_result, eval_metadata = _evaluate_for_training_report(
        config,
        rows,
        split=split,
        corrector=evaluation_corrector,
        reports_dir=reports_dir,
        report_metadata=report_metadata,
        report_options=EvaluationReportOptions(),
        limit=len(rows),
    )
    write_threshold_precision_recall_plot(
        threshold_sweep(evaluation_result.edit_scores, [0.5, 0.7, 0.8, 0.9, 0.95]),
        reports_dir / "threshold_precision_recall.png",
    )
    return {
        "status": "evaluated",
        "evaluation_split": split,
        "evaluation_count": len(rows),
        "evaluation_metrics": evaluation_result.metrics,
        "reports_dir": str(reports_dir),
        **eval_metadata,
        "report_paths": _report_paths(reports_dir),
    }


def train(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Prepare data and optionally run encoder fine-tuning."""

    config = load_config(config_path)
    reports_dir = Path(config.get("paths", {}).get("reports_dir", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    run_model_training = _run_model_training_enabled(config)
    model_training_disabled_source = _model_training_disabled_source(config)
    model_training_disabled = bool(model_training_disabled_source)
    rows = _load_training_rows(config)
    write_dataset_report(_full_dataset_stats(config, rows), reports_dir / "dataset_report.md")

    skip_feature_build = _skip_feature_build_for_no_training(model_training_disabled_source)
    feature_run: FeatureBuildRun | None = None
    if skip_feature_build:
        features = []
        feature_metadata: dict[str, Any] = {
            "feature_cache_enabled": False,
            "feature_cache_hit": False,
            "feature_cache_path": "",
            "feature_build_time_sec": 0.0,
            "features_count": 0,
            "word_candidate_positive_count": 0,
            "spelling_positive_count": 0,
            "hyphen_positive_count": 0,
            "split_join_positive_count": 0,
            "punctuation_positive_count": 0,
        }
    else:
        feature_run = _build_features_with_metadata(config, rows, split="train")
        features = feature_run.features
        feature_metadata = feature_run.metadata
        write_training_label_distribution_by_rule(
            features,
            reports_dir / "training_label_distribution_by_rule.csv",
        )
        feature_metadata.update(training_positive_counts(features))
    output_dir = config.get("paths", {}).get("adapter_output_dir", "models/adapters/latest")

    result: dict[str, Any] = {
        "status": "evaluation_prepared" if skip_feature_build else "features_prepared",
        "output_dir": output_dir,
        "feature_count": len(features),
        "feature_build_skipped": skip_feature_build,
        "feature_source_count": len(rows),
        "model_training_ran": False,
        "model_training_disabled": model_training_disabled,
        "model_training_disabled_source": model_training_disabled_source,
    }

    evaluation_split = _training_evaluation_split(config)
    all_evaluation_rows = _load_evaluation_rows(config, rows)
    fast_eval_feature_run: FeatureBuildRun | None = None
    fast_eval_rows: list[dict[str, Any]] = []
    if run_model_training and evaluation_runtime_config(config).fast_during_training:
        fast_eval_rows = limit_rows_for_evaluation(
            all_evaluation_rows,
            config,
            mode=EvaluationMode.FAST_DURING_TRAINING,
        )
        if fast_eval_rows:
            fast_eval_feature_run = _build_features_with_metadata(
                _model_eval_feature_config(config),
                fast_eval_rows,
                split=evaluation_split,
                limit=len(fast_eval_rows),
            )

    if run_model_training:
        result.update(
            _run_model_training_with_fast_eval(
                config,
                features,
                fast_eval_rows=fast_eval_rows,
                fast_eval_features=fast_eval_feature_run.features if fast_eval_feature_run is not None else None,
            )
        )

    save_training_artifacts(
        output_dir,
        config=config,
        label_mappings=config.get("labels", {}),
        thresholds=config.get("thresholds", {}),
    )
    evaluation_rows = limit_rows_for_evaluation(
        all_evaluation_rows,
        config,
        mode=EvaluationMode.FULL_AFTER_TRAINING,
    )
    corrector, backend_metadata = _select_evaluation_corrector(
        config,
        model_training_ran=bool(result["model_training_ran"]),
        model_training_disabled_source=model_training_disabled_source,
    )
    print(
        f"Evaluating {len(evaluation_rows)} {evaluation_split} examples "
        f"with {corrector.__class__.__name__} ({backend_metadata['evaluation_backend']})..."
    )
    evaluation_result, eval_metadata = _evaluate_for_training_report(
        config,
        evaluation_rows,
        split=evaluation_split,
        corrector=corrector,
        reports_dir=reports_dir,
        report_metadata=backend_metadata,
        report_options=EvaluationReportOptions(),
        limit=len(evaluation_rows),
    )
    evaluation_metrics = evaluation_result.metrics
    checkpoint_metric = str(config.get("training", {}).get("checkpoint_metric", "combined_score"))
    tracker = BestMetricTracker(checkpoint_metric)
    checkpoint_source_metrics = result.get("fast_evaluation_metrics") or evaluation_metrics
    is_best_checkpoint = tracker.update(checkpoint_source_metrics)
    checkpoint_metric_value = float(checkpoint_source_metrics.get(checkpoint_metric, 0.0))
    losses = [float(result.get("train_loss", 0.0))]
    write_loss_curve(losses, reports_dir / "loss_curves.png")
    write_threshold_precision_recall_plot(
        threshold_sweep(evaluation_result.edit_scores, [0.5, 0.7, 0.8, 0.9, 0.95]),
        reports_dir / "threshold_precision_recall.png",
    )
    write_training_report(
        {
            "feature_count": float(len(features)),
            "feature_build_skipped": skip_feature_build,
            "feature_source_count": float(len(rows)),
            "evaluation_split": evaluation_split,
            "evaluation_count": float(len(evaluation_rows)),
            "model_training_ran": float(result["model_training_ran"]),
            "model_training_disabled": model_training_disabled,
            "model_training_disabled_source": model_training_disabled_source,
            **feature_metadata,
            **_dataset_training_metadata(config),
            **_training_sanity_metadata(config),
            **backend_metadata,
            **eval_metadata,
            **_training_loss_metadata(result),
            "checkpoint_metric": checkpoint_metric,
            "checkpoint_metric_value": checkpoint_metric_value,
            "checkpoint_metric_source": "fast_eval" if result.get("fast_evaluation_metrics") else "full_eval",
            "is_best_checkpoint": float(is_best_checkpoint),
            **evaluation_metrics,
        },
        reports_dir / "training_report.md",
    )
    result.update(
        {
            "evaluation_count": len(evaluation_rows),
            "evaluation_split": evaluation_split,
            "evaluation_metrics": evaluation_metrics,
            **backend_metadata,
            **eval_metadata,
            "checkpoint_metric": checkpoint_metric,
            "checkpoint_metric_value": checkpoint_metric_value,
            "is_best_checkpoint": is_best_checkpoint,
            **feature_metadata,
            "reports_dir": str(reports_dir),
            "report_paths": _report_paths(reports_dir),
        }
    )
    print(f"Reports written to: {reports_dir}")
    return result


def _load_training_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    data_config = config.get("data", {})
    processed_path = data_config.get("processed_train_path")
    if processed_path and Path(processed_path).exists():
        frame = pd.read_csv(processed_path)
        if "split" in frame.columns:
            frame = frame[frame["split"] == "train"]
        max_examples = int(config.get("training", {}).get("max_train_examples", len(frame)))
        return frame.head(max_examples).to_dict("records")

    clean_texts = data_config.get(
        "debug_clean_texts",
        ["Я не знаю, что делать.", "Во-первых, это важно.", "Чистый текст."],
    )
    frame = build_synthetic_dataset(clean_texts, seed=int(data_config.get("synthetic_seed", 13)))
    max_examples = int(config.get("training", {}).get("max_train_examples", len(frame)))
    return frame.head(max_examples).to_dict("records")


def _full_dataset_stats(config: dict[str, Any], fallback_rows: list[dict[str, Any]]) -> dict[str, int]:
    processed_path = config.get("data", {}).get("processed_train_path")
    if processed_path and Path(processed_path).exists():
        return dataset_stats(pd.read_csv(processed_path, usecols=["is_clean", "is_synthetic", "split", "error_types"]))
    return dataset_stats(pd.DataFrame(fallback_rows))


def _dataset_training_metadata(config: dict[str, Any]) -> dict[str, Any]:
    data_config = config.get("data", {})
    dataset_path_value = str(data_config.get("processed_train_path") or "")
    manifest_path_value = str(data_config.get("manifest_path") or "")
    dataset_path = Path(dataset_path_value) if dataset_path_value else None
    manifest_path = Path(manifest_path_value) if manifest_path_value else None
    metadata: dict[str, Any] = {
        "dataset_path": dataset_path_value,
        "manifest_path": manifest_path_value,
        "manifest_verdict": "",
        "dataset_rows": 0,
        "split_counts": {},
    }
    if dataset_path is not None and dataset_path.exists():
        frame = pd.read_csv(dataset_path, usecols=lambda column: column == "split")
        metadata["dataset_rows"] = int(len(frame))
        if "split" in frame.columns:
            metadata["split_counts"] = {str(key): int(value) for key, value in frame["split"].value_counts().sort_index().items()}
    if manifest_path is not None and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metadata["manifest_verdict"] = str(manifest.get("verdict", manifest.get("final_verdict", "")))
        except json.JSONDecodeError:
            metadata["manifest_verdict"] = "INVALID_JSON"
    return metadata


def _training_sanity_metadata(config: dict[str, Any]) -> dict[str, Any]:
    training_config = config.get("training", {})
    batch_size = int(training_config.get("batch_size", 0) or 0)
    accumulation_steps = int(training_config.get("gradient_accumulation_steps", 0) or 0)
    warning = ""
    if batch_size == 64:
        warning = "batch_size=64 gives few optimizer steps; recommend batch_size=4 and gradient_accumulation_steps=4"
    return {
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "training_batch_warning": warning,
    }


def _training_loss_metadata(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "train_loss": float(result.get("train_loss", 0.0)),
        "word_loss_value": float(result.get("word_loss_value", 0.0)),
        "punctuation_loss_value": float(result.get("punctuation_loss_value", 0.0)),
        "confidence_loss_value": float(result.get("confidence_loss_value", 0.0)),
        "error_type_loss_value": float(result.get("error_type_loss_value", 0.0)),
    }


def _load_evaluation_rows(config: dict[str, Any], fallback_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _load_rows_for_split(config, split=_training_evaluation_split(config), fallback_rows=fallback_rows)


def _evaluate_for_training_report(
    config: dict[str, Any],
    evaluation_rows: list[dict[str, Any]],
    *,
    split: str,
    corrector: Any,
    reports_dir: Path,
    report_metadata: dict[str, Any],
    report_options: EvaluationReportOptions,
    limit: int | None,
    ) -> tuple[Any, dict[str, Any]]:
    batch_size = evaluation_batch_size(config)
    backend = getattr(corrector, "backend", None)
    if isinstance(corrector, TrainedModelCorrector) and hasattr(backend, "score_features_batched"):
        feature_config = _model_eval_feature_config(config)
        feature_run = _build_features_with_metadata(
            feature_config,
            evaluation_rows,
            split=split,
            limit=limit,
            tokenizer=getattr(backend, "tokenizer", None),
        )
        profiled = evaluate_features_detailed(
            evaluation_rows,
            feature_run.features,
            corrector=corrector,
            output_dir=reports_dir,
            metric_weights=config.get("metrics", {}).get("combined_score_weights"),
            report_metadata=report_metadata,
            report_options=report_options,
            batch_size=batch_size,
            mixed_precision=bool(config.get("training", {}).get("mixed_precision", True)),
            feature_load_time_sec=float(feature_run.cache_result.build_time_sec),
            feature_cache_hit=bool(feature_run.cache_result.hit),
            show_progress=bool(config.get("training", {}).get("show_progress", False)),
        )
        write_evaluation_profile_reports(profiled.profile, reports_dir)
        return profiled.evaluation, {
            "eval_feature_cache_enabled": bool(feature_run.cache_result.enabled),
            "eval_feature_cache_hit": bool(feature_run.cache_result.hit),
            "eval_feature_cache_path": str(feature_run.cache_result.path),
            "eval_feature_load_time_sec": round(float(feature_run.cache_result.build_time_sec), 3),
            **evaluation_profile_metadata(profiled.profile),
        }

    started = time.perf_counter()
    evaluation_result = evaluate_rows_detailed(
        evaluation_rows,
        corrector=corrector,
        output_dir=reports_dir,
        metric_weights=config.get("metrics", {}).get("combined_score_weights"),
        show_progress=bool(config.get("training", {}).get("show_progress", False)),
        report_metadata=report_metadata,
        candidate_generator=_candidate_recall_generator(corrector, config),
        candidate_recall_max_candidates=_candidate_recall_max_candidates(config),
        report_options=report_options,
    )
    elapsed = time.perf_counter() - started
    profile = _row_evaluation_profile(
        elapsed,
        batch_size=batch_size,
        count=len(evaluation_rows),
    )
    write_evaluation_profile_reports(profile, reports_dir)
    return evaluation_result, {
        "eval_feature_cache_enabled": False,
        "eval_feature_cache_hit": False,
        "eval_feature_cache_path": "",
        "eval_feature_load_time_sec": 0.0,
        **evaluation_profile_metadata(profile),
    }


def _model_eval_feature_config(config: dict[str, Any]) -> dict[str, Any]:
    cloned = copy.deepcopy(config)
    cloned.setdefault("evaluation", {})["feature_tokenizer"] = "model"
    return cloned


def _row_evaluation_profile(elapsed: float, *, batch_size: int, count: int) -> EvaluationProfile:
    return EvaluationProfile(
        total_eval_time_sec=elapsed,
        feature_load_time_sec=0.0,
        model_forward_time_sec=0.0,
        thresholding_time_sec=0.0,
        validation_time_sec=0.0,
        edit_realization_time_sec=0.0,
        metrics_time_sec=0.0,
        detailed_reports_time_sec=elapsed,
        rows_per_sec=count / elapsed if elapsed > 0 else 0.0,
        eval_batch_size=batch_size,
        gpu_available=_torch_gpu_available(),
        cuda_device=_torch_cuda_device(),
        feature_cache_hit=False,
        number_of_eval_examples=count,
    )


def _torch_gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _torch_cuda_device() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return ""
        return str(torch.cuda.get_device_name(torch.cuda.current_device()))
    except Exception:
        return ""


def _training_evaluation_split(config: dict[str, Any]) -> str:
    return str(config.get("training", {}).get("evaluation_split", "val"))


def _load_rows_for_split(config: dict[str, Any], *, split: str, fallback_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    data_config = config.get("data", {})
    processed_path = data_config.get("processed_train_path")
    training_config = config.get("training", {})
    if processed_path and Path(processed_path).exists():
        frame = pd.read_csv(processed_path)
        if "split" not in frame.columns:
            max_examples = int(training_config.get(_limit_key_for_split(split), len(frame)))
            return frame.head(max_examples).to_dict("records")
        split_rows = frame[frame["split"] == split]
        max_examples = int(training_config.get(_limit_key_for_split(split), len(split_rows)))
        split_rows = split_rows.head(max_examples)
        if not split_rows.empty:
            return split_rows.to_dict("records")
    return fallback_rows


def _limit_key_for_split(split: str) -> str:
    return {
        "train": "max_train_examples",
        "val": "max_val_examples",
        "test": "max_test_examples",
    }.get(split, f"max_{split}_examples")


def _build_evaluation_corrector(config: dict[str, Any], model_training_ran: bool):
    if model_training_ran:
        return TrainedModelCorrector.from_config(config)
    return Corrector.from_config(config)


def _select_evaluation_corrector(
    config: dict[str, Any],
    *,
    model_training_ran: bool,
    model_training_disabled_source: str,
) -> tuple[Any, dict[str, Any]]:
    metadata: dict[str, Any] = {
        "model_training_disabled": bool(model_training_disabled_source),
        "model_training_disabled_source": model_training_disabled_source,
    }
    if model_training_ran:
        return _build_evaluation_corrector(config, True), {
            **metadata,
            "evaluation_backend": "existing_checkpoint",
            "checkpoint_load_error": "",
        }

    if model_training_disabled_source and _configured_model_artifacts_exist(config):
        try:
            evaluation_config = _config_with_threshold_profile(config, "conservative")
            return TrainedModelCorrector.from_config(evaluation_config), {
                **metadata,
                "evaluation_backend": "existing_checkpoint",
                "checkpoint_load_error": "",
                "threshold_profile": _threshold_profile(evaluation_config),
            }
        except Exception as exc:
            return _build_evaluation_corrector(config, False), {
                **metadata,
                "evaluation_backend": "no_model",
                "checkpoint_load_error": f"{exc.__class__.__name__}: {exc}",
            }

    return _build_evaluation_corrector(config, False), {
        **metadata,
        "evaluation_backend": "no_model",
        "checkpoint_load_error": "",
    }


def _configured_model_artifacts_exist(config: dict[str, Any]) -> bool:
    paths = config.get("paths", {})
    adapter_dir = Path(paths.get("adapter_output_dir", "models/adapters/latest"))
    heads_path = Path(paths.get("heads_output_dir", "models/heads/latest")) / "heads.pt"
    return adapter_dir.exists() and heads_path.exists()


def _config_with_threshold_profile(config: dict[str, Any], profile: str) -> dict[str, Any]:
    cloned = copy.deepcopy(config)
    threshold_config = cloned.setdefault("thresholds", {})
    if profile in threshold_config:
        threshold_config["mode"] = profile
    return cloned


def _threshold_profile(config: dict[str, Any]) -> str:
    return str(config.get("thresholds", {}).get("mode", "balanced"))


def _active_thresholds(config: dict[str, Any]) -> dict[str, Any]:
    threshold_config = config.get("thresholds", {})
    mode = str(threshold_config.get("mode", "balanced"))
    return dict(threshold_config.get(mode, {}))


def _report_paths(reports_dir: Path) -> dict[str, str]:
    names = [
        "dataset_report.md",
        "training_report.md",
        "evaluation_summary.csv",
        "error_by_type.csv",
        "rule_precision_recall.csv",
        "error_by_rule.csv",
        "rule_worse_examples.csv",
        "candidate_recall_by_rule.csv",
        "gap_label_coverage_by_rule.csv",
        "clean_overcorrection_examples.csv",
        "dirty_worse_examples.csv",
        "accepted_edits.csv",
        "rejected_edits.csv",
        "loss_curves.png",
        "threshold_precision_recall.png",
        "feature_build_profile.csv",
        "feature_build_profile_summary.md",
        "feature_build_candidate_recall_by_rule.csv",
        "feature_build_gap_label_coverage_by_rule.csv",
        "training_label_distribution_by_rule.csv",
        "candidate_score_distribution_by_rule.csv",
        "evaluation_profile.csv",
        "evaluation_profile_summary.md",
    ]
    return {name: str(reports_dir / name) for name in names if (reports_dir / name).exists()}


def _candidate_recall_generator(corrector: Any, config: dict[str, Any]) -> CandidateGenerator:
    generator = getattr(corrector, "candidates", None)
    if isinstance(generator, CandidateGenerator):
        return generator
    return CandidateGenerator.from_config(config)


def _candidate_recall_max_candidates(config: dict[str, Any]) -> int:
    return int(config.get("model", {}).get("max_candidates", 16))


def _build_features(config: dict[str, Any], rows: list[dict[str, Any]]):
    return _build_features_with_metadata(config, rows, split="train").features


def _build_features_with_metadata(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    split: str = "train",
    force_cache: bool = False,
    limit: int | None = None,
    tokenizer: Any | None = None,
) -> FeatureBuildRun:
    training_config = config.get("training", {})
    builder_stats: dict[str, Any] = {}

    def builder():
        model_config = config.get("model", {})
        label_config = config.get("labels", {})
        feature_tokenizer = tokenizer or _feature_build_tokenizer(config)
        candidate_generator = CandidateGenerator.from_config(config, purpose="training_features")
        profiler = _feature_build_profiler(config, split=split, candidate_generator=candidate_generator)
        dictionary_policy_for_row = _dictionary_policy_for_training_row(config)
        features = build_features_from_rows(
            rows,
            tokenizer=feature_tokenizer,
            punctuation_label_map=label_config.get("punctuation", {}),
            punctuation_action_label_map=label_config.get("punctuation_actions", {}),
            error_type_label_map=label_config.get("error_types", {}),
            max_length=int(model_config.get("max_sequence_length", 192)),
            max_candidates=int(model_config.get("max_candidates", 32)),
            candidate_generator=candidate_generator,
            show_progress=bool(training_config.get("show_progress", False)),
            profiler=profiler,
            split=split,
            dictionary_policy_for_row=dictionary_policy_for_row,
        )
        dictionary_stats = candidate_generator.dictionary_cache_stats()
        builder_stats.update(
            {
                "dictionary_token_cache_hits": dictionary_stats["hits"],
                "dictionary_token_cache_misses": dictionary_stats["misses"],
                "dictionary_token_cache_hit_rate": round(float(dictionary_stats["hit_rate"]), 6),
                "dictionary_lexicon_size": _dictionary_lexicon_size(candidate_generator),
                "dictionary_max_choices_per_token": candidate_generator.dictionary_max_choices_per_token,
            }
        )
        profiler.write_reports(
            dictionary_cache_stats=dictionary_stats,
            feature_cache_stats={"enabled": bool(training_config.get("feature_cache", {}).get("enabled", False)), "hit": False},
        )
        return features

    cache_result = build_or_load_features(config, rows, split=split, builder=builder, force=force_cache, limit=limit)
    if not builder_stats:
        builder_stats.update(
            {
                "dictionary_token_cache_hits": 0,
                "dictionary_token_cache_misses": 0,
                "dictionary_token_cache_hit_rate": 0.0,
                "dictionary_lexicon_size": _dictionary_lexicon_size_from_config(config),
                "dictionary_max_choices_per_token": int(
                    config.get("dictionary", {}).get("max_choices_per_token", 50_000)
                ),
            }
        )
    metadata = {
        **cache_result.report_metrics(),
        **builder_stats,
        "syntax_enabled_for_feature_build": bool((training_config.get("feature_build", {}) or {}).get("enable_syntax", False)),
    }
    return FeatureBuildRun(features=cache_result.features, cache_result=cache_result, metadata=metadata)


def _feature_build_tokenizer(config: dict[str, Any]) -> Any:
    model_config = config.get("model", {})
    tokenizer = DebugTokenizer()
    use_model_tokenizer = str(config.get("evaluation", {}).get("feature_tokenizer", "")).lower() == "model"
    if _run_model_training_enabled(config) or use_model_tokenizer:
        tokenizer = load_tokenizer(
            EncoderLoadConfig(
                model_name=model_config.get("primary_encoder", "ai-forever/ruRoberta-large"),
                fallback_model_name=model_config.get("fallback_encoder", "ai-forever/ruRoberta-large"),
                local_files_only=bool(model_config.get("local_files_only", False)),
            )
        )
    return tokenizer


def _feature_build_profiler(
    config: dict[str, Any],
    *,
    split: str,
    candidate_generator: CandidateGenerator,
) -> FeatureBuildProfiler:
    feature_build_config = config.get("training", {}).get("feature_build", {}) or {}
    reports_dir = Path(config.get("paths", {}).get("reports_dir", "reports"))
    profile_output = feature_build_config.get("profile_output")
    output_path = Path(profile_output) if profile_output else reports_dir / "feature_build_profile.csv"
    if not output_path.is_absolute() and profile_output is None:
        output_path = reports_dir / output_path.name
    summary_path = output_path.with_name("feature_build_profile_summary.md")
    return FeatureBuildProfiler(
        enabled=bool(feature_build_config.get("profile", False)),
        output_path=output_path,
        summary_path=summary_path,
        sample_size=int(feature_build_config.get("profile_sample_size", 2000)),
        split=split,
        syntax_enabled=bool(feature_build_config.get("enable_syntax", False)),
        dictionary_lexicon_size=_dictionary_lexicon_size(candidate_generator) if bool(feature_build_config.get("profile", False)) else 0,
    )


def _dictionary_policy_for_training_row(config: dict[str, Any]):
    feature_build_config = config.get("training", {}).get("feature_build", {}) or {}
    dictionary_for_clean_rows = bool(feature_build_config.get("dictionary_for_clean_rows", True))
    dictionary_for_hard_negative_rows = bool(feature_build_config.get("dictionary_for_hard_negative_rows", True))

    def policy(row: dict[str, Any]) -> str:
        source_type = str(row.get("source_type", ""))
        if source_type == "clean_identity_from_open_clean" and not dictionary_for_clean_rows:
            return "unknown_only"
        if source_type == "hard_negative_from_open_clean" and not dictionary_for_hard_negative_rows:
            return "unknown_only"
        return "all"

    return policy


def _dictionary_lexicon_size(candidate_generator: CandidateGenerator) -> int:
    try:
        return len(candidate_generator._dictionary_lexicon())
    except Exception:
        return 0


def _dictionary_lexicon_size_from_config(config: dict[str, Any]) -> int:
    dictionary_config = config.get("dictionary", {})
    if not bool(dictionary_config.get("enabled", False)):
        return 0
    path = Path(dictionary_config.get("lexicon_path") or "")
    if not path.exists() or not path.is_file():
        return 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except Exception:
        return 0


def _skip_feature_build_for_no_training(model_training_disabled_source: str) -> bool:
    return model_training_disabled_source == DISABLE_MODEL_TRAINING_ENV


def _run_model_training_with_fast_eval(
    config: dict[str, Any],
    features: list[Any],
    *,
    fast_eval_rows: list[dict[str, Any]],
    fast_eval_features: list[Any] | None,
) -> dict[str, Any]:
    try:
        return _run_model_training(
            config,
            features,
            fast_eval_rows=fast_eval_rows,
            fast_eval_features=fast_eval_features,
        )
    except TypeError as exc:
        if "unexpected keyword" not in str(exc):
            raise
        return _run_model_training(config, features)


def _run_model_training(
    config: dict[str, Any],
    features,
    *,
    fast_eval_rows: list[dict[str, Any]] | None = None,
    fast_eval_features: list[Any] | None = None,
) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    import torch
    from torch.utils.data import DataLoader

    model_config = config.get("model", {})
    training_config = config.get("training", {})
    lora_config = model_config.get("lora", {})
    edit_model = CandidateAwareEditModel(
        EditModelConfig(
            model_name=model_config.get("primary_encoder", "ai-forever/ruRoberta-large"),
            fallback_model_name=model_config.get("fallback_encoder", "ai-forever/ruRoberta-large"),
            punctuation_label_count=len(config.get("labels", {}).get("punctuation", {})),
            punctuation_action_count=len(config.get("labels", {}).get("punctuation_actions", {})) or 5,
            error_type_count=len(config.get("labels", {}).get("error_types", {})),
            local_files_only=bool(model_config.get("local_files_only", False)),
            lora_enabled=bool(lora_config.get("enabled", True)),
            lora_r=int(lora_config.get("r", 8)),
            lora_alpha=int(lora_config.get("alpha", 16)),
            lora_dropout=float(lora_config.get("dropout", 0.05)),
        )
    )
    module = edit_model.module
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.to(device)

    collator = EditBatchCollator()

    def _collate(batch):
        collated = collator.collate(batch)
        labels = {key: value.to(device) for key, value in collated["labels"].items()}
        return {
            "input_ids": collated["input_ids"].to(device),
            "attention_mask": collated["attention_mask"].to(device),
            "candidate_spans": collated["candidate_spans"].to(device),
            "candidate_mask": collated["candidate_mask"].to(device),
            "candidate_replacement_ids": collated["candidate_replacement_ids"].to(device),
            "candidate_replacement_mask": collated["candidate_replacement_mask"].to(device),
            "punctuation_gap_indices": collated["punctuation_gap_indices"].to(device),
            "punctuation_right_gap_indices": collated["punctuation_right_gap_indices"].to(device),
            "punctuation_gap_mask": collated["punctuation_gap_mask"].to(device),
            "labels": labels,
        }

    loader = DataLoader(
        features,
        batch_size=int(training_config.get("batch_size", 2)),
        shuffle=True,
        collate_fn=_collate,
    )
    micro_steps = len(loader)
    accumulation_steps = int(training_config.get("gradient_accumulation_steps", 4))
    optimizer_steps = (micro_steps + accumulation_steps - 1) // accumulation_steps
    print(
        "Training setup: "
        f"{len(features)} examples, {micro_steps} micro-steps/epoch, "
        f"{optimizer_steps} optimizer steps/epoch, accumulation={accumulation_steps}"
    )
    optimizer = torch.optim.AdamW(module.parameters(), lr=float(training_config.get("learning_rate", 2e-4)))
    trainer = EditModelTrainer(
        module,
        optimizer,
        TrainLoopConfig(
            epochs=int(training_config.get("epochs", 1)),
            gradient_accumulation_steps=accumulation_steps,
            mixed_precision=bool(training_config.get("mixed_precision", True)),
            show_progress=bool(training_config.get("show_progress", True)),
            progress_log_every_steps=int(training_config.get("progress_log_every_steps", 1000)),
        ),
        loss_weights=training_config,
    )
    losses: list[float] = []
    fast_evaluation_metrics: dict[str, float] = {}
    for epoch_index in range(int(training_config.get("epochs", 1))):
        losses.append(trainer.train_epoch(loader))
        if fast_eval_rows and fast_eval_features:
            fast_corrector = TrainedModelCorrector(
                TorchCandidateModelBackend(
                    tokenizer=None,
                    module=module,
                    device=device,
                    punctuation_labels=config.get("labels", {}).get("punctuation", {}),
                    punctuation_action_labels=config.get("labels", {}).get("punctuation_actions", {}),
                    max_length=int(model_config.get("max_sequence_length", 128)),
                    max_candidates=int(model_config.get("max_candidates", 16)),
                ),
                thresholds=_active_thresholds(config),
                max_passes=int(config.get("decoder", {}).get("max_passes", 3)),
                candidate_generator=CandidateGenerator.from_config(config),
            )
            fast_eval = evaluate_features_detailed(
                fast_eval_rows,
                fast_eval_features,
                corrector=fast_corrector,
                output_dir=None,
                metric_weights=config.get("metrics", {}).get("combined_score_weights"),
                report_metadata=None,
                report_options=training_fast_report_options(config),
                batch_size=evaluation_batch_size(config),
                mixed_precision=bool(training_config.get("mixed_precision", True)),
                feature_load_time_sec=0.0,
                feature_cache_hit=False,
                show_progress=bool(training_config.get("show_progress", False)),
            )
            fast_evaluation_metrics = core_training_metrics(fast_eval.evaluation.metrics)
            print(
                f"Fast eval epoch {epoch_index + 1}: "
                f"{len(fast_eval_rows)} examples, "
                f"combined_score={fast_evaluation_metrics.get('combined_score', 0.0):.6f}"
            )
    heads_output_dir = Path(config.get("paths", {}).get("heads_output_dir", "models/heads/latest"))
    heads_output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(module.heads.state_dict(), heads_output_dir / "heads.pt")
    if hasattr(module.encoder, "save_pretrained"):
        module.encoder.save_pretrained(config.get("paths", {}).get("adapter_output_dir", "models/adapters/latest"))
    components = trainer.last_epoch_loss_components
    return {
        "status": "trained",
        "model_training_ran": True,
        "train_loss": losses[-1] if losses else 0.0,
        "word_loss_value": float(components.get("word_loss", 0.0)),
        "punctuation_loss_value": float(components.get("punctuation_loss", 0.0)),
        "confidence_loss_value": float(components.get("confidence_loss", 0.0)),
        "error_type_loss_value": float(components.get("error_type_loss", 0.0)),
        "fast_evaluation_metrics": fast_evaluation_metrics,
    }


def _run_model_training_enabled(config: dict[str, Any]) -> bool:
    if os.environ.get(DISABLE_MODEL_TRAINING_ENV) == "1":
        return False
    value = config.get("training", {}).get("run_model_training", False)
    if not isinstance(value, bool):
        raise ValueError("training.run_model_training must be true or false, not a string/value alias")
    return value


def _model_training_disabled_source(config: dict[str, Any]) -> str:
    if os.environ.get(DISABLE_MODEL_TRAINING_ENV) == "1":
        return DISABLE_MODEL_TRAINING_ENV
    value = config.get("training", {}).get("run_model_training", False)
    if value is False:
        return "training.run_model_training"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Russian edit-based corrector")
    parser.add_argument("config", nargs="?", default="configs/config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    print(train(args.config))


if __name__ == "__main__":
    main()
