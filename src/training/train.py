from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.candidates.candidate_generator import CandidateGenerator
from src.config.load_config import load_config
from src.data.dataset_builder import build_synthetic_dataset
from src.data.dataset_stats import dataset_stats
from src.evaluation.reports import write_loss_curve, write_threshold_precision_recall_plot, write_training_report
from src.evaluation.evaluate import evaluate_rows_detailed
from src.evaluation.reports import write_dataset_report
from src.evaluation.threshold_sweep import threshold_sweep
from src.inference.corrector import Corrector
from src.inference.model_corrector import TrainedModelCorrector
from src.model.edit_model import CandidateAwareEditModel, EditModelConfig
from src.model.encoder import EncoderLoadConfig, load_tokenizer
from src.training.save_load import save_training_artifacts
from src.training.tensorization import DebugTokenizer, EditBatchCollator, build_features_from_rows
from src.training.callbacks import BestMetricTracker
from src.training.trainer import EditModelTrainer, TrainLoopConfig


DISABLE_MODEL_TRAINING_ENV = "RUSSIAN_CORRECTOR_DISABLE_MODEL_TRAINING"


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
    evaluation_result = evaluate_rows_detailed(
        rows,
        corrector=evaluation_corrector,
        output_dir=reports_dir,
        metric_weights=config.get("metrics", {}).get("combined_score_weights"),
        show_progress=bool(config.get("training", {}).get("show_progress", False)),
        report_metadata={
            "evaluation_backend": "provided_corrector" if corrector is not None else "freshly_trained_model",
            "model_training_disabled": False,
            "model_training_disabled_source": "",
        },
        candidate_generator=_candidate_recall_generator(evaluation_corrector, config),
        candidate_recall_max_candidates=_candidate_recall_max_candidates(config),
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
        "report_paths": _report_paths(reports_dir),
    }


def train(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Prepare data and optionally run encoder fine-tuning."""

    config = load_config(config_path)
    reports_dir = Path(config.get("paths", {}).get("reports_dir", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_training_rows(config)
    write_dataset_report(_full_dataset_stats(config, rows), reports_dir / "dataset_report.md")

    features = _build_features(config, rows)
    output_dir = config.get("paths", {}).get("adapter_output_dir", "models/adapters/latest")

    run_model_training = _run_model_training_enabled(config)
    model_training_disabled_source = _model_training_disabled_source(config)
    model_training_disabled = bool(model_training_disabled_source)
    result: dict[str, Any] = {
        "status": "features_prepared",
        "output_dir": output_dir,
        "feature_count": len(features),
        "model_training_ran": False,
        "model_training_disabled": model_training_disabled,
        "model_training_disabled_source": model_training_disabled_source,
    }

    if run_model_training:
        result.update(_run_model_training(config, features))

    save_training_artifacts(
        output_dir,
        config=config,
        label_mappings=config.get("labels", {}),
        thresholds=config.get("thresholds", {}),
    )
    evaluation_split = _training_evaluation_split(config)
    evaluation_rows = _load_evaluation_rows(config, rows)
    corrector, backend_metadata = _select_evaluation_corrector(
        config,
        model_training_ran=bool(result["model_training_ran"]),
        model_training_disabled_source=model_training_disabled_source,
    )
    print(
        f"Evaluating {len(evaluation_rows)} {evaluation_split} examples "
        f"with {corrector.__class__.__name__} ({backend_metadata['evaluation_backend']})..."
    )
    evaluation_result = evaluate_rows_detailed(
        evaluation_rows,
        corrector=corrector,
        output_dir=reports_dir,
        metric_weights=config.get("metrics", {}).get("combined_score_weights"),
        show_progress=bool(config.get("training", {}).get("show_progress", False)),
        report_metadata=backend_metadata,
        candidate_generator=_candidate_recall_generator(corrector, config),
        candidate_recall_max_candidates=_candidate_recall_max_candidates(config),
    )
    evaluation_metrics = evaluation_result.metrics
    checkpoint_metric = str(config.get("training", {}).get("checkpoint_metric", "combined_score"))
    tracker = BestMetricTracker(checkpoint_metric)
    is_best_checkpoint = tracker.update(evaluation_metrics)
    checkpoint_metric_value = float(evaluation_metrics.get(checkpoint_metric, 0.0))
    losses = [float(result.get("train_loss", 0.0))]
    write_loss_curve(losses, reports_dir / "loss_curves.png")
    write_threshold_precision_recall_plot(
        threshold_sweep(evaluation_result.edit_scores, [0.5, 0.7, 0.8, 0.9, 0.95]),
        reports_dir / "threshold_precision_recall.png",
    )
    write_training_report(
        {
            "feature_count": float(len(features)),
            "evaluation_split": evaluation_split,
            "evaluation_count": float(len(evaluation_rows)),
            "model_training_ran": float(result["model_training_ran"]),
            "model_training_disabled": model_training_disabled,
            "model_training_disabled_source": model_training_disabled_source,
            **backend_metadata,
            "checkpoint_metric": checkpoint_metric,
            "checkpoint_metric_value": checkpoint_metric_value,
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
            "checkpoint_metric": checkpoint_metric,
            "checkpoint_metric_value": checkpoint_metric_value,
            "is_best_checkpoint": is_best_checkpoint,
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


def _load_evaluation_rows(config: dict[str, Any], fallback_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _load_rows_for_split(config, split=_training_evaluation_split(config), fallback_rows=fallback_rows)


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
            "evaluation_backend": "freshly_trained_model",
            "checkpoint_load_error": "",
        }

    if model_training_disabled_source and _configured_model_artifacts_exist(config):
        try:
            return TrainedModelCorrector.from_config(config), {
                **metadata,
                "evaluation_backend": "existing_model_checkpoint",
                "checkpoint_load_error": "",
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
    training_config = config.get("training", {})
    model_config = config.get("model", {})
    label_config = config.get("labels", {})
    tokenizer = DebugTokenizer()
    if _run_model_training_enabled(config):
        tokenizer = load_tokenizer(
            EncoderLoadConfig(
                model_name=model_config.get("primary_encoder", "ai-forever/ruRoberta-large"),
                fallback_model_name=model_config.get("fallback_encoder", "ai-forever/ruRoberta-large"),
                local_files_only=bool(model_config.get("local_files_only", False)),
            )
        )
    candidate_generator = CandidateGenerator.from_config(config)
    return build_features_from_rows(
        rows,
        tokenizer=tokenizer,
        punctuation_label_map=label_config.get("punctuation", {}),
        punctuation_action_label_map=label_config.get("punctuation_actions", {}),
        error_type_label_map=label_config.get("error_types", {}),
        max_length=int(model_config.get("max_sequence_length", 192)),
        max_candidates=int(model_config.get("max_candidates", 32)),
        candidate_generator=candidate_generator,
        show_progress=bool(training_config.get("show_progress", False)),
    )


def _run_model_training(config: dict[str, Any], features) -> dict[str, Any]:  # type: ignore[no-untyped-def]
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
    losses = [trainer.train_epoch(loader) for _ in range(int(training_config.get("epochs", 1)))]
    heads_output_dir = Path(config.get("paths", {}).get("heads_output_dir", "models/heads/latest"))
    heads_output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(module.heads.state_dict(), heads_output_dir / "heads.pt")
    if hasattr(module.encoder, "save_pretrained"):
        module.encoder.save_pretrained(config.get("paths", {}).get("adapter_output_dir", "models/adapters/latest"))
    return {"status": "trained", "model_training_ran": True, "train_loss": losses[-1] if losses else 0.0}


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
