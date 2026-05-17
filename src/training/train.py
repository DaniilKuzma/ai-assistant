from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd

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
    result: dict[str, Any] = {
        "status": "features_prepared",
        "output_dir": output_dir,
        "feature_count": len(features),
        "model_training_ran": False,
    }

    if run_model_training:
        result.update(_run_model_training(config, features))

    save_training_artifacts(
        output_dir,
        config=config,
        label_mappings=config.get("labels", {}),
        thresholds=config.get("thresholds", {}),
    )
    evaluation_rows = _load_evaluation_rows(config, rows)
    corrector = _build_evaluation_corrector(config, bool(result["model_training_ran"]))
    print(f"Evaluating {len(evaluation_rows)} examples with {corrector.__class__.__name__}...")
    evaluation_result = evaluate_rows_detailed(
        evaluation_rows,
        corrector=corrector,
        output_dir=reports_dir,
        metric_weights=config.get("metrics", {}).get("combined_score_weights"),
        show_progress=bool(config.get("training", {}).get("show_progress", False)),
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
            "evaluation_count": float(len(evaluation_rows)),
            "model_training_ran": float(result["model_training_ran"]),
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
            "evaluation_metrics": evaluation_metrics,
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
    data_config = config.get("data", {})
    processed_path = data_config.get("processed_train_path")
    training_config = config.get("training", {})
    if processed_path and Path(processed_path).exists():
        frame = pd.read_csv(processed_path)
        if "split" not in frame.columns:
            max_examples = int(training_config.get("max_val_examples", len(frame)))
            return frame.head(max_examples).to_dict("records")
        val = frame[frame["split"] == "val"].head(int(training_config.get("max_val_examples", 0)))
        if not val.empty:
            return val.to_dict("records")
    return fallback_rows


def _build_evaluation_corrector(config: dict[str, Any], model_training_ran: bool):
    if model_training_ran:
        return TrainedModelCorrector.from_config(config)
    return Corrector()


def _report_paths(reports_dir: Path) -> dict[str, str]:
    names = [
        "dataset_report.md",
        "training_report.md",
        "evaluation_summary.csv",
        "error_by_type.csv",
        "clean_overcorrection_examples.csv",
        "dirty_worse_examples.csv",
        "accepted_edits.csv",
        "rejected_edits.csv",
        "loss_curves.png",
        "threshold_precision_recall.png",
    ]
    return {name: str(reports_dir / name) for name in names if (reports_dir / name).exists()}


def _build_features(config: dict[str, Any], rows: list[dict[str, Any]]):
    training_config = config.get("training", {})
    model_config = config.get("model", {})
    label_config = config.get("labels", {})
    tokenizer = DebugTokenizer()
    if _run_model_training_enabled(config):
        tokenizer = load_tokenizer(
            EncoderLoadConfig(
                model_name=model_config.get("primary_encoder", "ai-forever/ruRoberta-large"),
                fallback_model_name=model_config.get("fallback_encoder", "ai-forever/ruBert-base"),
                local_files_only=bool(model_config.get("local_files_only", False)),
            )
        )
    return build_features_from_rows(
        rows,
        tokenizer=tokenizer,
        punctuation_label_map=label_config.get("punctuation", {}),
        error_type_label_map=label_config.get("error_types", {}),
        max_length=int(model_config.get("max_sequence_length", 192)),
        max_candidates=int(model_config.get("max_candidates", 32)),
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
            fallback_model_name=model_config.get("fallback_encoder", "ai-forever/ruBert-base"),
            punctuation_label_count=len(config.get("labels", {}).get("punctuation", {})),
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
    if os.environ.get("RUSSIAN_CORRECTOR_DISABLE_MODEL_TRAINING") == "1":
        return False
    value = config.get("training", {}).get("run_model_training", False)
    if not isinstance(value, bool):
        raise ValueError("training.run_model_training must be true or false, not a string/value alias")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Russian edit-based corrector")
    parser.add_argument("config", nargs="?", default="configs/config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    print(train(args.config))


if __name__ == "__main__":
    main()
