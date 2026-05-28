from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import json
import math
import shutil
import sys

import torch

from src.config.load_config import load_config
from src.model.edit_model import DirectEditModelConfig, DirectEditTaggerModel
from src.progress import ProgressReporter
from src.schema.labels import (
    BOUNDARY_AFTER_ID_TO_LABEL,
    BOUNDARY_AFTER_LABEL_TO_ID,
    BOUNDARY_BEFORE_ID_TO_LABEL,
    BOUNDARY_BEFORE_LABEL_TO_ID,
    GAP_ID_TO_LABEL,
    GAP_LABEL_TO_ID,
    RULE_ID_TO_LABEL,
    RULE_LABEL_TO_ID,
    TOKEN_ID_TO_LABEL,
    TOKEN_LABEL_TO_ID,
)
from src.training.online_dataset import FrozenJsonlDataset, OnlineGrammarDataset
from src.training.tensorization import DebugTokenizer, DirectBatchCollator


def train_model(
    config_path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
    debug_model: bool = False,
) -> dict[str, Any]:
    from torch.utils.data import DataLoader

    config = load_config(config_path)
    _apply_overrides(config, overrides or {})
    if (overrides or {}).get("smoke"):
        _apply_smoke_config(config, overrides or {})
    config["_debug_model"] = bool(debug_model)

    seed = int(config.get("generation", {}).get("seed", 13))
    torch.manual_seed(seed)

    print("[training] loading tokenizer and model", file=sys.stderr, flush=True)
    tokenizer = DebugTokenizer() if debug_model else _load_configured_tokenizer(config)
    model = _build_model(config, debug_model=debug_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    training = config.setdefault("training", {})
    generation = config.setdefault("generation", {})
    batch_size = max(1, int(training.get("batch_size", 32)))
    eval_batch_size = max(1, int(training.get("eval_batch_size", batch_size)))
    epochs = max(1, int(training.get("epochs", 1)))
    samples_per_epoch = max(1, int(generation.get("samples_per_epoch", batch_size)))
    gradient_accumulation_steps = max(1, int(training.get("gradient_accumulation_steps", 1)))
    steps_per_epoch = math.ceil(samples_per_epoch / batch_size)
    total_train_steps = steps_per_epoch * epochs
    total_optimizer_steps = math.ceil(steps_per_epoch / gradient_accumulation_steps) * epochs
    num_workers = max(0, int(training.get("num_workers", 0)))

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training.get("learning_rate", 5.0e-5)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    scheduler = _linear_warmup_scheduler(
        optimizer,
        total_steps=total_optimizer_steps,
        warmup_ratio=float(training.get("warmup_ratio", 0.0)),
    )
    amp_enabled = bool(training.get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    collator = DirectBatchCollator()
    train_dataset = OnlineGrammarDataset(config, tokenizer, split="train", seed=seed)
    print(
        "[training] "
        f"device={device} cuda_available={torch.cuda.is_available()} "
        f"epochs={epochs} samples_per_epoch={samples_per_epoch} "
        f"batch_size={batch_size} steps_per_epoch={steps_per_epoch} "
        f"total_train_steps={total_train_steps} amp={amp_enabled}",
        file=sys.stderr,
        flush=True,
    )

    loss_history: list[dict[str, float | int]] = []
    validation_history: list[dict[str, Any]] = []
    best_heads_state: dict[str, Any] | None = None
    last_heads_state: dict[str, Any] | None = None
    batches_seen = 0

    for epoch in range(1, epochs + 1):
        train_dataset.set_epoch(epoch - 1)
        dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=collator.collate,
            num_workers=num_workers,
        )
        epoch_result = _train_epoch(
            model,
            dataloader,
            optimizer,
            scheduler,
            scaler,
            device,
            training,
            steps_per_epoch,
            amp_enabled=amp_enabled,
            progress_label=f"train epoch {epoch}/{epochs}",
        )
        batches_seen += int(epoch_result["steps"])
        loss_history.append({"epoch": epoch, **epoch_result})
        last_heads_state = _heads_state_dict(model)

        if bool(training.get("validate_each_epoch", True)):
            validation = _validate(
                model,
                config,
                tokenizer,
                collator,
                device,
                eval_batch_size=eval_batch_size,
                amp_enabled=amp_enabled,
                progress_label=f"validate epoch {epoch}/{epochs}",
            )
            validation["epoch"] = epoch
        else:
            validation = {"epoch": epoch, "status": "skipped", "reason": "validate_each_epoch=false"}
        validation_history.append(validation)

        selected_so_far = _select_best_checkpoint(validation_history, fallback_epoch=epoch)
        if int(selected_so_far["selected_epoch"]) == epoch:
            best_heads_state = _heads_state_dict(model)

        if bool(training.get("save_each_epoch", False)) or epochs == 1:
            _save_epoch_checkpoint(model, config, epoch, validation=validation)

    if best_heads_state is None:
        best_heads_state = last_heads_state or _heads_state_dict(model)

    selected = _select_best_checkpoint(validation_history, fallback_epoch=epochs)
    selected_epoch = int(selected["selected_epoch"])
    summary = {
        "epochs": epochs,
        "samples_per_epoch": samples_per_epoch,
        "batch_size": batch_size,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "total_train_steps": total_train_steps,
        "actual_train_steps": batches_seen,
        "best_epoch": selected_epoch,
        "best_metric": selected["selected_metric"],
        "selected_epoch": selected_epoch,
        "selected_metric": selected["selected_metric"],
        "selected_validation_loss": selected["selected_validation_loss"],
        "loss_history": loss_history,
        "validation_history": validation_history,
        "debug_model": bool(debug_model),
        "smoke": bool((overrides or {}).get("smoke", False)),
    }
    artifact_paths = _save_final_artifacts(model, config, best_heads_state, summary)
    return {**summary, **artifact_paths}


def _train_epoch(
    model: Any,
    dataloader: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    device: Any,
    training: Mapping[str, Any],
    steps_per_epoch: int,
    *,
    amp_enabled: bool,
    progress_label: str = "train",
    progress_sink: Any | None = None,
    progress_interval_seconds: float | None = None,
    progress_step_interval: int | None = None,
) -> dict[str, float | int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {
        "loss": 0.0,
        "token_edit_loss": 0.0,
        "gap_punctuation_loss": 0.0,
        "boundary_before_loss": 0.0,
        "boundary_after_loss": 0.0,
        "rule_loss": 0.0,
    }
    gradient_accumulation_steps = max(1, int(training.get("gradient_accumulation_steps", 1)))
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    pending_step = False
    steps = 0
    reporter = ProgressReporter(
        progress_label,
        total=steps_per_epoch,
        sink=progress_sink or _stderr_progress_sink,
        min_interval_seconds=_progress_interval_seconds(training, progress_interval_seconds),
        step_interval=_progress_step_interval(training, progress_step_interval, steps_per_epoch),
    )
    reporter.start()

    for step, batch in enumerate(dataloader, start=1):
        if step > steps_per_epoch:
            break
        inputs, labels = _move_batch(batch, device)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            outputs = model(**inputs)
            loss, components = _direct_losses(outputs, labels, training)
            scaled_loss = loss / gradient_accumulation_steps
        scaler.scale(scaled_loss).backward()
        pending_step = True

        for key in totals:
            value = loss if key == "loss" else components[key]
            totals[key] += float(value.detach().cpu())
        steps += 1
        reporter.update(
            steps,
            {
                "loss": totals["loss"] / max(1, steps),
                "token": totals["token_edit_loss"] / max(1, steps),
                "gap": totals["gap_punctuation_loss"] / max(1, steps),
                "before": totals["boundary_before_loss"] / max(1, steps),
                "after": totals["boundary_after_loss"] / max(1, steps),
                "rule": totals["rule_loss"] / max(1, steps),
                "lr": _current_lr(optimizer),
            },
        )

        if step % gradient_accumulation_steps == 0:
            _optimizer_step(model, optimizer, scheduler, scaler, max_grad_norm, amp_enabled)
            pending_step = False

    if pending_step:
        _optimizer_step(model, optimizer, scheduler, scaler, max_grad_norm, amp_enabled)

    denominator = max(1, steps)
    reporter.finish(
        {
            "loss": totals["loss"] / denominator,
            "token": totals["token_edit_loss"] / denominator,
            "gap": totals["gap_punctuation_loss"] / denominator,
            "before": totals["boundary_before_loss"] / denominator,
            "after": totals["boundary_after_loss"] / denominator,
            "rule": totals["rule_loss"] / denominator,
            "lr": _current_lr(optimizer),
        }
    )
    return {
        "steps": steps,
        "loss": totals["loss"] / denominator,
        "token_edit_loss": totals["token_edit_loss"] / denominator,
        "gap_punctuation_loss": totals["gap_punctuation_loss"] / denominator,
        "boundary_before_loss": totals["boundary_before_loss"] / denominator,
        "boundary_after_loss": totals["boundary_after_loss"] / denominator,
        "rule_loss": totals["rule_loss"] / denominator,
    }


def _validate(
    model: Any,
    config: Mapping[str, Any],
    tokenizer: Any,
    collator: DirectBatchCollator,
    device: Any,
    *,
    eval_batch_size: int,
    amp_enabled: bool,
    progress_label: str = "validate",
) -> dict[str, Any]:
    from torch.utils.data import DataLoader

    dataset = FrozenJsonlDataset(config, tokenizer, split="val")
    if len(dataset) == 0:
        return {
            "status": "skipped",
            "reason": f"missing_or_empty_eval_jsonl:{dataset.path}",
            "exact_match": None,
            "combined_score": None,
            "loss": None,
        }

    model.eval()
    total_loss = 0.0
    total_examples = 0
    exact_matches = 0
    dataloader = DataLoader(dataset, batch_size=eval_batch_size, collate_fn=collator.collate)
    reporter = ProgressReporter(
        progress_label,
        total=len(dataset),
        sink=_stderr_progress_sink,
        min_interval_seconds=_progress_interval_seconds(config.get("training", {}), None),
        step_interval=_progress_step_interval(config.get("training", {}), None, len(dataset)),
    )
    reporter.start()
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = _move_batch(batch, device)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(**inputs)
                loss, _components = _direct_losses(outputs, labels, config.get("training", {}))
            total_loss += float(loss.detach().cpu())
            exact = _batch_exact_matches(outputs, labels)
            exact_matches += int(exact.sum().detach().cpu().item())
            total_examples += int(inputs["input_ids"].shape[0])
            reporter.update(
                total_examples,
                {
                    "loss": total_loss / max(1, len(dataloader)),
                    "exact": exact_matches / max(1, total_examples),
                },
            )

    exact_match = exact_matches / max(1, total_examples)
    reporter.finish({"loss": total_loss / max(1, len(dataloader)), "exact": exact_match})
    return {
        "status": "ok",
        "examples": total_examples,
        "loss": total_loss / max(1, len(dataloader)),
        "exact_match": exact_match,
        "combined_score": exact_match,
    }


def _direct_losses(
    outputs: Mapping[str, Any],
    labels: Mapping[str, Any],
    training: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    token_edit_loss = _cross_entropy_ignore(
        outputs["token_edit_logits"],
        labels["token_edit_label_ids"],
        labels.get("sample_weight"),
    )
    gap_loss = _cross_entropy_ignore(
        outputs["gap_punctuation_logits"],
        labels["gap_label_ids"],
        labels.get("sample_weight"),
    )
    boundary_before_loss = _optional_cross_entropy_ignore(
        outputs,
        labels,
        "boundary_before_logits",
        "boundary_before_label_ids",
    )
    boundary_after_loss = _optional_cross_entropy_ignore(
        outputs,
        labels,
        "boundary_after_logits",
        "boundary_after_label_ids",
    )
    rule_loss = _cross_entropy_ignore(
        outputs["rule_logits"],
        labels["rule_tag_ids"],
        labels.get("sample_weight"),
    )
    total = (
        float(training.get("token_edit_loss_weight", 1.0)) * token_edit_loss
        + float(training.get("gap_punctuation_loss_weight", 1.0)) * gap_loss
        + float(training.get("boundary_before_loss_weight", 1.0)) * boundary_before_loss
        + float(training.get("boundary_after_loss_weight", 1.0)) * boundary_after_loss
        + float(training.get("rule_loss_weight", 1.0)) * rule_loss
    )
    return total, {
        "token_edit_loss": token_edit_loss,
        "gap_punctuation_loss": gap_loss,
        "boundary_before_loss": boundary_before_loss,
        "boundary_after_loss": boundary_after_loss,
        "rule_loss": rule_loss,
    }


def _optional_cross_entropy_ignore(
    outputs: Mapping[str, Any],
    labels: Mapping[str, Any],
    logits_key: str,
    labels_key: str,
) -> Any:
    logits = outputs.get(logits_key)
    target = labels.get(labels_key)
    if logits is None:
        reference = next(iter(outputs.values()))
        return reference.new_tensor(0.0) if hasattr(reference, "new_tensor") else 0.0
    if target is None:
        return logits.new_tensor(0.0)
    return _cross_entropy_ignore(logits, target, labels.get("sample_weight"))


def _cross_entropy_ignore(logits: Any, labels: Any, sample_weight: Any | None) -> Any:
    import torch.nn.functional as functional

    flat_logits = logits.reshape(-1, logits.shape[-1])
    labels = labels.to(logits.device)
    flat_labels = labels.reshape(-1)
    valid = flat_labels.ne(-100)
    if not valid.any():
        return logits.new_tensor(0.0)

    losses = functional.cross_entropy(flat_logits[valid], flat_labels[valid], reduction="none")
    if sample_weight is None:
        return losses.mean()

    weights = sample_weight.to(device=logits.device, dtype=losses.dtype)
    while weights.ndim < labels.ndim:
        weights = weights.unsqueeze(-1)
    flat_weights = weights.expand(labels.shape).reshape(-1)[valid]
    return (losses * flat_weights).sum() / flat_weights.sum().clamp_min(1.0)


def _batch_exact_matches(outputs: Mapping[str, Any], labels: Mapping[str, Any]) -> Any:
    token_exact = _head_exact(outputs["token_edit_logits"], labels["token_edit_label_ids"])
    gap_exact = _head_exact(outputs["gap_punctuation_logits"], labels["gap_label_ids"])
    boundary_before_exact = _optional_head_exact(outputs, labels, "boundary_before_logits", "boundary_before_label_ids")
    boundary_after_exact = _optional_head_exact(outputs, labels, "boundary_after_logits", "boundary_after_label_ids")
    rule_exact = _head_exact(outputs["rule_logits"], labels["rule_tag_ids"])
    return token_exact & gap_exact & boundary_before_exact & boundary_after_exact & rule_exact


def _optional_head_exact(outputs: Mapping[str, Any], labels: Mapping[str, Any], logits_key: str, labels_key: str) -> Any:
    logits = outputs.get(logits_key)
    target = labels.get(labels_key)
    if logits is None or target is None:
        reference = next(iter(outputs.values()))
        import torch

        return torch.ones(reference.shape[0], dtype=torch.bool, device=reference.device)
    return _head_exact(logits, target)


def _head_exact(logits: Any, labels: Any) -> Any:
    predictions = logits.argmax(dim=-1).to(labels.device)
    valid = labels.ne(-100)
    return (predictions.eq(labels) | ~valid).all(dim=1)


def _move_batch(batch: Mapping[str, Any], device: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = {
        key: value.to(device)
        for key, value in batch["labels"].items()
    }
    inputs = {
        key: value.to(device)
        for key, value in batch.items()
        if key != "labels"
    }
    return inputs, labels


def _optimizer_step(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    max_grad_norm: float,
    amp_enabled: bool,
) -> None:
    if amp_enabled:
        scaler.unscale_(optimizer)
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    scale_before = float(scaler.get_scale()) if amp_enabled and hasattr(scaler, "get_scale") else None
    scaler.step(optimizer)
    scaler.update()
    optimizer_stepped = True
    if scale_before is not None and hasattr(scaler, "get_scale"):
        optimizer_stepped = float(scaler.get_scale()) >= scale_before
    if optimizer_stepped:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def _current_lr(optimizer: Any) -> float:
    param_groups = getattr(optimizer, "param_groups", [])
    if not param_groups:
        return 0.0
    return float(param_groups[0].get("lr", 0.0))


def _progress_interval_seconds(training: Mapping[str, Any], override: float | None) -> float:
    if override is not None:
        return max(0.0, float(override))
    return max(0.0, float(training.get("progress_log_interval_seconds", 30.0)))


def _progress_step_interval(training: Mapping[str, Any], override: int | None, total_steps: int) -> int:
    if override is not None:
        return max(1, int(override))
    configured = training.get("progress_log_interval_steps")
    if configured is not None:
        return max(1, int(configured))
    return max(1, max(1, int(total_steps)) // 100)


def _stderr_progress_sink(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def _linear_warmup_scheduler(optimizer: Any, *, total_steps: int, warmup_ratio: float) -> Any:
    from torch.optim.lr_scheduler import LambdaLR

    warmup_steps = max(0, int(total_steps * max(0.0, warmup_ratio)))

    def lr_lambda(current_step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        step = current_step + 1
        if step <= warmup_steps:
            return step / warmup_steps
        return 1.0

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _load_configured_tokenizer(config: Mapping[str, Any]) -> Any:
    from src.model import encoder as encoder_module
    from src.model.encoder import EncoderLoadConfig

    model_config = config.get("model", {}) if isinstance(config, Mapping) else {}
    return encoder_module.load_tokenizer(
        EncoderLoadConfig(
            model_name=str(model_config.get("encoder") or model_config.get("primary_encoder") or "ai-forever/ruRoberta-large"),
            fallback_model_name=str(
                model_config.get("fallback_encoder")
                or model_config.get("encoder")
                or model_config.get("primary_encoder")
                or "ai-forever/ruRoberta-large"
            ),
            local_files_only=bool(model_config.get("local_files_only", False)),
        )
    )


def _build_model(config: Mapping[str, Any], *, debug_model: bool) -> Any:
    if debug_model:
        return DirectEditTaggerModel.from_encoder(
            _TinyDebugEncoder(),
            DirectEditModelConfig(lora_enabled=False),
        ).module
    return DirectEditTaggerModel(_direct_model_config(config)).module


def _direct_model_config(config: Mapping[str, Any]) -> DirectEditModelConfig:
    model_config = config.get("model", {}) if isinstance(config, Mapping) else {}
    lora = model_config.get("lora", {}) if isinstance(model_config, Mapping) else {}
    return DirectEditModelConfig(
        model_name=str(model_config.get("encoder") or model_config.get("primary_encoder") or "ai-forever/ruRoberta-large"),
        fallback_model_name=str(
            model_config.get("fallback_encoder")
            or model_config.get("encoder")
            or model_config.get("primary_encoder")
            or "ai-forever/ruRoberta-large"
        ),
        local_files_only=bool(model_config.get("local_files_only", False)),
        lora_enabled=bool(lora.get("enabled", True)) if isinstance(lora, Mapping) else True,
        lora_r=int(lora.get("r", 8)) if isinstance(lora, Mapping) else 8,
        lora_alpha=int(lora.get("alpha", 16)) if isinstance(lora, Mapping) else 16,
        lora_dropout=float(lora.get("dropout", 0.05)) if isinstance(lora, Mapping) else 0.05,
    )


class _TinyDebugEncoder(torch.nn.Module):
    def __init__(self, hidden_size: int = 32, vocab_size: int = 32768) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.vocab_size = vocab_size
        self.embeddings = torch.nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids: Any, attention_mask: Any | None = None, **_: Any) -> Any:
        input_ids = input_ids.clamp(min=0, max=self.vocab_size - 1)
        return SimpleNamespace(last_hidden_state=self.embeddings(input_ids))


def _apply_overrides(config: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    training = config.setdefault("training", {})
    generation = config.setdefault("generation", {})
    mapping = {
        "epochs": (training, "epochs"),
        "batch_size": (training, "batch_size"),
        "learning_rate": (training, "learning_rate"),
        "gradient_accumulation_steps": (training, "gradient_accumulation_steps"),
        "mixed_precision": (training, "mixed_precision"),
        "samples_per_epoch": (generation, "samples_per_epoch"),
    }
    for override_key, (section, config_key) in mapping.items():
        if override_key in overrides and overrides[override_key] is not None:
            section[config_key] = overrides[override_key]
    if "steps" in overrides and overrides["steps"] is not None:
        training["smoke_steps"] = int(overrides["steps"])


def _apply_smoke_config(config: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    training = config.setdefault("training", {})
    generation = config.setdefault("generation", {})
    batch_size = max(1, int(training.get("batch_size", 1)))
    steps = max(1, int(overrides.get("steps") or training.get("smoke_steps", 2)))
    training["epochs"] = 1
    training["smoke_steps"] = steps
    generation["samples_per_epoch"] = steps * batch_size


def _heads_state_dict(model: Any) -> dict[str, Any]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.heads.state_dict().items()
    }


def _select_best_checkpoint(validation_history: list[dict[str, Any]], *, fallback_epoch: int) -> dict[str, Any]:
    candidates: list[tuple[float, float, int, dict[str, Any]]] = []
    for row in validation_history:
        metric = row.get("exact_match")
        if not isinstance(metric, (float, int)):
            continue
        loss = row.get("loss")
        loss_value = float(loss) if isinstance(loss, (float, int)) else float("inf")
        epoch = int(row.get("epoch") or fallback_epoch)
        candidates.append((float(metric), -loss_value, epoch, row))

    if not candidates:
        return {
            "selected_epoch": int(fallback_epoch),
            "selected_metric": None,
            "selected_validation_loss": None,
        }

    metric, negative_loss, epoch, _row = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    return {
        "selected_epoch": int(epoch),
        "selected_metric": float(metric),
        "selected_validation_loss": float(-negative_loss),
    }


def _save_epoch_checkpoint(
    model: Any,
    config: Mapping[str, Any],
    epoch: int,
    *,
    validation: Mapping[str, Any] | None = None,
) -> None:
    import torch

    checkpoint_dir = _checkpoint_dir(config, epoch)
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(_heads_state_dict(model), checkpoint_dir / "heads.pt")
    (checkpoint_dir / "labels.json").write_text(
        json.dumps(_label_maps(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (checkpoint_dir / "architecture.json").write_text(
        json.dumps(
            {
                "architecture": "direct_edit_tagger_v1",
                "debug_model": bool(config.get("_debug_model", False)),
                "epoch": int(epoch),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    adapter_saved = _save_adapter_checkpoint(model, checkpoint_dir / "adapters")
    (checkpoint_dir / "checkpoint_meta.json").write_text(
        json.dumps(
            {
                "epoch": int(epoch),
                "heads_epoch": int(epoch),
                "adapter_epoch": int(epoch),
                "adapter_saved": adapter_saved,
                "validation": dict(validation or {}),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _save_final_artifacts(
    model: Any,
    config: Mapping[str, Any],
    heads_state: dict[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, str]:
    import torch

    selected_epoch = int(summary.get("selected_epoch") or summary.get("best_epoch") or 1)
    checkpoint_dir = _checkpoint_dir(config, selected_epoch)
    if not checkpoint_dir.exists():
        _write_checkpoint_from_state(model, config, selected_epoch, heads_state, summary)
    return _copy_selected_checkpoint_to_latest(config, selected_epoch=selected_epoch, summary=summary)


def _write_checkpoint_from_state(
    model: Any,
    config: Mapping[str, Any],
    epoch: int,
    heads_state: dict[str, Any],
    summary: Mapping[str, Any],
) -> None:
    import torch

    checkpoint_dir = _checkpoint_dir(config, epoch)
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(heads_state, checkpoint_dir / "heads.pt")
    (checkpoint_dir / "labels.json").write_text(
        json.dumps(_label_maps(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (checkpoint_dir / "architecture.json").write_text(
        json.dumps(
            {
                "architecture": "direct_edit_tagger_v1",
                "debug_model": bool(summary.get("debug_model", False)),
                "epoch": int(epoch),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    adapter_saved = _save_adapter_checkpoint(model, checkpoint_dir / "adapters")
    (checkpoint_dir / "checkpoint_meta.json").write_text(
        json.dumps(
            {
                "epoch": int(epoch),
                "heads_epoch": int(epoch),
                "adapter_epoch": int(epoch),
                "adapter_saved": adapter_saved,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _copy_selected_checkpoint_to_latest(
    config: Mapping[str, Any],
    *,
    selected_epoch: int,
    summary: Mapping[str, Any],
) -> dict[str, str]:
    checkpoint_dir = _checkpoint_dir(config, selected_epoch)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Selected checkpoint is missing: {checkpoint_dir}")

    paths = config.get("paths", {}) if isinstance(config, Mapping) else {}
    adapter_dir = Path(str(paths.get("adapter_output_dir", "models/adapters/latest")))
    heads_dir = Path(str(paths.get("heads_output_dir", "models/heads/latest")))
    heads_dir.mkdir(parents=True, exist_ok=True)

    for filename in ("heads.pt", "labels.json"):
        shutil.copy2(checkpoint_dir / filename, heads_dir / filename)

    architecture = json.loads((checkpoint_dir / "architecture.json").read_text(encoding="utf-8"))
    architecture["selected_epoch"] = int(selected_epoch)
    (heads_dir / "architecture.json").write_text(
        json.dumps(architecture, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    checkpoint_meta = json.loads((checkpoint_dir / "checkpoint_meta.json").read_text(encoding="utf-8"))
    checkpoint_meta["selected_epoch"] = int(selected_epoch)
    checkpoint_meta["heads_epoch"] = int(checkpoint_meta.get("heads_epoch") or selected_epoch)
    checkpoint_meta["adapter_epoch"] = int(checkpoint_meta.get("adapter_epoch") or selected_epoch)
    artifact_consistent = checkpoint_meta["heads_epoch"] == checkpoint_meta["adapter_epoch"] == int(selected_epoch)
    checkpoint_meta["artifact_consistency_check"] = artifact_consistent
    (heads_dir / "checkpoint_meta.json").write_text(
        json.dumps(checkpoint_meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (heads_dir / "selected_checkpoint_meta.json").write_text(
        json.dumps(
            {
                **checkpoint_meta,
                "selected_epoch": int(selected_epoch),
                "selected_checkpoint_dir": str(checkpoint_dir),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    source_adapter_dir = checkpoint_dir / "adapters"
    if source_adapter_dir.exists() and any(source_adapter_dir.iterdir()):
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
        shutil.copytree(source_adapter_dir, adapter_dir)
    elif adapter_dir.exists():
        shutil.rmtree(adapter_dir)

    summary_dict = dict(summary)
    summary_dict["selected_epoch"] = int(selected_epoch)
    summary_dict["artifact_consistency_check"] = artifact_consistent
    if isinstance(summary, dict):
        summary["artifact_consistency_check"] = artifact_consistent
    summary_path = heads_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps(summary_dict, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {
        "heads_path": str(heads_dir / "heads.pt"),
        "labels_path": str(heads_dir / "labels.json"),
        "summary_path": str(summary_path),
        "architecture_path": str(heads_dir / "architecture.json"),
        "checkpoint_meta_path": str(heads_dir / "checkpoint_meta.json"),
    }


def _save_adapter_checkpoint(model: Any, adapter_dir: Path) -> bool:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    encoder = getattr(model, "encoder", None)
    if _is_peft_encoder(encoder):
        encoder.save_pretrained(adapter_dir)
        return True
    return False


def _checkpoint_dir(config: Mapping[str, Any], epoch: int) -> Path:
    paths = config.get("paths", {}) if isinstance(config, Mapping) else {}
    root = Path(str(paths.get("checkpoint_output_dir", "models/checkpoints")))
    return root / f"epoch-{int(epoch)}"


def _is_peft_encoder(encoder: Any) -> bool:
    if encoder is None or not hasattr(encoder, "save_pretrained"):
        return False
    return hasattr(encoder, "peft_config") or encoder.__class__.__name__.lower().startswith("peft")


def _label_maps() -> dict[str, Any]:
    return {
        "token_label_to_id": dict(TOKEN_LABEL_TO_ID),
        "token_id_to_label": list(TOKEN_ID_TO_LABEL),
        "gap_label_to_id": dict(GAP_LABEL_TO_ID),
        "gap_id_to_label": list(GAP_ID_TO_LABEL),
        "boundary_before_label_to_id": dict(BOUNDARY_BEFORE_LABEL_TO_ID),
        "boundary_before_id_to_label": list(BOUNDARY_BEFORE_ID_TO_LABEL),
        "boundary_after_label_to_id": dict(BOUNDARY_AFTER_LABEL_TO_ID),
        "boundary_after_id_to_label": list(BOUNDARY_AFTER_ID_TO_LABEL),
        "rule_tag_to_id": dict(RULE_LABEL_TO_ID),
        "rule_id_to_label": list(RULE_ID_TO_LABEL),
    }


__all__ = ["train_model"]

