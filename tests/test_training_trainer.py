from __future__ import annotations

from pathlib import Path

import torch

from src.training.trainer import (
    _copy_selected_checkpoint_to_latest,
    _optimizer_step,
    _select_best_checkpoint,
    _train_epoch,
)


class TinyDirectModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_logits = torch.nn.Parameter(torch.zeros(3))
        self.gap_logits = torch.nn.Parameter(torch.zeros(3))
        self.rule_logits = torch.nn.Parameter(torch.zeros(3))

    def forward(
        self,
        input_ids,
        attention_mask=None,
        word_token_indices=None,
        word_token_mask=None,
        gap_left_indices=None,
        gap_right_indices=None,
        gap_mask=None,
    ):
        batch_size, word_count = word_token_indices.shape
        return {
            "token_edit_logits": self.token_logits.expand(batch_size, word_count, -1),
            "gap_punctuation_logits": self.gap_logits.expand(batch_size, word_count, -1),
            "rule_logits": self.rule_logits.expand(batch_size, word_count, -1),
        }


class CountingSGD(torch.optim.SGD):
    def __init__(self, params, *args, **kwargs) -> None:
        self.step_count = 0
        super().__init__(params, *args, **kwargs)

    def step(self, *args, **kwargs):
        self.step_count += 1
        return super().step(*args, **kwargs)


def test_direct_trainer_steps_final_partial_gradient_accumulation_batch() -> None:
    model = TinyDirectModel()
    optimizer = CountingSGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    result = _train_epoch(
        model,
        [_batch(), _batch(), _batch()],
        optimizer,
        scheduler,
        scaler,
        torch.device("cpu"),
        {
            "gradient_accumulation_steps": 2,
            "max_grad_norm": 1.0,
            "token_edit_loss_weight": 1.0,
            "gap_punctuation_loss_weight": 1.0,
            "rule_loss_weight": 1.0,
        },
        steps_per_epoch=3,
        amp_enabled=False,
    )

    assert optimizer.step_count == 2
    assert result["steps"] == 3
    assert result["loss"] > 0


def test_train_epoch_emits_progress_lines() -> None:
    model = TinyDirectModel()
    optimizer = CountingSGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    progress_lines: list[str] = []

    _train_epoch(
        model,
        [_batch(), _batch()],
        optimizer,
        scheduler,
        scaler,
        torch.device("cpu"),
        {
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "token_edit_loss_weight": 1.0,
            "gap_punctuation_loss_weight": 1.0,
            "rule_loss_weight": 1.0,
        },
        steps_per_epoch=2,
        amp_enabled=False,
        progress_label="train epoch 1/1",
        progress_sink=progress_lines.append,
        progress_interval_seconds=0.0,
        progress_step_interval=1,
    )

    assert progress_lines
    assert "train epoch 1/1" in progress_lines[0]
    assert any("100.0%" in line and "2/2" in line for line in progress_lines)


def test_amp_optimizer_step_does_not_advance_scheduler_when_scaler_skips_step() -> None:
    model = TinyDirectModel()
    optimizer = _FakeOptimizer()
    scheduler = _CountingScheduler()
    scaler = _SkippingScaler()

    _optimizer_step(
        model,
        optimizer,
        scheduler,
        scaler,
        max_grad_norm=0.0,
        amp_enabled=True,
    )

    assert scheduler.step_count == 0
    assert optimizer.zero_grad_count == 1


def test_checkpoint_selection_uses_lower_validation_loss_for_exact_match_tie() -> None:
    selected = _select_best_checkpoint(
        [
            {"epoch": 1, "status": "ok", "exact_match": 1.0, "loss": 0.02},
            {"epoch": 2, "status": "ok", "exact_match": 0.9998, "loss": 0.0001},
            {"epoch": 3, "status": "ok", "exact_match": 1.0, "loss": 0.00001},
        ],
        fallback_epoch=3,
    )

    assert selected["selected_epoch"] == 3
    assert selected["selected_metric"] == 1.0
    assert selected["selected_validation_loss"] == 0.00001


def test_latest_artifacts_are_copied_from_one_selected_checkpoint(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "models" / "checkpoints"
    heads_latest = tmp_path / "models" / "heads" / "latest"
    adapters_latest = tmp_path / "models" / "adapters" / "latest"
    for epoch in (1, 3):
        checkpoint = checkpoint_root / f"epoch-{epoch}"
        (checkpoint / "adapters").mkdir(parents=True)
        (checkpoint / "heads.pt").write_text(f"heads-{epoch}", encoding="utf-8")
        (checkpoint / "labels.json").write_text(f'{{"epoch": {epoch}}}\n', encoding="utf-8")
        (checkpoint / "architecture.json").write_text(
            '{"architecture": "direct_edit_tagger_v1", "debug_model": false}\n',
            encoding="utf-8",
        )
        (checkpoint / "checkpoint_meta.json").write_text(
            f'{{"epoch": {epoch}, "heads_epoch": {epoch}, "adapter_epoch": {epoch}}}\n',
            encoding="utf-8",
        )
        (checkpoint / "adapters" / "adapter.txt").write_text(f"adapter-{epoch}", encoding="utf-8")

    summary = {"best_epoch": 1, "epochs": 3, "debug_model": False}
    result = _copy_selected_checkpoint_to_latest(
        {
            "paths": {
                "checkpoint_output_dir": str(checkpoint_root),
                "heads_output_dir": str(heads_latest),
                "adapter_output_dir": str(adapters_latest),
            }
        },
        selected_epoch=1,
        summary=summary,
    )

    assert Path(result["heads_path"]).read_text(encoding="utf-8") == "heads-1"
    assert (adapters_latest / "adapter.txt").read_text(encoding="utf-8") == "adapter-1"
    architecture = __import__("json").loads((heads_latest / "architecture.json").read_text(encoding="utf-8"))
    meta = __import__("json").loads((heads_latest / "checkpoint_meta.json").read_text(encoding="utf-8"))
    selected_meta = __import__("json").loads((heads_latest / "selected_checkpoint_meta.json").read_text(encoding="utf-8"))

    assert architecture["selected_epoch"] == 1
    assert meta["heads_epoch"] == meta["adapter_epoch"] == 1
    assert selected_meta["selected_epoch"] == 1
    assert summary["artifact_consistency_check"] is True


def test_trainer_uses_current_torch_amp_api() -> None:
    source = Path("src/training/trainer.py").read_text(encoding="utf-8")

    assert "torch.cuda.amp." not in source
    assert "torch.amp." in source


def _batch() -> dict[str, object]:
    return {
        "input_ids": torch.tensor([[1, 2, 0, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.long),
        "word_token_indices": torch.tensor([[0, 1, 0, 0]], dtype=torch.long),
        "word_token_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.bool),
        "gap_left_indices": torch.tensor([[0, 1, 0, 0]], dtype=torch.long),
        "gap_right_indices": torch.tensor([[1, -1, -1, -1]], dtype=torch.long),
        "gap_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.bool),
        "labels": {
            "token_edit_label_ids": torch.tensor([[0, 1, -100, -100]], dtype=torch.long),
            "gap_label_ids": torch.tensor([[0, 1, -100, -100]], dtype=torch.long),
            "rule_tag_ids": torch.tensor([[0, 1, -100, -100]], dtype=torch.long),
            "sample_weight": torch.tensor([1.0], dtype=torch.float),
        },
    }


class _FakeOptimizer:
    def __init__(self) -> None:
        self.zero_grad_count = 0

    def zero_grad(self, *, set_to_none: bool) -> None:
        self.zero_grad_count += 1


class _CountingScheduler:
    def __init__(self) -> None:
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1


class _SkippingScaler:
    def __init__(self) -> None:
        self.scale = 1024.0

    def get_scale(self) -> float:
        return self.scale

    def unscale_(self, optimizer) -> None:
        pass

    def step(self, optimizer) -> None:
        pass

    def update(self) -> None:
        self.scale = 512.0
