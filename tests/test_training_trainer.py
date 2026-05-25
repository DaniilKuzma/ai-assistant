from __future__ import annotations

from pathlib import Path

import torch

from src.training.trainer import _train_epoch


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
