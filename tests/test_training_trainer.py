from pathlib import Path

import torch

from src.training.trainer import EditModelTrainer, TrainLoopConfig


class TinyEditModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, input_ids, attention_mask=None, candidate_spans=None, candidate_mask=None):
        batch_size, sequence_length = input_ids.shape
        candidate_count = candidate_spans.shape[1]
        return {
            "candidate_scores": self.logit.expand(batch_size, candidate_count),
            "punctuation_logits": self.logit.expand(batch_size, sequence_length, 3),
            "confidence_logits": self.logit.expand(batch_size, candidate_count),
            "error_type_logits": self.logit.expand(batch_size, candidate_count, 3),
        }


def test_trainer_steps_final_partial_gradient_accumulation_batch():
    model = TinyEditModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    step_count = 0
    original_step = optimizer.step

    def counted_step(*args, **kwargs):
        nonlocal step_count
        step_count += 1
        return original_step(*args, **kwargs)

    optimizer.step = counted_step
    trainer = EditModelTrainer(
        model,
        optimizer,
        TrainLoopConfig(gradient_accumulation_steps=2, mixed_precision=False, show_progress=False),
        {},
    )

    trainer.train_epoch([_batch(), _batch(), _batch()])

    assert step_count == 2


def test_trainer_uses_current_torch_amp_api():
    source = Path("src/training/trainer.py").read_text(encoding="utf-8")

    assert "torch.cuda.amp." not in source
    assert "torch.amp." in source


def _batch():
    return {
        "input_ids": torch.tensor([[1, 2, 0, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.long),
        "candidate_spans": torch.tensor([[[0, 1], [1, 2]]], dtype=torch.long),
        "candidate_mask": torch.tensor([[1, 1]], dtype=torch.bool),
        "labels": {
            "candidate_labels": torch.tensor([[1.0, 0.0]]),
            "candidate_mask": torch.tensor([[1, 1]], dtype=torch.bool),
            "punctuation_labels": torch.tensor([[0, 1, 0, 0]], dtype=torch.long),
            "punctuation_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.bool),
            "confidence_labels": torch.tensor([[1.0, 0.0]]),
            "error_type_labels": torch.tensor([[1, 0]], dtype=torch.long),
        },
    }
