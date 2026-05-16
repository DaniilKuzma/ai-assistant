from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.model.losses import multitask_loss


@dataclass(frozen=True)
class TrainLoopConfig:
    epochs: int = 1
    gradient_accumulation_steps: int = 4
    mixed_precision: bool = True
    show_progress: bool = True
    progress_log_every_steps: int = 1000


class EditModelTrainer:
    def __init__(self, model: Any, optimizer: Any, config: TrainLoopConfig, loss_weights: dict[str, float]) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.loss_weights = loss_weights

    def train_epoch(self, dataloader: Any) -> float:
        import torch

        self.model.train()
        total = 0.0
        steps = 0
        amp_enabled = self.config.mixed_precision and torch.cuda.is_available()
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        self.optimizer.zero_grad()

        progress = _with_progress(
            dataloader,
            enabled=self.config.show_progress,
            total=_safe_len(dataloader),
            description="Training epoch",
        )

        for step, batch in enumerate(progress, start=1):
            labels = batch.pop("labels")
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = self.model(**batch)
                loss = multitask_loss(outputs, labels, self.loss_weights) / self.config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if step % self.config.gradient_accumulation_steps == 0:
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()
            total += float(loss.detach().cpu())
            steps += 1
            if self.config.show_progress and _should_update_progress(step, self.config.progress_log_every_steps):
                _set_progress_postfix(progress, loss=float(loss.detach().cpu()), optimizer_steps=step // self.config.gradient_accumulation_steps)
        if steps % self.config.gradient_accumulation_steps != 0:
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad()
        return total / max(1, steps)


def _with_progress(iterable: Any, *, enabled: bool, total: int | None, description: str) -> Any:
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, total=total, desc=description, dynamic_ncols=True)


def _safe_len(value: Any) -> int | None:
    try:
        return len(value)
    except TypeError:
        return None


def _should_update_progress(step: int, log_every_steps: int) -> bool:
    return step == 1 or step % max(1, log_every_steps) == 0


def _set_progress_postfix(progress: Any, **values: float | int) -> None:
    if hasattr(progress, "set_postfix"):
        progress.set_postfix(values)
