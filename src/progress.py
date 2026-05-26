from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import sys
import time
from typing import Any


ProgressSink = Callable[[str], None]
Clock = Callable[[], float]


def stderr_progress_sink(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def format_progress_line(
    label: str,
    *,
    current: int,
    total: int,
    metrics: Mapping[str, Any] | None = None,
    elapsed_seconds: float = 0.0,
    width: int = 24,
) -> str:
    total = max(1, int(total))
    current = min(max(0, int(current)), total)
    ratio = current / total
    filled = min(width, int(round(ratio * width)))
    bar = "#" * filled + "-" * (width - filled)
    metric_text = _format_metrics(metrics or {})
    suffix = f" {metric_text}" if metric_text else ""
    return (
        f"{label} [{bar}] {ratio * 100:5.1f}% "
        f"{current}/{total} elapsed={_format_duration(elapsed_seconds)}{suffix}"
    )


@dataclass
class ProgressReporter:
    label: str
    total: int
    sink: ProgressSink = stderr_progress_sink
    clock: Clock = time.monotonic
    min_interval_seconds: float = 30.0
    step_interval: int | None = None
    width: int = 24
    _started_at: float = field(default=0.0, init=False)
    _last_emit_at: float = field(default=0.0, init=False)
    _last_emit_step: int = field(default=-1, init=False)

    def start(self) -> None:
        now = self.clock()
        self._started_at = now
        self._emit(0, {}, now)

    def update(self, current: int, metrics: Mapping[str, Any] | None = None, *, force: bool = False) -> None:
        current = min(max(0, int(current)), max(1, int(self.total)))
        now = self.clock()
        if force or current >= self.total or self._should_emit(current, now):
            self._emit(current, metrics or {}, now)

    def finish(self, metrics: Mapping[str, Any] | None = None) -> None:
        final_step = max(1, int(self.total))
        if self._last_emit_step == final_step:
            return
        self.update(final_step, metrics or {}, force=True)

    def _should_emit(self, current: int, now: float) -> bool:
        if current == self._last_emit_step:
            return False
        interval = self.step_interval
        if interval is None:
            interval = max(1, max(1, int(self.total)) // 100)
        if current - self._last_emit_step >= max(1, int(interval)):
            return True
        return now - self._last_emit_at >= max(0.0, float(self.min_interval_seconds))

    def _emit(self, current: int, metrics: Mapping[str, Any], now: float) -> None:
        self._last_emit_at = now
        self._last_emit_step = current
        self.sink(
            format_progress_line(
                self.label,
                current=current,
                total=max(1, int(self.total)),
                metrics=metrics,
                elapsed_seconds=max(0.0, now - self._started_at),
                width=self.width,
            )
        )


def _format_metrics(metrics: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


__all__ = ["ProgressReporter", "format_progress_line", "stderr_progress_sink"]
