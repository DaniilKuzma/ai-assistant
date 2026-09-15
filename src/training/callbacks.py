from __future__ import annotations


class BestMetricTracker:
    def __init__(self, metric_name: str = "combined_score") -> None:
        self.metric_name = metric_name
        self.best: float | None = None

    def update(self, metrics: dict[str, float]) -> bool:
        value = metrics.get(self.metric_name)
        if value is None:
            return False
        if self.best is None or value > self.best:
            self.best = value
            return True
        return False
