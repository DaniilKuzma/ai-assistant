from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RuntimeThresholds:
    token_edit: float = 0.70
    punctuation: float = 0.70
    min_margin: float = 0.0
    rule_thresholds: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "RuntimeThresholds":
        runtime = config.get("runtime", {}) if isinstance(config, Mapping) else {}
        raw = runtime.get("confidence_thresholds", {}) if isinstance(runtime, Mapping) else {}
        thresholds = raw if isinstance(raw, Mapping) else {}
        rule_thresholds = thresholds.get("rule_thresholds", {})
        if not isinstance(rule_thresholds, Mapping):
            rule_thresholds = {}

        known_keys = {"token_edit", "punctuation", "min_margin", "rule_thresholds"}
        inline_rule_thresholds = {
            str(key): float(value)
            for key, value in thresholds.items()
            if key not in known_keys and _is_number(value)
        }
        merged_rule_thresholds = {
            str(key): float(value)
            for key, value in rule_thresholds.items()
            if _is_number(value)
        }
        merged_rule_thresholds.update(inline_rule_thresholds)

        return cls(
            token_edit=float(thresholds.get("token_edit", 0.70)),
            punctuation=float(thresholds.get("punctuation", 0.70)),
            min_margin=float(thresholds.get("min_margin", 0.0)),
            rule_thresholds=merged_rule_thresholds,
        )

    def threshold_for(self, rule_id: str, edit_type: str) -> float:
        if rule_id in self.rule_thresholds:
            return self.rule_thresholds[rule_id]
        if edit_type == "punctuation":
            return self.punctuation
        return self.token_edit

    def should_apply(self, confidence: float, margin: float, rule_id: str, edit_type: str) -> bool:
        return confidence >= self.threshold_for(rule_id, edit_type) and margin >= self.min_margin


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


__all__ = ["RuntimeThresholds"]
