from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

from src.rule_layers.base import LayerRuleSpec


LayerCoverage = dict[str, dict[str, dict[str, dict[str, int]]]]


@dataclass(frozen=True)
class LayerCoverageReport:
    coverage: LayerCoverage
    warnings: list[str]
    errors: list[str]


def collect_layer_coverage(specs: Iterable[LayerRuleSpec]) -> LayerCoverage:
    coverage: LayerCoverage = {}
    for spec in specs:
        layer_bucket = coverage.setdefault(spec.layer, {})
        rule_bucket = layer_bucket.setdefault(spec.rule_id, {})
        for case in spec.cases:
            sub_bucket = rule_bucket.setdefault(case.sub_rule_id, {})
            sub_bucket[case.mode] = sub_bucket.get(case.mode, 0) + 1
    return coverage


def validate_layer_coverage(
    specs: Iterable[LayerRuleSpec],
    *,
    enabled_rule_ids: set[str] | frozenset[str] | None = None,
) -> LayerCoverageReport:
    spec_tuple = tuple(specs)
    coverage = collect_layer_coverage(spec_tuple)
    warnings: list[str] = []
    errors: list[str] = []

    if not spec_tuple:
        errors.append("No layer specs were loaded.")

    by_rule = {spec.rule_id: spec for spec in spec_tuple if spec.enabled}
    enabled = set(enabled_rule_ids) if enabled_rule_ids is not None else set(by_rule)
    for rule_id in sorted(enabled):
        spec = by_rule.get(rule_id)
        if spec is None:
            warnings.append(f"Enabled layer rule {rule_id!r} has no loaded spec.")
            continue
        if not spec.cases:
            warnings.append(f"Layer spec {rule_id!r} has no cases.")
            continue
        modes = {case.mode for case in spec.cases}
        if "positive" not in modes:
            errors.append(f"Layer rule {rule_id!r} must have positive coverage.")
        if not ({"hard_negative", "clean_identity"} & modes):
            errors.append(f"Layer rule {rule_id!r} must have hard_negative or clean_identity coverage.")

    return LayerCoverageReport(coverage=coverage, warnings=warnings, errors=errors)


__all__ = ["LayerCoverageReport", "collect_layer_coverage", "validate_layer_coverage"]
