from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

from src.rule_layers.base import LayerRuleSpec
from src.schema.labels import rule_tag_to_id


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
        try:
            rule_tag_to_id(rule_id)
        except ValueError:
            errors.append(f"Layer rule {rule_id!r} is missing from schema labels.")
        if not spec.metadata:
            errors.append(f"Layer rule {rule_id!r} must have non-empty metadata.")
        if not spec.cases:
            warnings.append(f"Layer spec {rule_id!r} has no cases.")
            continue
        modes = {case.mode for case in spec.cases}
        supports_positive = bool(spec.metadata.get("supports_positive", "positive" in modes))
        supports_hard_negative = bool(spec.metadata.get("supports_hard_negative", "hard_negative" in modes))
        supports_clean_identity = bool(spec.metadata.get("supports_clean_identity", "clean_identity" in modes))
        expected_rule_kind = "guard" if not supports_positive else "correction"
        if spec.metadata.get("supports_positive") is not None and supports_positive is not ("positive" in modes):
            errors.append(f"Layer rule {rule_id!r} supports_positive metadata does not match cases.")
        if spec.metadata.get("supports_hard_negative") is not None and supports_hard_negative is not ("hard_negative" in modes):
            errors.append(f"Layer rule {rule_id!r} supports_hard_negative metadata does not match cases.")
        if spec.metadata.get("supports_clean_identity") is not None and supports_clean_identity is not ("clean_identity" in modes):
            errors.append(f"Layer rule {rule_id!r} supports_clean_identity metadata does not match cases.")
        if spec.metadata.get("rule_kind") is not None and spec.metadata.get("rule_kind") != expected_rule_kind:
            errors.append(f"Layer rule {rule_id!r} rule_kind metadata does not match capabilities.")

        for case in spec.cases:
            if not case.metadata:
                errors.append(f"Layer rule {rule_id!r} case {case.sub_rule_id!r} must have non-empty metadata.")
            if case.metadata.get("rule_id") != rule_id:
                errors.append(f"Layer rule {rule_id!r} case {case.sub_rule_id!r} metadata rule_id mismatch.")
            if case.metadata.get("sub_rule_id") != case.sub_rule_id:
                errors.append(f"Layer rule {rule_id!r} case {case.sub_rule_id!r} metadata sub_rule_id mismatch.")

        if supports_positive and "positive" not in modes:
            errors.append(f"Layer rule {rule_id!r} must have positive coverage.")
        if not supports_positive and "positive" in modes:
            errors.append(f"Guard-only layer rule {rule_id!r} must not have positive coverage.")
        if not supports_positive and not {"hard_negative", "clean_identity"} <= modes:
            errors.append(f"Guard-only layer rule {rule_id!r} must have hard_negative and clean_identity coverage.")
        elif not ({"hard_negative", "clean_identity"} & modes):
            errors.append(f"Layer rule {rule_id!r} must have hard_negative or clean_identity coverage.")

    return LayerCoverageReport(coverage=coverage, warnings=warnings, errors=errors)


__all__ = ["LayerCoverageReport", "collect_layer_coverage", "validate_layer_coverage"]
