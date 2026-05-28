from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path
import re
from typing import Any

import yaml

from src.rule_layers.base import LayerDirectCase, LayerOperation, LayerRuleSpec
from src.rule_layers.spec_loader import DEFAULT_LAYERS_DIR
from src.schema.labels import token_label_to_id


LAYER = "casing"
FAMILY = "casing"
CASING_DIR = "casing"
MODES = ("positive", "hard_negative", "clean_identity")
RULE_IDS = frozenset(
    {
        "casing_sentence_start",
        "casing_person_names",
        "casing_geo_names",
        "casing_organizations",
        "casing_documents_events",
        "casing_common_lowercase",
        "casing_formal_you_guard",
    }
)
TOKEN_LABELS = frozenset({"CAPITALIZE", "LOWERCASE"})
DIRECT_TOKEN_LABELS = frozenset({"KEEP", *TOKEN_LABELS})
TEMPLATE_VAR_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_casing_specs(root: str | Path = DEFAULT_LAYERS_DIR) -> tuple[LayerRuleSpec, ...]:
    base = _casing_dir(root)
    if not base.exists():
        return ()

    grouped_cases: dict[str, list[LayerDirectCase]] = defaultdict(list)
    descriptions: dict[str, str] = {}
    explanations: dict[str, str] = {}
    weights: dict[str, float] = {}
    source_files: dict[str, list[str]] = defaultdict(list)

    for yaml_path in sorted(base.glob("*.yaml")):
        raw = _read_yaml(yaml_path)
        for raw_spec in _raw_specs(raw, yaml_path):
            compiled = _compiled_spec(raw_spec, yaml_path)
            grouped_cases[compiled.rule_id].extend(compiled.cases)
            descriptions.setdefault(compiled.rule_id, compiled.description)
            explanations.setdefault(compiled.rule_id, compiled.explanation)
            weights.setdefault(compiled.rule_id, compiled.weight)
            source_files[compiled.rule_id].append(yaml_path.name)

    return tuple(
        LayerRuleSpec(
            layer=LAYER,
            rule_id=rule_id,
            family=FAMILY,
            cases=tuple(cases),
            description=descriptions.get(rule_id, f"Casing cases for {rule_id}."),
            explanation=explanations.get(rule_id, rule_id),
            enabled=True,
            weight=weights.get(rule_id, 1.0),
            metadata={
                "source_files": tuple(dict.fromkeys(source_files[rule_id])),
                "layer": LAYER,
                "family": FAMILY,
                "rule_kind": _rule_kind(cases),
                "supports_positive": any(case.mode == "positive" for case in cases),
                "supports_hard_negative": any(case.mode == "hard_negative" for case in cases),
                "supports_clean_identity": any(case.mode == "clean_identity" for case in cases),
            },
        )
        for rule_id, cases in sorted(grouped_cases.items())
    )


class _CompiledSpec:
    def __init__(
        self,
        rule_id: str,
        cases: tuple[LayerDirectCase, ...],
        description: str,
        explanation: str,
        weight: float,
    ) -> None:
        self.rule_id = rule_id
        self.cases = cases
        self.description = description
        self.explanation = explanation
        self.weight = weight


def _compiled_spec(data: Mapping[str, Any], path: Path) -> _CompiledSpec:
    rule_id = _required_str(data, "rule_id", path)
    if rule_id not in RULE_IDS:
        raise ValueError(f"Unsupported casing rule_id {rule_id!r}: {path}")

    if data.get("gap_operations") or data.get("direct_gap_labels"):
        raise ValueError(f"Casing layer does not allow gap operations or labels: {path}")
    if (
        data.get("boundary_operations")
        or data.get("direct_boundary_before_labels")
        or data.get("direct_boundary_after_labels")
    ):
        raise ValueError(f"Casing layer does not allow boundary operations or labels: {path}")

    slots = _slots(data.get("slots"), path)
    slot_groups = _slot_groups(data.get("slot_groups"), path)
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, Mapping):
        raise ValueError(f"Casing cases must be grouped by mode: {path}")

    cases: list[LayerDirectCase] = []
    for mode in MODES:
        items = raw_cases.get(mode, [])
        if not isinstance(items, list):
            raise ValueError(f"Casing case group {mode!r} must be a list: {path}")
        for item in items:
            if not isinstance(item, Mapping):
                raise ValueError(f"Casing case in {mode!r} must be a mapping: {path}")
            cases.extend(_expanded_cases(item, path, rule_id, mode, slots, slot_groups))

    if rule_id == "casing_formal_you_guard" and any(case.mode == "positive" for case in cases):
        raise ValueError(f"{rule_id} is guard-only and must not define positive cases: {path}")

    return _CompiledSpec(
        rule_id=rule_id,
        cases=tuple(cases),
        description=str(data.get("description") or f"Casing cases for {rule_id}."),
        explanation=str(data.get("explanation") or rule_id),
        weight=_weight(data.get("weight"), path, default=1.0),
    )


def _expanded_cases(
    data: Mapping[str, Any],
    path: Path,
    rule_id: str,
    mode: str,
    spec_slots: Mapping[str, tuple[str, ...]],
    spec_slot_groups: Mapping[str, tuple[Mapping[str, str], ...]],
) -> tuple[LayerDirectCase, ...]:
    if data.get("gap_operations") or data.get("direct_gap_labels"):
        raise ValueError(f"Casing layer does not allow gap operations or labels: {path}")
    if (
        data.get("boundary_operations")
        or data.get("direct_boundary_before_labels")
        or data.get("direct_boundary_after_labels")
    ):
        raise ValueError(f"Casing layer does not allow boundary operations or labels: {path}")

    sub_rule_id = _required_str(data, "sub_rule_id", path)
    case_slots = {**spec_slots, **_slots(data.get("slots"), path)}
    case_slot_groups = {**spec_slot_groups, **_slot_groups(data.get("slot_groups"), path)}
    source_template = _case_template(data, "source", "source_template", path)
    target_template = _case_template(data, "target", "target_template", path)
    raw_token_operations = _operation_list(data.get("token_operations"), path)
    raw_direct_token_labels = tuple(str(label) for label in _string_list(data.get("direct_token_labels")))
    for label in raw_direct_token_labels:
        _validate_direct_token_label(label, path)

    variables = _case_variables(source_template, target_template, raw_token_operations)
    values = _case_values(variables, case_slots, case_slot_groups, path)

    cases: list[LayerDirectCase] = []
    for value_map in values:
        source_text = _render(source_template, value_map, path)
        target_text = _render(target_template, value_map, path)
        token_operations = tuple(
            _operation_from_mapping(item, path, value_map)
            for item in raw_token_operations
        )
        expected_token_count = _count(data.get("expected_token_edit_count"), token_operations, raw_direct_token_labels, path)
        _validate_case(
            mode,
            source_text,
            target_text,
            token_operations,
            raw_direct_token_labels,
            expected_token_count,
            path,
            sub_rule_id,
        )
        cases.append(
            LayerDirectCase(
                rule_id=rule_id,
                family=FAMILY,
                sub_rule_id=sub_rule_id,
                mode=mode,
                source_text=source_text,
                target_text=target_text,
                token_operations=token_operations,
                expected_token_edit_count=expected_token_count,
                expected_gap_edit_count=0,
                metadata=_metadata(data, path, rule_id, sub_rule_id, value_map, mode),
                weight=_weight(data.get("weight"), path, default=1.0),
                direct_token_labels=raw_direct_token_labels,
            )
        )
    return tuple(cases)


def _operation_from_mapping(
    data: Mapping[str, Any],
    path: Path,
    values: Mapping[str, str],
) -> LayerOperation:
    label = _required_str(data, "label", path)
    if label not in TOKEN_LABELS:
        raise ValueError(f"Unsupported casing token label {label!r}: {path}")
    token_label_to_id(label)
    kind = str(data.get("kind") or "token").strip()
    if kind not in {"token", "token_span"}:
        raise ValueError(f"Unsupported casing token operation kind {kind!r}: {path}")
    return LayerOperation(
        kind=kind,
        label=label,
        source_pattern=_render(str(data.get("source_pattern") or ""), values, path),
        target_pattern=_render(str(data.get("target_pattern") or ""), values, path),
        token_index=_optional_int(data.get("token_index")),
        token_start=_optional_int(data.get("token_start")),
        token_end=_optional_int(data.get("token_end")),
        metadata=dict(data.get("metadata") or {}),
    )


def _metadata(
    data: Mapping[str, Any],
    path: Path,
    rule_id: str,
    sub_rule_id: str,
    values: Mapping[str, str],
    mode: str,
) -> dict[str, Any]:
    raw = dict(data.get("metadata") or {})
    rendered = _render_metadata(raw, values, path)
    supports_positive = rule_id != "casing_formal_you_guard"
    rendered.update(
        {
            "layer": LAYER,
            "family": FAMILY,
            "rule_id": rule_id,
            "sub_rule_id": sub_rule_id,
            "case_id": sub_rule_id,
            "source_file": path.name,
            "phenomenon": str(rendered.get("phenomenon") or _default_phenomenon(rule_id)),
            "operation": str(rendered.get("operation") or _default_operation(rule_id, mode)),
            "casing_class": str(rendered.get("casing_class") or _default_casing_class(rule_id)),
            "rule_kind": "correction" if supports_positive else "guard",
            "supports_positive": supports_positive,
            "construction_family": FAMILY,
            "uses_construction_bank": True,
            "construction_id": str(rendered.get("construction_id") or sub_rule_id),
            "uses_safety_clauses": False,
            "safety_clauses": [],
        }
    )
    if rule_id == "casing_sentence_start" and mode == "positive":
        rendered["allowed_source_surface_failures"] = ["sentence_start_not_uppercase"]
    return rendered


def _validate_case(
    mode: str,
    source_text: str,
    target_text: str,
    token_operations: Sequence[LayerOperation],
    direct_token_labels: Sequence[str],
    expected_token_count: int,
    path: Path,
    sub_rule_id: str,
) -> None:
    active_direct_labels = [label for label in direct_token_labels if label != "KEEP"]
    active_count = len(token_operations) + len(active_direct_labels)
    if mode == "positive":
        if source_text == target_text:
            raise ValueError(f"Positive casing case has identical source and target: {path}:{sub_rule_id}")
        if active_count == 0:
            raise ValueError(f"Positive casing case has no token edits: {path}:{sub_rule_id}")
        if expected_token_count < 1:
            raise ValueError(f"Positive casing case must expect at least one token edit: {path}:{sub_rule_id}")
        return
    if source_text != target_text:
        raise ValueError(f"{mode} casing case must be identity: {path}:{sub_rule_id}")
    if active_count != 0:
        raise ValueError(f"{mode} casing case must not define active token edits: {path}:{sub_rule_id}")
    if expected_token_count != 0:
        raise ValueError(f"{mode} casing case must have zero expected token edits: {path}:{sub_rule_id}")


def _case_values(
    variables: Sequence[str],
    slots: Mapping[str, tuple[str, ...]],
    slot_groups: Mapping[str, tuple[Mapping[str, str], ...]],
    path: Path,
) -> tuple[dict[str, str], ...]:
    if not variables:
        return ({},)

    grouped_variants = [
        entries
        for entries in slot_groups.values()
        if any(name in variables for entry in entries for name in entry)
    ]
    base_maps: list[dict[str, str]] = [{}]
    for entries in grouped_variants:
        next_maps: list[dict[str, str]] = []
        for base, entry in product(base_maps, entries):
            overlap = set(base) & set(entry)
            if overlap and any(base[name] != entry[name] for name in overlap):
                continue
            next_maps.append({**base, **entry})
        base_maps = next_maps

    result: list[dict[str, str]] = []
    for base in base_maps:
        remaining = [name for name in variables if name not in base]
        missing = [name for name in remaining if name not in slots]
        if missing:
            raise ValueError(f"Casing template references missing slots {missing!r}: {path}")
        for values in product(*(slots[name] for name in remaining)):
            result.append({**base, **dict(zip(remaining, values, strict=True))})
    return tuple(result)


def _case_variables(
    source_template: str,
    target_template: str,
    operations: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    variables: list[str] = []
    values = [source_template, target_template]
    for operation in operations:
        values.append(str(operation.get("source_pattern") or ""))
        values.append(str(operation.get("target_pattern") or ""))
    for value in values:
        for name in TEMPLATE_VAR_RE.findall(value):
            if name not in variables:
                variables.append(name)
    return tuple(variables)


def _case_template(data: Mapping[str, Any], plain_key: str, template_key: str, path: Path) -> str:
    if plain_key in data:
        value = str(data.get(plain_key) or "")
    elif template_key in data:
        value = str(data.get(template_key) or "")
    else:
        raise ValueError(f"Casing case is missing {plain_key!r}: {path}")
    if not value.strip():
        raise ValueError(f"Casing case {plain_key!r} must not be empty: {path}")
    return value


def _operation_list(value: Any, path: Path) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"Casing token_operations must be a list: {path}")
    result: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"Casing token operation must be a mapping: {path}")
        result.append(item)
    return tuple(result)


def _slots(value: Any, path: Path) -> dict[str, tuple[str, ...]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Casing slots must be a mapping: {path}")
    result: dict[str, tuple[str, ...]] = {}
    for key, raw_values in value.items():
        name = str(key)
        entries = _slot_values(raw_values, path, name)
        if not entries:
            raise ValueError(f"Casing slot {name!r} must not be empty: {path}")
        result[name] = entries
    return result


def _slot_values(value: Any, path: Path, name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list):
        raise ValueError(f"Casing slot {name!r} must be a string or list: {path}")
    entries: list[str] = []
    for item in value:
        text = _required_str(item, "value", path) if isinstance(item, Mapping) else str(item).strip()
        if text:
            entries.append(text)
    return tuple(dict.fromkeys(entries))


def _slot_groups(value: Any, path: Path) -> dict[str, tuple[Mapping[str, str], ...]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Casing slot_groups must be a mapping: {path}")
    result: dict[str, tuple[Mapping[str, str], ...]] = {}
    for group_name, raw_entries in value.items():
        if not isinstance(raw_entries, list):
            raise ValueError(f"Casing slot group {group_name!r} must be a list: {path}")
        entries: list[Mapping[str, str]] = []
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, Mapping):
                raise ValueError(f"Casing slot group entry must be a mapping: {path}")
            rendered = {str(key): str(item).strip() for key, item in raw_entry.items() if str(item).strip()}
            if rendered:
                entries.append(rendered)
        if not entries:
            raise ValueError(f"Casing slot group {group_name!r} must not be empty: {path}")
        result[str(group_name)] = tuple(entries)
    return result


def _render(template: str, values: Mapping[str, str], path: Path) -> str:
    rendered = template
    for _ in range(5):
        changed = False

        def replace(match: re.Match[str]) -> str:
            nonlocal changed
            name = match.group(1)
            try:
                replacement = values[name]
            except KeyError as exc:
                raise ValueError(f"Missing template value {name!r}: {path}") from exc
            changed = True
            return replacement

        rendered = TEMPLATE_VAR_RE.sub(replace, rendered)
        if not changed or not TEMPLATE_VAR_RE.search(rendered):
            return rendered
    if TEMPLATE_VAR_RE.search(rendered):
        raise ValueError(f"Unresolved nested template in {template!r}: {path}")
    return rendered


def _render_metadata(value: Any, values: Mapping[str, str], path: Path) -> Any:
    if isinstance(value, str):
        return _render(value, values, path)
    if isinstance(value, list):
        return [_render_metadata(item, values, path) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _render_metadata(item, values, path) for key, item in value.items()}
    return value


def _raw_specs(raw: Mapping[str, Any], path: Path) -> tuple[Mapping[str, Any], ...]:
    if "rules" not in raw:
        return (raw,)
    rules = raw.get("rules")
    if not isinstance(rules, list):
        raise ValueError(f"Casing 'rules' must be a list: {path}")
    result: list[Mapping[str, Any]] = []
    for item in rules:
        if not isinstance(item, Mapping):
            raise ValueError(f"Casing rule entry must be a mapping: {path}")
        result.append(item)
    return tuple(result)


def _read_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Casing YAML must contain a mapping: {path}")
    return data


def _casing_dir(root: str | Path) -> Path:
    path = Path(root)
    if path.name == CASING_DIR:
        return path
    return path / CASING_DIR


def _required_str(data: Mapping[str, Any], key: str, path: Path) -> str:
    value = str(data.get(key) or "").strip()
    if not value:
        raise ValueError(f"Casing entry is missing {key!r}: {path}")
    return value


def _weight(value: Any, path: Path, *, default: float) -> float:
    if value is None:
        return default
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Casing weight must be numeric: {path}") from exc
    if weight < 0:
        raise ValueError(f"Casing weight must be non-negative: {path}")
    return weight


def _count(
    value: Any,
    operations: Sequence[LayerOperation],
    direct_token_labels: Sequence[str],
    path: Path,
) -> int:
    if value is None:
        return len(operations) + sum(1 for label in direct_token_labels if label != "KEEP")
    if isinstance(value, bool):
        raise ValueError(f"Casing expected edit counts must be integers: {path}")
    count = int(value)
    if count < 0:
        raise ValueError(f"Casing expected edit counts must be non-negative: {path}")
    return count


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("Casing token indexes must be integers.")
    return int(value)


def _string_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _validate_direct_token_label(label: str, path: Path) -> None:
    if label not in DIRECT_TOKEN_LABELS:
        raise ValueError(f"Unsupported casing direct token label {label!r}: {path}")
    token_label_to_id(label)


def _rule_kind(cases: Sequence[LayerDirectCase]) -> str:
    return "correction" if any(case.mode == "positive" for case in cases) else "guard"


def _default_phenomenon(rule_id: str) -> str:
    return {
        "casing_sentence_start": "sentence_start",
        "casing_person_names": "proper_names",
        "casing_geo_names": "geo_names",
        "casing_organizations": "organizations",
        "casing_documents_events": "documents_events",
        "casing_common_lowercase": "common_lowercase",
        "casing_formal_you_guard": "formal_you_guard",
    }.get(rule_id, "casing")


def _default_operation(rule_id: str, mode: str) -> str:
    if mode != "positive":
        return "identity_guard"
    if rule_id == "casing_common_lowercase":
        return "lowercase"
    return "capitalize"


def _default_casing_class(rule_id: str) -> str:
    return {
        "casing_sentence_start": "sentence_start",
        "casing_person_names": "person_name",
        "casing_geo_names": "geo_name",
        "casing_organizations": "organization_name",
        "casing_documents_events": "document_event_name",
        "casing_common_lowercase": "common_word",
        "casing_formal_you_guard": "formal_you",
    }.get(rule_id, "casing")


__all__ = ["load_casing_specs"]
