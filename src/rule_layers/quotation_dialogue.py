from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path
import re
from typing import Any

import yaml

from src.schema.labels import (
    boundary_after_label_to_id,
    boundary_before_label_to_id,
    gap_label_to_id,
    token_label_to_id,
)
from src.rule_layers.base import LayerDirectCase, LayerOperation, LayerRuleSpec
from src.rule_layers.spec_loader import DEFAULT_LAYERS_DIR


LAYER = "quotation_dialogue"
FAMILY = "quotation_dialogue"
QUOTATION_DIALOGUE_DIR = "quotation_dialogue"
MODES = ("positive", "hard_negative", "clean_identity")
RULE_IDS = frozenset(
    {
        "quote_pairing",
        "quote_normalization",
        "quote_extra_marks",
        "dialogue_author_before",
        "dialogue_speech_before_author",
        "dialogue_author_inside_speech",
        "dialogue_bracket_guards",
    }
)
PHENOMENA = frozenset({"quote_pairing", "quote_normalization", "dialogue", "bracket_guard"})
TOKEN_LABELS = frozenset({"CAPITALIZE"})
GAP_LABELS = frozenset({"COLON", "DOT", "COMMA_DASH"})
BOUNDARY_LABEL_ALIASES: Mapping[str, tuple[str, str]] = {
    "INSERT_OPEN_QUOTE_BEFORE": ("boundary_before", "INSERT_OPEN_QUOTE"),
    "DELETE_OPEN_QUOTE_BEFORE": ("boundary_before", "DELETE_OPEN_QUOTE"),
    "NORMALIZE_OPEN_QUOTE_BEFORE": ("boundary_before", "NORMALIZE_OPEN_QUOTE"),
    "INSERT_OPEN_BRACKET_BEFORE": ("boundary_before", "INSERT_OPEN_BRACKET"),
    "DELETE_OPEN_BRACKET_BEFORE": ("boundary_before", "DELETE_OPEN_BRACKET"),
    "INSERT_CLOSE_QUOTE_AFTER": ("boundary_after", "INSERT_CLOSE_QUOTE"),
    "DELETE_CLOSE_QUOTE_AFTER": ("boundary_after", "DELETE_CLOSE_QUOTE"),
    "NORMALIZE_CLOSE_QUOTE_AFTER": ("boundary_after", "NORMALIZE_CLOSE_QUOTE"),
    "INSERT_CLOSE_BRACKET_AFTER": ("boundary_after", "INSERT_CLOSE_BRACKET"),
    "DELETE_CLOSE_BRACKET_AFTER": ("boundary_after", "DELETE_CLOSE_BRACKET"),
}
TEMPLATE_VAR_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_quotation_dialogue_specs(
    root: str | Path = DEFAULT_LAYERS_DIR,
) -> tuple[LayerRuleSpec, ...]:
    base = _quotation_dialogue_dir(root)
    if not base.exists():
        return ()

    specs: list[LayerRuleSpec] = []
    for yaml_path in sorted(base.glob("*.yaml")):
        raw = _read_yaml(yaml_path)
        for raw_spec in _raw_specs(raw, yaml_path):
            specs.append(_spec_from_mapping(raw_spec, yaml_path))
    return tuple(specs)


def _spec_from_mapping(data: Mapping[str, Any], path: Path) -> LayerRuleSpec:
    rule_id = _required_str(data, "rule_id", path)
    if rule_id not in RULE_IDS:
        raise ValueError(f"Unsupported quotation_dialogue rule_id {rule_id!r}: {path}")

    slots = _slots(data.get("slots"), path)
    slot_groups = _slot_groups(data.get("slot_groups"), path)
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, Mapping):
        raise ValueError(f"Quotation dialogue cases must be grouped by mode: {path}")

    cases: list[LayerDirectCase] = []
    for mode in MODES:
        items = raw_cases.get(mode, [])
        if not isinstance(items, list):
            raise ValueError(f"Quotation dialogue case group {mode!r} must be a list: {path}")
        for item in items:
            if not isinstance(item, Mapping):
                raise ValueError(f"Quotation dialogue case in {mode!r} must be a mapping: {path}")
            cases.extend(_expanded_cases(item, path, rule_id, mode, slots, slot_groups))

    return LayerRuleSpec(
        layer=LAYER,
        rule_id=rule_id,
        family=FAMILY,
        cases=tuple(cases),
        description=str(data.get("description") or f"Quotation dialogue cases for {rule_id}."),
        explanation=str(data.get("explanation") or rule_id),
        enabled=bool(data.get("enabled", True)),
        weight=_weight(data.get("weight"), path, default=1.0),
        metadata={
            "source_file": path.name,
            "layer": LAYER,
            "family": FAMILY,
            "supports_positive": any(case.mode == "positive" for case in cases),
            "supports_hard_negative": any(case.mode == "hard_negative" for case in cases),
            "supports_clean_identity": any(case.mode == "clean_identity" for case in cases),
            "rule_kind": _rule_kind(cases),
            **dict(data.get("metadata") or {}),
        },
    )


def _expanded_cases(
    data: Mapping[str, Any],
    path: Path,
    rule_id: str,
    mode: str,
    spec_slots: Mapping[str, tuple[str, ...]],
    spec_slot_groups: Mapping[str, tuple[Mapping[str, str], ...]],
) -> tuple[LayerDirectCase, ...]:
    sub_rule_id = _required_str(data, "sub_rule_id", path)
    case_slots = {**spec_slots, **_slots(data.get("slots"), path)}
    case_slot_groups = {**spec_slot_groups, **_slot_groups(data.get("slot_groups"), path)}
    source_template = _case_template(data, "source", "source_template", path)
    target_template = _case_template(data, "target", "target_template", path)
    raw_token_operations = _operation_list(data.get("token_operations"), path, "token_operations")
    raw_gap_operations = _operation_list(data.get("gap_operations"), path, "gap_operations")
    raw_boundary_operations = _operation_list(data.get("boundary_operations"), path, "boundary_operations")
    variables = _case_variables(source_template, target_template, raw_token_operations, raw_gap_operations, raw_boundary_operations)
    values = _case_values(variables, case_slots, case_slot_groups, path)

    cases: list[LayerDirectCase] = []
    for value_map in values:
        source_text = _render(source_template, value_map, path)
        target_text = _render(target_template, value_map, path)
        token_operations = tuple(
            _operation_from_mapping(item, path, value_map, default_kind="token")
            for item in raw_token_operations
        )
        gap_operations = tuple(
            _operation_from_mapping(item, path, value_map, default_kind="gap")
            for item in raw_gap_operations
        )
        boundary_operations = tuple(
            _boundary_operation_from_mapping(item, path, value_map)
            for item in raw_boundary_operations
        )
        expected_token_count = _count(data.get("expected_token_edit_count"), token_operations, path)
        expected_gap_count = _count(data.get("expected_gap_edit_count"), gap_operations, path)
        expected_boundary_count = _count(data.get("expected_boundary_edit_count"), boundary_operations, path)
        _validate_case(
            mode,
            source_text,
            target_text,
            token_operations,
            gap_operations,
            boundary_operations,
            expected_token_count,
            expected_gap_count,
            expected_boundary_count,
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
                gap_operations=gap_operations,
                boundary_operations=boundary_operations,
                expected_token_edit_count=expected_token_count,
                expected_gap_edit_count=expected_gap_count,
                expected_boundary_edit_count=expected_boundary_count,
                metadata=_metadata(data, path, rule_id, sub_rule_id, value_map, expected_token_count, expected_gap_count, expected_boundary_count),
                weight=_weight(data.get("weight"), path, default=1.0),
            )
        )
    return tuple(cases)


def _operation_from_mapping(
    data: Mapping[str, Any],
    path: Path,
    values: Mapping[str, str],
    *,
    default_kind: str,
) -> LayerOperation:
    label = _required_str(data, "label", path)
    if default_kind == "token":
        if label not in TOKEN_LABELS:
            raise ValueError(f"Unsupported quotation_dialogue token label {label!r}: {path}")
        token_label_to_id(label)
    elif default_kind == "gap":
        if label not in GAP_LABELS:
            raise ValueError(f"Unsupported quotation_dialogue gap label {label!r}: {path}")
        gap_label_to_id(label)
    return LayerOperation(
        kind=str(data.get("kind") or default_kind),
        label=label,
        source_pattern=_render(str(data.get("source_pattern") or ""), values, path),
        target_pattern=_render(str(data.get("target_pattern") or ""), values, path),
        token_index=_optional_int(data.get("token_index")),
        token_start=_optional_int(data.get("token_start")),
        token_end=_optional_int(data.get("token_end")),
        metadata=dict(data.get("metadata") or {}),
    )


def _boundary_operation_from_mapping(
    data: Mapping[str, Any],
    path: Path,
    values: Mapping[str, str],
) -> LayerOperation:
    raw_label = _required_str(data, "label", path)
    try:
        kind, label = BOUNDARY_LABEL_ALIASES[raw_label]
    except KeyError as exc:
        raise ValueError(f"Unsupported quotation_dialogue boundary label {raw_label!r}: {path}") from exc
    if kind == "boundary_before":
        boundary_before_label_to_id(label)
    else:
        boundary_after_label_to_id(label)
    return LayerOperation(
        kind=kind,
        label=label,
        source_pattern=_render(str(data.get("source_pattern") or ""), values, path),
        target_pattern=_render(str(data.get("target_pattern") or ""), values, path),
        token_index=_optional_int(data.get("token_index")),
        token_start=_optional_int(data.get("token_start")),
        token_end=_optional_int(data.get("token_end")),
        metadata={"raw_label": raw_label, **dict(data.get("metadata") or {})},
    )


def _metadata(
    data: Mapping[str, Any],
    path: Path,
    rule_id: str,
    sub_rule_id: str,
    values: Mapping[str, str],
    expected_token_count: int,
    expected_gap_count: int,
    expected_boundary_count: int,
) -> dict[str, Any]:
    raw = dict(data.get("metadata") or {})
    rendered = _render_metadata(raw, values, path)
    phenomenon = str(rendered.get("phenomenon") or _default_phenomenon(rule_id))
    if phenomenon not in PHENOMENA:
        raise ValueError(f"Unsupported quotation_dialogue phenomenon {phenomenon!r}: {path}:{sub_rule_id}")
    total = expected_token_count + expected_gap_count + expected_boundary_count
    rendered.update(
        {
            "layer": LAYER,
            "family": FAMILY,
            "rule_id": rule_id,
            "source_file": path.name,
            "case_id": sub_rule_id,
            "sub_rule_id": sub_rule_id,
            "phenomenon": phenomenon,
            "operation": str(rendered.get("operation") or sub_rule_id),
            "expected_token_edit_count": expected_token_count,
            "expected_gap_edit_count": expected_gap_count,
            "expected_boundary_edit_count": expected_boundary_count,
            "expected_total_edit_count": total,
            "construction_family": FAMILY,
            "uses_construction_bank": True,
            "construction_id": str(rendered.get("construction_id") or sub_rule_id),
            "uses_safety_clauses": False,
            "safety_clauses": [],
        }
    )
    if rule_id.startswith("dialogue_") and target_patterns_missing_final_punctuation(rendered):
        rendered["allowed_source_surface_failures"] = ["missing_final_punctuation"]
    return rendered


def _validate_case(
    mode: str,
    source_text: str,
    target_text: str,
    token_operations: Sequence[LayerOperation],
    gap_operations: Sequence[LayerOperation],
    boundary_operations: Sequence[LayerOperation],
    expected_token_count: int,
    expected_gap_count: int,
    expected_boundary_count: int,
    path: Path,
    sub_rule_id: str,
) -> None:
    active_count = len(token_operations) + len(gap_operations) + len(boundary_operations)
    if mode == "positive":
        if source_text == target_text:
            raise ValueError(f"Positive quotation_dialogue case has identical source and target: {path}:{sub_rule_id}")
        if active_count == 0:
            raise ValueError(f"Positive quotation_dialogue case has no operations: {path}:{sub_rule_id}")
        return
    if source_text != target_text:
        raise ValueError(f"{mode} quotation_dialogue case must be identity: {path}:{sub_rule_id}")
    if active_count != 0:
        raise ValueError(f"{mode} quotation_dialogue case must not define operations: {path}:{sub_rule_id}")
    if expected_token_count != 0 or expected_gap_count != 0 or expected_boundary_count != 0:
        raise ValueError(f"{mode} quotation_dialogue case must have zero expected counts: {path}:{sub_rule_id}")


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
            raise ValueError(f"Quotation dialogue template references missing slots {missing!r}: {path}")
        for values in product(*(slots[name] for name in remaining)):
            result.append({**base, **dict(zip(remaining, values, strict=True))})
    return tuple(result)


def _case_variables(
    source_template: str,
    target_template: str,
    *operation_groups: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    variables: list[str] = []
    values = [source_template, target_template]
    for operations in operation_groups:
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
        raise ValueError(f"Quotation dialogue case is missing {plain_key!r}: {path}")
    if not value.strip():
        raise ValueError(f"Quotation dialogue case {plain_key!r} must not be empty: {path}")
    return value


def _operation_list(value: Any, path: Path, key: str) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"Quotation dialogue {key} must be a list: {path}")
    result: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"Quotation dialogue operation must be a mapping: {path}")
        result.append(item)
    return tuple(result)


def _slots(value: Any, path: Path) -> dict[str, tuple[str, ...]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Quotation dialogue slots must be a mapping: {path}")
    result: dict[str, tuple[str, ...]] = {}
    for key, raw_values in value.items():
        name = str(key)
        entries = _slot_values(raw_values, path, name)
        if not entries:
            raise ValueError(f"Quotation dialogue slot {name!r} must not be empty: {path}")
        result[name] = entries
    return result


def _slot_values(value: Any, path: Path, name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list):
        raise ValueError(f"Quotation dialogue slot {name!r} must be a string or list: {path}")
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
        raise ValueError(f"Quotation dialogue slot_groups must be a mapping: {path}")
    result: dict[str, tuple[Mapping[str, str], ...]] = {}
    for group_name, raw_entries in value.items():
        if not isinstance(raw_entries, list):
            raise ValueError(f"Quotation dialogue slot group {group_name!r} must be a list: {path}")
        entries: list[Mapping[str, str]] = []
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, Mapping):
                raise ValueError(f"Quotation dialogue slot group entry must be a mapping: {path}")
            rendered = {str(key): str(item).strip() for key, item in raw_entry.items() if str(item).strip()}
            if rendered:
                entries.append(rendered)
        if not entries:
            raise ValueError(f"Quotation dialogue slot group {group_name!r} must not be empty: {path}")
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

        next_value = TEMPLATE_VAR_RE.sub(replace, rendered)
        rendered = next_value
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
        raise ValueError(f"Quotation dialogue 'rules' must be a list: {path}")
    result: list[Mapping[str, Any]] = []
    for item in rules:
        if not isinstance(item, Mapping):
            raise ValueError(f"Quotation dialogue rule entry must be a mapping: {path}")
        result.append(item)
    return tuple(result)


def _read_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Quotation dialogue YAML must contain a mapping: {path}")
    return data


def _quotation_dialogue_dir(root: str | Path) -> Path:
    path = Path(root)
    if path.name == QUOTATION_DIALOGUE_DIR:
        return path
    return path / QUOTATION_DIALOGUE_DIR


def _required_str(data: Mapping[str, Any], key: str, path: Path) -> str:
    value = str(data.get(key) or "").strip()
    if not value:
        raise ValueError(f"Quotation dialogue entry is missing {key!r}: {path}")
    return value


def _weight(value: Any, path: Path, *, default: float) -> float:
    if value is None:
        return default
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Quotation dialogue weight must be numeric: {path}") from exc
    if weight < 0:
        raise ValueError(f"Quotation dialogue weight must be non-negative: {path}")
    return weight


def _count(value: Any, operations: Sequence[LayerOperation], path: Path) -> int:
    if value is None:
        return len(operations)
    if isinstance(value, bool):
        raise ValueError(f"Quotation dialogue expected edit counts must be integers: {path}")
    count = int(value)
    if count < 0:
        raise ValueError(f"Quotation dialogue expected edit counts must be non-negative: {path}")
    return count


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("Quotation dialogue token indexes must be integers.")
    return int(value)


def _default_phenomenon(rule_id: str) -> str:
    if rule_id in {"quote_pairing"}:
        return "quote_pairing"
    if rule_id in {"quote_normalization", "quote_extra_marks"}:
        return "quote_normalization"
    if rule_id == "dialogue_bracket_guards":
        return "bracket_guard"
    return "dialogue"


def target_patterns_missing_final_punctuation(metadata: Mapping[str, Any]) -> bool:
    return (
        int(metadata.get("expected_gap_edit_count") or 0) > 0
        and str(metadata.get("operation") or "").find("missing") >= 0
    )


def _rule_kind(cases: Sequence[LayerDirectCase]) -> str:
    return "correction" if any(case.mode == "positive" for case in cases) else "guard"


__all__ = ["load_quotation_dialogue_specs"]
