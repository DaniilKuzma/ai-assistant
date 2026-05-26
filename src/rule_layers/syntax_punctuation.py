from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path
import re
from typing import Any

import yaml

from src.rule_layers.base import LayerDirectCase, LayerOperation, LayerRuleSpec
from src.rule_layers.spec_loader import DEFAULT_LAYERS_DIR


LAYER = "syntax_punctuation"
FAMILY = "syntax_punctuation"
SYNTAX_PUNCTUATION_DIR = "syntax_punctuation"
MODES = ("positive", "hard_negative", "clean_identity")
RULE_IDS = frozenset(
    {
        "punct_final_marks",
        "punct_dash_syntax",
        "punct_homogeneous_extended",
        "punct_detached_definitions",
        "punct_detached_adverbials",
        "punct_comparative_turns",
        "punct_introductory_extended",
        "punct_address_interjection",
        "punct_complex_sentences",
        "punct_bsp",
        "punct_fixed_expression_guards",
    }
)
GAP_LABELS = frozenset(
    {
        "COMMA",
        "DASH",
        "COLON",
        "SEMICOLON",
        "DOT",
        "QUESTION",
        "EXCLAMATION",
        "ELLIPSIS",
        "DELETE_PUNCTUATION",
    }
)
TEMPLATE_VAR_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_syntax_punctuation_specs(
    root: str | Path = DEFAULT_LAYERS_DIR,
) -> tuple[LayerRuleSpec, ...]:
    base = _syntax_punctuation_dir(root)
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
        raise ValueError(f"Unsupported syntax punctuation rule_id {rule_id!r}: {path}")

    slots = _slots(data.get("slots"), path)
    cases: list[LayerDirectCase] = []
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, Mapping):
        raise ValueError(f"Syntax punctuation cases must be grouped by mode: {path}")
    for mode in MODES:
        items = raw_cases.get(mode, [])
        if not isinstance(items, list):
            raise ValueError(f"Syntax punctuation case group {mode!r} must be a list: {path}")
        for item in items:
            if not isinstance(item, Mapping):
                raise ValueError(f"Syntax punctuation case in {mode!r} must be a mapping: {path}")
            cases.extend(_expanded_cases(item, path, rule_id, mode, slots))

    return LayerRuleSpec(
        layer=str(data.get("layer") or LAYER),
        rule_id=rule_id,
        family=str(data.get("family") or FAMILY),
        cases=tuple(cases),
        description=str(data.get("description") or f"Syntax punctuation cases for {rule_id}."),
        explanation=str(data.get("explanation") or rule_id),
        enabled=bool(data.get("enabled", True)),
        weight=_weight(data.get("weight"), path, default=1.0),
        metadata={"source_file": path.name, **dict(data.get("metadata") or {})},
    )


def _expanded_cases(
    data: Mapping[str, Any],
    path: Path,
    rule_id: str,
    mode: str,
    spec_slots: Mapping[str, tuple[str, ...]],
) -> tuple[LayerDirectCase, ...]:
    if data.get("token_operations"):
        raise ValueError(f"Syntax punctuation layer does not allow token operations: {path}")
    if data.get("direct_token_labels"):
        raise ValueError(f"Syntax punctuation layer does not allow direct token labels: {path}")

    sub_rule_id = _required_str(data, "sub_rule_id", path)
    case_slots = {**spec_slots, **_slots(data.get("slots"), path)}
    source_template = _case_template(data, "source", "source_template", path)
    target_template = _case_template(data, "target", "target_template", path)
    raw_gap_operations = _operation_list(data.get("gap_operations"), path)
    variables = _case_variables(source_template, target_template, raw_gap_operations)
    values = _case_values(variables, case_slots, path)

    cases: list[LayerDirectCase] = []
    for value_map in values:
        source_text = _render(source_template, value_map, path)
        target_text = _render(target_template, value_map, path)
        gap_operations = tuple(
            _operation_from_mapping(item, path, value_map)
            for item in raw_gap_operations
        )
        expected_gap_count = _count(data.get("expected_gap_edit_count"), gap_operations)
        expected_token_count = _count(data.get("expected_token_edit_count"), ())
        if expected_token_count != 0:
            raise ValueError(f"Syntax punctuation examples cannot expect token edits: {path}")
        _validate_case_text(mode, source_text, target_text, gap_operations, path, sub_rule_id)
        cases.append(
            LayerDirectCase(
                rule_id=rule_id,
                family=FAMILY,
                sub_rule_id=sub_rule_id,
                mode=mode,
                source_text=source_text,
                target_text=target_text,
                gap_operations=gap_operations,
                expected_token_edit_count=expected_token_count,
                expected_gap_edit_count=expected_gap_count,
                metadata=_metadata(data, path, sub_rule_id, value_map),
                weight=_weight(data.get("weight"), path, default=1.0),
            )
        )
    return tuple(cases)


def _operation_from_mapping(
    data: Mapping[str, Any],
    path: Path,
    values: Mapping[str, str],
) -> LayerOperation:
    label = _required_str(data, "label", path)
    if label not in GAP_LABELS:
        raise ValueError(f"Unsupported syntax punctuation gap label {label!r}: {path}")
    return LayerOperation(
        kind="gap",
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
    sub_rule_id: str,
    values: Mapping[str, str],
) -> dict[str, Any]:
    raw = dict(data.get("metadata") or {})
    rendered = _render_metadata(raw, values, path)
    rendered.setdefault("case_id", sub_rule_id)
    rendered["source_file"] = path.name
    return rendered


def _render_metadata(value: Any, values: Mapping[str, str], path: Path) -> Any:
    if isinstance(value, str):
        return _render(value, values, path)
    if isinstance(value, list):
        return [_render_metadata(item, values, path) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _render_metadata(item, values, path) for key, item in value.items()}
    return value


def _validate_case_text(
    mode: str,
    source_text: str,
    target_text: str,
    gap_operations: Sequence[LayerOperation],
    path: Path,
    sub_rule_id: str,
) -> None:
    if mode == "positive":
        if source_text == target_text:
            raise ValueError(f"Positive syntax punctuation case has identical source and target: {path}:{sub_rule_id}")
        if not gap_operations:
            raise ValueError(f"Positive syntax punctuation case has no gap operations: {path}:{sub_rule_id}")
        return
    if source_text != target_text:
        raise ValueError(f"{mode} syntax punctuation case must be identity: {path}:{sub_rule_id}")


def _case_values(
    variables: Sequence[str],
    slots: Mapping[str, tuple[str, ...]],
    path: Path,
) -> tuple[dict[str, str], ...]:
    if not variables:
        return ({},)
    missing = [name for name in variables if name not in slots]
    if missing:
        raise ValueError(f"Syntax punctuation template references missing slots {missing!r}: {path}")
    return tuple(
        dict(zip(variables, values, strict=True))
        for values in product(*(slots[name] for name in variables))
    )


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
        raise ValueError(f"Syntax punctuation case is missing {plain_key!r}: {path}")
    if not value.strip():
        raise ValueError(f"Syntax punctuation case {plain_key!r} must not be empty: {path}")
    return value


def _operation_list(value: Any, path: Path) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"Syntax punctuation gap_operations must be a list: {path}")
    result: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"Syntax punctuation gap operation must be a mapping: {path}")
        result.append(item)
    return tuple(result)


def _slots(value: Any, path: Path) -> dict[str, tuple[str, ...]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Syntax punctuation slots must be a mapping: {path}")
    result: dict[str, tuple[str, ...]] = {}
    for key, raw_values in value.items():
        name = str(key)
        entries = _slot_values(raw_values, path, name)
        if not entries:
            raise ValueError(f"Syntax punctuation slot {name!r} must not be empty: {path}")
        result[name] = entries
    return result


def _slot_values(value: Any, path: Path, name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list):
        raise ValueError(f"Syntax punctuation slot {name!r} must be a string or list: {path}")
    entries: list[str] = []
    for item in value:
        if isinstance(item, Mapping):
            text = _required_str(item, "value", path)
        else:
            text = str(item).strip()
        if text:
            entries.append(text)
    return tuple(dict.fromkeys(entries))


def _render(template: str, values: Mapping[str, str], path: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        try:
            return values[name]
        except KeyError as exc:
            raise ValueError(f"Missing template value {name!r}: {path}") from exc

    return TEMPLATE_VAR_RE.sub(replace, template)


def _raw_specs(raw: Mapping[str, Any], path: Path) -> tuple[Mapping[str, Any], ...]:
    if "rules" not in raw:
        return (raw,)
    rules = raw.get("rules")
    if not isinstance(rules, list):
        raise ValueError(f"Syntax punctuation 'rules' must be a list: {path}")
    result: list[Mapping[str, Any]] = []
    for item in rules:
        if not isinstance(item, Mapping):
            raise ValueError(f"Syntax punctuation rule entry must be a mapping: {path}")
        result.append(item)
    return tuple(result)


def _read_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Syntax punctuation YAML must contain a mapping: {path}")
    return data


def _syntax_punctuation_dir(root: str | Path) -> Path:
    path = Path(root)
    if path.name == SYNTAX_PUNCTUATION_DIR:
        return path
    return path / SYNTAX_PUNCTUATION_DIR


def _required_str(data: Mapping[str, Any], key: str, path: Path) -> str:
    value = str(data.get(key) or "").strip()
    if not value:
        raise ValueError(f"Syntax punctuation entry is missing {key!r}: {path}")
    return value


def _weight(value: Any, path: Path, *, default: float) -> float:
    if value is None:
        return default
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Syntax punctuation weight must be numeric: {path}") from exc
    if weight < 0:
        raise ValueError(f"Syntax punctuation weight must be non-negative: {path}")
    return weight


def _count(value: Any, operations: Sequence[LayerOperation]) -> int:
    if value is None:
        return len(operations)
    if isinstance(value, bool):
        raise ValueError("Syntax punctuation expected edit counts must be integers.")
    count = int(value)
    if count < 0:
        raise ValueError("Syntax punctuation expected edit counts must be non-negative.")
    return count


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("Syntax punctuation token indexes must be integers.")
    return int(value)


__all__ = ["load_syntax_punctuation_specs"]
