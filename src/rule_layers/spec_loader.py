from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re
from typing import Any

import yaml

from src.grammar_gen.randomness import RandomSource
from src.rule_layers.base import LayerDirectCase, LayerOperation, LayerRuleSpec


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LAYERS_DIR = PROJECT_ROOT / "lexicon" / "layers"
SLOT_NAMES = frozenset(
    {
        "nouns",
        "verbs",
        "adjectives",
        "adverbs",
        "objects",
        "persons",
        "documents",
        "places",
    }
)
TEMPLATE_VAR_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_layer_specs(
    layer_name: str,
    *,
    root: str | Path = DEFAULT_LAYERS_DIR,
    rng: RandomSource | None = None,
) -> tuple[LayerRuleSpec, ...]:
    random_source = rng or RandomSource()
    specs: list[LayerRuleSpec] = []
    for yaml_path in _yaml_paths(root, layer_name):
        raw = _read_yaml(yaml_path)
        for raw_spec in _raw_specs(raw, yaml_path):
            specs.append(_spec_from_mapping(raw_spec, yaml_path, layer_name, random_source))
    return tuple(specs)


def _yaml_paths(root: str | Path, layer_name: str) -> tuple[Path, ...]:
    base = Path(root)
    layer_dir = base / layer_name
    if not layer_dir.exists():
        return ()
    if layer_dir.is_file():
        return (layer_dir,)
    return tuple(sorted(path for path in layer_dir.glob("*.yaml") if path.is_file()))


def _read_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Layer YAML must contain a mapping: {path}")
    return data


def _raw_specs(raw: Mapping[str, Any], path: Path) -> tuple[Mapping[str, Any], ...]:
    if "rules" not in raw:
        return (raw,)
    rules = raw.get("rules")
    if not isinstance(rules, list):
        raise ValueError(f"Layer YAML 'rules' must be a list: {path}")
    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(rules):
        if not isinstance(item, Mapping):
            raise ValueError(f"Layer rule at {path}:{index} must be a mapping.")
        result.append(item)
    return tuple(result)


def _spec_from_mapping(
    data: Mapping[str, Any],
    path: Path,
    layer_name: str,
    rng: RandomSource,
) -> LayerRuleSpec:
    rule_id = _required_str(data, "rule_id", path)
    family = _required_str(data, "family", path)
    layer = str(data.get("layer") or layer_name).strip()
    if not layer:
        raise ValueError(f"Layer spec is missing 'layer': {path}")
    slots = _slots(data.get("slots"), path)
    cases = tuple(
        _case_from_mapping(raw_case, path, rule_id, family, mode, slots, rng)
        for mode, raw_case in _iter_cases(data.get("cases"), path)
    )
    return LayerRuleSpec(
        layer=layer,
        rule_id=rule_id,
        family=family,
        cases=cases,
        description=str(data.get("description") or ""),
        explanation=str(data.get("explanation") or rule_id),
        enabled=bool(data.get("enabled", True)),
        weight=_weight(data.get("weight"), path, default=1.0),
        metadata=dict(data.get("metadata") or {}),
    )


def _iter_cases(value: Any, path: Path) -> tuple[tuple[str | None, Mapping[str, Any]], ...]:
    if value is None:
        return ()
    result: list[tuple[str | None, Mapping[str, Any]]] = []
    if isinstance(value, list):
        for index, item in enumerate(value):
            if not isinstance(item, Mapping):
                raise ValueError(f"Layer case at {path}:{index} must be a mapping.")
            result.append((None, item))
        return tuple(result)
    if isinstance(value, Mapping):
        ordered_modes = ("positive", "hard_negative", "clean_identity")
        extra_modes = tuple(str(mode) for mode in value if str(mode) not in ordered_modes)
        for mode in (*ordered_modes, *extra_modes):
            if mode not in value:
                continue
            items = value[mode]
            if str(mode) not in {"positive", "hard_negative", "clean_identity"}:
                raise ValueError(f"Unsupported layer case mode group {mode!r}: {path}")
            if not isinstance(items, list):
                raise ValueError(f"Layer case group {mode!r} must be a list: {path}")
            for index, item in enumerate(items):
                if not isinstance(item, Mapping):
                    raise ValueError(f"Layer case at {path}:{mode}[{index}] must be a mapping.")
                result.append((str(mode), item))
        return tuple(result)
    raise ValueError(f"Layer spec 'cases' must be a list or mapping: {path}")


def _case_from_mapping(
    data: Mapping[str, Any],
    path: Path,
    rule_id: str,
    family: str,
    grouped_mode: str | None,
    spec_slots: Mapping[str, tuple[tuple[str, float], ...]],
    rng: RandomSource,
) -> LayerDirectCase:
    mode = str(data.get("mode") or grouped_mode or "").strip()
    if not mode:
        raise ValueError(f"Layer case is missing 'mode': {path}")
    sub_rule_id = _required_str(data, "sub_rule_id", path)
    case_slots = {**spec_slots, **_slots(data.get("slots") or data.get("variables"), path)}
    values: dict[str, str] = {}
    source_text = _case_text(data, "source", "source_template", path, case_slots, values, rng)
    target_text = _case_text(data, "target", "target_template", path, case_slots, values, rng)
    token_operations = tuple(
        _operation_from_mapping(item, path, default_kind="token_span", slots=case_slots, values=values, rng=rng)
        for item in _operation_list(data.get("token_operations"), path, "token_operations")
    )
    gap_operations = tuple(
        _operation_from_mapping(item, path, default_kind="gap", slots=case_slots, values=values, rng=rng)
        for item in _operation_list(data.get("gap_operations"), path, "gap_operations")
    )
    return LayerDirectCase(
        rule_id=str(data.get("rule_id") or rule_id),
        family=str(data.get("family") or family),
        sub_rule_id=sub_rule_id,
        mode=mode,
        source_text=source_text,
        target_text=target_text,
        token_operations=token_operations,
        gap_operations=gap_operations,
        expected_token_edit_count=_count(data.get("expected_token_edit_count"), token_operations),
        expected_gap_edit_count=_count(data.get("expected_gap_edit_count"), gap_operations),
        metadata=dict(data.get("metadata") or {}),
        weight=_weight(data.get("weight"), path, default=1.0),
        direct_token_labels=tuple(str(label) for label in _string_list(data.get("direct_token_labels"))),
        direct_gap_labels=tuple(str(label) for label in _string_list(data.get("direct_gap_labels"))),
    )


def _case_text(
    data: Mapping[str, Any],
    plain_key: str,
    template_key: str,
    path: Path,
    slots: Mapping[str, tuple[tuple[str, float], ...]],
    values: dict[str, str],
    rng: RandomSource,
) -> str:
    if plain_key in data:
        text = str(data.get(plain_key) or "")
    elif template_key in data:
        text = _render_template(str(data.get(template_key) or ""), slots, values, rng, path)
    else:
        raise ValueError(f"Layer case is missing '{plain_key}' or '{template_key}': {path}")
    if not text.strip():
        raise ValueError(f"Layer case field '{plain_key}' must not be empty: {path}")
    return text


def _operation_list(value: Any, path: Path, key: str) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"Layer case '{key}' must be a list: {path}")
    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"Layer operation at {path}:{key}[{index}] must be a mapping.")
        result.append(item)
    return tuple(result)


def _operation_from_mapping(
    data: Mapping[str, Any],
    path: Path,
    *,
    default_kind: str,
    slots: Mapping[str, tuple[tuple[str, float], ...]],
    values: dict[str, str],
    rng: RandomSource,
) -> LayerOperation:
    label = _required_str(data, "label", path)
    return LayerOperation(
        kind=str(data.get("kind") or default_kind),
        label=label,
        source_pattern=_render_template(str(data.get("source_pattern") or ""), slots, values, rng, path),
        target_pattern=_render_template(str(data.get("target_pattern") or ""), slots, values, rng, path),
        token_index=_optional_int(data.get("token_index")),
        token_start=_optional_int(data.get("token_start")),
        token_end=_optional_int(data.get("token_end")),
        metadata=dict(data.get("metadata") or {}),
    )


def _slots(value: Any, path: Path) -> dict[str, tuple[tuple[str, float], ...]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Layer slots must be a mapping: {path}")
    result: dict[str, tuple[tuple[str, float], ...]] = {}
    for name, raw_values in value.items():
        key = str(name)
        if key not in SLOT_NAMES:
            raise ValueError(f"Unsupported layer slot {key!r}: {path}")
        entries = _slot_entries(raw_values, path, key)
        if not entries:
            raise ValueError(f"Layer slot {key!r} must not be empty: {path}")
        result[key] = entries
    return result


def _slot_entries(value: Any, path: Path, key: str) -> tuple[tuple[str, float], ...]:
    if isinstance(value, str):
        return ((value, 1.0),)
    if not isinstance(value, list):
        raise ValueError(f"Layer slot {key!r} must be a scalar or list: {path}")
    entries: list[tuple[str, float]] = []
    for item in value:
        if isinstance(item, Mapping):
            slot_value = _required_str(item, "value", path)
            entries.append((slot_value, _weight(item.get("weight"), path, default=1.0)))
        else:
            slot_value = str(item)
            if slot_value.strip():
                entries.append((slot_value, 1.0))
    return tuple(entries)


def _render_template(
    template: str,
    slots: Mapping[str, tuple[tuple[str, float], ...]],
    values: dict[str, str],
    rng: RandomSource,
    path: Path,
) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in slots:
            raise ValueError(f"Layer template references missing slot {name!r}: {path}")
        if name not in values:
            values[name] = rng.weighted_choice(slots[name])
        return values[name]

    return TEMPLATE_VAR_RE.sub(replace, template)


def _required_str(data: Mapping[str, Any], key: str, path: Path) -> str:
    value = str(data.get(key) or "").strip()
    if not value:
        raise ValueError(f"Layer spec is missing '{key}': {path}")
    return value


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item) for item in value if str(item).strip()]
    return []


def _weight(value: Any, path: Path, *, default: float) -> float:
    if value is None:
        return default
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Layer weight must be numeric: {path}") from exc
    if weight < 0:
        raise ValueError(f"Layer weight must be non-negative: {path}")
    return weight


def _count(value: Any, operations: Sequence[LayerOperation]) -> int:
    if value is None:
        return len(operations)
    if isinstance(value, bool):
        raise ValueError("Layer expected edit counts must be integers.")
    count = int(value)
    if count < 0:
        raise ValueError("Layer expected edit counts must be non-negative.")
    return count


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("Layer token indexes must be integers.")
    return int(value)


__all__ = ["DEFAULT_LAYERS_DIR", "load_layer_specs"]
