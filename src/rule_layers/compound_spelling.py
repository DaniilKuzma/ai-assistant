from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
import re
from typing import Any

import yaml

from src.rule_layers.base import LayerDirectCase, LayerOperation, LayerRuleSpec
from src.rule_layers.spec_loader import DEFAULT_LAYERS_DIR


FAMILY = "compound_spelling"
LAYER = "compound_spelling"
OPERATIONS = frozenset({"merge", "split", "hyphenate", "unhyphenate", "replace"})
COMPOUND_DIR = "compound_spelling"
SPEC_LIST_KEYS = ("entries", "cases", "rules")
TOKEN_RE = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*")

LEGACY_LABELS: Mapping[tuple[str, str, str], str] = {
    ("takzhe_tak_zhe", "так же", "также"): "MERGE_TAK_ZHE_TO_TAKZHE",
    ("takzhe_tak_zhe", "также", "так же"): "SPLIT_TAKZHE_TO_TAK_ZHE",
    ("tozhe_to_zhe", "то же", "тоже"): "MERGE_TO_ZHE_TO_TOZHE",
    ("tozhe_to_zhe", "тоже", "то же"): "SPLIT_TOZHE_TO_TO_ZHE",
    ("zato_za_to", "за то", "зато"): "MERGE_ZA_TO_TO_ZATO",
    ("zato_za_to", "зато", "за то"): "SPLIT_ZATO_TO_ZA_TO",
    ("ne_verb", "не знает", "не знает"): "SPLIT_NE_VERB",
    ("ne_verb", "не проверил", "не проверил"): "SPLIT_NE_VERB",
    ("ne_verb", "не видел", "не видел"): "SPLIT_NE_VERB",
}

PARTICLE_SUFFIX_LABELS = {
    "то": "HYPHENATE_PARTICLE_TO",
    "либо": "HYPHENATE_PARTICLE_LIBO",
    "нибудь": "HYPHENATE_PARTICLE_NIBUD",
}


def load_compound_spelling_specs(
    root: str | Path = DEFAULT_LAYERS_DIR,
) -> tuple[LayerRuleSpec, ...]:
    base = _compound_dir(root)
    if not base.exists():
        return ()

    grouped: dict[str, list[LayerDirectCase]] = defaultdict(list)
    descriptions: dict[str, str] = {}
    source_files: dict[str, list[str]] = defaultdict(list)

    for yaml_path in sorted(path for path in base.glob("*.yaml") if path.name != "corrections.yaml"):
        raw = _read_yaml(yaml_path)
        for entry in _entries(raw, yaml_path):
            compiled = _compile_entry(entry, yaml_path)
            grouped[compiled.rule_id].extend(compiled.cases)
            descriptions.setdefault(compiled.rule_id, compiled.description)
            source_files[compiled.rule_id].append(yaml_path.name)

    return tuple(
        LayerRuleSpec(
            layer=LAYER,
            rule_id=rule_id,
            family=FAMILY,
            cases=tuple(cases),
            description=descriptions.get(rule_id, f"Compound spelling cases for {rule_id}."),
            explanation=rule_id,
            enabled=True,
            weight=1.0,
            metadata={
                "source_files": tuple(dict.fromkeys(source_files[rule_id])),
                "layer": LAYER,
                "family": FAMILY,
                "supports_positive": any(case.mode == "positive" for case in cases),
                "supports_hard_negative": any(case.mode == "hard_negative" for case in cases),
                "supports_clean_identity": any(case.mode == "clean_identity" for case in cases),
                "rule_kind": _rule_kind(cases),
            },
        )
        for rule_id, cases in sorted(grouped.items())
    )


class _CompiledEntry:
    def __init__(self, rule_id: str, description: str, cases: tuple[LayerDirectCase, ...]) -> None:
        self.rule_id = rule_id
        self.description = description
        self.cases = cases


def _compile_entry(data: Mapping[str, Any], path: Path) -> _CompiledEntry:
    rule_id = _required_str(data, "rule_id", path)
    sub_rule_id = _required_str(data, "sub_rule_id", path)
    operation = _operation(_required_str(data, "operation", path), path)
    correct = _required_str(data, "correct", path)
    wrong_variants = _required_str_list(data, "wrong_variants", path)
    positive_templates = _required_str_list(data, "positive_templates", path)
    hard_negative_templates = _required_str_list(data, "hard_negative_templates", path)
    clean_templates = _optional_str_list(data.get("clean_templates"))
    semantic_hint = str(data.get("semantic_hint") or "")
    forbidden_contexts = tuple(_optional_str_list(data.get("forbidden_contexts")))

    if len(positive_templates) < 3:
        raise ValueError(f"{path}:{sub_rule_id} must define at least three positive_templates.")
    if len(hard_negative_templates) < 3:
        raise ValueError(f"{path}:{sub_rule_id} must define at least three hard_negative_templates.")

    cases: list[LayerDirectCase] = []
    for wrong in wrong_variants:
        _validate_replace_operation(operation, wrong, correct, path, sub_rule_id)
        for template in positive_templates:
            cases.append(
                _positive_case(
                    rule_id,
                    sub_rule_id,
                    operation,
                    correct,
                    wrong,
                    template,
                    semantic_hint,
                    forbidden_contexts,
                )
            )

    for template in hard_negative_templates:
        text = _render_template(template, correct=correct, wrong=wrong_variants[0])
        cases.append(
            _identity_case(
                rule_id,
                sub_rule_id,
                "hard_negative",
                text,
                operation,
                correct,
                correct,
                semantic_hint,
                forbidden_contexts,
            )
        )

    for template in clean_templates:
        text = _render_template(template, correct=correct, wrong=wrong_variants[0])
        cases.append(
            _identity_case(
                rule_id,
                sub_rule_id,
                "clean_identity",
                text,
                operation,
                correct,
                correct,
                semantic_hint,
                forbidden_contexts,
            )
        )

    return _CompiledEntry(
        rule_id=rule_id,
        description=str(data.get("description") or f"Compound spelling subrule {sub_rule_id}."),
        cases=tuple(cases),
    )


def _positive_case(
    rule_id: str,
    sub_rule_id: str,
    operation: str,
    correct: str,
    wrong: str,
    template: str,
    semantic_hint: str,
    forbidden_contexts: tuple[str, ...],
) -> LayerDirectCase:
    source_text = _render_template(template, correct=correct, wrong=wrong)
    target_text = _render_template(template, correct=correct, wrong=correct)
    label = _label_for(sub_rule_id, operation, source=wrong, target=correct)
    metadata = _metadata(
        rule_id=rule_id,
        sub_rule_id=sub_rule_id,
        operation=operation,
        source=wrong,
        target=correct,
        semantic_hint=semantic_hint,
        forbidden_contexts=forbidden_contexts,
    )
    return LayerDirectCase(
        rule_id=rule_id,
        family=FAMILY,
        sub_rule_id=sub_rule_id,
        mode="positive",
        source_text=source_text,
        target_text=target_text,
        token_operations=(
            LayerOperation(
                kind="token_span",
                label=label,
                source_pattern=wrong,
                target_pattern=correct,
                metadata={"operation": operation},
            ),
        ),
        expected_token_edit_count=1,
        expected_gap_edit_count=0,
        metadata=metadata,
    )


def _identity_case(
    rule_id: str,
    sub_rule_id: str,
    mode: str,
    text: str,
    operation: str,
    source: str,
    target: str,
    semantic_hint: str,
    forbidden_contexts: tuple[str, ...],
) -> LayerDirectCase:
    return LayerDirectCase(
        rule_id=rule_id,
        family=FAMILY,
        sub_rule_id=sub_rule_id,
        mode=mode,
        source_text=text,
        target_text=text,
        expected_token_edit_count=0,
        expected_gap_edit_count=0,
        metadata=_metadata(
            rule_id=rule_id,
            sub_rule_id=sub_rule_id,
            operation=operation,
            source=source,
            target=target,
            semantic_hint=semantic_hint,
            forbidden_contexts=forbidden_contexts,
        ),
    )


def _label_for(sub_rule_id: str, operation: str, *, source: str, target: str) -> str:
    normalized_source = _normalize_for_label(source)
    normalized_target = _normalize_for_label(target)
    legacy = LEGACY_LABELS.get((sub_rule_id, normalized_source, normalized_target))
    if legacy is not None:
        if legacy == "SPLIT_NE_VERB":
            return legacy
        return legacy

    if sub_rule_id == "ne_verb" and normalized_source.startswith("не") and " " not in normalized_source:
        return "SPLIT_NE_VERB"

    if normalized_source.startswith("кое ") and "-" in normalized_target:
        return "HYPHENATE_KOE"

    suffix_label = _particle_suffix_label(normalized_source, normalized_target)
    if suffix_label is not None:
        return suffix_label

    if normalized_source.startswith("по ") and normalized_target.startswith("по-"):
        return "HYPHENATE_PO_ADVERB"

    if operation == "replace":
        return "DICT_REPLACE"
    return "SPAN_REPLACE_BY_LEXICON"


def _particle_suffix_label(source: str, target: str) -> str | None:
    if "-" not in target:
        return None
    source_parts = source.split()
    if len(source_parts) < 2:
        return None
    return PARTICLE_SUFFIX_LABELS.get(source_parts[-1])


def _metadata(
    *,
    rule_id: str,
    sub_rule_id: str,
    operation: str,
    source: str,
    target: str,
    semantic_hint: str,
    forbidden_contexts: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "layer": LAYER,
        "family": FAMILY,
        "rule_id": rule_id,
        "sub_rule_id": sub_rule_id,
        "operation": operation,
        "source": source,
        "target": target,
        "semantic_hint": semantic_hint,
        "forbidden_contexts": list(forbidden_contexts),
    }


def _rule_kind(cases: Sequence[LayerDirectCase]) -> str:
    return "correction" if any(case.mode == "positive" for case in cases) else "guard"


def _compound_dir(root: str | Path) -> Path:
    path = Path(root)
    if path.name == COMPOUND_DIR:
        return path
    return path / COMPOUND_DIR


def _read_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Compound spelling YAML must contain a mapping: {path}")
    return data


def _entries(raw: Mapping[str, Any], path: Path) -> tuple[Mapping[str, Any], ...]:
    found_key = next((key for key in SPEC_LIST_KEYS if key in raw), "")
    if not found_key:
        return (raw,)
    value = raw.get(found_key)
    if not isinstance(value, list):
        raise ValueError(f"Compound spelling YAML {found_key!r} must be a list: {path}")
    entries: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"Compound spelling entry at {path}:{index} must be a mapping.")
        entries.append(item)
    return tuple(entries)


def _render_template(template: str, *, correct: str, wrong: str) -> str:
    return (
        template.replace("{Correct}", _capitalize(correct))
        .replace("{Wrong}", _capitalize(wrong))
        .replace("{correct}", correct)
        .replace("{wrong}", wrong)
    )


def _capitalize(value: str) -> str:
    if not value:
        return value
    return f"{value[0].upper()}{value[1:]}"


def _required_str(data: Mapping[str, Any], key: str, path: Path) -> str:
    value = str(data.get(key) or "").strip()
    if not value:
        raise ValueError(f"Compound spelling entry is missing {key!r}: {path}")
    return value


def _required_str_list(data: Mapping[str, Any], key: str, path: Path) -> tuple[str, ...]:
    values = tuple(_optional_str_list(data.get(key)))
    if not values:
        raise ValueError(f"Compound spelling entry is missing non-empty {key!r}: {path}")
    return values


def _optional_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _operation(value: str, path: Path) -> str:
    normalized = value.strip().lower()
    aliases = {
        "hyphen": "hyphenate",
        "unhyphen": "unhyphenate",
        "dehyphen": "unhyphenate",
        "join": "merge",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in OPERATIONS:
        raise ValueError(f"Unsupported compound spelling operation {value!r}: {path}")
    return normalized


def _validate_replace_operation(operation: str, source: str, target: str, path: Path, sub_rule_id: str) -> None:
    if operation != "replace":
        return
    if len(TOKEN_RE.findall(source)) != 1 or len(TOKEN_RE.findall(target)) != 1:
        raise ValueError(f"{path}:{sub_rule_id} replace operation must be single-token.")


def _normalize_for_label(value: str) -> str:
    return value.casefold().replace("ё", "е")


__all__ = ["load_compound_spelling_specs"]
