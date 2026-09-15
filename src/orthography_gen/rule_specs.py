from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORTHOGRAPHY_DIR = PROJECT_ROOT / "lexicon" / "orthography"

SITE_TYPES = frozenset(
    {"suffix", "root", "prefix", "ending", "particle", "hyphen", "split_join", "sign", "consonant"}
)
MODEL_ROLES = frozenset(
    {"deterministic_replace", "detect_and_replace", "context_disambiguation", "hard_negative_only"}
)


@dataclass(frozen=True)
class OrthographyRuleSpec:
    rule_id: str
    orfogrammka_id: str
    family: str
    pos: str
    site_type: str
    requires: tuple[str, ...]
    correct_patterns: tuple[str, ...]
    wrong_patterns: tuple[str, ...]
    explanation_id: str
    model_role: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "OrthographyRuleSpec":
        spec = cls(
            rule_id=_required_str(data, "rule_id"),
            orfogrammka_id=_required_str(data, "orfogrammka_id"),
            family=_required_str(data, "family"),
            pos=_required_str(data, "pos"),
            site_type=_required_str(data, "site_type"),
            requires=tuple(_string_list(data.get("requires"))),
            correct_patterns=tuple(_string_list(data.get("correct_patterns"))),
            wrong_patterns=tuple(_string_list(data.get("wrong_patterns"))),
            explanation_id=_required_str(data, "explanation_id"),
            model_role=_required_str(data, "model_role"),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.site_type not in SITE_TYPES:
            raise ValueError(f"Unsupported orthography site_type: {self.site_type!r}.")
        if self.model_role not in MODEL_ROLES:
            raise ValueError(f"Unsupported orthography model_role: {self.model_role!r}.")
        if not self.correct_patterns:
            raise ValueError(f"Orthography rule {self.rule_id!r} must define correct_patterns.")
        if not self.wrong_patterns:
            raise ValueError(f"Orthography rule {self.rule_id!r} must define wrong_patterns.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "orfogrammka_id": self.orfogrammka_id,
            "family": self.family,
            "pos": self.pos,
            "site_type": self.site_type,
            "requires": list(self.requires),
            "correct_patterns": list(self.correct_patterns),
            "wrong_patterns": list(self.wrong_patterns),
            "explanation_id": self.explanation_id,
            "model_role": self.model_role,
        }


def load_rule_specs(path: str | Path = DEFAULT_ORTHOGRAPHY_DIR) -> dict[str, OrthographyRuleSpec]:
    specs: dict[str, OrthographyRuleSpec] = {}
    for yaml_path in _yaml_paths(path):
        raw = _read_yaml(yaml_path)
        raw_spec = raw.get("rule_spec") if isinstance(raw, Mapping) else None
        if raw_spec is None:
            continue
        spec = OrthographyRuleSpec.from_mapping(raw_spec)
        if spec.rule_id in specs:
            raise ValueError(f"Duplicate orthography rule spec: {spec.rule_id}")
        specs[spec.rule_id] = spec
    return specs


def _yaml_paths(path: str | Path) -> tuple[Path, ...]:
    base = Path(path)
    if base.is_file():
        return (base,)
    if not base.exists():
        raise FileNotFoundError(f"Orthography lexicon directory does not exist: {base}")
    return tuple(sorted(base.glob("*.yaml")))


def _read_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Orthography YAML must contain a mapping: {path}")
    return data


def _required_str(data: Mapping[str, Any], key: str) -> str:
    value = str(data.get(key) or "").strip()
    if not value:
        raise ValueError(f"Orthography rule spec is missing {key!r}.")
    return value


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return []


__all__ = [
    "DEFAULT_ORTHOGRAPHY_DIR",
    "MODEL_ROLES",
    "OrthographyRuleSpec",
    "SITE_TYPES",
    "load_rule_specs",
]
