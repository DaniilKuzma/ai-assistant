from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

from src.rules.coverage_matrix import iter_coverage_entries, load_rules_coverage
from src.rules.rule_ids import UNKNOWN_RULE_ID, normalize_rule_id


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES_YAML_PATH = PROJECT_ROOT / "configs" / "rules.yaml"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

BLOCK_METADATA_ONLY = "BLOCK_METADATA_ONLY"
BLOCK_PLANNED = "BLOCK_PLANNED"
BLOCK_DISABLED = "BLOCK_DISABLED"
BLOCK_NEEDS_DICTIONARY = "BLOCK_NEEDS_DICTIONARY"
BLOCK_NEEDS_SYNTAX = "BLOCK_NEEDS_SYNTAX"
BLOCK_NEEDS_MORPHOLOGY = "BLOCK_NEEDS_MORPHOLOGY"
BLOCK_NEEDS_NER = "BLOCK_NEEDS_NER"
BLOCK_NO_CANDIDATE_PATH = "BLOCK_NO_CANDIDATE_PATH"
BLOCK_NO_CANDIDATE = "BLOCK_NO_CANDIDATE"
BLOCK_NO_SYNTHETIC_OPERATOR = "BLOCK_NO_SYNTHETIC_OPERATOR"
EVAL_ONLY = "EVAL_ONLY"
MINING_ONLY = "MINING_ONLY"
INCLUDE_AFTER_VALIDATOR = "INCLUDE_AFTER_VALIDATOR"
INCLUDE_AFTER_THRESHOLD_CALIBRATION = "INCLUDE_AFTER_THRESHOLD_CALIBRATION"
INCLUDE_AFTER_TRAINING = "INCLUDE_AFTER_TRAINING"
INCLUDE_NOW = "INCLUDE_NOW"

TRAINING_INCLUDE_DECISIONS = frozenset({INCLUDE_NOW})
REPORT_VISIBLE_INCLUDE_DECISIONS = frozenset(
    {
        INCLUDE_NOW,
        INCLUDE_AFTER_VALIDATOR,
        INCLUDE_AFTER_THRESHOLD_CALIBRATION,
        INCLUDE_AFTER_TRAINING,
        EVAL_ONLY,
        MINING_ONLY,
    }
)
STRONGER_EXISTING_DECISIONS = frozenset(
    {
        INCLUDE_AFTER_VALIDATOR,
        INCLUDE_AFTER_THRESHOLD_CALIBRATION,
        INCLUDE_AFTER_TRAINING,
        EVAL_ONLY,
        MINING_ONLY,
    }
)
BLOCK_DECISION_PREFIX = "BLOCK_"
SERVICE_RULE_IDS = frozenset(
    {
        "",
        UNKNOWN_RULE_ID,
        "clean_identity",
        "clean_identity_hard_negative",
        "hard_negative",
        "unknown_real_validated",
    }
)
DEPENDENCY_DECISIONS = {
    "dictionary": BLOCK_NEEDS_DICTIONARY,
    "syntax": BLOCK_NEEDS_SYNTAX,
    "morphology": BLOCK_NEEDS_MORPHOLOGY,
    "ner": BLOCK_NEEDS_NER,
}
DICTIONARY_CANDIDATE_RULE_IDS = frozenset(
    {
        "dictionary_fuzzy",
        "double_consonant_candidate",
        "keyboard_typo_candidate",
        "swapped_letters_candidate",
        "missing_letter_candidate",
        "extra_letter_candidate",
        "yo_e_candidate",
    }
)
NEVER_TRAIN_RULE_IDS = frozenset({"quote_open", "quote_close", "neural_punctuation"})


@dataclass(frozen=True)
class RuleCapability:
    taxonomy_key: str
    domain: str
    entry_type: str
    title: str
    orfogrammka_id: str
    project_rule_ids: list[str]
    implementation_status: str
    requires: list[str]
    executable: bool
    training_eligible: bool
    training_decision: str
    training_reason: str
    has_candidate_path: bool
    has_synthetic_support: bool
    has_hard_negative_support: bool
    has_validator_support: bool
    has_dictionary_support: bool
    has_syntax_support: bool
    has_morphology_support: bool
    has_ner_support: bool
    risk_level: str


def load_rule_capabilities(rules_yaml_path: str | Path) -> list[RuleCapability]:
    available_modules = _default_available_modules()
    return [capability_for_taxonomy_entry(entry, available_modules=available_modules) for entry in iter_taxonomy_rules(rules_yaml_path)]


def iter_taxonomy_rules(rules_yaml_path: str | Path) -> list[dict[str, Any]]:
    data = load_rules_coverage(rules_yaml_path)
    entries: list[dict[str, Any]] = []
    for domain, taxonomy_key, entry in iter_coverage_entries(data):
        row = dict(entry)
        row["domain"] = domain
        row["taxonomy_key"] = taxonomy_key
        row.setdefault("source_section", domain)
        row.setdefault("project_rule_ids", list(row.get("rules", []) or []))
        entries.append(row)
    return entries


def capability_for_taxonomy_entry(entry: dict[str, Any], available_modules: dict | None = None) -> RuleCapability:
    modules = dict(available_modules or _default_available_modules())
    taxonomy_key = str(entry.get("taxonomy_key") or entry.get("_matrix_key") or entry.get("key") or "")
    domain = str(entry.get("domain") or entry.get("source_section") or "")
    implementation = dict(entry.get("implementation", {}) or {})
    dataset = dict(entry.get("dataset", {}) or {})
    status = str(entry.get("status") or implementation.get("status") or entry.get("implementation_status") or "")
    entry_type = str(entry.get("entry_type") or "")
    project_rule_ids = _normalized_rule_ids(
        entry.get("project_rule_ids")
        or entry.get("rules")
        or implementation.get("rule_ids")
        or []
    )
    requires = _normalized_requires(entry.get("requires") or implementation.get("requires") or [])
    executable = bool(entry.get("executable", implementation.get("executable", False)))
    if status in {"implemented", "partial", "candidate_only", "model_required", "dictionary_model_required", "syntax_required", "ner_required"}:
        executable = executable or bool(project_rule_ids)

    has_dictionary_support = _module_bool(modules, "dictionary_supported", "has_dictionary_support")
    has_syntax_support = _module_bool(modules, "syntax_supported", "has_syntax_support")
    has_morphology_support = _module_bool(modules, "morphology_supported", "has_morphology_support")
    has_ner_support = _module_bool(modules, "ner_supported", "has_ner_support")
    candidate_rule_ids = _module_set(modules, "candidate_rule_ids")
    synthetic_rule_ids = _module_set(modules, "synthetic_rule_ids")
    hard_negative_rule_ids = _module_set(modules, "hard_negative_rule_ids")
    validator_rule_ids = _module_set(modules, "validator_rule_ids")

    has_candidate_path = bool(project_rule_ids) and any(rule_id in candidate_rule_ids for rule_id in project_rule_ids)
    has_synthetic_support = bool(project_rule_ids) and any(rule_id in synthetic_rule_ids for rule_id in project_rule_ids)
    has_hard_negative_support = bool(project_rule_ids) and any(rule_id in hard_negative_rule_ids for rule_id in project_rule_ids)
    has_validator_support = bool(dataset.get("current_validator_support", False)) or (
        bool(project_rule_ids) and any(rule_id in validator_rule_ids for rule_id in project_rule_ids)
    )

    initial = RuleCapability(
        taxonomy_key=taxonomy_key,
        domain=domain,
        entry_type=entry_type,
        title=str(entry.get("title") or ""),
        orfogrammka_id=str(entry.get("orfogrammka_id") or ""),
        project_rule_ids=project_rule_ids,
        implementation_status=status,
        requires=requires,
        executable=executable,
        training_eligible=False,
        training_decision=str(dataset.get("training_eligibility_decision") or ""),
        training_reason=str(dataset.get("training_eligibility_reason") or ""),
        has_candidate_path=has_candidate_path,
        has_synthetic_support=has_synthetic_support,
        has_hard_negative_support=has_hard_negative_support,
        has_validator_support=has_validator_support,
        has_dictionary_support=has_dictionary_support,
        has_syntax_support=has_syntax_support,
        has_morphology_support=has_morphology_support,
        has_ner_support=has_ner_support,
        risk_level=str(dataset.get("risk_level") or _risk_level(project_rule_ids, requires)),
    )
    decision, reason = resolve_training_decision(initial)
    return RuleCapability(
        **{
            **asdict(initial),
            "training_eligible": decision in TRAINING_INCLUDE_DECISIONS,
            "training_decision": decision,
            "training_reason": reason,
        }
    )


def resolve_training_decision(capability: RuleCapability) -> tuple[str, str]:
    existing_decision = _existing_decision(capability)
    requires = {item.lower() for item in capability.requires}
    rule_ids = set(capability.project_rule_ids)

    if capability.entry_type == "group" or capability.implementation_status == "metadata_only":
        return BLOCK_METADATA_ONLY, "metadata_only taxonomy entry"
    if capability.implementation_status == "planned":
        return BLOCK_PLANNED, "planned taxonomy entry has no executable training path"
    if capability.implementation_status == "disabled":
        return BLOCK_DISABLED, "rule disabled in taxonomy"
    if rule_ids & NEVER_TRAIN_RULE_IDS:
        return BLOCK_METADATA_ONLY, "normalization or neural bucket is not a bounded training rule"
    if "dictionary" in requires and not capability.has_dictionary_support:
        return BLOCK_NEEDS_DICTIONARY, "requires dictionary support but no loadable dictionary is configured"
    if "syntax" in requires and not capability.has_syntax_support:
        return BLOCK_NEEDS_SYNTAX, "requires syntax support but syntax analyzer is unavailable or disabled"
    if "morphology" in requires and not capability.has_morphology_support:
        return BLOCK_NEEDS_MORPHOLOGY, "requires morphology support but morphology analyzer is unavailable"
    if "ner" in requires and not capability.has_ner_support:
        return BLOCK_NEEDS_NER, "requires explicit NER support; optional syntax NER labels are not enough"
    if existing_decision.startswith(BLOCK_DECISION_PREFIX):
        return _normalize_existing_block(existing_decision), _existing_reason(capability) or "blocked by source taxonomy decision"
    if capability.executable and not capability.has_candidate_path:
        return BLOCK_NO_CANDIDATE_PATH, "executable taxonomy entry has no runtime candidate path"
    if not capability.executable:
        return MINING_ONLY, "taxonomy entry is not executable; keep visible for mining/reporting"
    if capability.has_candidate_path and not capability.has_synthetic_support:
        return EVAL_ONLY, "candidate path exists but no safe synthetic/corruption operator is available"
    if capability.has_candidate_path and capability.has_synthetic_support and not capability.has_validator_support:
        return INCLUDE_AFTER_VALIDATOR, "candidate and synthetic paths exist but validator support is missing"
    if capability.has_candidate_path and capability.has_synthetic_support and not capability.has_hard_negative_support:
        return EVAL_ONLY, "candidate and synthetic paths exist but hard-negative support is missing"
    if existing_decision in STRONGER_EXISTING_DECISIONS:
        return existing_decision, _existing_reason(capability) or "preserved stronger source taxonomy decision"
    if capability.has_candidate_path and capability.has_synthetic_support and capability.has_hard_negative_support and capability.has_validator_support:
        return INCLUDE_NOW, "candidate, synthetic, hard-negative, and validator gates are available"
    if capability.has_candidate_path:
        return EVAL_ONLY, "candidate path exists but training gates are incomplete"
    return BLOCK_NO_CANDIDATE_PATH, "no runtime candidate path"


def active_rule_ids_for_training(capabilities: list[RuleCapability]) -> list[str]:
    active: set[str] = set()
    for capability in capabilities:
        if capability.training_decision not in TRAINING_INCLUDE_DECISIONS:
            continue
        active.update(capability.project_rule_ids)
    return sorted(rule_id for rule_id in active if rule_id and rule_id not in SERVICE_RULE_IDS)


def blocked_rule_capabilities(capabilities: list[RuleCapability]) -> list[RuleCapability]:
    return [capability for capability in capabilities if capability.training_decision.startswith(BLOCK_DECISION_PREFIX)]


def capability_matrix_frame(capabilities: Iterable[RuleCapability]) -> pd.DataFrame:
    rows = [asdict(capability) for capability in capabilities]
    columns = list(RuleCapability.__dataclass_fields__)
    return pd.DataFrame(rows, columns=columns)


def write_rule_capability_reports(capabilities: list[RuleCapability], reports_dir: str | Path) -> dict[str, str]:
    output = Path(reports_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = _csv_ready_frame(capability_matrix_frame(capabilities))
    blocked = frame[frame["training_decision"].astype(str).str.startswith(BLOCK_DECISION_PREFIX)] if not frame.empty else frame
    eligibility = frame[frame["training_decision"].isin(REPORT_VISIBLE_INCLUDE_DECISIONS)] if not frame.empty else frame
    missing = _missing_module_frame(capabilities)
    paths = {
        "rule_capability_matrix": output / "rule_capability_matrix.csv",
        "rule_eligibility_report": output / "rule_eligibility_report.csv",
        "blocked_rules_report": output / "blocked_rules_report.csv",
        "missing_module_rules_report": output / "missing_module_rules_report.csv",
    }
    frame.to_csv(paths["rule_capability_matrix"], index=False)
    eligibility.to_csv(paths["rule_eligibility_report"], index=False)
    blocked.to_csv(paths["blocked_rules_report"], index=False)
    missing.to_csv(paths["missing_module_rules_report"], index=False)
    return {key: str(path) for key, path in paths.items()}


def capability_manifest_fields(capabilities: list[RuleCapability]) -> dict[str, Any]:
    decision_counts = Counter(capability.training_decision for capability in capabilities)
    blocked_rule_ids = sorted(
        {
            rule_id
            for capability in capabilities
            if capability.training_decision.startswith(BLOCK_DECISION_PREFIX)
            for rule_id in capability.project_rule_ids
        }
    )
    eval_only_rule_ids = sorted(
        {rule_id for capability in capabilities if capability.training_decision == EVAL_ONLY for rule_id in capability.project_rule_ids}
    )
    mining_only_rule_ids = sorted(
        {rule_id for capability in capabilities if capability.training_decision == MINING_ONLY for rule_id in capability.project_rule_ids}
    )
    return {
        "rule_capability_counts": dict(sorted(decision_counts.items())),
        "active_rule_ids": active_rule_ids_for_training(capabilities),
        "blocked_rule_ids": blocked_rule_ids,
        "eval_only_rule_ids": eval_only_rule_ids,
        "mining_only_rule_ids": mining_only_rule_ids,
        "missing_dictionary_rule_count": _missing_dependency_count(capabilities, "dictionary"),
        "missing_syntax_rule_count": _missing_dependency_count(capabilities, "syntax"),
        "missing_morphology_rule_count": _missing_dependency_count(capabilities, "morphology"),
        "missing_ner_rule_count": _missing_dependency_count(capabilities, "ner"),
        "capability_rule_ids": sorted({rule_id for capability in capabilities for rule_id in capability.project_rule_ids}),
    }


def capability_training_audit_errors(rule_ids: Iterable[str], capabilities: list[RuleCapability]) -> list[str]:
    capability_rule_ids = {rule_id for capability in capabilities for rule_id in capability.project_rule_ids}
    blocked_rule_ids = {
        rule_id
        for capability in capabilities
        if capability.training_decision.startswith(BLOCK_DECISION_PREFIX)
        for rule_id in capability.project_rule_ids
    }
    errors: list[str] = []
    for raw_rule_id in sorted({normalize_rule_id(rule_id) for rule_id in rule_ids if str(rule_id)}):
        if raw_rule_id in SERVICE_RULE_IDS:
            continue
        if raw_rule_id in blocked_rule_ids:
            errors.append(f"blocked_rule_in_training:{raw_rule_id}")
        if raw_rule_id not in capability_rule_ids:
            errors.append(f"rule_missing_from_capability_matrix:{raw_rule_id}")
    return errors


def _default_available_modules(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = _load_config(config_path)
    registry_rule_ids = _registry_rule_ids()
    operator_rule_ids = _operator_rule_ids()
    syntax_rule_ids = _syntax_supported_rule_ids()
    corruption_rule_ids = _rule_corruption_supported_rule_ids(registry_rule_ids)
    candidate_rule_ids = registry_rule_ids | operator_rule_ids | syntax_rule_ids | DICTIONARY_CANDIDATE_RULE_IDS
    synthetic_rule_ids = operator_rule_ids | syntax_rule_ids | corruption_rule_ids
    hard_negative_rule_ids = operator_rule_ids | syntax_rule_ids | _validator_supported_rule_ids()
    validator_rule_ids = _validator_supported_rule_ids() | syntax_rule_ids | {
        rule_id for rule_id in registry_rule_ids if rule_id not in NEVER_TRAIN_RULE_IDS
    }
    return {
        "candidate_rule_ids": candidate_rule_ids,
        "synthetic_rule_ids": synthetic_rule_ids,
        "hard_negative_rule_ids": hard_negative_rule_ids,
        "validator_rule_ids": validator_rule_ids,
        "dictionary_supported": _dictionary_supported(config),
        "syntax_supported": _syntax_supported(config),
        "morphology_supported": _morphology_supported(config),
        "ner_supported": _ner_supported(config),
    }


def _registry_rule_ids() -> set[str]:
    try:
        from src.rules.registry import all_rules

        return {str(rule.spec.id) for rule in all_rules() if getattr(rule, "spec", None) is not None}
    except Exception:
        return set()


def _operator_rule_ids() -> set[str]:
    try:
        from src.data.corruption_operators import build_default_operator_registry

        return set(build_default_operator_registry().rule_ids())
    except Exception:
        return set()


def _syntax_supported_rule_ids() -> set[str]:
    try:
        from src.rules.syntax_synthetic import SUPPORTED_SYNTAX_RULE_IDS

        return {str(rule_id) for rule_id in SUPPORTED_SYNTAX_RULE_IDS}
    except Exception:
        return set()


def _rule_corruption_supported_rule_ids(registry_rule_ids: set[str]) -> set[str]:
    supported: set[str] = set()
    try:
        from src.rules.registry import rule_by_id

        for rule_id in registry_rule_ids:
            rule = rule_by_id(rule_id)
            if rule is not None and getattr(rule, "generate_corruptions", None) is not None:
                supported.add(rule_id)
    except Exception:
        return supported
    return supported


def _validator_supported_rule_ids() -> set[str]:
    try:
        from src.validation import strict_validator as validator

        names = (
            "QUOTE_NORMALIZATION_RULE_IDS",
            "N_NN_RULE_IDS",
            "NE_SPLIT_JOIN_RULE_IDS",
            "NI_RULE_IDS",
            "DIRECT_SPEECH_RULE_IDS",
            "SYNTAX_PUNCTUATION_RULE_IDS",
            "ASYNDETIC_RULE_IDS",
            "RISKY_KNOWN_SOURCE_LEXICAL_RULE_IDS",
            "ALWAYS_UNSAFE_LEXICAL_RULE_IDS",
        )
        result: set[str] = set()
        for name in names:
            result.update(str(rule_id) for rule_id in getattr(validator, name, frozenset()))
        result.update(DICTIONARY_CANDIDATE_RULE_IDS)
        return result
    except Exception:
        return set(DICTIONARY_CANDIDATE_RULE_IDS)


def _load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _dictionary_supported(config: dict[str, Any]) -> bool:
    dictionary_config = dict(config.get("dictionary", {}) or {})
    if not bool(dictionary_config.get("enabled", False)):
        return False
    raw_path = dictionary_config.get("lexicon_path")
    if not raw_path:
        return False
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return False
    try:
        from src.config.dictionary import load_dictionary_lexicon

        return bool(load_dictionary_lexicon(path, max_entries=1))
    except Exception:
        return False


def _syntax_supported(config: dict[str, Any]) -> bool:
    syntax_config = dict((config.get("nlp", {}) or {}).get("syntax", {}) or {})
    if not bool(syntax_config.get("enabled", True)):
        return False
    return importlib.util.find_spec("src.nlp.syntax_analyzer") is not None and importlib.util.find_spec("natasha") is not None


def _morphology_supported(config: dict[str, Any]) -> bool:
    del config
    return importlib.util.find_spec("src.candidates.morphology") is not None and (
        importlib.util.find_spec("pymorphy3") is not None or importlib.util.find_spec("pymorphy2") is not None
    )


def _ner_supported(config: dict[str, Any]) -> bool:
    nlp_config = dict(config.get("nlp", {}) or {})
    ner_config = dict(nlp_config.get("ner", {}) or {})
    return bool(ner_config.get("enabled", False)) and importlib.util.find_spec("src.nlp.ner") is not None


def _normalized_rule_ids(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        rule_id = normalize_rule_id(value)
        if rule_id == UNKNOWN_RULE_ID or rule_id in result:
            continue
        result.append(rule_id)
    return result


def _normalized_requires(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        item = str(value).strip().lower()
        if item and item not in result:
            result.append(item)
    return result


def _module_set(modules: dict[str, Any], key: str) -> set[str]:
    raw = modules.get(key, set())
    if isinstance(raw, str):
        return {raw}
    return {str(item) for item in (raw or set())}


def _module_bool(modules: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        if key in modules:
            return bool(modules[key])
    return False


def _existing_decision(capability: RuleCapability) -> str:
    return str(capability.training_decision or "")


def _existing_reason(capability: RuleCapability) -> str:
    return str(capability.training_reason or "")


def _normalize_existing_block(decision: str) -> str:
    if decision == BLOCK_NO_CANDIDATE:
        return BLOCK_NO_CANDIDATE_PATH
    return decision


def _risk_level(rule_ids: Iterable[str], requires: Iterable[str]) -> str:
    req = {item.lower() for item in requires}
    ids = set(rule_ids)
    if "ner" in req or "syntax" in req:
        return "high"
    if ids & DICTIONARY_CANDIDATE_RULE_IDS or "dictionary" in req or "morphology" in req:
        return "medium"
    return "low"


def _missing_dependency_count(capabilities: list[RuleCapability], dependency: str) -> int:
    attr = {
        "dictionary": "has_dictionary_support",
        "syntax": "has_syntax_support",
        "morphology": "has_morphology_support",
        "ner": "has_ner_support",
    }[dependency]
    return sum(1 for capability in capabilities if dependency in capability.requires and not bool(getattr(capability, attr)))


def _missing_module_frame(capabilities: list[RuleCapability]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for capability in capabilities:
        missing = []
        if "dictionary" in capability.requires and not capability.has_dictionary_support:
            missing.append("dictionary")
        if "syntax" in capability.requires and not capability.has_syntax_support:
            missing.append("syntax")
        if "morphology" in capability.requires and not capability.has_morphology_support:
            missing.append("morphology")
        if "ner" in capability.requires and not capability.has_ner_support:
            missing.append("ner")
        if not missing:
            continue
        rows.append(
            {
                "taxonomy_key": capability.taxonomy_key,
                "domain": capability.domain,
                "title": capability.title,
                "project_rule_ids": json.dumps(capability.project_rule_ids, ensure_ascii=False),
                "missing_modules": json.dumps(missing, ensure_ascii=False),
                "training_decision": capability.training_decision,
                "training_reason": capability.training_reason,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["taxonomy_key", "domain", "title", "project_rule_ids", "missing_modules", "training_decision", "training_reason"],
    )


def _csv_ready_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    for column in ("project_rule_ids", "requires"):
        if column in result:
            result[column] = result[column].map(lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value)
    return result
