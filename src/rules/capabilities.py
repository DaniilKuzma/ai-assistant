from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import yaml

from src.config.candidate_dataset_config import candidate_dataset_rule_activation
from src.rules.coverage_matrix import iter_coverage_entries, load_rules_coverage
from src.rules.rule_ids import RULE_ID_ALIASES, UNKNOWN_RULE_ID, normalize_rule_id


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

PRODUCTION_READY_DECISIONS = frozenset({INCLUDE_NOW})
SAFE_TRAINING_CANDIDATE_DECISIONS = frozenset({INCLUDE_NOW, INCLUDE_AFTER_THRESHOLD_CALIBRATION, INCLUDE_AFTER_TRAINING})
VALIDATOR_DEPENDENT_DECISIONS = frozenset({INCLUDE_AFTER_VALIDATOR})
BLOCKED_DECISIONS = frozenset(
    {
        BLOCK_METADATA_ONLY,
        BLOCK_PLANNED,
        BLOCK_DISABLED,
        BLOCK_NEEDS_DICTIONARY,
        BLOCK_NEEDS_SYNTAX,
        BLOCK_NEEDS_MORPHOLOGY,
        BLOCK_NEEDS_NER,
        BLOCK_NO_CANDIDATE_PATH,
        BLOCK_NO_CANDIDATE,
        BLOCK_NO_SYNTHETIC_OPERATOR,
        MINING_ONLY,
        EVAL_ONLY,
    }
)
TRAINING_INCLUDE_DECISIONS = PRODUCTION_READY_DECISIONS
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
ACTIVATION_STAGE_BY_DECISION = {
    INCLUDE_NOW: "production_ready",
    INCLUDE_AFTER_THRESHOLD_CALIBRATION: "threshold_calibration",
    INCLUDE_AFTER_TRAINING: "needs_training",
    INCLUDE_AFTER_VALIDATOR: "validator_dependent",
}
_VALIDATOR_PROBE_CACHE: dict[tuple[str, int, int], dict[str, Any]] = {}
VALIDATOR_PROBE_REPORT_COLUMNS = [
    "rule_id",
    "status",
    "reason",
    "example_count",
    "passed_examples",
]
EXPANDED_ACTIVATION_CANDIDATE_COLUMNS = [
    "rule_id",
    "taxonomy_key",
    "title",
    "decision",
    "activation_bucket",
    "has_candidate_path",
    "has_synthetic_support",
    "has_syntax_support",
    "has_hard_negative_support",
    "has_validator_support",
    "empirical_validator_support",
    "generated_probe_count",
    "verifier_pass_count",
    "candidate_recall",
    "included",
    "reason",
]
EXPANDED_ACTIVATION_BLOCKED_COLUMNS = [
    "rule_id",
    "taxonomy_key",
    "title",
    "blocker",
    "reason",
    "missing_component",
]
ACTIVE_RULE_ACTIVATION_STAGE_COLUMNS = [
    "rule_id",
    "activation_stage",
    "production_ready",
    "source",
    "included",
    "reason",
]


@dataclass
class RuleActivationPolicy:
    mode: str
    include_decisions: set[str]
    include_after_validator_when_runtime_supported: bool
    require_candidate_path: bool
    require_synthetic_support: bool
    require_hard_negative_support: bool
    require_validator_support_or_empirical_pass: bool
    exclude_blocked: bool
    exclude_ner_required_without_ner: bool
    min_empirical_validator_passes: int
    max_empirical_validator_examples: int
    expected_min_production_ready_rule_count: int
    expected_min_training_candidate_rule_count: int
    target_training_candidate_rule_count: int
    fail_below_min_training_candidate_rule_count: bool
    warn_below_target_training_candidate_rule_count: bool
    expected_min_final_active_rule_count: int
    target_final_active_rule_count: int
    fail_below_final_active_rule_count: bool
    warn_below_target_final_active_rule_count: bool


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


def load_rule_capabilities(rules_yaml_path: str | Path, config: Mapping[str, Any] | None = None) -> list[RuleCapability]:
    available_modules = _default_available_modules(config=config)
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
    if rule_ids and rule_ids <= NEVER_TRAIN_RULE_IDS:
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


def activation_policy_from_config(config: Mapping[str, Any]) -> RuleActivationPolicy:
    raw = candidate_dataset_rule_activation(config)
    mode = str(raw.get("mode") or "strict").strip() or "strict"
    if "include_decisions" in raw:
        include_decisions = {str(item) for item in raw.get("include_decisions") or [] if str(item)}
    elif mode == "expanded_safe":
        include_decisions = set(SAFE_TRAINING_CANDIDATE_DECISIONS)
    else:
        include_decisions = set(PRODUCTION_READY_DECISIONS)
    return RuleActivationPolicy(
        mode=mode,
        include_decisions=include_decisions,
        include_after_validator_when_runtime_supported=_bool_config(
            raw,
            "include_after_validator_when_runtime_supported",
            mode == "expanded_safe",
        ),
        require_candidate_path=_bool_config(raw, "require_candidate_path", True),
        require_synthetic_support=_bool_config(raw, "require_synthetic_support", True),
        require_hard_negative_support=_bool_config(raw, "require_hard_negative_support", True),
        require_validator_support_or_empirical_pass=_bool_config(
            raw,
            "require_validator_support_or_empirical_pass",
            True,
        ),
        exclude_blocked=_bool_config(raw, "exclude_blocked", True),
        exclude_ner_required_without_ner=_bool_config(raw, "exclude_ner_required_without_ner", True),
        min_empirical_validator_passes=max(0, int(raw.get("min_empirical_validator_passes", 3) or 0)),
        max_empirical_validator_examples=max(1, int(raw.get("max_empirical_validator_examples", 10) or 10)),
        expected_min_production_ready_rule_count=max(0, int(raw.get("expected_min_production_ready_rule_count", 0) or 0)),
        expected_min_training_candidate_rule_count=max(0, int(raw.get("expected_min_training_candidate_rule_count", 0) or 0)),
        target_training_candidate_rule_count=max(0, int(raw.get("target_training_candidate_rule_count", 0) or 0)),
        fail_below_min_training_candidate_rule_count=_bool_config(
            raw,
            "fail_below_min_training_candidate_rule_count",
            False,
        ),
        warn_below_target_training_candidate_rule_count=_bool_config(
            raw,
            "warn_below_target_training_candidate_rule_count",
            False,
        ),
        expected_min_final_active_rule_count=max(0, int(raw.get("expected_min_final_active_rule_count", 0) or 0)),
        target_final_active_rule_count=max(0, int(raw.get("target_final_active_rule_count", 0) or 0)),
        fail_below_final_active_rule_count=_bool_config(raw, "fail_below_final_active_rule_count", False),
        warn_below_target_final_active_rule_count=_bool_config(raw, "warn_below_target_final_active_rule_count", False),
    )


def production_ready_rule_ids(capabilities: list[RuleCapability]) -> list[str]:
    return sorted(
        rule_id
        for rule_id in {
            rule_id
            for capability in capabilities
            if production_ready_flag_for_rule(capability)
            for rule_id in capability.project_rule_ids
        }
        if rule_id and rule_id not in SERVICE_RULE_IDS
    )


def training_candidate_rule_ids(
    capabilities: list[RuleCapability],
    policy: RuleActivationPolicy | None = None,
) -> list[str]:
    current_policy = policy or _strict_activation_policy()
    active: set[str] = set()
    for capability in capabilities:
        for rule_id in capability.project_rule_ids:
            result = _activation_result_for_rule(capability, rule_id, current_policy)
            if bool(result["include"]):
                active.add(rule_id)
    return sorted(rule_id for rule_id in active if rule_id and rule_id not in SERVICE_RULE_IDS)


def active_rule_ids_for_training(
    capabilities: list[RuleCapability],
    policy: RuleActivationPolicy | None = None,
) -> list[str]:
    return training_candidate_rule_ids(capabilities, policy=policy)


def expanded_training_candidate_rule_ids(
    capabilities: list[RuleCapability],
    policy: RuleActivationPolicy | None = None,
    config: Mapping[str, Any] | None = None,
) -> list[str]:
    current_policy = policy or _expanded_activation_policy(config)
    rows = expanded_activation_candidate_rows(capabilities, policy=current_policy, config=config)
    return sorted(str(row["rule_id"]) for row in rows if bool(row.get("included")))


def expanded_activation_source_map(
    capabilities: list[RuleCapability] | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, set[str]]:
    sources: dict[str, set[str]] = {}

    def add(rule_id: Any, source: str) -> None:
        normalized = normalize_rule_id(rule_id)
        if normalized in SERVICE_RULE_IDS:
            return
        sources.setdefault(normalized, set()).add(source)

    for capability in capabilities or []:
        for rule_id in capability.project_rule_ids:
            add(rule_id, "capability_matrix")

    for entry in iter_taxonomy_rules(DEFAULT_RULES_YAML_PATH):
        for rule_id in _normalized_rule_ids(
            entry.get("project_rule_ids")
            or entry.get("rules")
            or dict(entry.get("implementation", {}) or {}).get("rule_ids")
            or []
        ):
            add(rule_id, "configs_rules_yaml")

    for rule_id in _registry_rule_ids():
        add(rule_id, "rule_registry")
    for rule_id in _operator_rule_ids():
        add(rule_id, "corruption_operator_registry")
    for rule_id in _syntax_supported_rule_ids():
        add(rule_id, "syntax_module")
    for rule_id in _module_set(_default_available_modules(config=config), "candidate_rule_ids"):
        add(rule_id, "candidate_generator_registry")
    for rule_id in set(RULE_ID_ALIASES) | set(RULE_ID_ALIASES.values()):
        add(rule_id, "known_rule_ids")
    for rule_id in _legacy_candidate_backed_rule_ids():
        add(rule_id, "legacy_candidate_backed")

    return {rule_id: sources[rule_id] for rule_id in sorted(sources)}


def expanded_activation_candidate_rows(
    capabilities: list[RuleCapability],
    policy: RuleActivationPolicy | None = None,
    config: Mapping[str, Any] | None = None,
    evidence_by_rule: Mapping[str, Mapping[str, Any]] | None = None,
    final_active_rule_ids: Iterable[str] | None = None,
    under_quota_rule_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    current_policy = policy or _expanded_activation_policy(config)
    evidence = {normalize_rule_id(rule_id): dict(values) for rule_id, values in dict(evidence_by_rule or {}).items()}
    final_active_set = (
        {normalize_rule_id(rule_id) for rule_id in final_active_rule_ids if normalize_rule_id(rule_id)}
        if final_active_rule_ids is not None
        else None
    )
    under_quota_set = {normalize_rule_id(rule_id) for rule_id in under_quota_rule_ids or [] if normalize_rule_id(rule_id)}
    source_map = expanded_activation_source_map(capabilities, config=config)
    capability_by_rule = _best_capability_by_rule_id(capabilities)
    activation_by_rule = _best_activation_by_rule(activation_policy_rows(capabilities, current_policy))
    modules = _default_available_modules(config=config)
    rows: list[dict[str, Any]] = []
    for rule_id in sorted(source_map):
        if rule_id in SERVICE_RULE_IDS:
            continue
        capability = capability_by_rule.get(rule_id)
        activation = activation_by_rule.get(rule_id, {})
        gate_blocker = _expanded_gate_blocker(rule_id, capability, activation, current_policy)
        activation_include = bool(activation.get("include")) and not gate_blocker
        if final_active_set is None:
            included = activation_include
        else:
            included = rule_id in final_active_set
        if rule_id in under_quota_set:
            included = False
        row_evidence = evidence.get(rule_id, {})
        reason = _expanded_candidate_reason(
            included=included,
            activation=activation,
            gate_blocker=gate_blocker,
            under_quota=rule_id in under_quota_set,
            evidence=row_evidence,
        )
        probe_status = str(activation.get("validator_probe_status", ""))
        rows.append(
            {
                "rule_id": rule_id,
                "taxonomy_key": str(getattr(capability, "taxonomy_key", "") if capability is not None else ""),
                "title": str(getattr(capability, "title", "") if capability is not None else ""),
                "decision": str(getattr(capability, "training_decision", "") if capability is not None else ""),
                "activation_bucket": _expanded_activation_bucket(rule_id, capability, current_policy),
                "has_candidate_path": _expanded_component_flag(rule_id, capability, modules, "candidate_rule_ids", "has_candidate_path"),
                "has_synthetic_support": _expanded_component_flag(rule_id, capability, modules, "synthetic_rule_ids", "has_synthetic_support"),
                "has_syntax_support": bool(
                    (getattr(capability, "has_syntax_support", False) if capability is not None else False)
                    or rule_id in _module_set(modules, "synthetic_rule_ids")
                    and "syntax_module" in source_map.get(rule_id, set())
                ),
                "has_hard_negative_support": _expanded_component_flag(rule_id, capability, modules, "hard_negative_rule_ids", "has_hard_negative_support"),
                "has_validator_support": _expanded_component_flag(rule_id, capability, modules, "validator_rule_ids", "has_validator_support"),
                "empirical_validator_support": probe_status == "pass",
                "generated_probe_count": int(row_evidence.get("generated_probe_count", 0) or 0),
                "verifier_pass_count": int(row_evidence.get("verifier_pass_count", 0) or 0),
                "candidate_recall": float(row_evidence.get("candidate_recall", 0.0) or 0.0),
                "included": bool(included),
                "reason": reason,
                "source": ",".join(sorted(source_map.get(rule_id, set()))),
            }
        )
    return rows


def expanded_activation_blocked_rows(
    capabilities: list[RuleCapability],
    policy: RuleActivationPolicy | None = None,
    config: Mapping[str, Any] | None = None,
    evidence_by_rule: Mapping[str, Mapping[str, Any]] | None = None,
    final_active_rule_ids: Iterable[str] | None = None,
    under_quota_rule_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    current_policy = policy or _expanded_activation_policy(config)
    activation_by_rule = _best_activation_by_rule(activation_policy_rows(capabilities, current_policy))
    capability_by_rule = _best_capability_by_rule_id(capabilities)
    candidate_rows = expanded_activation_candidate_rows(
        capabilities,
        policy=current_policy,
        config=config,
        evidence_by_rule=evidence_by_rule,
        final_active_rule_ids=final_active_rule_ids,
        under_quota_rule_ids=under_quota_rule_ids,
    )
    rows: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        rule_id = str(candidate["rule_id"])
        if bool(candidate.get("included")):
            continue
        capability = capability_by_rule.get(rule_id)
        activation = activation_by_rule.get(rule_id, {})
        blocker = _expanded_gate_blocker(rule_id, capability, activation, current_policy) or str(candidate.get("reason") or "not_final_active")
        rows.append(
            {
                "rule_id": rule_id,
                "taxonomy_key": str(candidate.get("taxonomy_key", "")),
                "title": str(candidate.get("title", "")),
                "blocker": blocker,
                "reason": str(candidate.get("reason", "")),
                "missing_component": _missing_component_for_blocker(blocker),
            }
        )
    return rows


def activation_stage_for_rule(capability: RuleCapability, policy: RuleActivationPolicy) -> str:
    del policy
    decision = str(capability.training_decision or "")
    if decision in ACTIVATION_STAGE_BY_DECISION:
        return ACTIVATION_STAGE_BY_DECISION[decision]
    if decision.startswith(BLOCK_DECISION_PREFIX):
        return "blocked"
    if decision == EVAL_ONLY:
        return "eval_only"
    if decision == MINING_ONLY:
        return "mining_only"
    return "inactive"


def production_ready_flag_for_rule(capability: RuleCapability) -> bool:
    return str(capability.training_decision or "") in PRODUCTION_READY_DECISIONS


def blocked_rule_capabilities(capabilities: list[RuleCapability]) -> list[RuleCapability]:
    return [capability for capability in capabilities if capability.training_decision.startswith(BLOCK_DECISION_PREFIX)]


def capability_matrix_frame(capabilities: Iterable[RuleCapability]) -> pd.DataFrame:
    rows = [asdict(capability) for capability in capabilities]
    columns = list(RuleCapability.__dataclass_fields__)
    return pd.DataFrame(rows, columns=columns)


def write_rule_capability_reports(
    capabilities: list[RuleCapability],
    reports_dir: str | Path,
    policy: RuleActivationPolicy | None = None,
) -> dict[str, str]:
    current_policy = policy or _strict_activation_policy()
    output = Path(reports_dir)
    output.mkdir(parents=True, exist_ok=True)
    activation_rows = activation_policy_rows(capabilities, current_policy)
    activation_frame = pd.DataFrame(activation_rows)
    frame = _csv_ready_frame(_capability_matrix_with_activation(capabilities, activation_rows))
    blocked = frame[frame["training_decision"].astype(str).str.startswith(BLOCK_DECISION_PREFIX)] if not frame.empty else frame
    eligibility = activation_frame[
        activation_frame["training_decision"].isin(REPORT_VISIBLE_INCLUDE_DECISIONS)
    ] if not activation_frame.empty else activation_frame
    missing = _missing_module_frame(capabilities)
    policy_fields = capability_manifest_fields(capabilities, policy=current_policy)
    policy_report = pd.DataFrame([_policy_report_row(policy_fields)])
    validator_probe_rows = pd.DataFrame(
        policy_fields.get("validator_probe_summary", {}).get("rows", []),
        columns=VALIDATOR_PROBE_REPORT_COLUMNS,
    )
    paths = {
        "rule_capability_matrix": output / "rule_capability_matrix.csv",
        "rule_eligibility_report": output / "rule_eligibility_report.csv",
        "blocked_rules_report": output / "blocked_rules_report.csv",
        "missing_module_rules_report": output / "missing_module_rules_report.csv",
        "rule_activation_policy_report": output / "rule_activation_policy_report.csv",
        "training_candidate_rules": output / "training_candidate_rules.csv",
        "production_ready_rules": output / "production_ready_rules.csv",
        "validator_probe_report": output / "validator_probe_report.csv",
    }
    frame.to_csv(paths["rule_capability_matrix"], index=False)
    eligibility.to_csv(paths["rule_eligibility_report"], index=False)
    blocked.to_csv(paths["blocked_rules_report"], index=False)
    missing.to_csv(paths["missing_module_rules_report"], index=False)
    policy_report.to_csv(paths["rule_activation_policy_report"], index=False)
    activation_frame[activation_frame["training_candidate"].astype(bool)].to_csv(paths["training_candidate_rules"], index=False)
    activation_frame[activation_frame["production_ready"].astype(bool)].to_csv(paths["production_ready_rules"], index=False)
    validator_probe_rows.to_csv(paths["validator_probe_report"], index=False)
    return {key: str(path) for key, path in paths.items()}


def capability_manifest_fields(
    capabilities: list[RuleCapability],
    policy: RuleActivationPolicy | None = None,
    final_active_rule_ids: Iterable[str] | None = None,
    under_quota_rule_ids: Iterable[str] | None = None,
    expanded_activation_blocked_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    current_policy = policy or _strict_activation_policy()
    decision_counts = Counter(capability.training_decision for capability in capabilities)
    production_ids = production_ready_rule_ids(capabilities)
    training_ids = training_candidate_rule_ids(capabilities, policy=current_policy)
    default_active_ids = active_rule_ids_for_training(capabilities, policy=current_policy)
    final_ids = (
        sorted({normalize_rule_id(rule_id) for rule_id in final_active_rule_ids if normalize_rule_id(rule_id) not in SERVICE_RULE_IDS})
        if final_active_rule_ids is not None
        else default_active_ids
    )
    under_quota_ids = sorted({normalize_rule_id(rule_id) for rule_id in under_quota_rule_ids or [] if normalize_rule_id(rule_id) not in SERVICE_RULE_IDS})
    activation_rows = activation_policy_rows(capabilities, current_policy)
    validator_summary = _validator_probe_summary_from_rows(activation_rows)
    capability_by_rule = _best_capability_by_rule_id(capabilities)
    activation_stage_counter: Counter[str] = Counter()
    for rule_id in final_ids:
        capability = capability_by_rule.get(rule_id)
        if capability is None:
            continue
        activation_stage_counter[activation_stage_for_rule(capability, current_policy)] += 1
    activation_stage_counts = dict(sorted((stage, count) for stage, count in activation_stage_counter.items() if stage))
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
        "production_ready_rule_ids": production_ids,
        "production_ready_rule_count": len(production_ids),
        "training_candidate_rule_ids": training_ids,
        "training_candidate_rule_count": len(training_ids),
        "expanded_training_candidate_rule_ids": training_ids,
        "expanded_training_candidate_rule_count": len(training_ids),
        "active_rule_ids": final_ids,
        "active_rule_count": len(final_ids),
        "final_active_rule_ids": final_ids,
        "final_active_rule_count": len(final_ids),
        "under_quota_rule_ids": under_quota_ids,
        "activation_policy": _activation_policy_manifest(current_policy, len(production_ids), len(training_ids), len(final_ids)),
        "activation_stage_counts": activation_stage_counts,
        "expanded_activation_blocked_counts": dict(sorted((expanded_activation_blocked_counts or {}).items())),
        "target_training_candidate_rule_count": current_policy.target_training_candidate_rule_count,
        "target_final_active_rule_count": current_policy.target_final_active_rule_count,
        "validator_probe_summary": validator_summary,
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


def activation_policy_rows(capabilities: list[RuleCapability], policy: RuleActivationPolicy | None = None) -> list[dict[str, Any]]:
    current_policy = policy or _strict_activation_policy()
    rows: list[dict[str, Any]] = []
    for capability in capabilities:
        for rule_id in capability.project_rule_ids:
            rows.append(_activation_result_for_rule(capability, rule_id, current_policy))
    return rows


def annotate_activation_columns(
    frame: pd.DataFrame,
    capabilities: list[RuleCapability],
    policy: RuleActivationPolicy | None = None,
) -> pd.DataFrame:
    if frame.empty:
        result = frame.copy()
        if "activation_stage" not in result:
            result["activation_stage"] = []
        if "production_ready" not in result:
            result["production_ready"] = []
        return result
    current_policy = policy or _strict_activation_policy()
    capability_by_rule = _best_capability_by_rule_id(capabilities)
    result = frame.copy()
    stages: list[str] = []
    production_flags: list[bool] = []
    metadata_values: list[str] = []
    for row in result.to_dict("records"):
        rule_ids = _row_activation_rule_ids(row)
        active_rule_ids = [rule_id for rule_id in rule_ids if rule_id not in SERVICE_RULE_IDS]
        row_stages = {
            activation_stage_for_rule(capability_by_rule[rule_id], current_policy)
            for rule_id in active_rule_ids
            if rule_id in capability_by_rule
        }
        stage = "mixed" if len(row_stages) > 1 else (next(iter(row_stages)) if row_stages else "")
        production_ready = bool(active_rule_ids) and all(
            rule_id in capability_by_rule and production_ready_flag_for_rule(capability_by_rule[rule_id])
            for rule_id in active_rule_ids
        )
        metadata = _json_dict(row.get("metadata"))
        metadata["activation_stage"] = stage
        metadata["production_ready"] = production_ready
        stages.append(stage)
        production_flags.append(production_ready)
        metadata_values.append(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    result["activation_stage"] = stages
    result["production_ready"] = production_flags
    if "metadata" in result:
        result["metadata"] = metadata_values
    return result


def activation_stage_counts_from_frame(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty or "activation_stage" not in frame:
        return {}
    counts = Counter(str(value) for value in frame["activation_stage"].fillna("").tolist() if str(value))
    return dict(sorted(counts.items()))


def _expanded_activation_policy(config: Mapping[str, Any] | None = None) -> RuleActivationPolicy:
    if config is not None:
        return activation_policy_from_config(config)
    return activation_policy_from_config({"data": {"candidate_opportunity": {"rule_activation": {"mode": "expanded_safe"}}}})


def _best_activation_by_rule(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        rule_id = str(row.get("rule_id") or "")
        if not rule_id:
            continue
        current = result.get(rule_id)
        if current is None or _decision_priority(str(row.get("training_decision", ""))) < _decision_priority(
            str(current.get("training_decision", ""))
        ):
            result[rule_id] = dict(row)
            continue
        if current is not None and bool(row.get("include")) and not bool(current.get("include")):
            result[rule_id] = dict(row)
    return result


def _legacy_candidate_backed_rule_ids() -> set[str]:
    try:
        from src.data.training_dataset import LEGACY_CANDIDATE_BACKED_RULE_IDS

        return {normalize_rule_id(rule_id) for rule_id in LEGACY_CANDIDATE_BACKED_RULE_IDS}
    except Exception:
        return set()


def _expanded_activation_bucket(rule_id: str, capability: RuleCapability | None, policy: RuleActivationPolicy) -> str:
    if _expanded_gate_blocker(rule_id, capability, {}, policy):
        return "blocked"
    if capability is None:
        return "blocked"
    return activation_stage_for_rule(capability, policy)


def _expanded_component_flag(
    rule_id: str,
    capability: RuleCapability | None,
    modules: Mapping[str, Any],
    module_key: str,
    capability_attr: str,
) -> bool:
    if capability is not None and bool(getattr(capability, capability_attr, False)):
        return True
    return rule_id in _module_set(dict(modules), module_key)


def _expanded_gate_blocker(
    rule_id: str,
    capability: RuleCapability | None,
    activation: Mapping[str, Any],
    policy: RuleActivationPolicy,
) -> str:
    normalized_rule_id = normalize_rule_id(rule_id)
    if normalized_rule_id in SERVICE_RULE_IDS:
        return "service_rule"
    if normalized_rule_id in NEVER_TRAIN_RULE_IDS:
        return "broad_normalization_bucket"
    if capability is None:
        return "missing_capability_or_taxonomy"
    requires = {item.lower() for item in capability.requires}
    decision = str(capability.training_decision or "")
    if policy.exclude_ner_required_without_ner and "ner" in requires and not capability.has_ner_support:
        return "needs_NER"
    if decision.startswith(BLOCK_DECISION_PREFIX):
        return decision
    if decision == MINING_ONLY:
        return MINING_ONLY
    if decision == EVAL_ONLY and decision not in policy.include_decisions:
        return EVAL_ONLY
    reason = str(activation.get("activation_exclusion_reason", "") or "")
    if reason and not bool(activation.get("include")):
        return reason
    return _structural_gate_reason(capability, policy)


def _expanded_candidate_reason(
    *,
    included: bool,
    activation: Mapping[str, Any],
    gate_blocker: str,
    under_quota: bool,
    evidence: Mapping[str, Any],
) -> str:
    if included:
        return str(activation.get("activation_exclusion_reason") or "gates_passed")
    if under_quota:
        return str(evidence.get("reason") or "under_quota")
    if gate_blocker:
        return gate_blocker
    return str(activation.get("activation_exclusion_reason") or "not_final_active")


def _missing_component_for_blocker(blocker: str) -> str:
    text = str(blocker or "")
    if text in {"needs_NER", BLOCK_NEEDS_NER}:
        return "ner"
    if text in {"no_candidate_path", BLOCK_NO_CANDIDATE_PATH, BLOCK_NO_CANDIDATE}:
        return "candidate_path"
    if text in {"no_synthetic_support", BLOCK_NO_SYNTHETIC_OPERATOR}:
        return "synthetic_support"
    if text == "no_hard_negative_support":
        return "hard_negative_support"
    if "validator" in text:
        return "validator_support"
    if text == "broad_normalization_bucket":
        return "concrete_candidate_operation"
    if text == "missing_capability_or_taxonomy":
        return "capability_matrix"
    if text in {EVAL_ONLY, MINING_ONLY} or text.startswith(BLOCK_DECISION_PREFIX):
        return "activation_policy"
    return ""


def _strict_activation_policy() -> RuleActivationPolicy:
    return RuleActivationPolicy(
        mode="strict",
        include_decisions=set(PRODUCTION_READY_DECISIONS),
        include_after_validator_when_runtime_supported=False,
        require_candidate_path=True,
        require_synthetic_support=True,
        require_hard_negative_support=True,
        require_validator_support_or_empirical_pass=True,
        exclude_blocked=True,
        exclude_ner_required_without_ner=True,
        min_empirical_validator_passes=3,
        max_empirical_validator_examples=10,
        expected_min_production_ready_rule_count=0,
        expected_min_training_candidate_rule_count=0,
        target_training_candidate_rule_count=0,
        fail_below_min_training_candidate_rule_count=False,
        warn_below_target_training_candidate_rule_count=False,
        expected_min_final_active_rule_count=0,
        target_final_active_rule_count=0,
        fail_below_final_active_rule_count=False,
        warn_below_target_final_active_rule_count=False,
    )


def _activation_result_for_rule(
    capability: RuleCapability,
    rule_id: str,
    policy: RuleActivationPolicy,
) -> dict[str, Any]:
    normalized_rule_id = normalize_rule_id(rule_id)
    stage = activation_stage_for_rule(capability, policy)
    include = False
    reason = ""
    probe_row: dict[str, Any] | None = None
    decision = str(capability.training_decision or "")
    if normalized_rule_id in SERVICE_RULE_IDS:
        reason = "service_rule"
    elif normalized_rule_id in NEVER_TRAIN_RULE_IDS:
        reason = BLOCK_METADATA_ONLY
    elif policy.exclude_blocked and _decision_is_blocked(decision, policy):
        reason = decision or "blocked_decision"
    elif decision in VALIDATOR_DEPENDENT_DECISIONS:
        if not policy.include_after_validator_when_runtime_supported:
            reason = "validator_dependent_not_enabled"
        elif not _structural_gates_pass(capability, policy):
            reason = _structural_gate_reason(capability, policy)
        elif capability.has_validator_support:
            include = True
            reason = "runtime_validator_supported"
        else:
            probe_row = _empirical_validator_probe_for_capability(_single_rule_capability(capability, normalized_rule_id), policy)
            include = bool(probe_row.get("empirical_validator_support"))
            reason = "" if include else str(probe_row.get("reason") or "validator_probe_failed")
    elif decision in policy.include_decisions:
        if not _structural_gates_pass(capability, policy):
            reason = _structural_gate_reason(capability, policy)
        elif policy.require_validator_support_or_empirical_pass and not capability.has_validator_support:
            probe_row = _empirical_validator_probe_for_capability(_single_rule_capability(capability, normalized_rule_id), policy)
            include = bool(probe_row.get("empirical_validator_support"))
            reason = "" if include else str(probe_row.get("reason") or "validator_probe_failed")
        else:
            include = True
            reason = "policy_decision_included"
    else:
        reason = decision or "decision_not_in_policy"
    return {
        "taxonomy_key": capability.taxonomy_key,
        "domain": capability.domain,
        "rule_id": normalized_rule_id,
        "training_decision": decision,
        "training_reason": capability.training_reason,
        "activation_stage": stage,
        "production_ready": production_ready_flag_for_rule(capability),
        "training_candidate": bool(include),
        "include": bool(include),
        "include_in_training": bool(include),
        "activation_exclusion_reason": reason,
        "has_candidate_path": capability.has_candidate_path,
        "has_synthetic_support": capability.has_synthetic_support,
        "has_hard_negative_support": capability.has_hard_negative_support,
        "has_validator_support": capability.has_validator_support,
        "requires": json.dumps(capability.requires, ensure_ascii=False),
        "validator_probe_status": str((probe_row or {}).get("status", "")),
        "validator_probe_reason": str((probe_row or {}).get("reason", "")),
        "validator_probe_passed_examples": int((probe_row or {}).get("passed_examples", 0) or 0),
        "validator_probe_example_count": int((probe_row or {}).get("example_count", 0) or 0),
    }


def _structural_gates_pass(capability: RuleCapability, policy: RuleActivationPolicy) -> bool:
    return _structural_gate_reason(capability, policy) == ""


def _structural_gate_reason(capability: RuleCapability, policy: RuleActivationPolicy) -> str:
    if policy.exclude_ner_required_without_ner and "ner" in {item.lower() for item in capability.requires} and not capability.has_ner_support:
        return "needs_NER"
    if policy.require_candidate_path and not capability.has_candidate_path:
        return "no_candidate_path"
    if policy.require_synthetic_support and not capability.has_synthetic_support:
        return "no_synthetic_support"
    if policy.require_hard_negative_support and not capability.has_hard_negative_support:
        return "no_hard_negative_support"
    return ""


def _decision_is_blocked(decision: str, policy: RuleActivationPolicy) -> bool:
    if str(decision).startswith(BLOCK_DECISION_PREFIX):
        return True
    if decision in {EVAL_ONLY, MINING_ONLY} and decision not in policy.include_decisions:
        return True
    return decision in BLOCKED_DECISIONS and decision not in policy.include_decisions


def _empirical_validator_probe_for_capability(
    capability: RuleCapability,
    policy: RuleActivationPolicy,
) -> dict[str, Any]:
    rule_id = capability.project_rule_ids[0] if capability.project_rule_ids else ""
    cache_key = (rule_id, int(policy.min_empirical_validator_passes), int(policy.max_empirical_validator_examples))
    if cache_key in _VALIDATOR_PROBE_CACHE:
        return dict(_VALIDATOR_PROBE_CACHE[cache_key])
    examples, source_name = _validator_probe_examples(rule_id, policy.max_empirical_validator_examples)
    if not examples:
        row = _validator_probe_row(rule_id, "no_generation_probe_source", 0, 0, source_name, {})
        _VALIDATOR_PROBE_CACHE[cache_key] = row
        return dict(row)
    try:
        from src.candidates.candidate_generator import CandidateGenerator
        from src.data.atomic_verifier import verify_atomic_positive

        candidate_generator = CandidateGenerator.from_config(_load_config(DEFAULT_CONFIG_PATH))
    except Exception:
        candidate_generator = None
        from src.data.atomic_verifier import verify_atomic_positive

    passed = 0
    reasons: Counter[str] = Counter()
    for source, target in examples[: policy.max_empirical_validator_examples]:
        verification = verify_atomic_positive(source, target, rule_id, candidate_generator=candidate_generator)
        if verification.passed:
            passed += 1
        else:
            reasons[str(verification.reason)] += 1
    status = "pass" if passed >= policy.min_empirical_validator_passes else "fail"
    reason = "" if status == "pass" else "validator_probe_failed"
    row = _validator_probe_row(rule_id, reason, len(examples[: policy.max_empirical_validator_examples]), passed, source_name, reasons)
    row["status"] = status
    row["empirical_validator_support"] = status == "pass"
    _VALIDATOR_PROBE_CACHE[cache_key] = row
    return dict(row)


def _validator_probe_examples(rule_id: str, max_examples: int) -> tuple[list[tuple[str, str]], str]:
    examples = _operator_probe_examples(rule_id, max_examples)
    if examples:
        return examples, "operator"
    examples = _syntax_probe_examples(rule_id, max_examples)
    if examples:
        return examples, "syntax"
    examples = _backfill_probe_examples(rule_id, max_examples)
    if examples:
        return examples, "backfill"
    return [], ""


def _operator_probe_examples(rule_id: str, max_examples: int) -> list[tuple[str, str]]:
    try:
        from src.data.corruption_operators import build_default_operator_registry
        from src.data.matrix_eval_dataset import DEFAULT_CLEAN_SENTENCES

        operator = build_default_operator_registry().get(rule_id)
        if operator is None:
            return []
        examples: list[tuple[str, str]] = []
        for clean in DEFAULT_CLEAN_SENTENCES[: max(20, max_examples * 3)]:
            for opportunity in operator.find_opportunities(clean)[:3]:
                try:
                    result = operator.corrupt(clean, opportunity)
                except Exception:
                    continue
                if result.source != result.target:
                    examples.append((result.source, result.target))
                if len(examples) >= max_examples:
                    return examples
        return examples
    except Exception:
        return []


def _syntax_probe_examples(rule_id: str, max_examples: int) -> list[tuple[str, str]]:
    try:
        from src.rules.syntax_synthetic import build_syntax_eval_examples

        frame = build_syntax_eval_examples(selected_rule_ids=[rule_id], min_examples_per_rule=max_examples)
        if frame.empty:
            return []
        return [
            (str(row["source"]), str(row["target"]))
            for row in frame.head(max_examples).to_dict("records")
            if str(row.get("source", "")) != str(row.get("target", ""))
        ]
    except Exception:
        return []


def _backfill_probe_examples(rule_id: str, max_examples: int) -> list[tuple[str, str]]:
    try:
        from src.candidates.candidate_generator import CandidateGenerator
        from src.data.synthetic_generator import TargetedBackfillGenerator

        candidate_generator = CandidateGenerator.from_config(_load_config(DEFAULT_CONFIG_PATH))
        result = TargetedBackfillGenerator(candidate_generator, seed=13).generate_for_rule(rule_id, max_examples)
        return [(example.source, example.target) for example in result.examples[:max_examples]]
    except Exception:
        return []


def _validator_probe_row(
    rule_id: str,
    reason: str,
    example_count: int,
    passed_examples: int,
    source_name: str,
    reasons: Mapping[str, int],
) -> dict[str, Any]:
    status = "pass" if reason == "" else ("no_generation_probe_source" if reason == "no_generation_probe_source" else "fail")
    return {
        "rule_id": rule_id,
        "status": status,
        "reason": reason,
        "probe_source": source_name,
        "example_count": int(example_count),
        "passed_examples": int(passed_examples),
        "failed_examples": max(0, int(example_count) - int(passed_examples)),
        "empirical_validator_support": reason == "",
        "failure_reasons": json.dumps(dict(sorted(reasons.items())), ensure_ascii=False, sort_keys=True),
    }


def _validator_probe_summary_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    probe_rows = [
        {
            "rule_id": row["rule_id"],
            "status": row.get("validator_probe_status", ""),
            "reason": row.get("validator_probe_reason", ""),
            "example_count": row.get("validator_probe_example_count", 0),
            "passed_examples": row.get("validator_probe_passed_examples", 0),
        }
        for row in rows
        if row.get("validator_probe_status")
    ]
    passed = sorted(str(row["rule_id"]) for row in probe_rows if row.get("status") == "pass")
    failed_reasons = {
        str(row["rule_id"]): str(row.get("reason") or "validator_probe_failed")
        for row in probe_rows
        if row.get("status") != "pass"
    }
    return {
        "enabled": bool(probe_rows),
        "probed_rule_count": len(probe_rows),
        "passed_rule_ids": passed,
        "failed_rule_reasons": dict(sorted(failed_reasons.items())),
        "rows": probe_rows,
    }


def _activation_policy_manifest(
    policy: RuleActivationPolicy,
    production_ready_count: int,
    training_candidate_count: int,
    final_active_count: int | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    resolved_final_active_count = training_candidate_count if final_active_count is None else int(final_active_count)
    if production_ready_count < policy.expected_min_production_ready_rule_count:
        errors.append(
            f"production_ready_rule_count_below_min:{production_ready_count}<{policy.expected_min_production_ready_rule_count}"
        )
    if (
        policy.fail_below_min_training_candidate_rule_count
        and training_candidate_count < policy.expected_min_training_candidate_rule_count
    ):
        errors.append(
            "training_candidate_rule_count_below_min:"
            f"{training_candidate_count}<{policy.expected_min_training_candidate_rule_count}"
        )
    if (
        policy.warn_below_target_training_candidate_rule_count
        and policy.target_training_candidate_rule_count > 0
        and training_candidate_count < policy.target_training_candidate_rule_count
    ):
        warnings.append(
            "training_candidate_rule_count_below_target:"
            f"{training_candidate_count}<{policy.target_training_candidate_rule_count}"
        )
    if (
        policy.fail_below_final_active_rule_count
        and resolved_final_active_count < policy.expected_min_final_active_rule_count
    ):
        errors.append(
            "final_active_rule_count_below_min:"
            f"{resolved_final_active_count}<{policy.expected_min_final_active_rule_count}"
        )
    if (
        policy.warn_below_target_final_active_rule_count
        and policy.target_final_active_rule_count > 0
        and resolved_final_active_count < policy.target_final_active_rule_count
    ):
        warnings.append(
            "final_active_rule_count_below_target:"
            f"{resolved_final_active_count}<{policy.target_final_active_rule_count}"
        )
    return {
        "mode": policy.mode,
        "include_decisions": sorted(policy.include_decisions),
        "include_after_validator_when_runtime_supported": policy.include_after_validator_when_runtime_supported,
        "require_candidate_path": policy.require_candidate_path,
        "require_synthetic_support": policy.require_synthetic_support,
        "require_hard_negative_support": policy.require_hard_negative_support,
        "require_validator_support_or_empirical_pass": policy.require_validator_support_or_empirical_pass,
        "exclude_blocked": policy.exclude_blocked,
        "exclude_ner_required_without_ner": policy.exclude_ner_required_without_ner,
        "min_empirical_validator_passes": policy.min_empirical_validator_passes,
        "max_empirical_validator_examples": policy.max_empirical_validator_examples,
        "expected_min_production_ready_rule_count": policy.expected_min_production_ready_rule_count,
        "expected_min_training_candidate_rule_count": policy.expected_min_training_candidate_rule_count,
        "target_training_candidate_rule_count": policy.target_training_candidate_rule_count,
        "fail_below_min_training_candidate_rule_count": policy.fail_below_min_training_candidate_rule_count,
        "warn_below_target_training_candidate_rule_count": policy.warn_below_target_training_candidate_rule_count,
        "expected_min_final_active_rule_count": policy.expected_min_final_active_rule_count,
        "target_final_active_rule_count": policy.target_final_active_rule_count,
        "fail_below_final_active_rule_count": policy.fail_below_final_active_rule_count,
        "warn_below_target_final_active_rule_count": policy.warn_below_target_final_active_rule_count,
        "errors": errors,
        "warnings": warnings,
    }


def _policy_report_row(fields: dict[str, Any]) -> dict[str, Any]:
    policy = dict(fields.get("activation_policy", {}) or {})
    return {
        **policy,
        "include_decisions": json.dumps(policy.get("include_decisions", []), ensure_ascii=False),
        "errors": json.dumps(policy.get("errors", []), ensure_ascii=False),
        "warnings": json.dumps(policy.get("warnings", []), ensure_ascii=False),
        "production_ready_rule_count": int(fields.get("production_ready_rule_count", 0) or 0),
        "training_candidate_rule_count": int(fields.get("training_candidate_rule_count", 0) or 0),
        "active_rule_count": int(fields.get("active_rule_count", 0) or 0),
        "final_active_rule_count": int(fields.get("final_active_rule_count", fields.get("active_rule_count", 0)) or 0),
    }


def _capability_matrix_with_activation(capabilities: list[RuleCapability], activation_rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = capability_matrix_frame(capabilities)
    by_taxonomy: dict[str, list[dict[str, Any]]] = {}
    for row in activation_rows:
        by_taxonomy.setdefault(str(row["taxonomy_key"]), []).append(row)
    additions: list[dict[str, Any]] = []
    for capability in capabilities:
        rows = by_taxonomy.get(capability.taxonomy_key, [])
        additions.append(
            {
                "production_ready_rule_ids": json.dumps(
                    [row["rule_id"] for row in rows if row["production_ready"]],
                    ensure_ascii=False,
                ),
                "training_candidate_rule_ids": json.dumps(
                    [row["rule_id"] for row in rows if row["training_candidate"]],
                    ensure_ascii=False,
                ),
                "activation_stages": json.dumps(
                    sorted({row["activation_stage"] for row in rows if row["activation_stage"]}),
                    ensure_ascii=False,
                ),
                "activation_exclusion_reasons": json.dumps(
                    sorted({row["activation_exclusion_reason"] for row in rows if row["activation_exclusion_reason"]}),
                    ensure_ascii=False,
                ),
            }
        )
    for column in ("production_ready_rule_ids", "training_candidate_rule_ids", "activation_stages", "activation_exclusion_reasons"):
        frame[column] = [row[column] for row in additions]
    return frame


def _best_capability_by_rule_id(capabilities: list[RuleCapability]) -> dict[str, RuleCapability]:
    result: dict[str, RuleCapability] = {}
    for capability in capabilities:
        for rule_id in capability.project_rule_ids:
            current = result.get(rule_id)
            if current is None or _decision_priority(capability.training_decision) < _decision_priority(current.training_decision):
                result[rule_id] = capability
    return result


def _decision_priority(decision: str) -> int:
    order = {
        INCLUDE_NOW: 0,
        INCLUDE_AFTER_VALIDATOR: 1,
        INCLUDE_AFTER_THRESHOLD_CALIBRATION: 2,
        INCLUDE_AFTER_TRAINING: 3,
        EVAL_ONLY: 4,
        MINING_ONLY: 5,
    }
    text = str(decision or "")
    if text.startswith(BLOCK_DECISION_PREFIX):
        return 6
    return order.get(text, 7)


def _single_rule_capability(capability: RuleCapability, rule_id: str) -> RuleCapability:
    return RuleCapability(**{**asdict(capability), "project_rule_ids": [rule_id]})


def _row_activation_rule_ids(row: Mapping[str, Any]) -> list[str]:
    values = _json_list(row.get("rule_ids", []))
    result = [normalize_rule_id(value) for value in values if normalize_rule_id(value) != UNKNOWN_RULE_ID]
    if result:
        return result
    for key in ("target_rule_id", "rule_id"):
        rule_id = normalize_rule_id(row.get(key, ""))
        if rule_id != UNKNOWN_RULE_ID:
            return [rule_id]
    return []


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [part.strip() for part in value.split(",") if part.strip()]
        return parsed if isinstance(parsed, list) else []
    return []


def _bool_config(raw: Mapping[str, Any], key: str, default: bool) -> bool:
    if key not in raw:
        return default
    value = raw.get(key)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", "none", "null", ""}:
        return False
    return bool(value)


def _default_available_modules(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = dict(config or _load_config(config_path))
    registry_rule_ids = _registry_rule_ids()
    operator_rule_ids = _operator_rule_ids()
    syntax_supported = _syntax_supported(config)
    syntax_rule_ids = _syntax_supported_rule_ids() if syntax_supported else set()
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
        "syntax_supported": syntax_supported,
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
