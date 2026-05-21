from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

from src.candidates.candidate_generator import CandidateGenerator
from src.candidates.matching import candidate_matches_edit
from src.config.load_config import load_config
from src.data.matrix_eval_dataset import DEFAULT_CLEAN_SENTENCES, STATIC_PROBES
from src.data.synthetic_generator import TARGETED_BACKFILL_DICTIONARY_RULE_IDS, TargetedBackfillGenerator
from src.rules.coverage_matrix import iter_coverage_entries, load_rules_coverage
from src.rules.registry import all_rules, rule_by_id
from src.rules.rule_ids import UNKNOWN_RULE_ID, normalize_rule_id
from src.validation.diff_analyzer import DiffAnalyzer
from src.validation.edit_classifier import is_allowed_edit_type


INCLUDE_NOW = "INCLUDE_NOW"
INCLUDE_AFTER_VALIDATOR = "INCLUDE_AFTER_VALIDATOR"
INCLUDE_AFTER_THRESHOLD_CALIBRATION = "INCLUDE_AFTER_THRESHOLD_CALIBRATION"
INCLUDE_AFTER_TRAINING = "INCLUDE_AFTER_TRAINING"
BLOCK_METADATA_ONLY = "BLOCK_METADATA_ONLY"
BLOCK_PLANNED = "BLOCK_PLANNED"
BLOCK_NO_CANDIDATE = "BLOCK_NO_CANDIDATE"
BLOCK_NEEDS_SYNTAX = "BLOCK_NEEDS_SYNTAX"
BLOCK_NEEDS_DICTIONARY = "BLOCK_NEEDS_DICTIONARY"
BLOCK_NEEDS_NER = "BLOCK_NEEDS_NER"
BLOCK_DISABLED = "BLOCK_DISABLED"

INCLUDE_DECISIONS = {
    INCLUDE_NOW,
    INCLUDE_AFTER_VALIDATOR,
    INCLUDE_AFTER_THRESHOLD_CALIBRATION,
    INCLUDE_AFTER_TRAINING,
}
BLOCK_DECISIONS = {
    BLOCK_METADATA_ONLY,
    BLOCK_PLANNED,
    BLOCK_NO_CANDIDATE,
    BLOCK_NEEDS_SYNTAX,
    BLOCK_NEEDS_DICTIONARY,
    BLOCK_NEEDS_NER,
    BLOCK_DISABLED,
}
DECISION_ORDER = {
    INCLUDE_AFTER_VALIDATOR: 0,
    INCLUDE_AFTER_THRESHOLD_CALIBRATION: 1,
    INCLUDE_AFTER_TRAINING: 2,
    INCLUDE_NOW: 3,
}

ENTRY_ELIGIBLE_COLUMNS = [
    "matrix_key",
    "orfogrammka_id",
    "title",
    "rule_id",
    "status",
    "training_eligibility_decision",
    "reason",
    "risk_level",
    "requires",
    "current_candidate_path",
    "current_synthetic_support",
    "current_validator_support",
    "current_candidate_recall",
    "current_f1",
    "needs_before_training",
]
BLOCKED_COLUMNS = [
    "matrix_key",
    "title",
    "status",
    "blocked_reason",
    "requires",
    "recommended_next_action",
]
PROBE_COLUMNS = [
    "matrix_key",
    "rule_id",
    "example_source",
    "example_target",
    "generated_candidate_count",
    "matching_candidate_found",
    "candidate_recall",
    "decision",
    "reason",
]
VALIDATOR_NEEDED_COLUMNS = [
    "matrix_key",
    "rule_id",
    "title",
    "risk_level",
    "reason",
    "needs_before_training",
]

HIGH_RISK_RULE_IDS = frozenset(
    {
        "dictionary_fuzzy",
        "keyboard_typo_candidate",
        "missing_letter_candidate",
        "extra_letter_candidate",
        "swapped_letters_candidate",
        "double_consonant_candidate",
        "capitalization_sentence_start",
        "abbreviation_case_protection",
        "final_punctuation_default",
        "hyphen_po_adverbs",
        "tsya_soft_delete",
        "tsya_soft_insert",
        "quote_pair_balance",
        "bracket_pair_balance",
        "punctuation_delete_replace",
    }
)
CONTEXT_PAIR_RULE_IDS = frozenset(
    {
        "context_tak_zhe",
        "context_to_zhe",
        "context_chto_by",
        "context_za_to",
        "context_vsledstvie",
        "context_nesmotrya",
    }
)
MEDIUM_RISK_RULE_IDS = frozenset(
    {
        "direct_speech_colon",
        "direct_speech_quotes",
        "direct_speech_dash",
        "subject_predicate_dash",
        "enumeration_colon",
        "explanation_colon",
        "consequence_dash",
        "asyndetic_dash",
        "semicolon",
        "prefix_pre_pri",
        "n_nn_adjective",
        "n_nn_participle",
        "n_nn_deverbal_adjective",
        "n_nn_short_form",
        "ne_verb",
        "ne_adjective",
        "ne_adverb",
        "ne_participle",
        "pol_polu_compounds",
        "hyphen_whitelist",
    }
)
RISKY_VALIDATOR_RULE_IDS = HIGH_RISK_RULE_IDS | CONTEXT_PAIR_RULE_IDS
DICTIONARY_RULE_IDS = frozenset(TARGETED_BACKFILL_DICTIONARY_RULE_IDS) | {
    "prefix_pre_pri",
    "frequent_error_exact",
    "hyphen_whitelist",
    "n_nn_adjective",
    "n_nn_participle",
    "n_nn_deverbal_adjective",
    "n_nn_short_form",
    "pol_polu_compounds",
    "abbreviation_case_protection",
}
PROBE_PAIR_OVERRIDES: dict[str, tuple[tuple[str, str], ...]] = {
    "consequence_dash": (("Начался дождь отчет не приняли.", "Начался дождь— отчет не приняли."),),
    "asyndetic_dash": (("Солнце село архив закрыли.", "Солнце село— архив закрыли."),),
    "semicolon": (("Документ готов отчет отправлен.", "Документ готов; отчет отправлен."),),
    "frequent_error_exact": (("Жызнь прекрасна.", "Жизнь прекрасна."),),
    "hyphen_whitelist": (("Он сделал это по-умолчанию.", "Он сделал это по умолчанию."),),
    "direct_speech_dash": (("«Отчет готов» сказал редактор.", "«Отчет готов» — сказал редактор."),),
    "quote_pair_balance": (
        ("Редактор сказал: «отчет готов.", "Редактор сказал: «отчет готов»."),
        ("Редактор сказал: отчет готов».", "Редактор сказал: «отчет готов»."),
    ),
}


@dataclass(frozen=True)
class RuleAudit:
    rule_id: str
    matrix_key: str
    status: str
    requires: tuple[str, ...]
    candidate_path: bool
    synthetic_support: bool
    hard_negative_support: bool
    validator_support: bool
    candidate_recall: float | None
    gap_coverage: float | None
    current_f1: float | None
    risk_level: str
    decision: str
    reason: str
    production_ready_now: bool
    training_eligible_now: bool
    needs_before_training: tuple[str, ...]


@dataclass(frozen=True)
class ProbeAudit:
    matrix_key: str
    rule_id: str
    example_source: str
    example_target: str
    generated_candidate_count: int
    matching_candidate_found: bool
    candidate_recall: float | None
    decision: str
    reason: str


@dataclass(frozen=True)
class PreDatasetCapabilityResult:
    updated_config: dict[str, Any]
    entry_rows: list[dict[str, Any]]
    rule_rows: list[dict[str, Any]]
    probe_rows: list[dict[str, Any]]
    validator_rows: list[dict[str, Any]]
    summary: dict[str, Any]


def build_pre_dataset_capability(
    *,
    rules_config_path: str | Path = "configs/rules.yaml",
    matrix_reports_dir: str | Path = "reports/matrix_eval",
    config_path: str | Path = "configs/config.yaml",
    clean_pool_path: str | Path = "data/processed/clean_sentence_pool.csv.gz",
) -> PreDatasetCapabilityResult:
    config = load_rules_coverage(rules_config_path)
    updated = deepcopy(config)
    registry_ids = {rule.spec.id for rule in all_rules()}
    rule_to_matrix = _rule_to_matrix(config)
    matrix_state = _matrix_state(matrix_reports_dir)
    probes = _probe_rules(
        sorted(rule_to_matrix),
        rule_to_matrix=rule_to_matrix,
        config_path=config_path,
        clean_pool_path=clean_pool_path,
    )
    rule_audits = {
        rule_id: _audit_rule(
            rule_id=rule_id,
            matrix_key=rule_to_matrix.get(rule_id, ""),
            status=_status_for_rule(config, rule_id),
            requires=_requires_for_rule(config, rule_id),
            probe=probes.get(rule_id),
            matrix_state=matrix_state.get(rule_id, {}),
            registry_ids=registry_ids,
        )
        for rule_id in sorted(rule_to_matrix)
    }

    entry_rows: list[dict[str, Any]] = []
    rule_rows: list[dict[str, Any]] = []
    validator_rows: list[dict[str, Any]] = []
    for section, matrix_key, entry in iter_coverage_entries(config):
        raw_entry = config[section][matrix_key]
        dataset_update, entry_row = _audit_entry(section, matrix_key, raw_entry, rule_audits)
        updated[section][matrix_key]["dataset"] = {
            **(updated[section][matrix_key].get("dataset") or {}),
            **dataset_update,
        }
        entry_rows.append(entry_row)
        for rule_id in raw_entry.get("implementation", {}).get("rule_ids", []) or []:
            audit = rule_audits.get(normalize_rule_id(rule_id))
            if audit is None:
                continue
            rule_rows.append(_rule_report_row(raw_entry, matrix_key, audit))
            if audit.training_eligible_now and audit.decision == INCLUDE_AFTER_VALIDATOR:
                validator_rows.append(_validator_row(raw_entry, matrix_key, audit))

    probe_rows = [_probe_row(probe) for probe in probes.values()]
    summary = _summary(config, entry_rows, rule_rows)
    return PreDatasetCapabilityResult(
        updated_config=updated,
        entry_rows=entry_rows,
        rule_rows=rule_rows,
        probe_rows=probe_rows,
        validator_rows=validator_rows,
        summary=summary,
    )


def write_pre_dataset_capability_outputs(
    *,
    rules_config_path: str | Path = "configs/rules.yaml",
    output_dir: str | Path = "reports/pre_dataset_capability",
    matrix_reports_dir: str | Path = "reports/matrix_eval",
    config_path: str | Path = "configs/config.yaml",
    clean_pool_path: str | Path = "data/processed/clean_sentence_pool.csv.gz",
    update_rules_yaml: bool = True,
) -> dict[str, str]:
    result = build_pre_dataset_capability(
        rules_config_path=rules_config_path,
        matrix_reports_dir=matrix_reports_dir,
        config_path=config_path,
        clean_pool_path=clean_pool_path,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    eligible = pd.DataFrame(result.rule_rows)
    if eligible.empty:
        eligible = pd.DataFrame(columns=ENTRY_ELIGIBLE_COLUMNS)
    else:
        eligible = eligible[eligible["training_eligible_now"].astype(bool)]
        eligible = eligible.reindex(columns=ENTRY_ELIGIBLE_COLUMNS)
    blocked = pd.DataFrame(_blocked_rows(result.entry_rows), columns=BLOCKED_COLUMNS)
    probes = pd.DataFrame(result.probe_rows, columns=PROBE_COLUMNS)
    validators = pd.DataFrame(result.validator_rows, columns=VALIDATOR_NEEDED_COLUMNS)

    paths = {
        "candidate_path_probe": output / "candidate_path_probe.csv",
        "dataset_eligible_rules": output / "dataset_eligible_rules.csv",
        "dataset_blocked_rules": output / "dataset_blocked_rules.csv",
        "validator_needed_training_rules": output / "validator_needed_training_rules.csv",
        "current_capability_summary": output / "current_capability_summary.md",
        "current_capability_dataset_plan": output / "current_capability_dataset_plan.md",
        "rules_yaml_update_report": output / "rules_yaml_update_report.md",
    }
    probes.to_csv(paths["candidate_path_probe"], index=False)
    eligible.to_csv(paths["dataset_eligible_rules"], index=False)
    blocked.to_csv(paths["dataset_blocked_rules"], index=False)
    validators.to_csv(paths["validator_needed_training_rules"], index=False)
    paths["current_capability_summary"].write_text(_summary_markdown(result.summary), encoding="utf-8")
    paths["current_capability_dataset_plan"].write_text(_dataset_plan_markdown(result.rule_rows), encoding="utf-8")
    paths["rules_yaml_update_report"].write_text(
        _rules_yaml_update_markdown(result.summary, rules_config_path),
        encoding="utf-8",
    )
    if update_rules_yaml:
        Path(rules_config_path).write_text(
            yaml.safe_dump(result.updated_config, allow_unicode=True, sort_keys=False, width=120),
            encoding="utf-8",
        )
    return {key: str(path) for key, path in paths.items()}


def _audit_rule(
    *,
    rule_id: str,
    matrix_key: str,
    status: str,
    requires: tuple[str, ...],
    probe: ProbeAudit | None,
    matrix_state: dict[str, Any],
    registry_ids: set[str],
) -> RuleAudit:
    rule = rule_by_id(rule_id)
    candidate_path = bool(probe and probe.matching_candidate_found) and rule_id in registry_ids and rule is not None
    probe_recall = probe.candidate_recall if probe and probe.candidate_recall is not None else None
    report_recall = _float_or_none(matrix_state.get("candidate_recall"))
    candidate_recall = _best_recall(probe_recall, report_recall if candidate_path else None)
    gap_coverage = _float_or_none(matrix_state.get("gap_coverage"))
    current_f1 = _float_or_none(matrix_state.get("f1"))
    risk_level = _risk_level(rule_id, requires)
    validator_support = _validator_support(rule_id, requires, matrix_state)
    hard_negative_support = rule_id in RISKY_VALIDATOR_RULE_IDS or bool(int(_number(matrix_state.get("false_positive"), 0)) == 0)
    synthetic_support = candidate_path

    if not candidate_path:
        decision, reason = _block_decision_for_missing_candidate(status, requires)
        return RuleAudit(
            rule_id=rule_id,
            matrix_key=matrix_key,
            status=status,
            requires=requires,
            candidate_path=False,
            synthetic_support=False,
            hard_negative_support=hard_negative_support,
            validator_support=validator_support,
            candidate_recall=candidate_recall,
            gap_coverage=gap_coverage,
            current_f1=current_f1,
            risk_level=risk_level,
            decision=decision,
            reason=reason,
            production_ready_now=False,
            training_eligible_now=False,
            needs_before_training=("none",),
        )

    if candidate_recall is None or candidate_recall < 0.85:
        return RuleAudit(
            rule_id=rule_id,
            matrix_key=matrix_key,
            status=status,
            requires=requires,
            candidate_path=True,
            synthetic_support=synthetic_support,
            hard_negative_support=hard_negative_support,
            validator_support=validator_support,
            candidate_recall=candidate_recall,
            gap_coverage=gap_coverage,
            current_f1=current_f1,
            risk_level=risk_level,
            decision=BLOCK_NO_CANDIDATE,
            reason="candidate recall below 0.85 on probe/eval examples",
            production_ready_now=False,
            training_eligible_now=False,
            needs_before_training=("none",),
        )

    eval_decision = str(matrix_state.get("decision") or "")
    if rule_id in RISKY_VALIDATOR_RULE_IDS or eval_decision == "NEEDS_VALIDATOR":
        decision = INCLUDE_AFTER_VALIDATOR
        reason = "candidate path verified; validator guards and hard negatives are required before dataset build"
        needs = ("validator", "hard_negatives")
    elif eval_decision == "READY_NEXT_DATASET":
        decision = INCLUDE_NOW
        reason = "candidate path and matrix safety gates are ready for dataset inclusion"
        needs = ("none",)
    elif eval_decision == "NEEDS_THRESHOLD":
        decision = INCLUDE_AFTER_THRESHOLD_CALIBRATION
        reason = "candidate path verified; current blocker is threshold calibration"
        needs = ("threshold_calibration",)
    elif eval_decision == "NEEDS_MORE_TRAINING":
        decision = INCLUDE_AFTER_TRAINING
        reason = "candidate path verified; current blocker is model training distribution"
        needs = ("more_training",)
    else:
        decision = INCLUDE_AFTER_TRAINING
        reason = "candidate path verified by probe; rule needs training/eval coverage before production activation"
        needs = ("more_training",)

    production_ready = decision == INCLUDE_NOW
    return RuleAudit(
        rule_id=rule_id,
        matrix_key=matrix_key,
        status=status,
        requires=requires,
        candidate_path=True,
        synthetic_support=synthetic_support,
        hard_negative_support=hard_negative_support,
        validator_support=validator_support,
        candidate_recall=candidate_recall,
        gap_coverage=gap_coverage,
        current_f1=current_f1,
        risk_level=risk_level,
        decision=decision,
        reason=reason,
        production_ready_now=production_ready,
        training_eligible_now=True,
        needs_before_training=needs,
    )


def _audit_entry(
    section: str,
    matrix_key: str,
    entry: dict[str, Any],
    rule_audits: dict[str, RuleAudit],
) -> tuple[dict[str, Any], dict[str, Any]]:
    impl = entry.get("implementation") or {}
    dataset = entry.get("dataset") or {}
    status = str(impl.get("status") or "")
    entry_type = str(entry.get("entry_type") or "")
    requires = tuple(str(item) for item in impl.get("requires", []) or [])
    rule_ids = [normalize_rule_id(rule_id) for rule_id in impl.get("rule_ids", []) or []]
    audits = [rule_audits[rule_id] for rule_id in rule_ids if rule_id in rule_audits]

    if entry_type == "group" or status == "metadata_only":
        return _blocked_entry_update(dataset, BLOCK_METADATA_ONLY, "metadata_only", requires), _entry_row(
            section, matrix_key, entry, BLOCK_METADATA_ONLY, "metadata_only", False, False, requires, audits
        )
    if status == "planned":
        return _blocked_entry_update(dataset, BLOCK_PLANNED, "planned", requires), _entry_row(
            section, matrix_key, entry, BLOCK_PLANNED, "planned", False, False, requires, audits
        )
    if status == "disabled":
        return _blocked_entry_update(dataset, BLOCK_DISABLED, "disabled", requires), _entry_row(
            section, matrix_key, entry, BLOCK_DISABLED, "disabled", False, False, requires, audits
        )
    if not audits:
        decision, reason = _block_decision_for_missing_candidate(status, requires)
        return _blocked_entry_update(dataset, decision, reason, requires), _entry_row(
            section, matrix_key, entry, decision, reason, False, False, requires, audits
        )

    eligible_audits = [audit for audit in audits if audit.training_eligible_now]
    if not eligible_audits:
        decision = _dominant_block_decision([audit.decision for audit in audits], status, requires)
        reason = "; ".join(sorted({audit.reason for audit in audits})) or "no candidate-backed rule_id"
        return _blocked_entry_update(dataset, decision, reason, requires, audits), _entry_row(
            section, matrix_key, entry, decision, reason, False, False, requires, audits
        )

    decision = min((audit.decision for audit in eligible_audits), key=lambda value: DECISION_ORDER.get(value, 99))
    reason = "; ".join(sorted({audit.reason for audit in eligible_audits}))
    production_ready = bool(eligible_audits) and all(audit.production_ready_now for audit in eligible_audits)
    update = {
        "production_ready_now": production_ready,
        "training_eligible_now": True,
        "training_eligibility_decision": decision,
        "training_eligibility_reason": reason,
        "current_candidate_path": any(audit.candidate_path for audit in eligible_audits),
        "current_synthetic_support": any(audit.synthetic_support for audit in eligible_audits),
        "current_hard_negative_support": any(audit.hard_negative_support for audit in eligible_audits),
        "current_validator_support": any(audit.validator_support for audit in eligible_audits),
        "current_candidate_recall": _max_or_none(audit.candidate_recall for audit in eligible_audits),
        "current_gap_coverage": _max_or_none(audit.gap_coverage for audit in eligible_audits),
        "risk_level": _max_risk(audit.risk_level for audit in eligible_audits),
        "needs_before_training": _needs_union(audit.needs_before_training for audit in eligible_audits),
    }
    return update, _entry_row(section, matrix_key, entry, decision, reason, True, production_ready, requires, eligible_audits)


def _blocked_entry_update(
    dataset: dict[str, Any],
    decision: str,
    reason: str,
    requires: Iterable[str],
    audits: Iterable[RuleAudit] = (),
) -> dict[str, Any]:
    audit_list = list(audits)
    return {
        "production_ready_now": False,
        "training_eligible_now": False,
        "training_eligibility_decision": decision,
        "training_eligibility_reason": reason,
        "current_candidate_path": any(audit.candidate_path for audit in audit_list),
        "current_synthetic_support": any(audit.synthetic_support for audit in audit_list),
        "current_hard_negative_support": any(audit.hard_negative_support for audit in audit_list),
        "current_validator_support": any(audit.validator_support for audit in audit_list),
        "current_candidate_recall": _max_or_none(audit.candidate_recall for audit in audit_list),
        "current_gap_coverage": _max_or_none(audit.gap_coverage for audit in audit_list),
        "risk_level": _max_risk((audit.risk_level for audit in audit_list), fallback=_risk_from_requires(tuple(requires))),
        "needs_before_training": ["none"],
    }


def _probe_rules(
    rule_ids: list[str],
    *,
    rule_to_matrix: dict[str, str],
    config_path: str | Path,
    clean_pool_path: str | Path,
) -> dict[str, ProbeAudit]:
    default_generator = _candidate_generator_from_config(config_path)
    fast_generator = CandidateGenerator(dictionary_lexicon=(), dictionary_limit=0, syntax_provider=lambda _text: ())
    dictionary_generator = _dictionary_probe_generator(default_generator)
    analyzer = DiffAnalyzer()
    result: dict[str, ProbeAudit] = {}
    for rule_id in rule_ids:
        generator = _generator_for_rule(rule_id, default_generator, fast_generator, dictionary_generator)
        result[rule_id] = _probe_rule(
            rule_id,
            matrix_key=rule_to_matrix.get(rule_id, ""),
            generator=generator,
            fallback_generator=default_generator,
            analyzer=analyzer,
            clean_pool_path=clean_pool_path,
        )
    return result


def _probe_rule(
    rule_id: str,
    *,
    matrix_key: str,
    generator: CandidateGenerator,
    fallback_generator: CandidateGenerator,
    analyzer: DiffAnalyzer,
    clean_pool_path: str | Path,
) -> ProbeAudit:
    rejected: ProbeAudit | None = None
    for source, target in _probe_pairs(rule_id, fallback_generator, clean_pool_path):
        probe = _probe_pair(rule_id, matrix_key, source, target, generator, analyzer)
        if probe.matching_candidate_found:
            return probe
        rejected = probe if rejected is None or probe.generated_candidate_count > rejected.generated_candidate_count else rejected
    return rejected or ProbeAudit(
        matrix_key=matrix_key,
        rule_id=rule_id,
        example_source="",
        example_target="",
        generated_candidate_count=0,
        matching_candidate_found=False,
        candidate_recall=None,
        decision=BLOCK_NO_CANDIDATE,
        reason="no bounded probe example",
    )


def _probe_pairs(
    rule_id: str,
    generator: CandidateGenerator,
    clean_pool_path: str | Path,
) -> Iterable[tuple[str, str]]:
    yield from PROBE_PAIR_OVERRIDES.get(rule_id, ())
    yield from STATIC_PROBES.get(rule_id, ())
    backfill = TargetedBackfillGenerator(generator, seed=29).generate_for_rule(rule_id, 1)
    for example in backfill.examples:
        yield example.source, example.target
    rule = rule_by_id(rule_id)
    if rule is not None and hasattr(rule, "generate_corruptions"):
        for target in _clean_probe_sentences(clean_pool_path):
            try:
                corruptions = rule.generate_corruptions(target)
            except Exception:
                corruptions = []
            for corruption in corruptions:
                if normalize_rule_id(getattr(corruption, "rule_id", "")) == rule_id:
                    source = corruption.apply(target)
                    if source != target:
                        yield source, target


def _probe_pair(
    rule_id: str,
    matrix_key: str,
    source: str,
    target: str,
    generator: CandidateGenerator,
    analyzer: DiffAnalyzer,
) -> ProbeAudit:
    candidates = generator.generate(source)
    matching = _matching_candidate_found(rule_id, source, target, candidates, analyzer)
    return ProbeAudit(
        matrix_key=matrix_key,
        rule_id=rule_id,
        example_source=source,
        example_target=target,
        generated_candidate_count=len(candidates),
        matching_candidate_found=matching,
        candidate_recall=1.0 if matching else 0.0,
        decision=INCLUDE_AFTER_TRAINING if matching else BLOCK_NO_CANDIDATE,
        reason="candidate-backed probe matched" if matching else "candidate generator did not match probe target",
    )


def _matching_candidate_found(rule_id: str, source: str, target: str, candidates: list[Any], analyzer: DiffAnalyzer) -> bool:
    edits = [edit for edit in analyzer.analyze(source, target, candidates=candidates) if is_allowed_edit_type(edit.edit_type)]
    for candidate in candidates:
        if normalize_rule_id(getattr(candidate, "rule_id", "")) != rule_id:
            continue
        if _apply_candidate(source, candidate) == target:
            return True
        if any(candidate_matches_edit(candidate, edit) for edit in edits):
            return True
    return False


def _candidate_generator_from_config(config_path: str | Path) -> CandidateGenerator:
    path = Path(config_path)
    if path.exists():
        try:
            return CandidateGenerator.from_config(load_config(path))
        except Exception:
            pass
    return CandidateGenerator(dictionary_lexicon=(), dictionary_limit=0)


def _dictionary_probe_generator(default_generator: CandidateGenerator) -> CandidateGenerator:
    return CandidateGenerator(
        dictionary_provider=getattr(default_generator, "dictionary_provider", None),
        dictionary_limit=getattr(default_generator, "dictionary_limit", 2),
        dictionary_min_score=getattr(default_generator, "dictionary_min_score", 85),
        dictionary_yo_e_enabled=getattr(default_generator, "dictionary_yo_e_enabled", False),
        syntax_provider=lambda _text: (),
        dictionary_token_cache_enabled=getattr(default_generator, "dictionary_token_cache_enabled", True),
        dictionary_token_cache_max_size=getattr(default_generator, "dictionary_token_cache_max_size", 200_000),
        dictionary_max_choices_per_token=getattr(default_generator, "dictionary_max_choices_per_token", 50_000),
        dictionary_use_second_letter_index=getattr(default_generator, "dictionary_use_second_letter_index", False),
    )


def _generator_for_rule(
    rule_id: str,
    default_generator: CandidateGenerator,
    fast_generator: CandidateGenerator,
    dictionary_generator: CandidateGenerator,
) -> CandidateGenerator:
    rule = rule_by_id(rule_id)
    requires = tuple(getattr(getattr(rule, "spec", None), "requires", ()) or ())
    if rule_id in TARGETED_BACKFILL_DICTIONARY_RULE_IDS or rule_id == "yo_e_candidate":
        return dictionary_generator
    if "ner" in requires:
        return default_generator
    return fast_generator


def _matrix_state(matrix_reports_dir: str | Path) -> dict[str, dict[str, Any]]:
    base = Path(matrix_reports_dir)
    paths = [
        base / "matrix_rule_eval_summary.csv",
        base / "working_eval" / "matrix_rule_eval_summary.csv",
    ]
    summary = next((_read_csv(path) for path in paths if path.exists()), pd.DataFrame())
    if summary.empty or "rule_id" not in summary.columns:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in summary.to_dict("records"):
        rule_id = normalize_rule_id(row.get("rule_id", ""))
        if rule_id and rule_id != UNKNOWN_RULE_ID:
            result[rule_id] = row
    return result


def _rule_to_matrix(config: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for _section, matrix_key, entry in iter_coverage_entries(config):
        for raw_rule_id in entry.get("rules", []) or []:
            rule_id = normalize_rule_id(raw_rule_id)
            if rule_id != UNKNOWN_RULE_ID:
                result[rule_id] = matrix_key
    return result


def _status_for_rule(config: dict[str, Any], rule_id: str) -> str:
    for _section, _matrix_key, entry in iter_coverage_entries(config):
        if rule_id in [normalize_rule_id(item) for item in entry.get("rules", []) or []]:
            return str(entry.get("status") or "")
    return ""


def _requires_for_rule(config: dict[str, Any], rule_id: str) -> tuple[str, ...]:
    for _section, _matrix_key, entry in iter_coverage_entries(config):
        if rule_id in [normalize_rule_id(item) for item in entry.get("rules", []) or []]:
            return tuple(str(item) for item in entry.get("requires", []) or [])
    rule = rule_by_id(rule_id)
    return tuple(str(item) for item in getattr(getattr(rule, "spec", None), "requires", ()) or ())


def _block_decision_for_missing_candidate(status: str, requires: Iterable[str]) -> tuple[str, str]:
    requires_set = {str(item).lower() for item in requires}
    if status == "disabled":
        return BLOCK_DISABLED, "disabled"
    if "ner" in requires_set:
        return BLOCK_NEEDS_NER, "requires NER/protected-span candidate path"
    if "syntax" in requires_set:
        return BLOCK_NEEDS_SYNTAX, "requires syntax-backed candidate path"
    if "dictionary" in requires_set:
        return BLOCK_NEEDS_DICTIONARY, "requires dictionary-backed candidate path"
    return BLOCK_NO_CANDIDATE, "no candidate path"


def _dominant_block_decision(decisions: Iterable[str], status: str, requires: Iterable[str]) -> str:
    decision_set = set(decisions)
    for decision in (BLOCK_DISABLED, BLOCK_NEEDS_NER, BLOCK_NEEDS_SYNTAX, BLOCK_NEEDS_DICTIONARY, BLOCK_NO_CANDIDATE):
        if decision in decision_set:
            return decision
    return _block_decision_for_missing_candidate(status, requires)[0]


def _rule_report_row(entry: dict[str, Any], matrix_key: str, audit: RuleAudit) -> dict[str, Any]:
    return {
        "matrix_key": matrix_key,
        "orfogrammka_id": str(entry.get("orfogrammka_id") or ""),
        "title": str(entry.get("title") or ""),
        "rule_id": audit.rule_id,
        "status": audit.status,
        "training_eligible_now": audit.training_eligible_now,
        "training_eligibility_decision": audit.decision,
        "reason": audit.reason,
        "risk_level": audit.risk_level,
        "requires": _json_list(audit.requires),
        "current_candidate_path": audit.candidate_path,
        "current_synthetic_support": audit.synthetic_support,
        "current_validator_support": audit.validator_support,
        "current_candidate_recall": audit.candidate_recall,
        "current_f1": audit.current_f1,
        "needs_before_training": _json_list(audit.needs_before_training),
    }


def _entry_row(
    section: str,
    matrix_key: str,
    entry: dict[str, Any],
    decision: str,
    reason: str,
    training_eligible: bool,
    production_ready: bool,
    requires: Iterable[str],
    audits: Iterable[RuleAudit],
) -> dict[str, Any]:
    audit_list = list(audits)
    impl = entry.get("implementation") or {}
    return {
        "section": section,
        "matrix_key": matrix_key,
        "orfogrammka_id": str(entry.get("orfogrammka_id") or ""),
        "title": str(entry.get("title") or ""),
        "entry_type": str(entry.get("entry_type") or ""),
        "status": str(impl.get("status") or ""),
        "rule_ids": [audit.rule_id for audit in audit_list],
        "production_ready_now": production_ready,
        "training_eligible_now": training_eligible,
        "training_eligibility_decision": decision,
        "training_eligibility_reason": reason,
        "current_candidate_path": any(audit.candidate_path for audit in audit_list),
        "current_synthetic_support": any(audit.synthetic_support for audit in audit_list),
        "current_validator_support": any(audit.validator_support for audit in audit_list),
        "current_candidate_recall": _max_or_none(audit.candidate_recall for audit in audit_list),
        "current_f1": _max_or_none(audit.current_f1 for audit in audit_list),
        "risk_level": _max_risk((audit.risk_level for audit in audit_list), fallback=_risk_from_requires(tuple(requires))),
        "requires": _json_list(tuple(requires)),
    }


def _validator_row(entry: dict[str, Any], matrix_key: str, audit: RuleAudit) -> dict[str, Any]:
    return {
        "matrix_key": matrix_key,
        "rule_id": audit.rule_id,
        "title": str(entry.get("title") or ""),
        "risk_level": audit.risk_level,
        "reason": audit.reason,
        "needs_before_training": _json_list(audit.needs_before_training),
    }


def _probe_row(probe: ProbeAudit) -> dict[str, Any]:
    return {
        "matrix_key": probe.matrix_key,
        "rule_id": probe.rule_id,
        "example_source": probe.example_source,
        "example_target": probe.example_target,
        "generated_candidate_count": probe.generated_candidate_count,
        "matching_candidate_found": probe.matching_candidate_found,
        "candidate_recall": probe.candidate_recall,
        "decision": probe.decision,
        "reason": probe.reason,
    }


def _blocked_rows(entry_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in entry_rows:
        if row["training_eligible_now"]:
            continue
        rows.append(
            {
                "matrix_key": row["matrix_key"],
                "title": row["title"],
                "status": row["status"],
                "blocked_reason": row["training_eligibility_decision"],
                "requires": row["requires"],
                "recommended_next_action": _recommended_next_action(row["training_eligibility_decision"]),
            }
        )
    return rows


def _summary(config: dict[str, Any], entry_rows: list[dict[str, Any]], rule_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(len(config.get(section, {})) for section in ("orthography", "punctuation"))
    metadata_only = sum(1 for row in entry_rows if row["entry_type"] == "group" or row["status"] == "metadata_only")
    production_ready = sum(1 for row in entry_rows if row["production_ready_now"])
    training_eligible = sum(1 for row in entry_rows if row["training_eligible_now"])
    legacy_eligible = sum(
        1
        for section in ("orthography", "punctuation")
        for entry in config.get(section, {}).values()
        if (entry.get("dataset") or {}).get("eligible_now")
    )
    decision_counts = Counter(row["training_eligibility_decision"] for row in entry_rows)
    eligible_rule_rows = [row for row in rule_rows if row["training_eligible_now"]]
    active_rule_ids = sorted({row["rule_id"] for row in eligible_rule_rows})
    recommended_size = _recommended_dataset_size(active_rule_ids, eligible_rule_rows)
    return {
        "total_taxonomy_entries": total,
        "metadata_only_count": metadata_only,
        "true_leaf_actionable_entries_count": total - metadata_only,
        "production_ready_now_count": production_ready,
        "training_eligible_now_count": training_eligible,
        "legacy_eligible_now_count": legacy_eligible,
        "training_eligible_active_rule_count": len(active_rule_ids),
        "recommended_dataset_active_rule_count": len(active_rule_ids),
        "recommended_dataset_size": recommended_size,
        "active_rule_ids": active_rule_ids,
        **{decision: decision_counts.get(decision, 0) for decision in sorted(INCLUDE_DECISIONS | BLOCK_DECISIONS)},
    }


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Current Pre-Dataset Capability Summary",
        "",
        f"- total taxonomy entries: {summary['total_taxonomy_entries']}",
        f"- metadata_only count: {summary['metadata_only_count']}",
        f"- true leaf/actionable entries count: {summary['true_leaf_actionable_entries_count']}",
        f"- production_ready_now count: {summary['production_ready_now_count']}",
        f"- training_eligible_now count: {summary['training_eligible_now_count']}",
        f"- training_eligible active rule_id count: {summary['training_eligible_active_rule_count']}",
        f"- INCLUDE_NOW count: {summary[INCLUDE_NOW]}",
        f"- INCLUDE_AFTER_VALIDATOR count: {summary[INCLUDE_AFTER_VALIDATOR]}",
        f"- INCLUDE_AFTER_THRESHOLD_CALIBRATION count: {summary[INCLUDE_AFTER_THRESHOLD_CALIBRATION]}",
        f"- INCLUDE_AFTER_TRAINING count: {summary[INCLUDE_AFTER_TRAINING]}",
        f"- BLOCK_NO_CANDIDATE count: {summary[BLOCK_NO_CANDIDATE]}",
        f"- BLOCK_NEEDS_SYNTAX count: {summary[BLOCK_NEEDS_SYNTAX]}",
        f"- BLOCK_NEEDS_DICTIONARY count: {summary[BLOCK_NEEDS_DICTIONARY]}",
        f"- BLOCK_NEEDS_NER count: {summary[BLOCK_NEEDS_NER]}",
        f"- BLOCK_METADATA_ONLY count: {summary[BLOCK_METADATA_ONLY]}",
        f"- BLOCK_PLANNED count: {summary[BLOCK_PLANNED]}",
        f"- BLOCK_DISABLED count: {summary[BLOCK_DISABLED]}",
        f"- recommended dataset active rule_id count: {summary['recommended_dataset_active_rule_count']}",
        f"- recommended dataset size estimate: {summary['recommended_dataset_size']}",
        "",
    ]
    if summary["training_eligible_now_count"] < 55 and 60 <= summary["training_eligible_active_rule_count"] <= 75:
        lines.extend(
            [
                "## Count Interpretation",
                "",
                "The taxonomy-entry count is below 55 because several Orfogrammka entries group multiple executable rule_ids.",
                "This is an explicit audited justification: dataset planning should use the active rule_id count, not only the taxonomy-entry count.",
                "",
            ]
        )
    if summary["training_eligible_active_rule_count"] <= summary["legacy_eligible_now_count"]:
        lines.extend(
            [
                "## Eligibility Expansion Note",
                "",
                "The audit did not expand active rule_id coverage beyond the legacy strict entry count; investigate candidate probes before dataset build.",
                "",
            ]
        )
    return "\n".join(lines)


def _dataset_plan_markdown(rule_rows: list[dict[str, Any]]) -> str:
    eligible = [row for row in rule_rows if row["training_eligible_now"]]
    by_decision: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_decision[str(row["training_eligibility_decision"])].append(row)
    lines = [
        "# Current Capability Dataset Plan",
        "",
        "This plan is for a future dataset build only. It does not build data, train, change checkpoints, or change production thresholds.",
        "",
        "## Include Rule IDs",
        "",
    ]
    for decision in (INCLUDE_NOW, INCLUDE_AFTER_VALIDATOR, INCLUDE_AFTER_THRESHOLD_CALIBRATION, INCLUDE_AFTER_TRAINING):
        rows = sorted(by_decision.get(decision, []), key=lambda row: str(row["rule_id"]))
        lines.extend([f"### {decision}", ""])
        if not rows:
            lines.extend(["- none", ""])
            continue
        for row in rows:
            quota = _quota_for_risk(str(row["risk_level"]))
            needs = ", ".join(json.loads(row["needs_before_training"])) if str(row["needs_before_training"]).startswith("[") else str(row["needs_before_training"])
            lines.append(f"- {row['rule_id']}: risk={row['risk_level']}; quota={quota}; needs={needs}")
        lines.append("")
    lines.extend(
        [
            "## Hard Negatives Needed",
            "",
            "- dictionary/typo candidates: clean known words, proper names, morphology-compatible distractors",
            "- capitalization and abbreviations: sentence-initial traps, acronym/proper-name casing traps",
            "- punctuation: clean punctuation contexts, quote/bracket balance traps, direct-speech traps",
            "- context pairs: clean minimal pairs for each split/join alternative",
            "",
            "## Do Not Include",
            "",
            "- metadata-only taxonomy groups",
            "- planned or disabled rows",
            "- rows blocked by syntax, dictionary, NER, or no candidate path",
        ]
    )
    return "\n".join(lines) + "\n"


def _rules_yaml_update_markdown(summary: dict[str, Any], rules_config_path: str | Path) -> str:
    return "\n".join(
        [
            "# Rules YAML Pre-Dataset Update Report",
            "",
            f"- updated file: `{rules_config_path}`",
            "- updated only `dataset:` metadata fields",
            "- no dataset build, training, checkpoint, or production threshold mutation was performed",
            f"- total taxonomy entries: {summary['total_taxonomy_entries']}",
            f"- production_ready_now count: {summary['production_ready_now_count']}",
            f"- training_eligible_now count: {summary['training_eligible_now_count']}",
            f"- training_eligible active rule_id count: {summary['training_eligible_active_rule_count']}",
            "",
            "PRE_DATASET_RULES_YAML_UPDATE_COMPLETE",
            "",
        ]
    )


def _recommended_next_action(decision: str) -> str:
    return {
        BLOCK_METADATA_ONLY: "keep as taxonomy heading only",
        BLOCK_PLANNED: "implement bounded candidate path before training data",
        BLOCK_NO_CANDIDATE: "add candidate generator and candidate-backed probes",
        BLOCK_NEEDS_SYNTAX: "add bounded syntax-backed candidate implementation",
        BLOCK_NEEDS_DICTIONARY: "add dictionary/lexicon candidate support and probes",
        BLOCK_NEEDS_NER: "add NER/protected-span candidate path and hard negatives",
        BLOCK_DISABLED: "keep disabled until explicit policy decision",
    }.get(decision, "manual review")


def _risk_level(rule_id: str, requires: Iterable[str]) -> str:
    if rule_id in HIGH_RISK_RULE_IDS or rule_id in CONTEXT_PAIR_RULE_IDS:
        return "high"
    if rule_id in MEDIUM_RISK_RULE_IDS:
        return "medium"
    return _risk_from_requires(tuple(requires))


def _risk_from_requires(requires: tuple[str, ...]) -> str:
    req = {item.lower() for item in requires}
    if "ner" in req or "syntax" in req:
        return "high"
    if "dictionary" in req:
        return "medium"
    return "low"


def _validator_support(rule_id: str, requires: Iterable[str], matrix_state: dict[str, Any]) -> bool:
    if rule_id in RISKY_VALIDATOR_RULE_IDS:
        return bool(_number(matrix_state.get("rejected_count"), 0) > 0 or _number(matrix_state.get("false_positive"), 0) == 0)
    return bool("validator" in {str(item).lower() for item in requires} or _number(matrix_state.get("rejected_count"), 0) > 0)


def _max_risk(values: Iterable[str], fallback: str = "low") -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    selected = [value for value in values if value in order]
    if not selected:
        return fallback
    return max(selected, key=lambda value: order[value])


def _needs_union(values: Iterable[Iterable[str]]) -> list[str]:
    ordered = []
    for value in values:
        for item in value:
            if item == "none":
                continue
            if item not in ordered:
                ordered.append(item)
    return ordered or ["none"]


def _best_recall(probe_recall: float | None, report_recall: float | None) -> float | None:
    values = [value for value in (probe_recall, report_recall) if value is not None]
    return max(values) if values else None


def _max_or_none(values: Iterable[float | None]) -> float | None:
    selected = [float(value) for value in values if value is not None]
    return max(selected) if selected else None


def _recommended_dataset_size(active_rule_ids: list[str], rows: list[dict[str, Any]]) -> int:
    risk_by_rule = {str(row["rule_id"]): str(row["risk_level"]) for row in rows}
    return sum(_quota_for_risk(risk_by_rule.get(rule_id, "medium")) for rule_id in active_rule_ids)


def _quota_for_risk(risk: str) -> int:
    if risk == "high":
        return 120
    if risk == "medium":
        return 80
    return 50


def _clean_probe_sentences(clean_pool_path: str | Path) -> list[str]:
    path = Path(clean_pool_path)
    if path.exists():
        try:
            frame = pd.read_csv(path, usecols=lambda column: column in {"text", "sentence", "target", "source"})
            for column in ("text", "sentence", "target", "source"):
                if column in frame.columns:
                    values = [str(value) for value in frame[column].dropna().head(50).tolist() if str(value).strip()]
                    if values:
                        return values
        except Exception:
            pass
    return list(DEFAULT_CLEAN_SENTENCES)


def _read_csv(path: str | Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _number(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _apply_candidate(source: str, candidate: Any) -> str:
    start = int(getattr(candidate, "start", -1))
    end = int(getattr(candidate, "end", -1))
    if start < 0 or end < start:
        return source
    return source[:start] + str(getattr(candidate, "replacement", "")) + source[end:]


def _json_list(values: Iterable[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False)
