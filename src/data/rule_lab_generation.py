from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import itertools
import json
from pathlib import Path
import random
import re
from difflib import SequenceMatcher
from typing import Any, Iterable, Mapping

import yaml

from src.candidates.candidate_generator import Candidate
from src.config.candidate_dataset_config import candidate_dataset_paths, candidate_dataset_rule_quota, candidate_dataset_value
from src.data.atomic_verifier import AtomicVerificationResult, verify_atomic_positive
from src.data.dataset_contract import (
    DATASET_CONTRACT,
    HARD_NEGATIVE_OPEN,
    LAYER_ATOMIC_HARD_NEGATIVE,
    LAYER_ATOMIC_POSITIVE,
    SYNTHETIC_OPEN_CLEAN,
)
from src.data.dataset_quality import clean_or_hard_quality_reasons, normalized_pair_hash, normalize_pair
from src.data.training_quality_audit import contains_artificial_marker_text, plain_quote_bracket_balance_reasons
from src.rules.rule_ids import UNKNOWN_RULE_ID, normalize_rule_id


SOURCE_RULE_LAB = "rule_lab"
SERVICE_HARD_NEGATIVE_RULE_ID = "clean_identity_hard_negative"
HARD_NEGATIVE_ERROR_TYPE = "hard_negative"


@dataclass(frozen=True)
class RuleLabTemplate:
    template_id: str
    target: str = ""
    source: str = ""
    text: str = ""
    mutation: dict[str, Any] = field(default_factory=dict)
    required_slots: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0


@dataclass(frozen=True)
class RuleLabSlotLibrary:
    name: str
    values: list[Any] = field(default_factory=list)
    source: str = ""
    morphology_tags: list[str] = field(default_factory=list)
    case: str = ""
    gender: str = ""
    number: str = ""
    animacy: str = ""


@dataclass(frozen=True)
class RuleLabRecipe:
    rule_id: str
    family: str = ""
    enabled: bool = False
    positive_templates: list[RuleLabTemplate] = field(default_factory=list)
    hard_negative_templates: list[RuleLabTemplate] = field(default_factory=list)
    slots: dict[str, RuleLabSlotLibrary] = field(default_factory=dict)
    slot_sources: dict[str, Any] = field(default_factory=dict)
    morphology_constraints: dict[str, Any] = field(default_factory=dict)
    context_injection: dict[str, Any] = field(default_factory=dict)
    mutation: dict[str, Any] = field(default_factory=dict)
    min_required: int = 0
    preferred: int = 0
    max_total: int = 0
    max_per_template_share: float = 1.0
    max_per_slot_value_share: float = 1.0
    seed: int = 13
    disabled_reason: str = ""


@dataclass(frozen=True)
class RuleLabDiversityStats:
    rule_id: str
    generated_count: int = 0
    accepted_count: int = 0
    unique_source_count: int = 0
    unique_target_count: int = 0
    template_count: int = 0
    dominant_template_id: str = ""
    dominant_template_share: float = 0.0
    dominant_slot_value_share: float = 0.0
    duplicate_count: int = 0
    near_duplicate_count: int = 0
    status: str = "empty"
    reason: str = ""


@dataclass(frozen=True)
class RuleLabGenerationResult:
    atomic_positive_rows: list[dict[str, Any]] = field(default_factory=list)
    hard_negative_rows: list[dict[str, Any]] = field(default_factory=list)
    rejection_rows: list[dict[str, Any]] = field(default_factory=list)
    diversity_rows: list[RuleLabDiversityStats] = field(default_factory=list)
    template_validation_rows: list[dict[str, Any]] = field(default_factory=list)
    slot_usage_rows: list[dict[str, Any]] = field(default_factory=list)
    generation_rows: list[dict[str, Any]] = field(default_factory=list)
    recipe_status_rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class _RenderedPositive:
    rule_id: str
    template_id: str
    source: str
    target: str
    slot_values: dict[str, str]
    source_fragment: str
    target_fragment: str
    context_injected: bool = False


@dataclass(frozen=True)
class _RenderedHardNegative:
    rule_id: str
    template_id: str
    text: str
    slot_values: dict[str, str]


_PLACEHOLDER_RE = re.compile(r"{([A-Za-z0-9_.]+)}")


def load_rule_lab_recipes(path: str | Path) -> dict[str, RuleLabRecipe]:
    recipe_path = Path(path)
    if not recipe_path.exists():
        return {}
    raw = yaml.safe_load(recipe_path.read_text(encoding="utf-8")) or {}
    raw_rules = raw.get("rules", {}) if isinstance(raw, Mapping) else {}
    recipes: dict[str, RuleLabRecipe] = {}
    for raw_rule_id, raw_recipe in dict(raw_rules or {}).items():
        if not isinstance(raw_recipe, Mapping):
            continue
        rule_id = normalize_rule_id(str(raw_recipe.get("rule_id") or raw_rule_id))
        if rule_id == UNKNOWN_RULE_ID:
            continue
        recipes[rule_id] = _recipe_from_mapping(rule_id, raw_recipe)
    return recipes


def generate_rule_lab_rows(
    rule_ids: Iterable[str],
    candidate_generator: Any,
    config: Mapping[str, Any],
    *,
    clean_rows: Iterable[Mapping[str, Any]] | None = None,
    existing_positive_rows: Iterable[Mapping[str, Any]] | None = None,
    existing_hard_negative_rows: Iterable[Mapping[str, Any]] | None = None,
    positive_counts_by_rule: Mapping[str, int] | None = None,
    hard_counts_by_rule: Mapping[str, int] | None = None,
    positive_target: int | None = None,
    hard_target: int | None = None,
) -> RuleLabGenerationResult:
    rule_lab_config = _rule_lab_config(config)
    if not _truthy(rule_lab_config.get("enabled", False)) or not _truthy(rule_lab_config.get("fill_underfilled_rules", True)):
        return RuleLabGenerationResult()

    recipes = load_rule_lab_recipes(_recipe_path(config))
    quota = candidate_dataset_rule_quota(config)
    positive_goal = _target_value(positive_target, quota, "preferred_atomic_positives_per_active_rule", 50)
    hard_goal = _target_value(hard_target, quota, "min_hard_negatives_per_active_rule", 0)
    max_generated_per_rule = max(0, int(rule_lab_config.get("max_generated_per_rule", positive_goal) or 0))
    max_hard_per_rule = max(0, int(rule_lab_config.get("max_hard_negatives_per_rule", hard_goal) or 0))
    fail_on_low_diversity = _truthy(rule_lab_config.get("fail_on_low_diversity", True))

    clean_contexts = _clean_context_values(clean_rows or ())
    seen_positive = _positive_keys(existing_positive_rows or ())
    seen_hard = _hard_negative_keys(existing_hard_negative_rows or ())
    positive_counts = Counter({normalize_rule_id(rule_id): int(count or 0) for rule_id, count in dict(positive_counts_by_rule or {}).items()})
    hard_counts = Counter({normalize_rule_id(rule_id): int(count or 0) for rule_id, count in dict(hard_counts_by_rule or {}).items()})

    positive_rows: list[dict[str, Any]] = []
    hard_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    template_validation_rows: list[dict[str, Any]] = []
    slot_usage_counter: Counter[tuple[str, str, str, str]] = Counter()
    generated_counter: Counter[str] = Counter()
    accepted_positive_contexts: dict[str, list[_RenderedPositive]] = defaultdict(list)
    accepted_positive_rows_by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    diversity_rows: list[RuleLabDiversityStats] = []
    generation_rows: list[dict[str, Any]] = []
    recipe_status_rows: list[dict[str, Any]] = []

    normalized_rule_ids = _normalized_rule_ids(rule_ids)
    for rule_id in normalized_rule_ids:
        recipe = recipes.get(rule_id)
        if recipe is None:
            recipe_status_rows.append(_recipe_status_row(rule_id, None, "missing", "recipe_missing"))
            generation_rows.append(_generation_row(rule_id, 0, 0, 0, 0, "missing", "recipe_missing"))
            continue
        if not recipe.enabled:
            reason = recipe.disabled_reason or "disabled"
            recipe_status_rows.append(_recipe_status_row(rule_id, recipe, "disabled", reason))
            generation_rows.append(_generation_row(rule_id, 0, 0, 0, 0, "disabled", reason))
            continue

        recipe_status_rows.append(_recipe_status_row(rule_id, recipe, "enabled", ""))
        template_validation_rows.extend(_validate_recipe_templates(recipe))

        positive_requested = _positive_request_count(
            rule_id,
            recipe,
            positive_counts,
            positive_goal,
            max_generated_per_rule,
        )
        hard_requested = _hard_request_count(rule_id, hard_counts, hard_goal, max_hard_per_rule)

        generated_positive = 0
        accepted_positive = 0
        accepted_context_injected = 0
        max_context_injected = _context_injection_limit(recipe, rule_lab_config, positive_requested)
        for rendered in _render_positive_candidates(recipe, clean_contexts, positive_requested):
            generated_positive += 1
            generated_counter[rule_id] += 1
            if accepted_positive >= positive_requested:
                break
            reason = _rendered_positive_rejection(rendered, seen_positive, accepted_positive_contexts[rule_id])
            if not reason and rendered.context_injected and accepted_context_injected >= max_context_injected:
                reason = "context_injection_share_exceeded"
            if reason:
                rejection_rows.append(_rejection_row(rendered.rule_id, rendered.source, rendered.target, reason, "positive_render", rendered.template_id))
                continue
            verification = verify_atomic_positive(
                rendered.source,
                rendered.target,
                rendered.rule_id,
                candidate_generator=candidate_generator,
            )
            reason = _verification_rejection_reason(verification)
            if reason:
                rejection_rows.append(_rejection_row(rendered.rule_id, rendered.source, rendered.target, reason, "positive_verification", rendered.template_id))
                continue
            key = _positive_key(rendered.source, rendered.target, rendered.rule_id)
            seen_positive.add(key)
            row = _atomic_positive_row(rendered, verification)
            positive_rows.append(row)
            accepted_positive_contexts[rule_id].append(rendered)
            accepted_positive_rows_by_rule[rule_id].append(row)
            positive_counts[rule_id] += 1
            accepted_positive += 1
            if rendered.context_injected:
                accepted_context_injected += 1
            for slot_name, slot_value in rendered.slot_values.items():
                slot_usage_counter[(rule_id, rendered.template_id, slot_name, slot_value)] += 1

        accepted_hard = 0
        generated_hard = 0
        for rendered in _render_hard_negative_candidates(recipe, hard_requested):
            generated_hard += 1
            if accepted_hard >= hard_requested:
                break
            quality_reasons = _hard_negative_quality_failure_reasons(rendered.text)
            if quality_reasons:
                for reason in quality_reasons:
                    rejection_rows.append(
                        _rejection_row(
                            rendered.rule_id,
                            rendered.text,
                            rendered.text,
                            f"hard_negative_quality_failed:{reason}",
                            "hard_negative_quality",
                            rendered.template_id,
                        )
                    )
                continue
            candidate = _matching_hard_negative_candidate(rendered.text, rendered.rule_id, candidate_generator)
            if candidate is None:
                rejection_rows.append(_rejection_row(rendered.rule_id, rendered.text, rendered.text, "candidate_missing", "hard_negative_candidate", rendered.template_id))
                continue
            key = _hard_negative_key(rendered.text, rendered.rule_id)
            if key in seen_hard:
                rejection_rows.append(_rejection_row(rendered.rule_id, rendered.text, rendered.text, "duplicate_hard_negative", "hard_negative_dedupe", rendered.template_id))
                continue
            seen_hard.add(key)
            hard_rows.append(_hard_negative_row(rendered, candidate))
            hard_counts[rule_id] += 1
            accepted_hard += 1
            for slot_name, slot_value in rendered.slot_values.items():
                slot_usage_counter[(rule_id, rendered.template_id, slot_name, slot_value)] += 1

        diversity = _diversity_stats_for_rule(
            rule_id,
            generated_count=generated_counter[rule_id],
            rendered_rows=accepted_positive_contexts[rule_id],
            max_per_template_share=_recipe_template_share(recipe, rule_lab_config),
            max_per_slot_value_share=_recipe_slot_share(recipe, rule_lab_config),
        )
        if fail_on_low_diversity and accepted_positive_contexts[rule_id] and diversity.status == "blocked":
            positive_rows = [row for row in positive_rows if str(row.get("rule_id")) != rule_id]
            hard_rows = [row for row in hard_rows if str(row.get("target_rule_id")) != rule_id]
            positive_counts[rule_id] -= len(accepted_positive_rows_by_rule[rule_id])
            hard_counts[rule_id] -= accepted_hard
            rejection_rows.append(_rejection_row(rule_id, "", "", f"diversity_gate_failed:{diversity.reason}", "diversity_gate", ""))
            accepted_positive = 0
            accepted_hard = 0
        diversity_rows.append(diversity)
        generation_rows.append(
            _generation_row(
                rule_id,
                positive_requested,
                accepted_positive,
                hard_requested,
                accepted_hard,
                "ok" if accepted_positive or accepted_hard else "empty",
                "" if accepted_positive or accepted_hard else "no_rows_generated",
            )
        )

    slot_usage_rows = [
        {
            "rule_id": rule_id,
            "template_id": template_id,
            "slot_name": slot_name,
            "slot_value": slot_value,
            "count": count,
        }
        for (rule_id, template_id, slot_name, slot_value), count in sorted(slot_usage_counter.items())
    ]
    return RuleLabGenerationResult(
        atomic_positive_rows=positive_rows,
        hard_negative_rows=hard_rows,
        rejection_rows=rejection_rows,
        diversity_rows=diversity_rows,
        template_validation_rows=template_validation_rows,
        slot_usage_rows=slot_usage_rows,
        generation_rows=generation_rows,
        recipe_status_rows=recipe_status_rows,
    )


def rule_lab_recipe_rule_ids(config: Mapping[str, Any]) -> set[str]:
    if not _truthy(_rule_lab_config(config).get("enabled", False)):
        return set()
    return set(load_rule_lab_recipes(_recipe_path(config)))


def _recipe_from_mapping(rule_id: str, raw: Mapping[str, Any]) -> RuleLabRecipe:
    slots = {
        str(name): _slot_library(str(name), value)
        for name, value in dict(raw.get("slots", {}) or {}).items()
    }
    diversity = dict(raw.get("diversity", {}) or {})
    return RuleLabRecipe(
        rule_id=rule_id,
        family=str(raw.get("family") or ""),
        enabled=_truthy(raw.get("enabled", False)),
        positive_templates=[_template_from_mapping(item) for item in list(raw.get("positive_templates", []) or [])],
        hard_negative_templates=[_template_from_mapping(item) for item in list(raw.get("hard_negative_templates", []) or [])],
        slots=slots,
        slot_sources=dict(raw.get("slot_sources", {}) or {}),
        morphology_constraints=dict(raw.get("morphology_constraints", {}) or {}),
        context_injection=dict(raw.get("context_injection", {}) or {}),
        mutation=dict(raw.get("mutation", {}) or {}),
        min_required=max(0, int(raw.get("min_required", 0) or 0)),
        preferred=max(0, int(raw.get("preferred", 0) or 0)),
        max_total=max(0, int(raw.get("max_total", 0) or 0)),
        max_per_template_share=float(diversity.get("max_per_template_share", raw.get("max_per_template_share", 1.0)) or 1.0),
        max_per_slot_value_share=float(diversity.get("max_per_slot_value_share", raw.get("max_per_slot_value_share", 1.0)) or 1.0),
        seed=int(raw.get("seed", 13) or 13),
        disabled_reason=str(raw.get("disabled_reason") or ""),
    )


def _template_from_mapping(raw: Any) -> RuleLabTemplate:
    if isinstance(raw, str):
        return RuleLabTemplate(template_id=_template_id_from_text(raw), target=raw)
    item = dict(raw or {})
    return RuleLabTemplate(
        template_id=str(item.get("template_id") or _template_id_from_text(str(item.get("target") or item.get("text") or ""))),
        target=str(item.get("target") or ""),
        source=str(item.get("source") or ""),
        text=str(item.get("text") or ""),
        mutation=dict(item.get("mutation", {}) or {}),
        required_slots=[str(value) for value in list(item.get("required_slots", []) or [])],
        constraints=dict(item.get("constraints", {}) or {}),
        weight=float(item.get("weight", 1.0) or 1.0),
    )


def _slot_library(name: str, raw: Any) -> RuleLabSlotLibrary:
    if isinstance(raw, list):
        return RuleLabSlotLibrary(name=name, values=list(raw))
    item = dict(raw or {})
    return RuleLabSlotLibrary(
        name=name,
        values=list(item.get("values", []) or []),
        source=str(item.get("source") or ""),
        morphology_tags=[str(value) for value in list(item.get("morphology_tags", []) or [])],
        case=str(item.get("case") or ""),
        gender=str(item.get("gender") or ""),
        number=str(item.get("number") or ""),
        animacy=str(item.get("animacy") or ""),
    )


def _render_positive_candidates(recipe: RuleLabRecipe, clean_contexts: list[str], requested: int) -> list[_RenderedPositive]:
    if requested <= 0:
        return []
    rendered: list[_RenderedPositive] = []
    for template in recipe.positive_templates:
        variants = _rendered_template_variants(recipe, template, clean_contexts=clean_contexts, hard_negative=False)
        rendered.extend(_positive_from_variant(recipe, template, variant) for variant in variants)
    return _deterministic_sample(rendered, recipe.seed, len(rendered))


def _render_hard_negative_candidates(recipe: RuleLabRecipe, requested: int) -> list[_RenderedHardNegative]:
    if requested <= 0:
        return []
    rendered: list[_RenderedHardNegative] = []
    for template in recipe.hard_negative_templates:
        for variant in _rendered_template_variants(recipe, template, clean_contexts=[], hard_negative=True):
            text = str(variant.get("text") or variant.get("target") or "")
            if text and not _has_unfilled_slot(text):
                rendered.append(
                    _RenderedHardNegative(
                        rule_id=recipe.rule_id,
                        template_id=template.template_id,
                        text=text,
                        slot_values=dict(variant.get("slot_values", {}) or {}),
                    )
                )
    return _deterministic_sample(rendered, recipe.seed + 101, len(rendered))


def _positive_from_variant(recipe: RuleLabRecipe, template: RuleLabTemplate, variant: Mapping[str, Any]) -> _RenderedPositive:
    target = str(variant.get("target") or "")
    explicit_source = str(variant.get("source") or "")
    mutation = dict(template.mutation or recipe.mutation or {})
    source_fragment = str(mutation.get("source_fragment") or "")
    target_fragment = str(mutation.get("target_fragment") or "")
    source = explicit_source
    if not source and source_fragment and target_fragment and target_fragment in target:
        source = target.replace(target_fragment, source_fragment, 1)
    return _RenderedPositive(
        rule_id=recipe.rule_id,
        template_id=template.template_id,
        source=source,
        target=target,
        slot_values=dict(variant.get("slot_values", {}) or {}),
        source_fragment=source_fragment,
        target_fragment=target_fragment,
        context_injected=bool(variant.get("context_injected", False)),
    )


def _rendered_template_variants(
    recipe: RuleLabRecipe,
    template: RuleLabTemplate,
    *,
    clean_contexts: list[str],
    hard_negative: bool,
) -> list[dict[str, Any]]:
    raw_templates = {
        "target": template.target,
        "source": template.source,
        "text": template.text,
    }
    placeholders = _placeholders(raw_templates.values())
    slot_values = _slot_values_for_placeholders(recipe, placeholders, clean_contexts=clean_contexts)
    required_roots = {_slot_root(value) for value in template.required_slots}
    required_roots.update(_slot_root(value) for value in placeholders)
    missing = sorted(root for root in required_roots if root not in slot_values)
    if missing:
        return []
    roots = sorted(slot_values)
    combinations = itertools.product(*(slot_values[root] for root in roots))
    variants: list[dict[str, Any]] = []
    for values in combinations:
        context = {root: value for root, value in zip(roots, values)}
        rendered = {key: _render_text(text, context) for key, text in raw_templates.items()}
        rendered["slot_values"] = _flat_slot_values(context)
        rendered["context_injected"] = "context" in context
        if hard_negative and not rendered.get("text") and rendered.get("target"):
            rendered["text"] = rendered["target"]
        if any(_has_unfilled_slot(str(value)) for value in rendered.values() if isinstance(value, str)):
            continue
        variants.append(rendered)
    return variants


def _slot_values_for_placeholders(
    recipe: RuleLabRecipe,
    placeholders: set[str],
    *,
    clean_contexts: list[str],
) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = {}
    roots = {_slot_root(value) for value in placeholders}
    for root in roots:
        if root == "context":
            if _truthy(recipe.context_injection.get("enabled", False)):
                result[root] = list(clean_contexts)
            continue
        library = recipe.slots.get(root)
        if library is not None and library.values:
            result[root] = list(library.values)
    return result


def _render_text(template: str, values: Mapping[str, Any]) -> str:
    if not template:
        return ""

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        root = _slot_root(key)
        field_name = _slot_field(key)
        value = values.get(root)
        if value is None:
            return match.group(0)
        if field_name:
            if isinstance(value, Mapping):
                return str(value.get(field_name, match.group(0)))
            return match.group(0)
        return str(value)

    return _PLACEHOLDER_RE.sub(replace, template)


def _flat_slot_values(values: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for slot_name, value in values.items():
        if isinstance(value, Mapping):
            for field_name, field_value in value.items():
                result[f"{slot_name}.{field_name}"] = str(field_value)
        else:
            result[slot_name] = str(value)
    return result


def _rendered_positive_rejection(
    rendered: _RenderedPositive,
    seen_positive: set[tuple[str, str, str]],
    accepted: list[_RenderedPositive],
) -> str:
    if not rendered.source or not rendered.target:
        return "empty_source_or_target"
    if _has_unfilled_slot(rendered.source) or _has_unfilled_slot(rendered.target):
        return "unfilled_slot"
    if rendered.source == rendered.target:
        return "identity_pair"
    key = _positive_key(rendered.source, rendered.target, rendered.rule_id)
    if key in seen_positive:
        return "duplicate_pair"
    if _near_duplicate(rendered.target, [item.target for item in accepted]):
        return "near_duplicate_pair"
    return ""


def _verification_rejection_reason(verification: AtomicVerificationResult) -> str:
    if not verification.passed:
        return verification.reason or "atomic_verification_failed"
    if int(verification.gold_edit_count) != 1:
        return "non_atomic_edit_count"
    if not verification.candidate_present:
        return "candidate_missing"
    if not verification.strict_validator_passed:
        return "strict_validator_rejected"
    if not verification.target_quality_passed:
        return "target_quality_failed"
    if int(verification.extra_edit_count) != 0:
        return "extra_edits_present"
    return ""


def _atomic_positive_row(rendered: _RenderedPositive, verification: AtomicVerificationResult) -> dict[str, Any]:
    matched = dict(verification.matched_candidate or {})
    edits = [_edit_payload(edit, rendered.rule_id) for edit in verification.edits]
    pair_hash = normalized_pair_hash(rendered.source, rendered.target)
    metadata = {
        "dataset_contract": DATASET_CONTRACT,
        "dataset_layer": LAYER_ATOMIC_POSITIVE,
        "source_type": SYNTHETIC_OPEN_CLEAN,
        "rule_ids": [rendered.rule_id],
        "is_atomic": True,
        "is_stress": False,
        "count_toward_rule_quota": True,
        "loss_weight": 1.0,
        "gold_edit_count": 1,
        "candidate_present": True,
        "candidate_rule_ids": list(verification.candidate_rule_ids),
        "strict_validator_passed": True,
        "target_quality_pass": True,
        "target_quality_passed": True,
        "extra_edit_count": int(verification.extra_edit_count),
        "matched_candidate": matched,
        "atomic_verification": asdict(verification),
        "target_family": rendered.rule_id,
        "activation_source": SOURCE_RULE_LAB,
        "generation_sources": [SOURCE_RULE_LAB],
        "generation_strategy": SOURCE_RULE_LAB,
        "error_bearing_sentence_source": "corpus" if rendered.context_injected else "fallback_template",
        "original_clean_sentence": rendered.target,
        "source_name": SOURCE_RULE_LAB,
        "source_row_id": rendered.template_id,
        "miner_name": SOURCE_RULE_LAB,
        "template_id": rendered.template_id,
        "slot_values": rendered.slot_values,
        "source_fragment": rendered.source_fragment,
        "target_fragment": rendered.target_fragment,
        "normalized_pair_hash": pair_hash,
    }
    error_type = _error_type_from_verification(verification)
    return {
        "source": rendered.source,
        "target": rendered.target,
        "split": "train",
        "source_type": SYNTHETIC_OPEN_CLEAN,
        "error_type": error_type,
        "rule_ids": json.dumps([rendered.rule_id], ensure_ascii=False),
        "edits": json.dumps(edits, ensure_ascii=False),
        "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        "original_clean_source": rendered.target,
        "source_corpus": SOURCE_RULE_LAB,
        "source_subcorpus": rendered.rule_id,
        "is_hard_negative": False,
        "is_real_pair": False,
        "template_id": rendered.template_id,
        "normalized_pair_hash": pair_hash,
        "error_types": json.dumps([error_type], ensure_ascii=False),
        "source_dataset": SOURCE_RULE_LAB,
        "is_clean": False,
        "is_synthetic": True,
        "domain": "open_clean",
        "rule_id": rendered.rule_id,
        "edit_operations": json.dumps(edits, ensure_ascii=False),
        "dataset_contract": DATASET_CONTRACT,
        "dataset_layer": LAYER_ATOMIC_POSITIVE,
        "is_atomic": True,
        "is_stress": False,
        "count_toward_rule_quota": True,
        "gold_edit_count": 1,
        "loss_weight": 1.0,
        "target_rule_id": rendered.rule_id,
        "candidate_source": str(matched.get("source") or ""),
        "candidate_replacement": str(matched.get("replacement") or ""),
        "candidate_start": int(matched.get("start", -1) if matched.get("start", -1) != "" else -1),
        "candidate_end": int(matched.get("end", -1) if matched.get("end", -1) != "" else -1),
        "verification_status": "passed",
        "rejection_reason": "",
        "activation_source": SOURCE_RULE_LAB,
    }


def _hard_negative_row(rendered: _RenderedHardNegative, candidate: Candidate) -> dict[str, Any]:
    candidate_source = str(getattr(candidate, "source", ""))
    candidate_replacement = str(getattr(candidate, "replacement", ""))
    candidate_start = int(getattr(candidate, "start", -1))
    candidate_end = int(getattr(candidate, "end", -1))
    pair_hash = normalized_pair_hash(rendered.text, rendered.text)
    metadata = {
        "dataset_contract": DATASET_CONTRACT,
        "dataset_layer": LAYER_ATOMIC_HARD_NEGATIVE,
        "source_type": HARD_NEGATIVE_OPEN,
        "rule_ids": [SERVICE_HARD_NEGATIVE_RULE_ID],
        "is_atomic": True,
        "is_hard_negative": True,
        "target_rule_id": rendered.rule_id,
        "candidate_rule_id": rendered.rule_id,
        "candidate_source": candidate_source,
        "candidate_replacement": candidate_replacement,
        "candidate_start": candidate_start,
        "candidate_end": candidate_end,
        "candidate_edit_type": str(getattr(candidate, "edit_type", "")),
        "candidate_mode": str(getattr(candidate, "mode", "")),
        "candidate_requires_model": bool(getattr(candidate, "requires_model", False)),
        "candidate_requires_scoring": bool(getattr(candidate, "requires_scoring", False)),
        "candidate_group": str(getattr(candidate, "group", "")),
        "count_toward_rule_quota": False,
        "loss_weight": 1.0,
        "gold_edit_count": 0,
        "activation_source": SOURCE_RULE_LAB,
        "generation_sources": [SOURCE_RULE_LAB],
        "generation_strategy": SOURCE_RULE_LAB,
        "hard_negative_source": SOURCE_RULE_LAB,
        "expected_accepted_edits": 0,
        "original_clean_sentence": rendered.text,
        "source_name": SOURCE_RULE_LAB,
        "source_row_id": rendered.template_id,
        "miner_name": SOURCE_RULE_LAB,
        "template_id": rendered.template_id,
        "slot_values": rendered.slot_values,
    }
    return {
        "source": rendered.text,
        "target": rendered.text,
        "split": "train",
        "source_type": HARD_NEGATIVE_OPEN,
        "error_type": HARD_NEGATIVE_ERROR_TYPE,
        "rule_ids": json.dumps([SERVICE_HARD_NEGATIVE_RULE_ID], ensure_ascii=False),
        "edits": "[]",
        "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        "original_clean_source": rendered.text,
        "source_corpus": SOURCE_RULE_LAB,
        "source_subcorpus": rendered.rule_id,
        "is_hard_negative": True,
        "is_real_pair": False,
        "template_id": rendered.template_id,
        "normalized_pair_hash": pair_hash,
        "error_types": json.dumps([HARD_NEGATIVE_ERROR_TYPE], ensure_ascii=False),
        "source_dataset": SOURCE_RULE_LAB,
        "is_clean": True,
        "is_synthetic": True,
        "domain": "open_clean",
        "rule_id": SERVICE_HARD_NEGATIVE_RULE_ID,
        "edit_operations": "[]",
        "dataset_contract": DATASET_CONTRACT,
        "dataset_layer": LAYER_ATOMIC_HARD_NEGATIVE,
        "is_atomic": True,
        "is_stress": False,
        "count_toward_rule_quota": False,
        "loss_weight": 1.0,
        "gold_edit_count": 0,
        "target_rule_id": rendered.rule_id,
        "candidate_source": candidate_source,
        "candidate_replacement": candidate_replacement,
        "candidate_start": candidate_start,
        "candidate_end": candidate_end,
        "verification_status": "passed",
        "rejection_reason": "",
        "activation_source": SOURCE_RULE_LAB,
    }


def _matching_hard_negative_candidate(text: str, rule_id: str, candidate_generator: Any) -> Candidate | None:
    try:
        candidates = list(candidate_generator.generate(text))
    except Exception:
        return None
    for candidate in candidates:
        if normalize_rule_id(getattr(candidate, "rule_id", "")) != rule_id:
            continue
        if _candidate_is_non_keep(candidate):
            return candidate
    return None


def _candidate_is_non_keep(candidate: Any) -> bool:
    edit_type = str(getattr(candidate, "edit_type", "") or "").lower()
    action = str(getattr(candidate, "action", "") or "").upper()
    if edit_type == "keep" or action in {"KEEP", "KEEP_NONE", "KEEP_EXISTING"}:
        return False
    return str(getattr(candidate, "source", "")) != str(getattr(candidate, "replacement", "")) or action in {"INSERT", "DELETE", "REPLACE"}


def _hard_negative_quality_failure_reasons(text: str) -> list[str]:
    reasons: list[str] = []
    if contains_artificial_marker_text(text, text):
        reasons.append("artificial_marker")
    balance_reasons = plain_quote_bracket_balance_reasons(text)
    reasons.extend(balance_reasons)
    quality_reasons = clean_or_hard_quality_reasons(text)
    if balance_reasons:
        quality_reasons = [reason for reason in quality_reasons if reason != "unbalanced_quote_or_bracket"]
    reasons.extend(quality_reasons)
    return list(dict.fromkeys(reasons))


def _diversity_stats_for_rule(
    rule_id: str,
    *,
    generated_count: int,
    rendered_rows: list[_RenderedPositive],
    max_per_template_share: float,
    max_per_slot_value_share: float,
) -> RuleLabDiversityStats:
    accepted_count = len(rendered_rows)
    if accepted_count <= 0:
        return RuleLabDiversityStats(rule_id=rule_id, generated_count=generated_count, status="empty", reason="no_accepted_rows")
    sources = [normalize_pair(row.source) for row in rendered_rows]
    targets = [normalize_pair(row.target) for row in rendered_rows]
    template_counts = Counter(row.template_id for row in rendered_rows)
    slot_counts: Counter[tuple[str, str]] = Counter()
    for row in rendered_rows:
        for slot_name, slot_value in row.slot_values.items():
            slot_counts[(slot_name, slot_value)] += 1
    dominant_template_id = template_counts.most_common(1)[0][0] if template_counts else ""
    dominant_template_share = _dominant_share(list(template_counts.elements()))
    dominant_slot_value_share = max((count / accepted_count for count in slot_counts.values()), default=0.0)
    duplicate_count = accepted_count - len(set(sources))
    near_duplicate_count = _near_duplicate_count(targets)
    reasons: list[str] = []
    if duplicate_count != 0 or len(set(sources)) != accepted_count:
        reasons.append("duplicate_count")
    if dominant_template_share > max_per_template_share:
        reasons.append(f"dominant_template_share:{dominant_template_share:.3f}>{max_per_template_share:.3f}")
    if dominant_slot_value_share > max_per_slot_value_share:
        reasons.append(f"dominant_slot_value_share:{dominant_slot_value_share:.3f}>{max_per_slot_value_share:.3f}")
    status = "blocked" if reasons else "ok"
    return RuleLabDiversityStats(
        rule_id=rule_id,
        generated_count=generated_count,
        accepted_count=accepted_count,
        unique_source_count=len(set(sources)),
        unique_target_count=len(set(targets)),
        template_count=len(template_counts),
        dominant_template_id=dominant_template_id,
        dominant_template_share=dominant_template_share,
        dominant_slot_value_share=dominant_slot_value_share,
        duplicate_count=duplicate_count,
        near_duplicate_count=near_duplicate_count,
        status=status,
        reason=",".join(reasons),
    )


def _validate_recipe_templates(recipe: RuleLabRecipe) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kind, templates in (("positive", recipe.positive_templates), ("hard_negative", recipe.hard_negative_templates)):
        for template in templates:
            placeholders = _placeholders((template.source, template.target, template.text))
            missing = sorted(
                root
                for root in {_slot_root(value) for value in placeholders}
                if root != "context" and root not in recipe.slots
            )
            status = "ok" if not missing else "invalid"
            reason = "" if not missing else "missing_slots:" + ",".join(missing)
            rows.append(
                {
                    "rule_id": recipe.rule_id,
                    "template_id": template.template_id,
                    "template_kind": kind,
                    "status": status,
                    "reason": reason,
                }
            )
    return rows


def _generation_row(rule_id: str, positive_requested: int, positive_generated: int, hard_requested: int, hard_generated: int, status: str, reason: str) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "positive_requested": positive_requested,
        "positive_generated": positive_generated,
        "hard_negative_requested": hard_requested,
        "hard_negative_generated": hard_generated,
        "status": status,
        "reason": reason,
    }


def _recipe_status_row(rule_id: str, recipe: RuleLabRecipe | None, status: str, reason: str) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "enabled": bool(recipe.enabled) if recipe is not None else False,
        "status": status,
        "reason": reason,
        "disabled_reason": recipe.disabled_reason if recipe is not None else "",
        "positive_template_count": len(recipe.positive_templates) if recipe is not None else 0,
        "hard_negative_template_count": len(recipe.hard_negative_templates) if recipe is not None else 0,
    }


def _rejection_row(rule_id: str, source: str, target: str, reason: str, stage: str, template_id: str) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "source": source[:500],
        "target": target[:500],
        "reason": reason,
        "stage": stage,
        "template_id": template_id,
        "source_name": SOURCE_RULE_LAB,
        "source_row_id": template_id,
        "miner_name": SOURCE_RULE_LAB,
    }


def _positive_request_count(
    rule_id: str,
    recipe: RuleLabRecipe,
    counts: Mapping[str, int],
    target: int,
    max_generated_per_rule: int,
) -> int:
    deficit = max(0, target - int(counts.get(rule_id, 0) or 0))
    if recipe.preferred:
        deficit = min(deficit, max(0, recipe.preferred - int(counts.get(rule_id, 0) or 0)))
    cap = max_generated_per_rule
    if recipe.max_total:
        cap = min(cap, recipe.max_total)
    return max(0, min(deficit, cap))


def _hard_request_count(rule_id: str, counts: Mapping[str, int], target: int, max_hard_per_rule: int) -> int:
    return max(0, min(max(0, target - int(counts.get(rule_id, 0) or 0)), max_hard_per_rule))


def _recipe_template_share(recipe: RuleLabRecipe, config: Mapping[str, Any]) -> float:
    recipe_limit = float(recipe.max_per_template_share or 1.0)
    config_limit = float(config.get("max_per_template_share", 1.0) or 1.0)
    return min(recipe_limit, config_limit)


def _recipe_slot_share(recipe: RuleLabRecipe, config: Mapping[str, Any]) -> float:
    recipe_limit = float(recipe.max_per_slot_value_share or 1.0)
    config_limit = float(config.get("max_per_slot_value_share", 1.0) or 1.0)
    return min(recipe_limit, config_limit)


def _context_injection_limit(recipe: RuleLabRecipe, config: Mapping[str, Any], requested: int) -> int:
    if requested <= 0 or not _truthy(recipe.context_injection.get("enabled", False)):
        return 0
    configured = recipe.context_injection.get("max_share")
    if configured is None:
        configured = config.get("context_injection_max_share", 0.5)
    share = max(0.0, min(1.0, float(configured or 0.0)))
    if share <= 0:
        return 0
    return max(1, int(requested * share + 0.999999))


def _recipe_path(config: Mapping[str, Any]) -> Path:
    paths = candidate_dataset_paths(config)
    return Path(paths.get("rule_lab_recipes_config") or "configs/rule_lab_recipes.yaml")


def _rule_lab_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(candidate_dataset_value(config, "rule_lab", {}) or {})


def _target_value(explicit: int | None, quota: Mapping[str, Any], key: str, default: int) -> int:
    if explicit is not None:
        return max(0, int(explicit or 0))
    return max(0, int(quota.get(key, default) or 0))


def _normalized_rule_ids(rule_ids: Iterable[str]) -> list[str]:
    result: list[str] = []
    for rule_id in rule_ids:
        normalized = normalize_rule_id(str(rule_id))
        if normalized == UNKNOWN_RULE_ID or normalized in result:
            continue
        result.append(normalized)
    return result


def _positive_keys(rows: Iterable[Mapping[str, Any]]) -> set[tuple[str, str, str]]:
    return {
        _positive_key(str(row.get("source", "")), str(row.get("target", "")), str(row.get("rule_id") or row.get("target_rule_id") or ""))
        for row in rows
        if normalize_rule_id(str(row.get("rule_id") or row.get("target_rule_id") or "")) != UNKNOWN_RULE_ID
    }


def _hard_negative_keys(rows: Iterable[Mapping[str, Any]]) -> set[tuple[str, str, str]]:
    result: set[tuple[str, str, str]] = set()
    for row in rows:
        rule_id = normalize_rule_id(str(row.get("target_rule_id") or ""))
        if rule_id == UNKNOWN_RULE_ID:
            continue
        result.add(_hard_negative_key(str(row.get("source", "")), rule_id))
    return result


def _positive_key(source: str, target: str, rule_id: str) -> tuple[str, str, str]:
    return (normalize_pair(source), normalize_pair(target), normalize_rule_id(rule_id))


def _hard_negative_key(text: str, rule_id: str) -> tuple[str, str, str]:
    return (normalize_pair(text), normalize_pair(text), normalize_rule_id(rule_id))


def _clean_context_values(clean_rows: Iterable[Mapping[str, Any]]) -> list[str]:
    values: list[str] = []
    for row in clean_rows:
        text = ""
        for key in ("text", "target", "source"):
            raw = row.get(key)
            if raw:
                text = str(raw).strip()
                break
        if not text:
            continue
        text = re.sub(r"[.!?…]+$", "", text).strip()
        if text and text not in values:
            values.append(text)
    return values


def _deterministic_sample(items: list[Any], seed: int, limit: int) -> list[Any]:
    if limit <= 0:
        return []
    result = list(items)
    random.Random(seed).shuffle(result)
    return result[:limit]


def _placeholders(values: Iterable[str]) -> set[str]:
    result: set[str] = set()
    for value in values:
        result.update(match.group(1) for match in _PLACEHOLDER_RE.finditer(str(value or "")))
    return result


def _slot_root(value: str) -> str:
    return str(value).split(".", 1)[0]


def _slot_field(value: str) -> str:
    parts = str(value).split(".", 1)
    return parts[1] if len(parts) == 2 else ""


def _has_unfilled_slot(text: str) -> bool:
    return bool(_PLACEHOLDER_RE.search(text))


def _near_duplicate(text: str, existing: list[str]) -> bool:
    normalized = normalize_pair(text)
    for item in existing:
        other = normalize_pair(item)
        if normalized == other:
            return True
        if SequenceMatcher(None, normalized, other).ratio() >= 0.985:
            return True
    return False


def _near_duplicate_count(values: list[str]) -> int:
    near = 0
    for index, left in enumerate(values):
        for right in values[index + 1 :]:
            if left != right and SequenceMatcher(None, left, right).ratio() >= 0.985:
                near += 1
    return near


def _dominant_share(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    return max(counts.values()) / len(values)


def _edit_payload(edit: Mapping[str, Any], rule_id: str) -> dict[str, Any]:
    payload = dict(edit)
    payload["rule_id"] = normalize_rule_id(str(payload.get("rule_id") or rule_id))
    return payload


def _error_type_from_verification(verification: AtomicVerificationResult) -> str:
    if not verification.edits:
        return "spelling"
    edit_type = str(verification.edits[0].get("edit_type", ""))
    if edit_type.startswith("punctuation_"):
        return "punctuation"
    if edit_type in {"split_word", "join_words"}:
        return "split_join"
    return "spelling"


def _template_id_from_text(text: str) -> str:
    digest = abs(hash(text)) % 1_000_000
    return f"template_{digest}"


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
