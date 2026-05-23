from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
import itertools
import json
import random
from collections.abc import Iterable, Mapping
from typing import Any

from src.candidates.candidate_generator import Candidate, CandidateGenerator
from src.candidates.matching import candidate_matches_edit
from src.data.atomic_verifier import gold_edits_for_pair
from src.data.dataset_contract import DATASET_CONTRACT, LAYER_STRESS_MULTI_ERROR, SYNTHETIC_OPEN_CLEAN
from src.data.dataset_quality import normalized_pair_hash
from src.rules.rule_ids import UNKNOWN_RULE_ID, normalize_rule_id
from src.validation.edit_classifier import coarse_error_type
from src.validation.strict_validator import StrictValidator


MAX_OPPORTUNITIES_PER_RULE = 8
MAX_COMBO_ATTEMPTS_PER_SENTENCE = 200


@dataclass(frozen=True)
class StressGenerationResult:
    rows: list[dict[str, Any]]
    attempt_rows: list[dict[str, Any]]
    rejection_rows: list[dict[str, Any]]
    counts_by_rule_combo: dict[str, int]


@dataclass(frozen=True)
class _RuleOpportunity:
    rule_id: str
    operator: Any
    opportunity: Any


def generate_multi_error_stress_rows(
    clean_rows: list[dict[str, Any]],
    registry: Any,
    rule_ids: list[str],
    candidate_generator: CandidateGenerator | Any,
    target_count: int,
    seed: int,
    max_errors_per_sentence: int = 3,
    loss_weight: float = 0.4,
) -> StressGenerationResult:
    target = max(0, int(target_count))
    if target <= 0:
        return StressGenerationResult([], [], [], {})

    max_errors = max(2, min(3, int(max_errors_per_sentence)))
    randomizer = random.Random(seed)
    generator = candidate_generator or CandidateGenerator()
    target_rule_ids = _normalized_rule_ids(rule_ids)
    operators: dict[str, Any] = {}
    rejection_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    counts_by_rule_combo: Counter[str] = Counter()
    seen_hashes: set[str] = set()

    for rule_id in target_rule_ids:
        operator = registry.get(rule_id) if registry is not None else None
        if operator is None:
            rejection_rows.append(_rejection_row(-1, "", "", rule_id, "operator_missing", stage="registry"))
            continue
        operators[rule_id] = operator

    indexed_rows = list(enumerate(clean_rows))
    randomizer.shuffle(indexed_rows)
    for row_index, clean_row in indexed_rows:
        if len(rows) >= target:
            break
        clean_target = _clean_target_from_row(clean_row)
        if not clean_target:
            rejection_rows.append(_rejection_row(row_index, "", "", "", "missing_clean_target", stage="clean_row"))
            continue

        opportunities, opportunity_errors = _find_opportunities(clean_target, operators, row_index)
        rejection_rows.extend(opportunity_errors)
        rule_count = len({item.rule_id for item in opportunities})
        if rule_count < 2:
            rejection_rows.append(
                _rejection_row(row_index, clean_target, clean_target, "", "insufficient_rule_opportunities", stage="opportunity")
            )
            continue

        combos = _candidate_combinations(opportunities, max_errors=max_errors, randomizer=randomizer)
        if not combos:
            rejection_rows.append(
                _rejection_row(row_index, clean_target, clean_target, "", "overlapping_opportunities", stage="opportunity")
            )
            continue

        for combo in combos:
            if len(rows) >= target:
                break
            rule_combo = _combo_key(item.rule_id for item in combo)
            accepted_row, rejection = _build_stress_row(
                row_index=row_index,
                clean_row=clean_row,
                target=clean_target,
                selected=combo,
                candidate_generator=generator,
                max_errors=max_errors,
                loss_weight=loss_weight,
            )
            if accepted_row is not None:
                pair_hash = str(accepted_row["normalized_pair_hash"])
                if pair_hash in seen_hashes:
                    rejection = _rejection_row(
                        row_index,
                        str(accepted_row.get("source", "")),
                        str(accepted_row.get("target", "")),
                        rule_combo,
                        "duplicate_pair",
                        stage="dedupe",
                    )
                    accepted_row = None
                else:
                    seen_hashes.add(pair_hash)

            attempt_rows.append(
                {
                    "row_index": row_index,
                    "rule_combo": rule_combo,
                    "target": clean_target[:300],
                    "accepted": accepted_row is not None,
                    "reason": "" if accepted_row is not None else str((rejection or {}).get("reason", "")),
                }
            )
            if accepted_row is None:
                if rejection is not None:
                    rejection_rows.append(rejection)
                continue
            rows.append(accepted_row)
            counts_by_rule_combo[rule_combo] += 1

    return StressGenerationResult(
        rows=rows,
        attempt_rows=attempt_rows,
        rejection_rows=rejection_rows,
        counts_by_rule_combo=dict(sorted(counts_by_rule_combo.items())),
    )


def _find_opportunities(
    target: str,
    operators: Mapping[str, Any],
    row_index: int,
) -> tuple[list[_RuleOpportunity], list[dict[str, Any]]]:
    opportunities: list[_RuleOpportunity] = []
    rejections: list[dict[str, Any]] = []
    for rule_id, operator in operators.items():
        try:
            found = list(operator.find_opportunities(target))
        except Exception as exc:
            rejections.append(
                _rejection_row(
                    row_index,
                    target,
                    target,
                    rule_id,
                    "operator_opportunity_error",
                    stage="opportunity",
                    error=exc,
                )
            )
            continue
        for opportunity in found[:MAX_OPPORTUNITIES_PER_RULE]:
            if _valid_opportunity_span(target, opportunity):
                opportunities.append(_RuleOpportunity(rule_id, operator, opportunity))
    return opportunities, rejections


def _candidate_combinations(
    opportunities: list[_RuleOpportunity],
    *,
    max_errors: int,
    randomizer: random.Random,
) -> list[tuple[_RuleOpportunity, ...]]:
    combos: list[tuple[_RuleOpportunity, ...]] = []
    max_size = min(max_errors, len({item.rule_id for item in opportunities}))
    for size in range(max_size, 1, -1):
        for combo in itertools.combinations(opportunities, size):
            if len({item.rule_id for item in combo}) != size:
                continue
            if not _non_overlapping(combo):
                continue
            combos.append(combo)
    randomizer.shuffle(combos)
    return combos[:MAX_COMBO_ATTEMPTS_PER_SENTENCE]


def _build_stress_row(
    *,
    row_index: int,
    clean_row: Mapping[str, Any],
    target: str,
    selected: tuple[_RuleOpportunity, ...],
    candidate_generator: Any,
    max_errors: int,
    loss_weight: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    rule_ids = sorted({item.rule_id for item in selected})
    rule_combo = _combo_key(rule_ids)
    source = target
    try:
        for item in sorted(selected, key=lambda value: (_opportunity_start(value.opportunity), _opportunity_end(value.opportunity)), reverse=True):
            source = str(item.operator.corrupt(source, item.opportunity).source)
    except Exception as exc:
        return None, _rejection_row(row_index, source, target, rule_combo, "operator_corruption_error", stage="corruption", error=exc)

    if source == target:
        return None, _rejection_row(row_index, source, target, rule_combo, "identity_pair", stage="verification")

    try:
        candidates = list(candidate_generator.generate(source))
    except Exception as exc:
        return None, _rejection_row(row_index, source, target, rule_combo, "candidate_generation_error", stage="candidate", error=exc)

    edits = gold_edits_for_pair(source, target, candidates=candidates)
    if len(edits) <= 1:
        return None, _rejection_row(
            row_index,
            source,
            target,
            rule_combo,
            "not_multi_error",
            stage="verification",
            gold_edit_count=len(edits),
        )
    if len(edits) > max_errors:
        return None, _rejection_row(
            row_index,
            source,
            target,
            rule_combo,
            "too_many_gold_edits",
            stage="verification",
            gold_edit_count=len(edits),
        )
    if len(edits) != len(selected):
        return None, _rejection_row(
            row_index,
            source,
            target,
            rule_combo,
            "edit_count_mismatch",
            stage="verification",
            gold_edit_count=len(edits),
        )

    matched_candidates = _matched_candidates_for_edits(candidates, edits)
    if len(matched_candidates) != len(edits):
        return None, _rejection_row(
            row_index,
            source,
            target,
            rule_combo,
            "candidate_missing",
            stage="candidate",
            gold_edit_count=len(edits),
            candidate_rule_ids=_candidate_rule_ids(candidates),
        )

    validation = StrictValidator().validate(source, target, trusted_edits=matched_candidates)
    accepted_edits = validation.accepted_edits
    strict_passed = validation.apply_accepted() == target and all(
        any(candidate_matches_edit(candidate, accepted) for accepted in accepted_edits) for candidate in matched_candidates
    )
    if not strict_passed:
        return None, _rejection_row(
            row_index,
            source,
            target,
            rule_combo,
            "strict_validator_rejected",
            stage="strict_validator",
            gold_edit_count=len(edits),
            candidate_rule_ids=_candidate_rule_ids(candidates),
        )

    candidate_rule_ids = _candidate_rule_ids(matched_candidates)
    edit_dicts = [_edit_to_dict(edit) for edit in edits]
    error_types = sorted({coarse_error_type(edit.edit_type) for edit in edits if coarse_error_type(edit.edit_type) != "unknown"}) or ["mixed"]
    metadata = _json_dict(clean_row.get("metadata"))
    metadata.update(
        {
            "dataset_contract": DATASET_CONTRACT,
            "dataset_layer": LAYER_STRESS_MULTI_ERROR,
            "source_type": SYNTHETIC_OPEN_CLEAN,
            "operator_based_generation": True,
            "candidate_present": True,
            "candidate_rule_ids": candidate_rule_ids,
            "strict_validator_passed": True,
            "generation_strategy": "multi_error_stress",
            "error_bearing_sentence_source": "corpus",
            "is_atomic": False,
            "is_stress": True,
            "count_toward_rule_quota": False,
            "loss_weight": float(loss_weight),
            "gold_edit_count": len(edits),
            "stress_rule_ids": rule_ids,
            "rule_ids": rule_ids,
            "selected_opportunities": [_opportunity_to_dict(item) for item in selected],
            "matched_candidates": [_candidate_to_dict(candidate) for candidate in matched_candidates],
            "original_clean_sentence": target,
        }
    )
    source_corpus = str(clean_row.get("source_name") or clean_row.get("source_corpus") or clean_row.get("source_dataset") or "")
    source_subcorpus = str(clean_row.get("source_subcorpus") or clean_row.get("source_subdataset") or "")
    pair_hash = normalized_pair_hash(source, target)
    return (
        {
            "source": source,
            "target": target,
            "split": "train",
            "source_type": SYNTHETIC_OPEN_CLEAN,
            "error_type": error_types[0] if len(error_types) == 1 else "mixed",
            "rule_ids": json.dumps(rule_ids, ensure_ascii=False),
            "edits": json.dumps(edit_dicts, ensure_ascii=False),
            "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            "original_clean_source": target,
            "source_corpus": source_corpus,
            "source_subcorpus": source_subcorpus,
            "is_hard_negative": False,
            "is_real_pair": False,
            "template_id": "",
            "normalized_pair_hash": pair_hash,
            "error_types": json.dumps(error_types, ensure_ascii=False),
            "source_dataset": source_corpus or SYNTHETIC_OPEN_CLEAN,
            "is_clean": False,
            "is_synthetic": True,
            "domain": str(clean_row.get("domain") or "open_clean"),
            "rule_id": rule_ids[0] if rule_ids else UNKNOWN_RULE_ID,
            "edit_operations": json.dumps(edit_dicts, ensure_ascii=False),
            "dataset_contract": DATASET_CONTRACT,
            "dataset_layer": LAYER_STRESS_MULTI_ERROR,
            "is_atomic": False,
            "is_stress": True,
            "count_toward_rule_quota": False,
            "loss_weight": float(loss_weight),
            "gold_edit_count": len(edits),
            "target_rule_id": rule_ids[0] if rule_ids else UNKNOWN_RULE_ID,
            "candidate_source": "",
            "candidate_replacement": "",
            "candidate_start": -1,
            "candidate_end": -1,
            "verification_status": "accepted",
            "rejection_reason": "",
        },
        None,
    )


def _matched_candidates_for_edits(candidates: list[Candidate], edits: list[Any]) -> list[Candidate]:
    matched: list[Candidate] = []
    used_indexes: set[int] = set()
    for edit in edits:
        match_index = -1
        for index, candidate in enumerate(candidates):
            if index in used_indexes:
                continue
            if candidate_matches_edit(candidate, edit):
                match_index = index
                break
        if match_index < 0:
            return matched
        used_indexes.add(match_index)
        matched.append(candidates[match_index])
    return matched


def _normalized_rule_ids(rule_ids: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for raw_rule_id in rule_ids:
        rule_id = normalize_rule_id(raw_rule_id)
        if rule_id == UNKNOWN_RULE_ID or rule_id in result:
            continue
        result.append(rule_id)
    return result


def _clean_target_from_row(row: Mapping[str, Any]) -> str:
    for column in ("text", "target", "source"):
        value = row.get(column)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _valid_opportunity_span(target: str, opportunity: Any) -> bool:
    start = _opportunity_start(opportunity)
    end = _opportunity_end(opportunity)
    return 0 <= start <= end <= len(target)


def _non_overlapping(combo: tuple[_RuleOpportunity, ...]) -> bool:
    spans = sorted((_opportunity_start(item.opportunity), _opportunity_end(item.opportunity)) for item in combo)
    for (left_start, left_end), (right_start, right_end) in zip(spans, spans[1:], strict=False):
        if left_start == left_end and right_start == right_end:
            if left_start == right_start:
                return False
            continue
        if left_end > right_start:
            return False
    return True


def _opportunity_start(opportunity: Any) -> int:
    return int(getattr(opportunity, "start", -1))


def _opportunity_end(opportunity: Any) -> int:
    return int(getattr(opportunity, "end", -1))


def _combo_key(rule_ids: Iterable[Any]) -> str:
    return "+".join(sorted({str(rule_id) for rule_id in rule_ids if str(rule_id)}))


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


def _edit_to_dict(edit: Any) -> dict[str, Any]:
    if is_dataclass(edit):
        return asdict(edit)
    return {
        "source": str(getattr(edit, "source", "")),
        "replacement": str(getattr(edit, "replacement", "")),
        "edit_type": str(getattr(edit, "edit_type", "")),
        "start": int(getattr(edit, "start", -1)),
        "end": int(getattr(edit, "end", -1)),
        "status": str(getattr(edit, "status", "")),
        "reason": str(getattr(edit, "reason", "")),
        "confidence": float(getattr(edit, "confidence", 0.0)),
        "rule_id": str(getattr(edit, "rule_id", "")),
    }


def _candidate_to_dict(candidate: Candidate) -> dict[str, Any]:
    return {
        "source": candidate.source,
        "replacement": candidate.replacement,
        "edit_type": candidate.edit_type,
        "start": candidate.start,
        "end": candidate.end,
        "confidence": candidate.confidence,
        "requires_model": candidate.requires_model,
        "rule_id": candidate.rule_id,
        "mode": candidate.mode,
    }


def _opportunity_to_dict(item: _RuleOpportunity) -> dict[str, Any]:
    opportunity = item.opportunity
    payload = asdict(opportunity) if is_dataclass(opportunity) else dict(getattr(opportunity, "__dict__", {}))
    payload["rule_id"] = item.rule_id
    return payload


def _candidate_rule_ids(candidates: list[Candidate]) -> list[str]:
    return sorted(
        {
            normalize_rule_id(getattr(candidate, "rule_id", ""))
            for candidate in candidates
            if normalize_rule_id(getattr(candidate, "rule_id", "")) != UNKNOWN_RULE_ID
        }
    )


def _rejection_row(
    row_index: int,
    source: str,
    target: str,
    rule_id: str,
    reason: str,
    *,
    stage: str,
    gold_edit_count: int = 0,
    candidate_rule_ids: list[str] | None = None,
    error: Exception | None = None,
) -> dict[str, Any]:
    row = {
        "row_index": int(row_index),
        "rule_id": rule_id,
        "reason": reason,
        "stage": stage,
        "source": source[:300],
        "target": target[:300],
        "gold_edit_count": int(gold_edit_count),
        "candidate_rule_ids": json.dumps(candidate_rule_ids or [], ensure_ascii=False),
        "error": "",
    }
    if error is not None:
        row["error"] = f"{type(error).__name__}: {error}"
    return row
