from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from src.candidates.candidate_generator import Candidate, CandidateGenerator
from src.candidates.candidate_ranking import rank_candidates_for_budget
from src.candidates.matching import candidate_matches_edit
from src.data.dataset_quality import positive_target_quality_pass
from src.rules.rule_ids import UNKNOWN_RULE_ID, normalize_rule_id
from src.validation.diff_analyzer import DiffAnalyzer, Edit
from src.validation.edit_classifier import is_allowed_edit_type
from src.validation.strict_validator import StrictValidator


@dataclass(frozen=True)
class AtomicVerificationResult:
    passed: bool
    reason: str
    source: str
    target: str
    rule_id: str
    gold_edit_count: int
    edits: list[dict[str, Any]]
    candidate_present: bool
    candidate_rule_ids: list[str]
    matched_candidate: dict[str, Any] | None
    strict_validator_passed: bool
    target_quality_passed: bool
    extra_edit_count: int


def gold_edits_for_pair(source: str, target: str, candidates: list[Candidate] | None = None) -> list[Edit]:
    return DiffAnalyzer().analyze(source, target, candidates=candidates)


def candidate_applies_to_target(source: str, target: str, candidate: Candidate) -> bool:
    start = int(getattr(candidate, "start", -1))
    end = int(getattr(candidate, "end", -1))
    if start < 0 or end < start or end > len(source):
        return False
    candidate_source = str(getattr(candidate, "source", ""))
    if candidate_source and source[start:end].lower() != candidate_source.lower():
        return False
    if not candidate_source and start != end:
        return False
    replacement = str(getattr(candidate, "replacement", ""))
    return source[:start] + replacement + source[end:] == target


def find_matching_candidate(
    source: str,
    target: str,
    rule_id: str,
    candidate_generator: CandidateGenerator | None = None,
    max_candidates: int | None = None,
) -> Candidate | None:
    candidates = _generated_candidates(source, candidate_generator, max_candidates=max_candidates)
    edits = gold_edits_for_pair(source, target, candidates=candidates)
    return _matching_candidate_from_candidates(source, target, normalize_rule_id(rule_id), candidates, edits)


def verify_atomic_positive(
    source: str,
    target: str,
    rule_id: str,
    candidate_generator: CandidateGenerator | None = None,
    require_strict_validator: bool = True,
) -> AtomicVerificationResult:
    candidates = _generated_candidates(source, candidate_generator)
    edits = gold_edits_for_pair(source, target, candidates=candidates)
    return _verify_atomic_positive_with_candidates(
        source,
        target,
        rule_id,
        candidates,
        edits,
        require_strict_validator=require_strict_validator,
    )


def verify_candidate_covered_pair(
    source: str,
    target: str,
    rule_ids: Iterable[str] | None = None,
    candidate_generator: CandidateGenerator | None = None,
) -> AtomicVerificationResult:
    candidates = _generated_candidates(source, candidate_generator)
    edits = gold_edits_for_pair(source, target, candidates=candidates)
    normalized_rule_ids = _candidate_covered_rule_ids(source, target, rule_ids, candidates, edits)
    if not normalized_rule_ids:
        return _verify_atomic_positive_with_candidates(source, target, UNKNOWN_RULE_ID, candidates, edits)

    first_result: AtomicVerificationResult | None = None
    for candidate_rule_id in normalized_rule_ids:
        result = _verify_atomic_positive_with_candidates(source, target, candidate_rule_id, candidates, edits)
        if result.passed:
            return result
        if first_result is None:
            first_result = result
    return first_result or _verify_atomic_positive_with_candidates(source, target, UNKNOWN_RULE_ID, candidates, edits)


def verification_to_metadata(result: AtomicVerificationResult) -> dict[str, Any]:
    return {
        "atomic_verification": {
            "passed": result.passed,
            "reason": result.reason,
            "rule_id": result.rule_id,
            "gold_edit_count": result.gold_edit_count,
            "edits": result.edits,
            "candidate_present": result.candidate_present,
            "candidate_rule_ids": result.candidate_rule_ids,
            "matched_candidate": result.matched_candidate,
            "strict_validator_passed": result.strict_validator_passed,
            "target_quality_passed": result.target_quality_passed,
            "extra_edit_count": result.extra_edit_count,
        }
    }


def _verify_atomic_positive_with_candidates(
    source: str,
    target: str,
    rule_id: str,
    candidates: list[Candidate],
    edits: list[Edit],
    *,
    require_strict_validator: bool = True,
) -> AtomicVerificationResult:
    normalized_rule_id = normalize_rule_id(rule_id)
    candidate_rule_ids = _candidate_rule_ids(candidates)
    target_quality_passed = positive_target_quality_pass(target)

    if source == target:
        return _result(
            False,
            "identity_pair",
            source,
            target,
            normalized_rule_id,
            edits,
            candidate_rule_ids=candidate_rule_ids,
            target_quality_passed=target_quality_passed,
        )
    if normalized_rule_id == UNKNOWN_RULE_ID:
        return _result(
            False,
            "unknown_rule",
            source,
            target,
            normalized_rule_id,
            edits,
            candidate_rule_ids=candidate_rule_ids,
            target_quality_passed=target_quality_passed,
        )
    if len(edits) != 1:
        return _result(
            False,
            "non_atomic_edit_count",
            source,
            target,
            normalized_rule_id,
            edits,
            candidate_rule_ids=candidate_rule_ids,
            target_quality_passed=target_quality_passed,
        )

    edit = edits[0]
    if not is_allowed_edit_type(edit.edit_type):
        return _result(
            False,
            "unsupported_edit_type",
            source,
            target,
            normalized_rule_id,
            edits,
            candidate_rule_ids=candidate_rule_ids,
            target_quality_passed=target_quality_passed,
        )

    matched_candidate = _matching_candidate_from_candidates(source, target, normalized_rule_id, candidates, edits)
    if matched_candidate is None:
        return _result(
            False,
            "candidate_missing",
            source,
            target,
            normalized_rule_id,
            edits,
            candidate_rule_ids=candidate_rule_ids,
            target_quality_passed=target_quality_passed,
        )

    strict_validator_passed = True
    extra_edit_count = 0
    if require_strict_validator:
        validation = StrictValidator().validate(source, target, trusted_edits=[matched_candidate])
        accepted_edits = validation.accepted_edits
        matched_accepted_count = sum(1 for accepted in accepted_edits if candidate_matches_edit(matched_candidate, accepted))
        extra_edit_count = sum(1 for accepted in accepted_edits if not candidate_matches_edit(matched_candidate, accepted))
        strict_validator_passed = validation.apply_accepted() == target and matched_accepted_count == 1 and extra_edit_count == 0
        if not strict_validator_passed:
            return _result(
                False,
                "strict_validator_rejected",
                source,
                target,
                normalized_rule_id,
                edits,
                candidate_rule_ids=candidate_rule_ids,
                matched_candidate=matched_candidate,
                strict_validator_passed=False,
                target_quality_passed=target_quality_passed,
                extra_edit_count=extra_edit_count,
            )

    if not target_quality_passed:
        return _result(
            False,
            "target_quality_failed",
            source,
            target,
            normalized_rule_id,
            edits,
            candidate_rule_ids=candidate_rule_ids,
            matched_candidate=matched_candidate,
            strict_validator_passed=strict_validator_passed,
            target_quality_passed=False,
            extra_edit_count=extra_edit_count,
        )

    return _result(
        True,
        "ok",
        source,
        target,
        normalized_rule_id,
        edits,
        candidate_rule_ids=candidate_rule_ids,
        matched_candidate=matched_candidate,
        strict_validator_passed=strict_validator_passed,
        target_quality_passed=True,
        extra_edit_count=extra_edit_count,
    )


def _generated_candidates(
    source: str,
    candidate_generator: CandidateGenerator | None,
    *,
    max_candidates: int | None = None,
) -> list[Candidate]:
    generator = candidate_generator or CandidateGenerator()
    candidates = list(generator.generate(source))
    if max_candidates is not None:
        return rank_candidates_for_budget(candidates, int(max_candidates))
    return candidates


def _matching_candidate_from_candidates(
    source: str,
    target: str,
    rule_id: str,
    candidates: list[Candidate],
    edits: list[Edit],
) -> Candidate | None:
    normalized_rule_id = normalize_rule_id(rule_id)
    for candidate in candidates:
        if normalize_rule_id(getattr(candidate, "rule_id", "")) != normalized_rule_id:
            continue
        if not candidate_applies_to_target(source, target, candidate):
            continue
        if any(candidate_matches_edit(candidate, edit) for edit in edits):
            return candidate
    return None


def _candidate_covered_rule_ids(
    source: str,
    target: str,
    rule_ids: Iterable[str] | None,
    candidates: list[Candidate],
    edits: list[Edit],
) -> list[str]:
    explicit = _normalized_rule_ids(rule_ids or ())
    if explicit:
        return explicit

    result: list[str] = []
    for edit in edits:
        rule_id = normalize_rule_id(edit.rule_id)
        if rule_id != UNKNOWN_RULE_ID and rule_id not in result:
            result.append(rule_id)
    for candidate in candidates:
        rule_id = normalize_rule_id(candidate.rule_id)
        if rule_id == UNKNOWN_RULE_ID or rule_id in result:
            continue
        if candidate_applies_to_target(source, target, candidate) and any(candidate_matches_edit(candidate, edit) for edit in edits):
            result.append(rule_id)
    return result


def _normalized_rule_ids(rule_ids: Iterable[str]) -> list[str]:
    result: list[str] = []
    for rule_id in rule_ids:
        normalized = normalize_rule_id(rule_id)
        if normalized == UNKNOWN_RULE_ID or normalized in result:
            continue
        result.append(normalized)
    return result


def _candidate_rule_ids(candidates: list[Candidate]) -> list[str]:
    return sorted(
        {
            normalize_rule_id(candidate.rule_id)
            for candidate in candidates
            if normalize_rule_id(candidate.rule_id) != UNKNOWN_RULE_ID
        }
    )


def _result(
    passed: bool,
    reason: str,
    source: str,
    target: str,
    rule_id: str,
    edits: list[Edit],
    *,
    candidate_rule_ids: list[str],
    matched_candidate: Candidate | None = None,
    strict_validator_passed: bool = False,
    target_quality_passed: bool = False,
    extra_edit_count: int = 0,
) -> AtomicVerificationResult:
    return AtomicVerificationResult(
        passed=passed,
        reason=reason,
        source=source,
        target=target,
        rule_id=rule_id,
        gold_edit_count=len(edits),
        edits=[_edit_to_dict(edit) for edit in edits],
        candidate_present=matched_candidate is not None,
        candidate_rule_ids=list(candidate_rule_ids),
        matched_candidate=_candidate_to_dict(matched_candidate) if matched_candidate is not None else None,
        strict_validator_passed=strict_validator_passed,
        target_quality_passed=target_quality_passed,
        extra_edit_count=extra_edit_count,
    )


def _edit_to_dict(edit: Edit) -> dict[str, Any]:
    return {
        "source": edit.source,
        "replacement": edit.replacement,
        "edit_type": edit.edit_type,
        "start": edit.start,
        "end": edit.end,
        "status": edit.status,
        "reason": edit.reason,
        "confidence": edit.confidence,
        "rule_id": edit.rule_id,
    }


def _candidate_to_dict(candidate: Candidate) -> dict[str, Any]:
    result: dict[str, Any] = {
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
    optional_fields = {
        "action": candidate.action,
        "label": candidate.label,
        "gap_index": candidate.gap_index,
        "requires": list(candidate.requires),
        "group": candidate.group,
        "edit_domain": candidate.edit_domain,
        "syntax_family": candidate.syntax_family,
        "subtype": candidate.subtype,
        "trigger_text": candidate.trigger_text,
        "trigger_lemma": candidate.trigger_lemma,
        "trigger_pos": candidate.trigger_pos,
        "gap_kind": candidate.gap_kind,
        "evidence": candidate.evidence,
        "confidence_source": candidate.confidence_source,
        "implementation_group": candidate.implementation_group,
        "constraint_group": candidate.constraint_group,
        "bundle_id": candidate.bundle_id,
        "metadata": candidate.metadata,
    }
    for key, value in optional_fields.items():
        if not _is_empty_metadata_value(value):
            result[key] = value
    return result


def _is_empty_metadata_value(value: Any) -> bool:
    return value is None or value == "" or value == () or value == [] or value == {}
