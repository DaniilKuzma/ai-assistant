from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import pandas as pd

from src.candidates.candidate_generator import Candidate
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
from src.validation.strict_validator import TRAINING_CONTEXT_SPLIT_JOIN_BY_RULE


MINER_NAME = "corpus_backed_split_join"
SERVICE_HARD_NEGATIVE_RULE_ID = "clean_identity_hard_negative"
HARD_NEGATIVE_ERROR_TYPE = "hard_negative"
SUPPORTED_CONTEXT_RULES = tuple(TRAINING_CONTEXT_SPLIT_JOIN_BY_RULE)


@dataclass(frozen=True)
class RuleDataSourceStats:
    rule_id: str
    corpus_mined_positive_count: int = 0
    syntax_mined_positive_count: int = 0
    morphology_mined_positive_count: int = 0
    real_pattern_replay_positive_count: int = 0
    rule_lab_positive_count: int = 0
    corpus_mined_hard_negative_count: int = 0
    rule_lab_hard_negative_count: int = 0
    total_atomic_positive_count: int = 0
    total_hard_negative_count: int = 0


@dataclass(frozen=True)
class RuleStructuralDiversityStats:
    rule_id: str
    unique_source_count: int = 0
    unique_target_count: int = 0
    unique_left_context_count: int = 0
    unique_right_context_count: int = 0
    unique_sentence_pattern_count: int = 0
    unique_pos_context_count: int = 0
    unique_dependency_pattern_count: int = 0
    dominant_template_share: float = 0.0
    dominant_source_type_share: float = 0.0
    rule_lab_share: float = 0.0
    near_duplicate_count: int = 0
    status: str = "empty"
    reason: str = ""


@dataclass(frozen=True)
class RuleDataCompilerResult:
    atomic_positive_rows: list[dict[str, Any]] = field(default_factory=list)
    hard_negative_rows: list[dict[str, Any]] = field(default_factory=list)
    rejection_rows: list[dict[str, Any]] = field(default_factory=list)
    source_stats_rows: list[RuleDataSourceStats] = field(default_factory=list)
    diversity_stats_rows: list[RuleStructuralDiversityStats] = field(default_factory=list)
    underfilled_rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class CandidateTargetContext:
    rule_id: str
    target: str
    source_fragment: str
    target_fragment: str
    source_name: str
    source_row_id: str
    miner_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HardNegativeContext:
    rule_id: str
    text: str
    source_name: str
    source_row_id: str
    miner_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


class RuleExampleMiner:
    rule_id: str
    source_name: str

    def mine_positive_targets(self, clean_rows: list[dict[str, Any]], config: dict[str, Any]) -> list[CandidateTargetContext]:
        raise NotImplementedError

    def corrupt_target(self, context: CandidateTargetContext) -> tuple[str, str]:
        raise NotImplementedError

    def mine_hard_negatives(self, clean_rows: list[dict[str, Any]], config: dict[str, Any]) -> list[HardNegativeContext]:
        raise NotImplementedError


@dataclass(frozen=True)
class CorpusBackedSplitJoinMiner(RuleExampleMiner):
    rule_id: str
    source_fragment: str
    target_fragment: str
    source_name: str = "corpus_mined"

    def mine_positive_targets(self, clean_rows: list[dict[str, Any]], config: dict[str, Any]) -> list[CandidateTargetContext]:
        limit = _per_rule_scan_limit(config)
        contexts: list[CandidateTargetContext] = []
        for row in clean_rows[:limit]:
            target = _clean_text_from_row(row)
            if not target:
                continue
            for match in _phrase_matches(target, self.target_fragment):
                contexts.append(
                    CandidateTargetContext(
                        rule_id=self.rule_id,
                        target=target,
                        source_fragment=_match_case(target[match.start() : match.end()], self.source_fragment),
                        target_fragment=target[match.start() : match.end()],
                        source_name=str(row.get("source_name") or row.get("source_corpus") or "open_clean"),
                        source_row_id=_row_id(row),
                        miner_name=MINER_NAME,
                        metadata=_source_metadata(row, match.start(), match.end()),
                    )
                )
        return contexts

    def corrupt_target(self, context: CandidateTargetContext) -> tuple[str, str]:
        start = int(context.metadata.get("target_start", -1))
        end = int(context.metadata.get("target_end", -1))
        target = context.target
        source = target[:start] + context.source_fragment + target[end:]
        return source, target

    def mine_hard_negatives(self, clean_rows: list[dict[str, Any]], config: dict[str, Any]) -> list[HardNegativeContext]:
        limit = _per_rule_scan_limit(config)
        contexts: list[HardNegativeContext] = []
        for row in clean_rows[:limit]:
            text = _clean_text_from_row(row)
            if not text:
                continue
            if self.source_fragment.lower() not in text.lower():
                continue
            contexts.append(
                HardNegativeContext(
                    rule_id=self.rule_id,
                    text=text,
                    source_name=str(row.get("source_name") or row.get("source_corpus") or "open_clean"),
                    source_row_id=_row_id(row),
                    miner_name=MINER_NAME,
                    metadata=_source_metadata(row, -1, -1),
                )
            )
        return contexts


def compile_rule_data(
    clean_rows: Iterable[Mapping[str, Any]] | pd.DataFrame,
    active_or_candidate_rule_ids: Iterable[str],
    candidate_generator: Any,
    config: dict[str, Any],
    existing_rows: Iterable[Mapping[str, Any]] | pd.DataFrame | None = None,
    target_counts: Mapping[str, Any] | None = None,
) -> RuleDataCompilerResult:
    rows = _records(clean_rows)
    existing = _records(existing_rows if existing_rows is not None else [])
    rule_ids = _supported_rule_ids(active_or_candidate_rule_ids)
    miners = [_miner_for_rule(rule_id) for rule_id in rule_ids]
    positive_target = _positive_target(config, target_counts)
    hard_target = _hard_negative_target(config, target_counts)

    accepted_positive: list[dict[str, Any]] = []
    accepted_hard: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    seen_positive = _existing_positive_keys(existing)
    seen_hard = _existing_hard_negative_keys(existing)
    positive_counts: Counter[str] = Counter()
    hard_counts: Counter[str] = Counter()

    for miner in miners:
        for context in miner.mine_positive_targets(rows, config):
            if positive_counts[context.rule_id] >= positive_target:
                break
            source, target = miner.corrupt_target(context)
            verification = verify_atomic_positive(source, target, context.rule_id, candidate_generator=candidate_generator)
            reason = _positive_rejection_reason(verification, context)
            if reason:
                rejections.append(_rejection_row(context.rule_id, source, target, reason, "positive_verification", context))
                continue
            key = _positive_key(source, target, context.rule_id)
            if key in seen_positive:
                rejections.append(_rejection_row(context.rule_id, source, target, "duplicate_pair", "dedupe", context))
                continue
            seen_positive.add(key)
            accepted_positive.append(_atomic_positive_row(source, target, context, verification))
            positive_counts[context.rule_id] += 1

        for context in miner.mine_hard_negatives(rows, config):
            if hard_counts[context.rule_id] >= hard_target:
                break
            text = context.text
            quality_reasons = _hard_negative_quality_failure_reasons(text)
            if quality_reasons:
                for reason in quality_reasons:
                    rejections.append(_rejection_row(context.rule_id, text, text, f"hard_negative_quality_failed:{reason}", "hard_negative_quality", context))
                continue
            candidate = _matching_hard_negative_candidate(text, context.rule_id, candidate_generator)
            if candidate is None:
                rejections.append(_rejection_row(context.rule_id, text, text, "candidate_missing", "hard_negative_candidate", context))
                continue
            key = _hard_negative_key(text, context.rule_id)
            if key in seen_hard:
                rejections.append(_rejection_row(context.rule_id, text, text, "duplicate_hard_negative", "dedupe", context))
                continue
            seen_hard.add(key)
            accepted_hard.append(_hard_negative_row(text, context, candidate))
            hard_counts[context.rule_id] += 1

    source_stats = _source_stats(rule_ids, accepted_positive, accepted_hard)
    diversity_stats = _diversity_stats(rule_ids, accepted_positive, config)
    underfilled = _underfilled_rows(rule_ids, positive_counts, hard_counts, positive_target, hard_target)
    return RuleDataCompilerResult(
        atomic_positive_rows=accepted_positive,
        hard_negative_rows=accepted_hard,
        rejection_rows=rejections,
        source_stats_rows=source_stats,
        diversity_stats_rows=diversity_stats,
        underfilled_rows=underfilled,
    )


def write_rule_data_compiler_reports(result: RuleDataCompilerResult, reports_dir: str | Path) -> None:
    output_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _frame_with_columns(
        [asdict(row) for row in result.source_stats_rows],
        [field.name for field in fields(RuleDataSourceStats)],
    ).to_csv(
        output_dir / "rule_data_source_report.csv",
        index=False,
    )
    _frame_with_columns(
        [asdict(row) for row in result.diversity_stats_rows],
        [field.name for field in fields(RuleStructuralDiversityStats)],
    ).to_csv(
        output_dir / "rule_structural_diversity_report.csv",
        index=False,
    )
    _frame_with_columns(
        result.rejection_rows,
        [
            "rule_id",
            "source",
            "target",
            "reason",
            "stage",
            "source_name",
            "source_row_id",
            "miner_name",
        ],
    ).to_csv(
        output_dir / "rule_miner_rejection_report.csv",
        index=False,
    )
    _frame_with_columns(
        result.underfilled_rows,
        [
            "rule_id",
            "atomic_positive_count",
            "atomic_positive_target",
            "hard_negative_count",
            "hard_negative_target",
            "status",
            "reason",
        ],
    ).to_csv(
        output_dir / "rule_underfilled_backlog.csv",
        index=False,
    )


def _frame_with_columns(rows: Iterable[Mapping[str, Any]], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(list(rows)).reindex(columns=columns)


def _miner_for_rule(rule_id: str) -> CorpusBackedSplitJoinMiner:
    source_fragment, target_fragment = TRAINING_CONTEXT_SPLIT_JOIN_BY_RULE[rule_id]
    return CorpusBackedSplitJoinMiner(rule_id=rule_id, source_fragment=source_fragment, target_fragment=target_fragment)


def _supported_rule_ids(rule_ids: Iterable[str]) -> list[str]:
    result: list[str] = []
    for rule_id in rule_ids:
        normalized = normalize_rule_id(rule_id)
        if normalized in TRAINING_CONTEXT_SPLIT_JOIN_BY_RULE and normalized not in result:
            result.append(normalized)
    return result


def _positive_rejection_reason(verification: AtomicVerificationResult, context: CandidateTargetContext) -> str:
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
    matched = dict(verification.matched_candidate or {})
    if normalize_rule_id(str(matched.get("rule_id") or "")) != context.rule_id:
        return "candidate_rule_mismatch"
    if str(matched.get("source") or "").lower() != context.source_fragment.lower():
        return "candidate_fragment_mismatch"
    if str(matched.get("replacement") or "").lower() != context.target_fragment.lower():
        return "candidate_fragment_mismatch"
    return ""


def _atomic_positive_row(
    source: str,
    target: str,
    context: CandidateTargetContext,
    verification: AtomicVerificationResult,
) -> dict[str, Any]:
    matched = dict(verification.matched_candidate or {})
    edits = [_edit_payload(edit, context.rule_id) for edit in verification.edits]
    pair_hash = normalized_pair_hash(source, target)
    metadata = {
        "dataset_contract": DATASET_CONTRACT,
        "dataset_layer": LAYER_ATOMIC_POSITIVE,
        "source_type": SYNTHETIC_OPEN_CLEAN,
        "rule_ids": [context.rule_id],
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
        "target_family": context.rule_id,
        "activation_source": "corpus_mined",
        "generation_sources": ["corpus_mined"],
        "generation_strategy": "corpus_backed_rule_data_compiler",
        "error_bearing_sentence_source": "corpus",
        "original_clean_sentence": target,
        "source_name": context.source_name,
        "source_row_id": context.source_row_id,
        "source_subcorpus": str(context.metadata.get("source_subcorpus", "")),
        "source_doc_id": str(context.metadata.get("source_doc_id", "")),
        "sentence_id": str(context.metadata.get("sentence_id", "")),
        "clean_hash": str(context.metadata.get("hash", "")),
        "miner_name": context.miner_name,
        "source_fragment": context.source_fragment,
        "target_fragment": context.target_fragment,
        "normalized_pair_hash": pair_hash,
    }
    error_type = "split_join"
    return {
        "source": source,
        "target": target,
        "split": "train",
        "source_type": SYNTHETIC_OPEN_CLEAN,
        "error_type": error_type,
        "rule_ids": json.dumps([context.rule_id], ensure_ascii=False),
        "edits": json.dumps(edits, ensure_ascii=False),
        "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        "original_clean_source": target,
        "source_corpus": context.source_name,
        "source_subcorpus": str(context.metadata.get("source_subcorpus", "")),
        "is_hard_negative": False,
        "is_real_pair": False,
        "template_id": "",
        "normalized_pair_hash": pair_hash,
        "error_types": json.dumps([error_type], ensure_ascii=False),
        "source_dataset": context.source_name,
        "is_clean": False,
        "is_synthetic": True,
        "domain": str(context.metadata.get("domain") or "open_clean"),
        "rule_id": context.rule_id,
        "edit_operations": json.dumps(edits, ensure_ascii=False),
        "dataset_contract": DATASET_CONTRACT,
        "dataset_layer": LAYER_ATOMIC_POSITIVE,
        "is_atomic": True,
        "is_stress": False,
        "count_toward_rule_quota": True,
        "gold_edit_count": 1,
        "loss_weight": 1.0,
        "target_rule_id": context.rule_id,
        "candidate_source": str(matched.get("source") or ""),
        "candidate_replacement": str(matched.get("replacement") or ""),
        "candidate_start": int(matched.get("start", -1) if matched.get("start", -1) != "" else -1),
        "candidate_end": int(matched.get("end", -1) if matched.get("end", -1) != "" else -1),
        "verification_status": "passed",
        "rejection_reason": "",
        "activation_source": "corpus_mined",
    }


def _hard_negative_row(text: str, context: HardNegativeContext, candidate: Candidate) -> dict[str, Any]:
    candidate_source = str(getattr(candidate, "source", ""))
    candidate_replacement = str(getattr(candidate, "replacement", ""))
    candidate_start = int(getattr(candidate, "start", -1))
    candidate_end = int(getattr(candidate, "end", -1))
    pair_hash = normalized_pair_hash(text, text)
    metadata = {
        "dataset_contract": DATASET_CONTRACT,
        "dataset_layer": LAYER_ATOMIC_HARD_NEGATIVE,
        "source_type": HARD_NEGATIVE_OPEN,
        "rule_ids": [SERVICE_HARD_NEGATIVE_RULE_ID],
        "is_atomic": True,
        "is_hard_negative": True,
        "target_rule_id": context.rule_id,
        "candidate_rule_id": context.rule_id,
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
        "activation_source": "corpus_mined",
        "generation_sources": ["corpus_mined"],
        "generation_strategy": "corpus_backed_rule_data_compiler",
        "hard_negative_source": "corpus_mined",
        "expected_accepted_edits": 0,
        "clean_or_hard_quality_reasons": [],
        "original_clean_sentence": text,
        "source_name": context.source_name,
        "source_row_id": context.source_row_id,
        "miner_name": context.miner_name,
    }
    return {
        "source": text,
        "target": text,
        "split": "train",
        "source_type": HARD_NEGATIVE_OPEN,
        "error_type": HARD_NEGATIVE_ERROR_TYPE,
        "rule_ids": json.dumps([SERVICE_HARD_NEGATIVE_RULE_ID], ensure_ascii=False),
        "edits": "[]",
        "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        "original_clean_source": text,
        "source_corpus": context.source_name,
        "source_subcorpus": str(context.metadata.get("source_subcorpus", "")),
        "is_hard_negative": True,
        "is_real_pair": False,
        "template_id": "",
        "normalized_pair_hash": pair_hash,
        "error_types": json.dumps([HARD_NEGATIVE_ERROR_TYPE], ensure_ascii=False),
        "source_dataset": context.source_name,
        "is_clean": True,
        "is_synthetic": False,
        "domain": str(context.metadata.get("domain") or "open_clean"),
        "rule_id": SERVICE_HARD_NEGATIVE_RULE_ID,
        "edit_operations": "[]",
        "dataset_contract": DATASET_CONTRACT,
        "dataset_layer": LAYER_ATOMIC_HARD_NEGATIVE,
        "is_atomic": True,
        "is_stress": False,
        "count_toward_rule_quota": False,
        "loss_weight": 1.0,
        "gold_edit_count": 0,
        "target_rule_id": context.rule_id,
        "candidate_source": candidate_source,
        "candidate_replacement": candidate_replacement,
        "candidate_start": candidate_start,
        "candidate_end": candidate_end,
        "verification_status": "passed",
        "rejection_reason": "",
        "activation_source": "corpus_mined",
    }


def _matching_hard_negative_candidate(text: str, rule_id: str, candidate_generator: Any) -> Candidate | None:
    try:
        candidates = list(candidate_generator.generate(text))
    except Exception:
        return None
    for candidate in candidates:
        if normalize_rule_id(getattr(candidate, "rule_id", "")) != rule_id:
            continue
        if not _candidate_is_non_keep(candidate):
            continue
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


def _source_stats(rule_ids: list[str], positives: list[dict[str, Any]], hard_rows: list[dict[str, Any]]) -> list[RuleDataSourceStats]:
    positive_counts = Counter(str(row.get("rule_id", "")) for row in positives)
    hard_counts = Counter(str(row.get("target_rule_id", "")) for row in hard_rows)
    return [
        RuleDataSourceStats(
            rule_id=rule_id,
            corpus_mined_positive_count=int(positive_counts.get(rule_id, 0)),
            corpus_mined_hard_negative_count=int(hard_counts.get(rule_id, 0)),
            total_atomic_positive_count=int(positive_counts.get(rule_id, 0)),
            total_hard_negative_count=int(hard_counts.get(rule_id, 0)),
        )
        for rule_id in rule_ids
    ]


def _diversity_stats(rule_ids: list[str], positives: list[dict[str, Any]], config: dict[str, Any]) -> list[RuleStructuralDiversityStats]:
    by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in positives:
        by_rule[str(row.get("rule_id", ""))].append(row)
    compiler_config = ((config.get("data", {}) or {}).get("rule_data_compiler", {}) or {})
    min_patterns = int(compiler_config.get("min_unique_sentence_patterns_per_rule", 0) or 0)
    min_left = int(compiler_config.get("min_unique_left_contexts_per_rule", 0) or 0)
    min_right = int(compiler_config.get("min_unique_right_contexts_per_rule", 0) or 0)
    result: list[RuleStructuralDiversityStats] = []
    for rule_id in rule_ids:
        rows = by_rule.get(rule_id, [])
        if not rows:
            result.append(RuleStructuralDiversityStats(rule_id=rule_id, status="empty", reason="no_atomic_positive_rows"))
            continue
        sources = [str(row.get("source", "")) for row in rows]
        targets = [str(row.get("target", "")) for row in rows]
        left_contexts = [_left_context(row) for row in rows]
        right_contexts = [_right_context(row) for row in rows]
        patterns = [_sentence_pattern(row.get("target", "")) for row in rows]
        source_types = [str(row.get("activation_source") or _metadata(row).get("activation_source") or "") for row in rows]
        dominant_source_type_share = _dominant_share(source_types)
        dominant_template_share = _dominant_share([normalize_pair(text) for text in targets])
        near_duplicates = _near_duplicate_count(targets)
        status = "ok"
        reasons: list[str] = []
        if min_patterns and len(set(patterns)) < min_patterns:
            status = "warning"
            reasons.append("low_unique_sentence_patterns")
        if min_left and len(set(left_contexts)) < min_left:
            status = "warning"
            reasons.append("low_unique_left_contexts")
        if min_right and len(set(right_contexts)) < min_right:
            status = "warning"
            reasons.append("low_unique_right_contexts")
        result.append(
            RuleStructuralDiversityStats(
                rule_id=rule_id,
                unique_source_count=len(set(normalize_pair(text) for text in sources)),
                unique_target_count=len(set(normalize_pair(text) for text in targets)),
                unique_left_context_count=len(set(left_contexts)),
                unique_right_context_count=len(set(right_contexts)),
                unique_sentence_pattern_count=len(set(patterns)),
                unique_pos_context_count=0,
                unique_dependency_pattern_count=0,
                dominant_template_share=dominant_template_share,
                dominant_source_type_share=dominant_source_type_share,
                rule_lab_share=0.0,
                near_duplicate_count=near_duplicates,
                status=status,
                reason=",".join(reasons),
            )
        )
    return result


def _underfilled_rows(
    rule_ids: list[str],
    positive_counts: Counter[str],
    hard_counts: Counter[str],
    positive_target: int,
    hard_target: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rule_id in rule_ids:
        positive_count = int(positive_counts.get(rule_id, 0))
        hard_count = int(hard_counts.get(rule_id, 0))
        reasons: list[str] = []
        if positive_count < positive_target:
            reasons.append("atomic_positive_under_target")
        if hard_count < hard_target:
            reasons.append("hard_negative_under_target")
        if reasons:
            rows.append(
                {
                    "rule_id": rule_id,
                    "atomic_positive_count": positive_count,
                    "atomic_positive_target": positive_target,
                    "hard_negative_count": hard_count,
                    "hard_negative_target": hard_target,
                    "status": "underfilled",
                    "reason": ",".join(reasons),
                }
            )
    return rows


def _rejection_row(
    rule_id: str,
    source: str,
    target: str,
    reason: str,
    stage: str,
    context: CandidateTargetContext | HardNegativeContext,
) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "source": source[:500],
        "target": target[:500],
        "reason": reason,
        "stage": stage,
        "source_name": context.source_name,
        "source_row_id": context.source_row_id,
        "miner_name": context.miner_name,
    }


def _positive_key(source: str, target: str, rule_id: str) -> tuple[str, str, str]:
    return (normalize_pair(source), normalize_pair(target), normalize_rule_id(rule_id))


def _hard_negative_key(text: str, rule_id: str) -> tuple[str, str, str]:
    return (normalize_pair(text), normalize_pair(text), normalize_rule_id(rule_id))


def _existing_positive_keys(rows: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    result: set[tuple[str, str, str]] = set()
    for row in rows:
        rule_id = normalize_rule_id(str(row.get("rule_id") or _metadata(row).get("target_rule_id") or ""))
        if rule_id == UNKNOWN_RULE_ID:
            continue
        result.add(_positive_key(str(row.get("source", "")), str(row.get("target", "")), rule_id))
    return result


def _existing_hard_negative_keys(rows: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    result: set[tuple[str, str, str]] = set()
    for row in rows:
        rule_id = normalize_rule_id(str(row.get("target_rule_id") or _metadata(row).get("target_rule_id") or ""))
        if rule_id == UNKNOWN_RULE_ID:
            continue
        result.add(_hard_negative_key(str(row.get("source", "")), rule_id))
    return result


def _records(rows_or_frame: Iterable[Mapping[str, Any]] | pd.DataFrame) -> list[dict[str, Any]]:
    if isinstance(rows_or_frame, pd.DataFrame):
        return rows_or_frame.fillna("").to_dict("records")
    return [dict(row) for row in rows_or_frame]


def _clean_text_from_row(row: Mapping[str, Any]) -> str:
    for key in ("text", "target", "source"):
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _source_metadata(row: Mapping[str, Any], start: int, end: int) -> dict[str, Any]:
    return {
        "target_start": start,
        "target_end": end,
        "source_subcorpus": str(row.get("source_subcorpus") or ""),
        "source_doc_id": str(row.get("source_doc_id") or ""),
        "sentence_id": str(row.get("sentence_id") or ""),
        "hash": str(row.get("hash") or ""),
        "domain": str(row.get("domain") or "open_clean"),
    }


def _row_id(row: Mapping[str, Any]) -> str:
    for key in ("source_row_id", "sentence_id", "hash"):
        value = row.get(key)
        if value:
            return str(value)
    return ""


def _phrase_matches(text: str, phrase: str) -> list[re.Match[str]]:
    pattern = re.compile(r"(?<![А-Яа-яЁё])" + re.escape(phrase) + r"(?![А-Яа-яЁё])", re.IGNORECASE)
    return list(pattern.finditer(text))


def _match_case(sample: str, phrase: str) -> str:
    if sample[:1].isupper():
        return phrase[:1].upper() + phrase[1:]
    return phrase


def _edit_payload(edit: dict[str, Any], rule_id: str) -> dict[str, Any]:
    payload = dict(edit)
    payload["rule_id"] = normalize_rule_id(str(payload.get("rule_id") or rule_id))
    return payload


def _positive_target(config: dict[str, Any], target_counts: Mapping[str, Any] | None) -> int:
    if target_counts and "atomic_positive" in target_counts:
        return max(0, int(target_counts.get("atomic_positive") or 0))
    quota = ((config.get("data", {}) or {}).get("rule_quota", {}) or {})
    return max(1, int(quota.get("preferred_atomic_positives_per_active_rule", 50) or 50))


def _hard_negative_target(config: dict[str, Any], target_counts: Mapping[str, Any] | None) -> int:
    if target_counts and "atomic_hard_negative" in target_counts:
        return max(0, int(target_counts.get("atomic_hard_negative") or 0))
    quota = ((config.get("data", {}) or {}).get("rule_quota", {}) or {})
    return max(1, int(quota.get("min_hard_negatives_per_active_rule", 50) or 50))


def _per_rule_scan_limit(config: dict[str, Any]) -> int:
    compiler_config = ((config.get("data", {}) or {}).get("rule_data_compiler", {}) or {})
    return max(1, int(compiler_config.get("max_scan_rows_per_rule", 50_000) or 50_000))


def _metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("metadata", {})
    if isinstance(raw, Mapping):
        return dict(raw)
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _left_context(row: Mapping[str, Any]) -> str:
    source = str(row.get("source", ""))
    start = int(row.get("candidate_start", -1) or -1)
    if start < 0:
        return ""
    return normalize_pair(source[max(0, start - 40) : start])


def _right_context(row: Mapping[str, Any]) -> str:
    source = str(row.get("source", ""))
    end = int(row.get("candidate_end", -1) or -1)
    if end < 0:
        return ""
    return normalize_pair(source[end : end + 40])


def _sentence_pattern(text: Any) -> str:
    value = normalize_pair(str(text))
    value = re.sub(r"[а-яё]+", "W", value, flags=re.IGNORECASE)
    value = re.sub(r"\d+", "N", value)
    return value


def _dominant_share(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    return max(counts.values()) / len(values)


def _near_duplicate_count(texts: list[str]) -> int:
    normalized = [normalize_pair(text) for text in texts]
    duplicates = sum(max(0, count - 1) for count in Counter(normalized).values())
    near = 0
    for index, left in enumerate(normalized):
        left_tokens = set(left.split())
        for right in normalized[index + 1 :]:
            right_tokens = set(right.split())
            if not left_tokens or not right_tokens:
                continue
            overlap = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
            if left != right and overlap >= 0.92:
                near += 1
    return duplicates + near
