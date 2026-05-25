from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import pandas as pd

from src.candidates.candidate_generator import Candidate
from src.config.candidate_dataset_config import candidate_dataset_rule_quota, candidate_dataset_value
from src.data.atomic_verifier import AtomicVerificationResult, verify_atomic_positive
from src.data.dataset_contract import (
    DATASET_CONTRACT,
    HARD_NEGATIVE_OPEN,
    LAYER_ATOMIC_HARD_NEGATIVE,
    LAYER_ATOMIC_POSITIVE,
    SYNTHETIC_OPEN_CLEAN,
)
from src.data.dataset_quality import clean_or_hard_quality_reasons, normalized_pair_hash, normalize_pair
from src.data.rule_lab_generation import RuleLabGenerationResult, generate_rule_lab_rows, rule_lab_recipe_rule_ids
from src.data.training_quality_audit import contains_artificial_marker_text, plain_quote_bracket_balance_reasons
from src.rules.rule_ids import UNKNOWN_RULE_ID, normalize_rule_id
from src.validation.strict_validator import TRAINING_CONTEXT_SPLIT_JOIN_BY_RULE


MINER_NAME = "corpus_backed_split_join"
SOURCE_CORPUS_MINED = "corpus_mined"
SOURCE_SYNTAX_MINED = "syntax_mined"
SOURCE_MORPHOLOGY_MINED = "morphology_mined"
SOURCE_REAL_PATTERN_REPLAY = "real_pattern_replay"
SOURCE_RULE_LAB = "rule_lab"
DEFAULT_SOURCE_PRIORITY = (
    SOURCE_CORPUS_MINED,
    SOURCE_SYNTAX_MINED,
    SOURCE_MORPHOLOGY_MINED,
    SOURCE_REAL_PATTERN_REPLAY,
    SOURCE_RULE_LAB,
)
SERVICE_HARD_NEGATIVE_RULE_ID = "clean_identity_hard_negative"
HARD_NEGATIVE_ERROR_TYPE = "hard_negative"
SUPPORTED_CONTEXT_RULES = tuple(TRAINING_CONTEXT_SPLIT_JOIN_BY_RULE)
SUPPORTED_MORPHOLOGY_RULES = ("tsya_soft_insert", "tsya_soft_delete", "n_nn_adjective")
SUPPORTED_SYNTAX_PUNCTUATION_RULES = (
    "comma_subordinate",
    "homogeneous_comma",
    "introductory_comma",
    "detached_adverbial_comma",
    "detached_participial_comma",
    "direct_speech_dash",
    "direct_speech_colon",
    "enumeration_colon",
    "explanation_colon",
    "asyndetic_dash",
    "subject_predicate_dash",
)
DEFAULT_REAL_PATTERN_PATHS = (
    "data/processed/real_error_pairs_mining.csv.gz",
    "data/processed/real_error_pairs_rejected.csv.gz",
    "data/processed/real_error_pairs_atomic.csv.gz",
)


@dataclass(frozen=True)
class RuleDataSourceStats:
    rule_id: str
    corpus_mined_positive_count: int = 0
    syntax_mined_positive_count: int = 0
    morphology_mined_positive_count: int = 0
    real_pattern_replay_positive_count: int = 0
    rule_lab_positive_count: int = 0
    total_atomic_positive_count: int = 0
    corpus_mined_hard_negative_count: int = 0
    rule_lab_hard_negative_count: int = 0
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
    rule_lab_generation_rows: list[dict[str, Any]] = field(default_factory=list)
    rule_lab_rejection_rows: list[dict[str, Any]] = field(default_factory=list)
    rule_lab_diversity_rows: list[Any] = field(default_factory=list)
    rule_lab_template_validation_rows: list[dict[str, Any]] = field(default_factory=list)
    rule_lab_slot_usage_rows: list[dict[str, Any]] = field(default_factory=list)
    rule_lab_recipe_status_rows: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    audit_errors: list[str] = field(default_factory=list)


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
    activation_source: str = SOURCE_CORPUS_MINED
    generation_strategy: str = "rule_data_compiler"
    error_type: str = ""


@dataclass(frozen=True)
class HardNegativeContext:
    rule_id: str
    text: str
    source_name: str
    source_row_id: str
    miner_name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    activation_source: str = SOURCE_CORPUS_MINED


class RuleExampleMiner:
    rule_id: str
    source_name: str
    activation_source: str

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
    source_name: str = SOURCE_CORPUS_MINED
    activation_source: str = SOURCE_CORPUS_MINED

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
                        activation_source=self.activation_source,
                        generation_strategy="corpus_backed_rule_data_compiler",
                        error_type="split_join",
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
                    activation_source=self.activation_source,
                )
            )
        return contexts


@dataclass(frozen=True)
class MorphologyBackedMiner(RuleExampleMiner):
    rule_id: str
    source_name: str = SOURCE_MORPHOLOGY_MINED
    activation_source: str = SOURCE_MORPHOLOGY_MINED

    def mine_positive_targets(self, clean_rows: list[dict[str, Any]], config: dict[str, Any]) -> list[CandidateTargetContext]:
        if not _morphology_available():
            return [
                CandidateTargetContext(
                    rule_id=self.rule_id,
                    target=_clean_text_from_row(clean_rows[0]) if clean_rows else "",
                    source_fragment="",
                    target_fragment="",
                    source_name=self.source_name,
                    source_row_id="",
                    miner_name=self.activation_source,
                    metadata={"forced_rejection_reason": "morphology_unavailable"},
                    activation_source=self.activation_source,
                    generation_strategy="morphology_backed_rule_data_compiler",
                    error_type="spelling",
                )
            ]

        limit = _per_rule_scan_limit(config)
        contexts: list[CandidateTargetContext] = []
        for row in clean_rows[:limit]:
            target = _clean_text_from_row(row)
            if not target:
                continue
            for start, end, target_fragment, source_fragment in _morphology_mutations(self.rule_id, target):
                contexts.append(
                    CandidateTargetContext(
                        rule_id=self.rule_id,
                        target=target,
                        source_fragment=source_fragment,
                        target_fragment=target_fragment,
                        source_name=str(row.get("source_name") or row.get("source_corpus") or "open_clean"),
                        source_row_id=_row_id(row),
                        miner_name=self.activation_source,
                        metadata=_source_metadata(row, start, end),
                        activation_source=self.activation_source,
                        generation_strategy="morphology_backed_rule_data_compiler",
                        error_type="spelling",
                    )
                )
        return contexts

    def corrupt_target(self, context: CandidateTargetContext) -> tuple[str, str]:
        start = int(context.metadata.get("target_start", -1))
        end = int(context.metadata.get("target_end", -1))
        target = context.target
        return target[:start] + context.source_fragment + target[end:], target

    def mine_hard_negatives(self, clean_rows: list[dict[str, Any]], config: dict[str, Any]) -> list[HardNegativeContext]:
        del clean_rows, config
        return []


@dataclass(frozen=True)
class SyntaxBackedMiner(RuleExampleMiner):
    rule_id: str
    source_name: str = SOURCE_SYNTAX_MINED
    activation_source: str = SOURCE_SYNTAX_MINED

    def mine_positive_targets(self, clean_rows: list[dict[str, Any]], config: dict[str, Any]) -> list[CandidateTargetContext]:
        operator = _syntax_operator_for_rule(self.rule_id)
        if operator is None:
            return [self._empty_opportunity_context(clean_rows, "opportunities_seen_0")]

        limit = _per_rule_scan_limit(config)
        contexts: list[CandidateTargetContext] = []
        opportunities_seen = 0
        for row in clean_rows[:limit]:
            target = _clean_text_from_row(row)
            if not target:
                continue
            try:
                opportunities = list(operator.find_opportunities(target))
            except Exception:
                continue
            opportunities_seen += len(opportunities)
            for opportunity in opportunities:
                try:
                    corruption = operator.corrupt(target, opportunity)
                except Exception:
                    continue
                source = str(getattr(corruption, "source", ""))
                if not source or source == target:
                    continue
                target_fragment = str(getattr(corruption, "target_form", "") or getattr(opportunity, "text", "") or "")
                source_fragment = str(getattr(corruption, "error_form", "") or "")
                metadata = _source_metadata(row, int(getattr(opportunity, "start", -1)), int(getattr(opportunity, "end", -1)))
                metadata.update(
                    {
                        "source_text": source,
                        "syntax_family": str(getattr(opportunity, "syntax_family", "")),
                        "opportunity": asdict(opportunity) if hasattr(opportunity, "__dataclass_fields__") else {},
                    }
                )
                contexts.append(
                    CandidateTargetContext(
                        rule_id=self.rule_id,
                        target=target,
                        source_fragment=source_fragment,
                        target_fragment=target_fragment,
                        source_name=str(row.get("source_name") or row.get("source_corpus") or "open_clean"),
                        source_row_id=_row_id(row),
                        miner_name=self.activation_source,
                        metadata=metadata,
                        activation_source=self.activation_source,
                        generation_strategy="syntax_backed_rule_data_compiler",
                        error_type="punctuation",
                    )
                )
        if not contexts and opportunities_seen == 0:
            return [self._empty_opportunity_context(clean_rows, "opportunities_seen_0")]
        return contexts

    def _empty_opportunity_context(self, clean_rows: list[dict[str, Any]], reason: str) -> CandidateTargetContext:
        target = _clean_text_from_row(clean_rows[0]) if clean_rows else ""
        return CandidateTargetContext(
            rule_id=self.rule_id,
            target=target,
            source_fragment="",
            target_fragment="",
            source_name=self.source_name,
            source_row_id="",
            miner_name=self.activation_source,
            metadata={"forced_rejection_reason": reason},
            activation_source=self.activation_source,
            generation_strategy="syntax_backed_rule_data_compiler",
            error_type="punctuation",
        )

    def corrupt_target(self, context: CandidateTargetContext) -> tuple[str, str]:
        source = str(context.metadata.get("source_text") or "")
        return source, context.target

    def mine_hard_negatives(self, clean_rows: list[dict[str, Any]], config: dict[str, Any]) -> list[HardNegativeContext]:
        del clean_rows, config
        return []


@dataclass(frozen=True)
class ReplayPattern:
    rule_id: str
    source_fragment: str
    target_fragment: str
    edit_type: str
    source_file: str = ""


@dataclass(frozen=True)
class RealPatternReplayMiner(RuleExampleMiner):
    rule_id: str
    patterns: tuple[ReplayPattern, ...]
    unsafe_count: int = 0
    source_name: str = SOURCE_REAL_PATTERN_REPLAY
    activation_source: str = SOURCE_REAL_PATTERN_REPLAY

    def mine_positive_targets(self, clean_rows: list[dict[str, Any]], config: dict[str, Any]) -> list[CandidateTargetContext]:
        limit = _per_rule_scan_limit(config)
        contexts: list[CandidateTargetContext] = []
        if self.unsafe_count and not self.patterns:
            contexts.append(self._forced_rejection_context(clean_rows, "pattern_replay_unsafe"))
        for pattern in self.patterns:
            for row in clean_rows[:limit]:
                target = _clean_text_from_row(row)
                if not target:
                    continue
                for start, end, source_text, source_fragment, target_fragment in _apply_replay_pattern(target, pattern):
                    metadata = _source_metadata(row, start, end)
                    metadata.update(
                        {
                            "source_text": source_text,
                            "pattern_source_file": pattern.source_file,
                            "pattern_edit_type": pattern.edit_type,
                        }
                    )
                    contexts.append(
                        CandidateTargetContext(
                            rule_id=self.rule_id,
                            target=target,
                            source_fragment=source_fragment,
                            target_fragment=target_fragment,
                            source_name=str(row.get("source_name") or row.get("source_corpus") or "open_clean"),
                            source_row_id=_row_id(row),
                            miner_name=self.activation_source,
                            metadata=metadata,
                            activation_source=self.activation_source,
                            generation_strategy="real_pattern_replay_rule_data_compiler",
                            error_type=_error_type_from_pattern(pattern),
                        )
                    )
        if self.patterns and not contexts:
            contexts.append(self._forced_rejection_context(clean_rows, "pattern_replay_unsafe"))
        return contexts

    def _forced_rejection_context(self, clean_rows: list[dict[str, Any]], reason: str) -> CandidateTargetContext:
        target = _clean_text_from_row(clean_rows[0]) if clean_rows else ""
        return CandidateTargetContext(
            rule_id=self.rule_id,
            target=target,
            source_fragment="",
            target_fragment="",
            source_name=self.source_name,
            source_row_id="",
            miner_name=self.activation_source,
            metadata={"forced_rejection_reason": reason},
            activation_source=self.activation_source,
            generation_strategy="real_pattern_replay_rule_data_compiler",
        )

    def corrupt_target(self, context: CandidateTargetContext) -> tuple[str, str]:
        return str(context.metadata.get("source_text") or ""), context.target

    def mine_hard_negatives(self, clean_rows: list[dict[str, Any]], config: dict[str, Any]) -> list[HardNegativeContext]:
        del clean_rows, config
        return []


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
    rule_ids = _supported_rule_ids(active_or_candidate_rule_ids, config)
    source_priority = _source_priority(config)
    real_patterns_by_rule, real_unsafe_by_rule = _real_patterns_by_rule(config, rule_ids)
    positive_target = _positive_target(config, target_counts)
    hard_target = _hard_negative_target(config, target_counts)

    accepted_positive: list[dict[str, Any]] = []
    accepted_hard: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    seen_positive = _existing_positive_keys(existing)
    seen_hard = _existing_hard_negative_keys(existing)
    positive_counts = _existing_positive_counts(existing)
    hard_counts = _existing_hard_negative_counts(existing)
    rule_lab_result = RuleLabGenerationResult()
    warnings: list[str] = []
    audit_errors: list[str] = []

    for source_key in source_priority:
        if source_key == SOURCE_RULE_LAB:
            rule_lab_result = generate_rule_lab_rows(
                rule_ids,
                candidate_generator,
                config,
                clean_rows=rows,
                existing_positive_rows=[*existing, *accepted_positive],
                existing_hard_negative_rows=[*existing, *accepted_hard],
                positive_counts_by_rule=positive_counts,
                hard_counts_by_rule=hard_counts,
                positive_target=positive_target,
                hard_target=hard_target,
            )
            accepted_positive.extend(rule_lab_result.atomic_positive_rows)
            accepted_hard.extend(rule_lab_result.hard_negative_rows)
            rejections.extend(rule_lab_result.rejection_rows)
            for row in rule_lab_result.atomic_positive_rows:
                rule_id = normalize_rule_id(str(row.get("rule_id") or row.get("target_rule_id") or ""))
                if rule_id != UNKNOWN_RULE_ID and _counts_toward_rule_quota(row):
                    positive_counts[rule_id] += 1
                    seen_positive.add(_positive_key(str(row.get("source", "")), str(row.get("target", "")), rule_id))
            for row in rule_lab_result.hard_negative_rows:
                rule_id = normalize_rule_id(str(row.get("target_rule_id") or _metadata(row).get("target_rule_id") or ""))
                if rule_id != UNKNOWN_RULE_ID:
                    hard_counts[rule_id] += 1
                    seen_hard.add(_hard_negative_key(str(row.get("source", "")), rule_id))
            continue

        miners = _miners_for_source(source_key, rule_ids, real_patterns_by_rule, real_unsafe_by_rule)
        for miner in miners:
            for context in miner.mine_positive_targets(rows, config):
                if positive_counts[context.rule_id] >= positive_target:
                    break
                source, target = miner.corrupt_target(context)
                forced_reason = str(context.metadata.get("forced_rejection_reason") or "")
                if forced_reason:
                    rejections.append(_rejection_row(context.rule_id, source, target, forced_reason, "positive_mining", context))
                    continue
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

    share_by_rule = _rule_lab_share_by_rule(rule_ids, [*existing, *accepted_positive])
    high_share_rule_ids = _rule_lab_high_share_rule_ids(share_by_rule, config)
    if high_share_rule_ids:
        warnings.extend(f"rule_lab_share_high:{rule_id}" for rule_id in high_share_rule_ids)
        if _fail_on_high_rule_lab_share(config):
            for rule_id in high_share_rule_ids:
                blocked_count = sum(1 for row in accepted_positive if _row_activation_source(row) == SOURCE_RULE_LAB and _row_rule_id(row) == rule_id)
                if blocked_count:
                    rejections.append(
                        {
                            "rule_id": rule_id,
                            "source": "",
                            "target": "",
                            "reason": f"rule_lab_share_high:{share_by_rule.get(rule_id, 0.0):.3f}",
                            "stage": "rule_lab_share_gate",
                            "source_name": SOURCE_RULE_LAB,
                            "source_row_id": "",
                            "miner_name": SOURCE_RULE_LAB,
                        }
                    )
                    positive_counts[rule_id] -= blocked_count
            accepted_positive = [
                row
                for row in accepted_positive
                if not (_row_activation_source(row) == SOURCE_RULE_LAB and _row_rule_id(row) in set(high_share_rule_ids))
            ]
            audit_errors.extend(f"rule_lab_share_high:{rule_id}" for rule_id in high_share_rule_ids)
            share_by_rule = _rule_lab_share_by_rule(rule_ids, [*existing, *accepted_positive])

    source_stats = _source_stats(rule_ids, accepted_positive, accepted_hard)
    diversity_stats = _diversity_stats(rule_ids, accepted_positive, config, rule_lab_share_by_rule=share_by_rule)
    underfilled = _underfilled_rows(
        rule_ids,
        positive_counts,
        hard_counts,
        min_positive_target=_min_positive_required(config, target_counts),
        preferred_positive_target=positive_target,
        hard_target=hard_target,
        rejections=rejections,
    )
    return RuleDataCompilerResult(
        atomic_positive_rows=accepted_positive,
        hard_negative_rows=accepted_hard,
        rejection_rows=rejections,
        source_stats_rows=source_stats,
        diversity_stats_rows=diversity_stats,
        underfilled_rows=underfilled,
        rule_lab_generation_rows=rule_lab_result.generation_rows,
        rule_lab_rejection_rows=rule_lab_result.rejection_rows,
        rule_lab_diversity_rows=rule_lab_result.diversity_rows,
        rule_lab_template_validation_rows=rule_lab_result.template_validation_rows,
        rule_lab_slot_usage_rows=rule_lab_result.slot_usage_rows,
        rule_lab_recipe_status_rows=rule_lab_result.recipe_status_rows,
        warnings=_dedupe_ordered(warnings),
        audit_errors=_dedupe_ordered(audit_errors),
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
        _aggregated_rejection_rows(result.rejection_rows),
        [
            "rule_id",
            "miner_name",
            "reason",
            "count",
            "example_source",
            "example_target",
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
            "hard_negative_count",
            "min_atomic_required",
            "preferred_atomic",
            "min_hard_required",
            "top_rejection_reason",
            "recommended_next_action",
        ],
    ).to_csv(
        output_dir / "rule_underfilled_backlog.csv",
        index=False,
    )
    _frame_with_columns(
        result.rule_lab_generation_rows,
        [
            "rule_id",
            "positive_requested",
            "positive_generated",
            "hard_negative_requested",
            "hard_negative_generated",
            "status",
            "reason",
        ],
    ).to_csv(output_dir / "rule_lab_generation_report.csv", index=False)
    _frame_with_columns(
        result.rule_lab_rejection_rows,
        [
            "rule_id",
            "source",
            "target",
            "reason",
            "stage",
            "template_id",
            "source_name",
            "source_row_id",
            "miner_name",
        ],
    ).to_csv(output_dir / "rule_lab_rejection_report.csv", index=False)
    _frame_with_columns(
        _dataclass_or_mapping_rows(result.rule_lab_diversity_rows),
        [
            "rule_id",
            "generated_count",
            "accepted_count",
            "unique_source_count",
            "unique_target_count",
            "template_count",
            "dominant_template_id",
            "dominant_template_share",
            "dominant_slot_value_share",
            "duplicate_count",
            "near_duplicate_count",
            "status",
            "reason",
        ],
    ).to_csv(output_dir / "rule_lab_diversity_report.csv", index=False)
    _frame_with_columns(
        result.rule_lab_template_validation_rows,
        ["rule_id", "template_id", "template_kind", "status", "reason"],
    ).to_csv(output_dir / "rule_lab_template_validation_report.csv", index=False)
    _frame_with_columns(
        result.rule_lab_slot_usage_rows,
        ["rule_id", "template_id", "slot_name", "slot_value", "count"],
    ).to_csv(output_dir / "rule_lab_slot_usage_report.csv", index=False)
    _frame_with_columns(
        result.rule_lab_recipe_status_rows,
        [
            "rule_id",
            "enabled",
            "status",
            "reason",
            "disabled_reason",
            "positive_template_count",
            "hard_negative_template_count",
        ],
    ).to_csv(output_dir / "rule_lab_recipe_status_report.csv", index=False)


def _frame_with_columns(rows: Iterable[Mapping[str, Any]], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(list(rows)).reindex(columns=columns)


def _dataclass_or_mapping_rows(rows: Iterable[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, Mapping):
            result.append(dict(row))
        elif hasattr(row, "__dataclass_fields__"):
            result.append(asdict(row))
    return result


def _source_priority(config: dict[str, Any]) -> list[str]:
    compiler_config = dict(candidate_dataset_value(config, "rule_data_compiler", {}) or {})
    configured = compiler_config.get("source_priority") or DEFAULT_SOURCE_PRIORITY
    result: list[str] = []
    for source in configured:
        source_key = str(source)
        if source_key == SOURCE_RULE_LAB:
            continue
        if source_key in DEFAULT_SOURCE_PRIORITY and source_key not in result:
            result.append(source_key)
    for source_key in DEFAULT_SOURCE_PRIORITY:
        if source_key == SOURCE_RULE_LAB:
            continue
        if source_key not in result:
            result.append(source_key)
    result.append(SOURCE_RULE_LAB)
    return result


def _miners_for_source(
    source_key: str,
    rule_ids: list[str],
    real_patterns_by_rule: Mapping[str, tuple[ReplayPattern, ...]],
    real_unsafe_by_rule: Mapping[str, int],
) -> list[RuleExampleMiner]:
    if source_key == SOURCE_CORPUS_MINED:
        return [_miner_for_rule(rule_id) for rule_id in rule_ids if rule_id in TRAINING_CONTEXT_SPLIT_JOIN_BY_RULE]
    if source_key == SOURCE_SYNTAX_MINED:
        return [SyntaxBackedMiner(rule_id=rule_id) for rule_id in rule_ids if rule_id in SUPPORTED_SYNTAX_PUNCTUATION_RULES]
    if source_key == SOURCE_MORPHOLOGY_MINED:
        return [MorphologyBackedMiner(rule_id=rule_id) for rule_id in rule_ids if rule_id in SUPPORTED_MORPHOLOGY_RULES]
    if source_key == SOURCE_REAL_PATTERN_REPLAY:
        return [
            RealPatternReplayMiner(
                rule_id=rule_id,
                patterns=tuple(real_patterns_by_rule.get(rule_id, ())),
                unsafe_count=int(real_unsafe_by_rule.get(rule_id, 0)),
            )
            for rule_id in rule_ids
            if real_patterns_by_rule.get(rule_id) or real_unsafe_by_rule.get(rule_id, 0)
        ]
    return []


def _miner_for_rule(rule_id: str) -> CorpusBackedSplitJoinMiner:
    source_fragment, target_fragment = TRAINING_CONTEXT_SPLIT_JOIN_BY_RULE[rule_id]
    return CorpusBackedSplitJoinMiner(rule_id=rule_id, source_fragment=source_fragment, target_fragment=target_fragment)


def _supported_rule_ids(rule_ids: Iterable[str], config: dict[str, Any]) -> list[str]:
    result: list[str] = []
    real_patterns_enabled = bool(_real_pattern_paths(config))
    rule_lab_enabled_rule_ids = rule_lab_recipe_rule_ids(config)
    for rule_id in rule_ids:
        normalized = normalize_rule_id(rule_id)
        if normalized == UNKNOWN_RULE_ID or normalized in result:
            continue
        if (
            normalized in TRAINING_CONTEXT_SPLIT_JOIN_BY_RULE
            or normalized in SUPPORTED_MORPHOLOGY_RULES
            or normalized in SUPPORTED_SYNTAX_PUNCTUATION_RULES
            or normalized in rule_lab_enabled_rule_ids
            or real_patterns_enabled
        ):
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
    activation_source = context.activation_source or SOURCE_CORPUS_MINED
    generation_strategy = context.generation_strategy or f"{activation_source}_rule_data_compiler"
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
        "activation_source": activation_source,
        "generation_sources": [activation_source],
        "generation_strategy": generation_strategy,
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
    error_type = context.error_type or _error_type_from_verification(verification)
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
        "activation_source": activation_source,
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
        "activation_source": context.activation_source,
        "generation_sources": [context.activation_source],
        "generation_strategy": f"{context.activation_source}_rule_data_compiler",
        "hard_negative_source": context.activation_source,
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
        "activation_source": context.activation_source,
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


def _morphology_available() -> bool:
    try:
        from src.candidates.morphology import morph_analyzer

        morph_analyzer()
    except Exception:
        return False
    return True


def _morphology_mutations(rule_id: str, target: str) -> list[tuple[int, int, str, str]]:
    mutations: list[tuple[int, int, str, str]] = []
    for match in re.finditer(r"[А-Яа-яЁё]+", target):
        target_fragment = match.group(0)
        lower = target_fragment.lower()
        if rule_id == "tsya_soft_insert" and lower.endswith("ться") and _morphology_known(lower):
            dirty = lower[: -len("ться")] + "тся"
        elif rule_id == "tsya_soft_delete" and lower.endswith("тся") and _morphology_known(lower):
            dirty = lower[: -len("тся")] + "ться"
        elif rule_id == "n_nn_adjective" and "нн" in lower and _morphology_known(lower):
            for index in _double_en_indexes(lower):
                dirty = lower[:index] + "н" + lower[index + 2 :]
                mutations.append((match.start(), match.end(), target_fragment, _match_case(target_fragment, dirty)))
            continue
        else:
            continue
        if dirty != lower:
            mutations.append((match.start(), match.end(), target_fragment, _match_case(target_fragment, dirty)))
    return mutations


def _double_en_indexes(word: str) -> list[int]:
    result: list[int] = []
    index = word.find("нн")
    while index >= 0:
        result.append(index)
        index = word.find("нн", index + 2)
    return result


def _morphology_known(word: str) -> bool:
    try:
        from src.candidates.morphology import parses
    except Exception:
        return False
    try:
        return any(getattr(parse, "is_known", False) for parse in parses(word))
    except Exception:
        return False


def _syntax_operator_for_rule(rule_id: str) -> Any | None:
    if rule_id not in SUPPORTED_SYNTAX_PUNCTUATION_RULES:
        return None
    try:
        from src.data.corruption_operators import build_default_operator_registry
    except Exception:
        return None
    try:
        return build_default_operator_registry().get(rule_id)
    except Exception:
        return None


def _real_pattern_paths(config: dict[str, Any]) -> list[Path]:
    compiler_config = dict(candidate_dataset_value(config, "rule_data_compiler", {}) or {})
    configured = compiler_config.get("real_pattern_paths")
    if configured is None and not compiler_config:
        return []
    raw_paths = configured if configured is not None else DEFAULT_REAL_PATTERN_PATHS
    paths: list[Path] = []
    for raw_path in raw_paths:
        path = Path(str(raw_path))
        if path.exists():
            paths.append(path)
    return paths


def _real_patterns_by_rule(config: dict[str, Any], rule_ids: Iterable[str]) -> tuple[dict[str, tuple[ReplayPattern, ...]], dict[str, int]]:
    rule_set = {normalize_rule_id(rule_id) for rule_id in rule_ids}
    patterns: dict[str, list[ReplayPattern]] = defaultdict(list)
    unsafe: Counter[str] = Counter()
    for path in _real_pattern_paths(config):
        try:
            frame = pd.read_csv(path, low_memory=False).fillna("")
        except Exception:
            continue
        for row in frame.to_dict("records"):
            rule_id = _real_pattern_rule_id(row)
            if rule_id not in rule_set:
                continue
            pattern = _safe_replay_pattern(row, source_file=str(path))
            if pattern is None:
                unsafe[rule_id] += 1
                continue
            patterns[rule_id].append(pattern)
    return {rule_id: tuple(items) for rule_id, items in patterns.items()}, dict(unsafe)


def _real_pattern_rule_id(row: Mapping[str, Any]) -> str:
    direct = normalize_rule_id(str(row.get("rule_id") or ""))
    if direct != UNKNOWN_RULE_ID:
        return direct
    for rule_id in _json_list(row.get("rule_ids")):
        normalized = normalize_rule_id(str(rule_id))
        if normalized != UNKNOWN_RULE_ID:
            return normalized
    for rule_id in _json_list(row.get("candidate_rule_ids")):
        normalized = normalize_rule_id(str(rule_id))
        if normalized != UNKNOWN_RULE_ID:
            return normalized
    return UNKNOWN_RULE_ID


def _safe_replay_pattern(row: Mapping[str, Any], *, source_file: str) -> ReplayPattern | None:
    rule_id = _real_pattern_rule_id(row)
    source = str(row.get("source", "")).strip()
    target = str(row.get("target", "")).strip()
    if not source or not target or source == target:
        return None
    if _real_gold_edit_count(row) != 1:
        return None

    token_pattern = _single_token_replacement_pattern(source, target, rule_id, source_file)
    if token_pattern is not None:
        return token_pattern
    split_join_pattern = _split_join_pattern(source, target, rule_id, source_file)
    if split_join_pattern is not None:
        return split_join_pattern
    punctuation_pattern = _punctuation_pattern(source, target, rule_id, source_file)
    if punctuation_pattern is not None:
        return punctuation_pattern
    return None


def _single_token_replacement_pattern(source: str, target: str, rule_id: str, source_file: str) -> ReplayPattern | None:
    source_tokens = _word_tokens_with_spans(source)
    target_tokens = _word_tokens_with_spans(target)
    if len(source_tokens) != len(target_tokens):
        return None
    diffs = [(left, right) for left, right in zip(source_tokens, target_tokens) if left[0].lower() != right[0].lower()]
    if len(diffs) != 1:
        return None
    left, right = diffs[0]
    source_without = source[: left[1]] + right[0] + source[left[2] :]
    if normalize_pair(source_without) != normalize_pair(target):
        return None
    return ReplayPattern(rule_id=rule_id, source_fragment=left[0], target_fragment=right[0], edit_type="single_token_replacement", source_file=source_file)


def _split_join_pattern(source: str, target: str, rule_id: str, source_file: str) -> ReplayPattern | None:
    source_words = [token[0] for token in _word_tokens_with_spans(source)]
    target_words = [token[0] for token in _word_tokens_with_spans(target)]
    if len(source_words) == len(target_words) + 1:
        for index in range(len(target_words)):
            if source_words[index].lower() + source_words[index + 1].lower() != target_words[index].lower():
                continue
            source_fragment = source_words[index] + " " + source_words[index + 1]
            target_fragment = target_words[index]
            if normalize_pair(source.replace(source_fragment, target_fragment, 1)) == normalize_pair(target):
                return ReplayPattern(rule_id=rule_id, source_fragment=source_fragment, target_fragment=target_fragment, edit_type="split_join", source_file=source_file)
    if len(target_words) == len(source_words) + 1:
        for index in range(len(source_words)):
            if target_words[index].lower() + target_words[index + 1].lower() != source_words[index].lower():
                continue
            source_fragment = source_words[index]
            target_fragment = target_words[index] + " " + target_words[index + 1]
            if normalize_pair(source.replace(source_fragment, target_fragment, 1)) == normalize_pair(target):
                return ReplayPattern(rule_id=rule_id, source_fragment=source_fragment, target_fragment=target_fragment, edit_type="split_join", source_file=source_file)
    return None


def _punctuation_pattern(source: str, target: str, rule_id: str, source_file: str) -> ReplayPattern | None:
    try:
        from src.validation.diff_analyzer import DiffAnalyzer
    except Exception:
        return None
    edits = DiffAnalyzer().analyze(source, target, candidates=[])
    if len(edits) != 1:
        return None
    edit = edits[0]
    if not str(edit.edit_type).startswith("punctuation_"):
        return None
    return ReplayPattern(rule_id=rule_id, source_fragment=str(edit.source), target_fragment=str(edit.replacement), edit_type=str(edit.edit_type), source_file=source_file)


def _word_tokens_with_spans(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(0), match.start(), match.end()) for match in re.finditer(r"[А-Яа-яЁё]+", text)]


def _apply_replay_pattern(target: str, pattern: ReplayPattern) -> list[tuple[int, int, str, str, str]]:
    results: list[tuple[int, int, str, str, str]] = []
    if not pattern.target_fragment:
        return results
    for match in _fragment_matches(target, pattern.target_fragment):
        target_fragment = target[match.start() : match.end()]
        source_fragment = _match_case(target_fragment, pattern.source_fragment)
        source = target[: match.start()] + source_fragment + target[match.end() :]
        if source != target:
            results.append((match.start(), match.end(), source, source_fragment, target_fragment))
    return results


def _fragment_matches(text: str, fragment: str) -> list[re.Match[str]]:
    if re.fullmatch(r"[А-Яа-яЁё ]+", fragment):
        pattern = re.compile(r"(?<![А-Яа-яЁё])" + re.escape(fragment) + r"(?![А-Яа-яЁё])", re.IGNORECASE)
    else:
        pattern = re.compile(re.escape(fragment))
    return list(pattern.finditer(text))


def _real_gold_edit_count(row: Mapping[str, Any]) -> int:
    for key in ("gold_edit_count", "edit_count"):
        value = row.get(key)
        if value not in (None, ""):
            try:
                return max(0, int(float(value)))
            except (TypeError, ValueError):
                pass
    edits = _json_list(row.get("edits") or row.get("edit_operations"))
    return len(edits)


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _error_type_from_pattern(pattern: ReplayPattern) -> str:
    if pattern.edit_type.startswith("punctuation_"):
        return "punctuation"
    if pattern.edit_type == "split_join":
        return "split_join"
    return "spelling"


def _error_type_from_verification(verification: AtomicVerificationResult) -> str:
    if not verification.edits:
        return "spelling"
    edit_type = str(verification.edits[0].get("edit_type", ""))
    if edit_type.startswith("punctuation_"):
        return "punctuation"
    if edit_type in {"split_word", "join_words"}:
        return "split_join"
    return "spelling"


def _source_stats(rule_ids: list[str], positives: list[dict[str, Any]], hard_rows: list[dict[str, Any]]) -> list[RuleDataSourceStats]:
    positive_counts = Counter((str(row.get("rule_id", "")), str(row.get("activation_source") or _metadata(row).get("activation_source") or "")) for row in positives)
    total_positive_counts = Counter(str(row.get("rule_id", "")) for row in positives)
    hard_counts = Counter((str(row.get("target_rule_id", "")), str(row.get("activation_source") or _metadata(row).get("activation_source") or "")) for row in hard_rows)
    total_hard_counts = Counter(str(row.get("target_rule_id", "")) for row in hard_rows)
    return [
        RuleDataSourceStats(
            rule_id=rule_id,
            corpus_mined_positive_count=int(positive_counts.get((rule_id, SOURCE_CORPUS_MINED), 0)),
            syntax_mined_positive_count=int(positive_counts.get((rule_id, SOURCE_SYNTAX_MINED), 0)),
            morphology_mined_positive_count=int(positive_counts.get((rule_id, SOURCE_MORPHOLOGY_MINED), 0)),
            real_pattern_replay_positive_count=int(positive_counts.get((rule_id, SOURCE_REAL_PATTERN_REPLAY), 0)),
            rule_lab_positive_count=int(positive_counts.get((rule_id, SOURCE_RULE_LAB), 0)),
            total_atomic_positive_count=int(total_positive_counts.get(rule_id, 0)),
            corpus_mined_hard_negative_count=int(hard_counts.get((rule_id, SOURCE_CORPUS_MINED), 0)),
            rule_lab_hard_negative_count=int(hard_counts.get((rule_id, SOURCE_RULE_LAB), 0)),
            total_hard_negative_count=int(total_hard_counts.get(rule_id, 0)),
        )
        for rule_id in rule_ids
    ]


def _diversity_stats(
    rule_ids: list[str],
    positives: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    rule_lab_share_by_rule: Mapping[str, float] | None = None,
) -> list[RuleStructuralDiversityStats]:
    by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in positives:
        by_rule[str(row.get("rule_id", ""))].append(row)
    compiler_config = dict(candidate_dataset_value(config, "rule_data_compiler", {}) or {})
    min_patterns = int(compiler_config.get("min_unique_sentence_patterns_per_rule", 0) or 0)
    min_left = int(compiler_config.get("min_unique_left_contexts_per_rule", 0) or 0)
    min_right = int(compiler_config.get("min_unique_right_contexts_per_rule", 0) or 0)
    result: list[RuleStructuralDiversityStats] = []
    for rule_id in rule_ids:
        rows = by_rule.get(rule_id, [])
        if not rows:
            result.append(
                RuleStructuralDiversityStats(
                    rule_id=rule_id,
                    rule_lab_share=float((rule_lab_share_by_rule or {}).get(rule_id, 0.0) or 0.0),
                    status="empty",
                    reason="no_atomic_positive_rows",
                )
            )
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
                rule_lab_share=float((rule_lab_share_by_rule or {}).get(rule_id, 0.0) or 0.0),
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
    min_positive_target: int,
    preferred_positive_target: int,
    hard_target: int,
    rejections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rejection_reasons_by_rule: dict[str, Counter[str]] = defaultdict(Counter)
    for rejection in rejections:
        rejection_reasons_by_rule[str(rejection.get("rule_id", ""))][str(rejection.get("reason", ""))] += 1
    for rule_id in rule_ids:
        positive_count = int(positive_counts.get(rule_id, 0))
        hard_count = int(hard_counts.get(rule_id, 0))
        top_reason = ""
        if positive_count < min_positive_target:
            top_reason = _top_rejection_reason(rejection_reasons_by_rule.get(rule_id, Counter())) or "opportunities_seen_0"
        if hard_count < hard_target:
            if not top_reason or positive_count >= min_positive_target:
                top_reason = "hard_negative_count_under_min"
        if positive_count < min_positive_target or hard_count < hard_target:
            rows.append(
                {
                    "rule_id": rule_id,
                    "atomic_positive_count": positive_count,
                    "hard_negative_count": hard_count,
                    "min_atomic_required": min_positive_target,
                    "preferred_atomic": preferred_positive_target,
                    "min_hard_required": hard_target,
                    "top_rejection_reason": top_reason,
                    "recommended_next_action": _recommended_next_action(top_reason),
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


def _aggregated_rejection_rows(rejections: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for rejection in rejections:
        key = (
            str(rejection.get("rule_id", "")),
            str(rejection.get("miner_name", "")),
            str(rejection.get("reason", "")),
        )
        if key not in grouped:
            grouped[key] = {
                "rule_id": key[0],
                "miner_name": key[1],
                "reason": key[2],
                "count": 0,
                "example_source": str(rejection.get("source", ""))[:500],
                "example_target": str(rejection.get("target", ""))[:500],
            }
        grouped[key]["count"] += 1
    return sorted(grouped.values(), key=lambda row: (row["rule_id"], row["miner_name"], row["reason"]))


def _top_rejection_reason(reasons: Counter[str]) -> str:
    if not reasons:
        return ""
    return reasons.most_common(1)[0][0]


def _recommended_next_action(reason: str) -> str:
    mapping = {
        "candidate_missing": "fix_candidate_generator_or_rule_mapping",
        "strict_validator_rejected": "fix_strict_validator_support",
        "non_atomic_edit_count": "fix_operator_or_miner_to_generate_atomic_edits",
        "target_quality_failed": "fix_template_or_context_quality",
        "unsupported_edit_type": "extend_diff_or_candidate_matching",
        "opportunities_seen_0": "add_rule_specific_miner_or_templates",
        "hard_negative_count_under_min": "add_hard_negative_miner_or_templates",
    }
    return mapping.get(reason, "inspect_rule_data_compiler_rejections")


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


def _existing_positive_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        if _row_dataset_layer(row) != LAYER_ATOMIC_POSITIVE:
            continue
        if not _counts_toward_rule_quota(row):
            continue
        rule_id = _row_rule_id(row)
        if rule_id != UNKNOWN_RULE_ID:
            counts[rule_id] += 1
    return counts


def _existing_hard_negative_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        if _row_dataset_layer(row) != LAYER_ATOMIC_HARD_NEGATIVE and not _bool_value(row.get("is_hard_negative") or _metadata(row).get("is_hard_negative")):
            continue
        rule_id = normalize_rule_id(str(row.get("target_rule_id") or _metadata(row).get("target_rule_id") or ""))
        if rule_id != UNKNOWN_RULE_ID:
            counts[rule_id] += 1
    return counts


def _rule_lab_share_by_rule(rule_ids: list[str], rows: list[dict[str, Any]]) -> dict[str, float]:
    total_counts: Counter[str] = Counter()
    rule_lab_counts: Counter[str] = Counter()
    rule_id_set = set(rule_ids)
    for row in rows:
        if _row_dataset_layer(row) != LAYER_ATOMIC_POSITIVE:
            continue
        if not _counts_toward_rule_quota(row):
            continue
        rule_id = _row_rule_id(row)
        if rule_id == UNKNOWN_RULE_ID or rule_id not in rule_id_set:
            continue
        total_counts[rule_id] += 1
        if _row_activation_source(row) == SOURCE_RULE_LAB:
            rule_lab_counts[rule_id] += 1
    return {
        rule_id: (float(rule_lab_counts.get(rule_id, 0)) / float(total_counts[rule_id]))
        for rule_id in rule_ids
        if total_counts.get(rule_id, 0)
    }


def _rule_lab_high_share_rule_ids(rule_lab_share_by_rule: Mapping[str, float], config: dict[str, Any]) -> list[str]:
    rule_lab_config = dict(candidate_dataset_value(config, "rule_lab", {}) or {})
    compiler_config = dict(candidate_dataset_value(config, "rule_data_compiler", {}) or {})
    threshold = float(rule_lab_config.get("max_rule_lab_share_per_rule", compiler_config.get("max_rule_lab_share_per_rule", 0.30)) or 0.30)
    return sorted(rule_id for rule_id, share in rule_lab_share_by_rule.items() if threshold >= 0 and float(share) > threshold)


def _fail_on_high_rule_lab_share(config: dict[str, Any]) -> bool:
    rule_lab_config = dict(candidate_dataset_value(config, "rule_lab", {}) or {})
    compiler_config = dict(candidate_dataset_value(config, "rule_data_compiler", {}) or {})
    return _bool_value(rule_lab_config.get("fail_on_high_rule_lab_share", compiler_config.get("fail_on_high_rule_lab_share", False)))


def _row_rule_id(row: Mapping[str, Any]) -> str:
    metadata = _metadata(row)
    for value in (row.get("target_rule_id"), row.get("rule_id"), metadata.get("target_rule_id")):
        rule_id = normalize_rule_id(str(value or ""))
        if rule_id != UNKNOWN_RULE_ID and rule_id != SERVICE_HARD_NEGATIVE_RULE_ID:
            return rule_id
    rule_ids = row.get("rule_ids") or metadata.get("rule_ids")
    for value in _list_value(rule_ids):
        rule_id = normalize_rule_id(str(value or ""))
        if rule_id != UNKNOWN_RULE_ID and rule_id != SERVICE_HARD_NEGATIVE_RULE_ID:
            return rule_id
    return UNKNOWN_RULE_ID


def _row_dataset_layer(row: Mapping[str, Any]) -> str:
    return str(row.get("dataset_layer") or _metadata(row).get("dataset_layer") or "")


def _row_activation_source(row: Mapping[str, Any]) -> str:
    return str(row.get("activation_source") or _metadata(row).get("activation_source") or "")


def _counts_toward_rule_quota(row: Mapping[str, Any]) -> bool:
    metadata = _metadata(row)
    if "count_toward_rule_quota" in row:
        return _bool_value(row.get("count_toward_rule_quota"))
    if "count_toward_rule_quota" in metadata:
        return _bool_value(metadata.get("count_toward_rule_quota"))
    return True


def _list_value(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return [value]
    if isinstance(parsed, list):
        return list(parsed)
    return [parsed]


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _dedupe_ordered(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in result:
            result.append(text)
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
    quota = candidate_dataset_rule_quota(config)
    return max(1, int(quota.get("preferred_atomic_positives_per_active_rule", 50) or 50))


def _min_positive_required(config: dict[str, Any], target_counts: Mapping[str, Any] | None) -> int:
    if target_counts and "atomic_positive" in target_counts:
        return max(0, int(target_counts.get("atomic_positive") or 0))
    quota = candidate_dataset_rule_quota(config)
    return max(1, int(quota.get("min_atomic_positives_per_active_rule", _positive_target(config, target_counts)) or 1))


def _hard_negative_target(config: dict[str, Any], target_counts: Mapping[str, Any] | None) -> int:
    if target_counts and "atomic_hard_negative" in target_counts:
        return max(0, int(target_counts.get("atomic_hard_negative") or 0))
    quota = candidate_dataset_rule_quota(config)
    return max(1, int(quota.get("min_hard_negatives_per_active_rule", 50) or 50))


def _per_rule_scan_limit(config: dict[str, Any]) -> int:
    compiler_config = dict(candidate_dataset_value(config, "rule_data_compiler", {}) or {})
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
