from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

import pandas as pd

from src.candidates.candidate_generator import Candidate, CandidateGenerator
from src.data.dataset_contract import DATASET_CONTRACT, HARD_NEGATIVE_OPEN, LAYER_ATOMIC_HARD_NEGATIVE
from src.data.dataset_quality import clean_or_hard_quality_reasons, normalized_pair_hash
from src.data.training_quality_audit import contains_artificial_marker_text, plain_quote_bracket_balance_reasons
from src.rules.rule_ids import UNKNOWN_RULE_ID, normalize_rule_id


SERVICE_HARD_NEGATIVE_RULE_ID = "clean_identity_hard_negative"
HARD_NEGATIVE_ERROR_TYPE = "hard_negative"

FALLBACK_TEMPLATES_BY_RULE: dict[str, str] = {
    "context_chto_by": "Что бы ты ни сказал, комиссия проверит документы завтра.",
    "context_tak_zhe": "Так же, как раньше, редакция проверила отчёт утром.",
    "context_to_zhe": "То же самое решение комиссия приняла вечером.",
    "context_za_to": "За то решение отвечала отдельная рабочая группа.",
    "context_nesmotrya": "Не смотря по сторонам, курьер быстро прошёл через двор.",
    "context_vsledstvie": "В следствие по делу внесли новые документы.",
}

COVERAGE_REPORT_COLUMNS = [
    "rule_id",
    "min_per_rule",
    "preferred_per_rule",
    "generated_count",
    "clean_count",
    "fallback_count",
    "rejected_count",
    "status",
]
REJECTION_REPORT_COLUMNS = [
    "rule_id",
    "source",
    "reason",
    "stage",
    "candidate_rule_ids",
    "error",
]


@dataclass(frozen=True)
class HardNegativeGenerationResult:
    rows: list[dict[str, Any]]
    attempt_rows: list[dict[str, Any]]
    rejection_rows: list[dict[str, Any]]
    counts_by_rule: dict[str, int]


def generate_atomic_hard_negatives(
    clean_rows: list[dict[str, Any]],
    rule_ids: list[str],
    candidate_generator: CandidateGenerator | Any,
    min_per_rule: int,
    preferred_per_rule: int,
    seed: int,
    max_scan_rows: int,
    fallback_templates_enabled: bool = True,
) -> HardNegativeGenerationResult:
    target_rule_ids = _normalized_target_rule_ids(rule_ids)
    min_target = max(0, int(min_per_rule))
    preferred_target = max(min_target, int(preferred_per_rule))
    generator = candidate_generator or CandidateGenerator()
    counts_by_rule = {rule_id: 0 for rule_id in target_rule_ids}
    accepted_by_source: Counter[tuple[str, str]] = Counter()
    rejected_by_rule: Counter[str] = Counter()
    scan_counts = {rule_id: 0 for rule_id in target_rule_ids}
    rejection_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int, str]] = set()

    shuffled = [dict(row) for row in clean_rows]
    random.Random(seed).shuffle(shuffled)
    scan_limit = max(0, int(max_scan_rows))

    for clean_row in shuffled[:scan_limit]:
        if _all_rules_satisfied(counts_by_rule, preferred_target):
            break
        text = _clean_text_from_row(clean_row)
        if not text:
            continue
        try:
            candidates = list(generator.generate(text))
        except Exception as exc:
            for rule_id in _underfilled_rules(counts_by_rule, preferred_target):
                rejection_rows.append(
                    _rejection_row(
                        rule_id,
                        text,
                        "candidate_generation_error",
                        "clean_pool",
                        error=exc,
                    )
                )
                rejected_by_rule[rule_id] += 1
            continue
        candidate_rule_ids = _candidate_rule_ids(candidates)
        for rule_id in target_rule_ids:
            if counts_by_rule[rule_id] < preferred_target:
                scan_counts[rule_id] += 1
        for candidate in candidates:
            rule_id = normalize_rule_id(getattr(candidate, "rule_id", ""))
            if rule_id not in counts_by_rule or counts_by_rule[rule_id] >= preferred_target:
                continue
            if not _is_non_keep_candidate(candidate):
                continue
            quality_reasons = _hard_negative_quality_failure_reasons(text, text)
            if quality_reasons:
                for reason in quality_reasons:
                    rejection_rows.append(
                        _rejection_row(
                            rule_id,
                            text,
                            f"hard_negative_quality_failed:{reason}",
                            "clean_pool",
                            candidate_rule_ids=candidate_rule_ids,
                        )
                    )
                    rejected_by_rule[rule_id] += 1
                continue
            dedupe_key = _dedupe_key(rule_id, text, candidate)
            if dedupe_key in seen:
                rejection_rows.append(
                    _rejection_row(
                        rule_id,
                        text,
                        "duplicate_candidate",
                        "clean_pool",
                        candidate_rule_ids=candidate_rule_ids,
                    )
                )
                rejected_by_rule[rule_id] += 1
                continue
            seen.add(dedupe_key)
            rows.append(
                _hard_negative_row(
                    text,
                    candidate,
                    target_rule_id=rule_id,
                    clean_row=clean_row,
                    source_kind="clean_pool",
                )
            )
            counts_by_rule[rule_id] += 1
            accepted_by_source[(rule_id, "clean_pool")] += 1

    if fallback_templates_enabled:
        for rule_id in _underfilled_rules(counts_by_rule, preferred_target):
            template = FALLBACK_TEMPLATES_BY_RULE.get(rule_id)
            if template is None:
                rejection_rows.append(_rejection_row(rule_id, "", "fallback_template_missing", "fallback_template"))
                rejected_by_rule[rule_id] += 1
                continue
            try:
                candidates = list(generator.generate(template))
            except Exception as exc:
                rejection_rows.append(
                    _rejection_row(
                        rule_id,
                        template,
                        "candidate_generation_error",
                        "fallback_template",
                        error=exc,
                    )
                )
                rejected_by_rule[rule_id] += 1
                continue
            candidate_rule_ids = _candidate_rule_ids(candidates)
            accepted = False
            for candidate in candidates:
                candidate_rule_id = normalize_rule_id(getattr(candidate, "rule_id", ""))
                if candidate_rule_id != rule_id or not _is_non_keep_candidate(candidate):
                    continue
                quality_reasons = _hard_negative_quality_failure_reasons(template, template)
                if quality_reasons:
                    for reason in quality_reasons:
                        rejection_rows.append(
                            _rejection_row(
                                rule_id,
                                template,
                                f"hard_negative_quality_failed:{reason}",
                                "fallback_template",
                                candidate_rule_ids=candidate_rule_ids,
                            )
                        )
                        rejected_by_rule[rule_id] += 1
                    continue
                dedupe_key = _dedupe_key(rule_id, template, candidate)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                rows.append(
                    _hard_negative_row(
                        template,
                        candidate,
                        target_rule_id=rule_id,
                        clean_row={
                            "source_name": "fallback_hard_negative_template",
                            "source_subcorpus": rule_id,
                            "domain": "open_clean",
                        },
                        source_kind="fallback_template",
                    )
                )
                counts_by_rule[rule_id] += 1
                accepted_by_source[(rule_id, "fallback_template")] += 1
                accepted = True
                if counts_by_rule[rule_id] >= preferred_target:
                    break
            if not accepted:
                rejection_rows.append(
                    _rejection_row(
                        rule_id,
                        template,
                        "candidate_not_generated_for_template",
                        "fallback_template",
                        candidate_rule_ids=candidate_rule_ids,
                    )
                )
                rejected_by_rule[rule_id] += 1

    attempt_rows = [
        {
            "rule_id": rule_id,
            "min_per_rule": min_target,
            "preferred_per_rule": preferred_target,
            "scanned_clean_rows": int(scan_counts.get(rule_id, 0)),
            "generated_clean": int(accepted_by_source.get((rule_id, "clean_pool"), 0)),
            "generated_fallback": int(accepted_by_source.get((rule_id, "fallback_template"), 0)),
            "generated_count": int(counts_by_rule.get(rule_id, 0)),
            "rejected_count": int(rejected_by_rule.get(rule_id, 0)),
            "status": _coverage_status(int(counts_by_rule.get(rule_id, 0)), min_target, preferred_target),
        }
        for rule_id in target_rule_ids
    ]
    return HardNegativeGenerationResult(
        rows=rows,
        attempt_rows=attempt_rows,
        rejection_rows=rejection_rows,
        counts_by_rule=counts_by_rule,
    )


def write_hard_negative_reports(result: HardNegativeGenerationResult, reports_dir: str | Path) -> None:
    output_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    attempts = {str(row.get("rule_id")): row for row in result.attempt_rows}
    rejection_counts = Counter(str(row.get("rule_id")) for row in result.rejection_rows)
    rule_ids = sorted(set(result.counts_by_rule) | set(attempts) | set(rejection_counts))
    coverage_rows: list[dict[str, Any]] = []
    for rule_id in rule_ids:
        attempt = attempts.get(rule_id, {})
        generated_count = int(result.counts_by_rule.get(rule_id, attempt.get("generated_count", 0)) or 0)
        min_target = int(attempt.get("min_per_rule", 0) or 0)
        preferred_target = int(attempt.get("preferred_per_rule", min_target) or min_target)
        coverage_rows.append(
            {
                "rule_id": rule_id,
                "min_per_rule": min_target,
                "preferred_per_rule": preferred_target,
                "generated_count": generated_count,
                "clean_count": int(attempt.get("generated_clean", 0) or 0),
                "fallback_count": int(attempt.get("generated_fallback", 0) or 0),
                "rejected_count": int(rejection_counts.get(rule_id, attempt.get("rejected_count", 0)) or 0),
                "status": _coverage_status(generated_count, min_target, preferred_target),
            }
        )

    pd.DataFrame(coverage_rows, columns=COVERAGE_REPORT_COLUMNS).to_csv(
        output_dir / "hard_negative_coverage_report.csv",
        index=False,
    )
    pd.DataFrame(result.rejection_rows, columns=REJECTION_REPORT_COLUMNS).to_csv(
        output_dir / "hard_negative_rejection_report.csv",
        index=False,
    )


def _normalized_target_rule_ids(rule_ids: list[str]) -> list[str]:
    result: list[str] = []
    for rule_id in rule_ids:
        normalized = normalize_rule_id(rule_id)
        if normalized == UNKNOWN_RULE_ID or normalized in result:
            continue
        result.append(normalized)
    return result


def _clean_text_from_row(row: Mapping[str, Any]) -> str:
    for column in ("text", "target", "source"):
        value = row.get(column)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _all_rules_satisfied(counts_by_rule: dict[str, int], preferred_target: int) -> bool:
    return all(count >= preferred_target for count in counts_by_rule.values())


def _underfilled_rules(counts_by_rule: dict[str, int], preferred_target: int) -> list[str]:
    return [rule_id for rule_id, count in counts_by_rule.items() if count < preferred_target]


def _is_non_keep_candidate(candidate: Candidate) -> bool:
    edit_type = str(getattr(candidate, "edit_type", "") or "").lower()
    action = str(getattr(candidate, "action", "") or "").upper()
    if edit_type == "keep" or action in {"KEEP", "KEEP_NONE", "KEEP_EXISTING"}:
        return False
    source = str(getattr(candidate, "source", ""))
    replacement = str(getattr(candidate, "replacement", ""))
    return source != replacement or action in {"INSERT", "DELETE", "REPLACE"}


def _candidate_rule_ids(candidates: list[Candidate]) -> list[str]:
    return sorted({normalize_rule_id(getattr(candidate, "rule_id", "")) for candidate in candidates})


def _dedupe_key(rule_id: str, sentence: str, candidate: Candidate) -> tuple[str, str, int, int, str]:
    return (
        rule_id,
        sentence,
        int(getattr(candidate, "start", -1)),
        int(getattr(candidate, "end", -1)),
        str(getattr(candidate, "replacement", "")),
    )


def _hard_negative_quality_failure_reasons(source: str, target: str) -> list[str]:
    reasons: list[str] = []
    if source != target:
        reasons.append("non_identity")
    if contains_artificial_marker_text(source, target):
        reasons.append("artificial_marker")
    balance_reasons = plain_quote_bracket_balance_reasons(source)
    reasons.extend(balance_reasons)
    quality_reasons = clean_or_hard_quality_reasons(source)
    if balance_reasons:
        quality_reasons = [reason for reason in quality_reasons if reason != "unbalanced_quote_or_bracket"]
    reasons.extend(quality_reasons)
    return list(dict.fromkeys(reasons))


def _hard_negative_row(
    text: str,
    candidate: Candidate,
    *,
    target_rule_id: str,
    clean_row: Mapping[str, Any],
    source_kind: str,
) -> dict[str, Any]:
    candidate_source = str(getattr(candidate, "source", ""))
    candidate_replacement = str(getattr(candidate, "replacement", ""))
    candidate_start = int(getattr(candidate, "start", -1))
    candidate_end = int(getattr(candidate, "end", -1))
    rule_ids = [SERVICE_HARD_NEGATIVE_RULE_ID]
    metadata = {
        "dataset_contract": DATASET_CONTRACT,
        "dataset_layer": LAYER_ATOMIC_HARD_NEGATIVE,
        "source_type": HARD_NEGATIVE_OPEN,
        "rule_ids": rule_ids,
        "is_atomic": True,
        "is_hard_negative": True,
        "target_rule_id": target_rule_id,
        "candidate_rule_id": target_rule_id,
        "candidate_source": candidate_source,
        "candidate_replacement": candidate_replacement,
        "candidate_start": candidate_start,
        "candidate_end": candidate_end,
        "candidate_edit_type": str(getattr(candidate, "edit_type", "")),
        "candidate_action": str(getattr(candidate, "action", "")),
        "candidate_mode": str(getattr(candidate, "mode", "")),
        "candidate_requires_model": bool(getattr(candidate, "requires_model", False)),
        "candidate_requires_scoring": bool(getattr(candidate, "requires_scoring", False)),
        "candidate_group": str(getattr(candidate, "group", "")),
        "count_toward_rule_quota": False,
        "loss_weight": 1.0,
        "gold_edit_count": 0,
        "generation_strategy": "atomic_hard_negative",
        "hard_negative_source": source_kind,
        "expected_accepted_edits": 0,
        "original_clean_sentence": text,
    }
    source_corpus = str(clean_row.get("source_name") or clean_row.get("source_corpus") or "")
    source_subcorpus = str(clean_row.get("source_subcorpus") or "")
    domain = str(clean_row.get("domain") or "open_clean")
    pair_hash = normalized_pair_hash(text, text)
    return {
        "source": text,
        "target": text,
        "split": "train",
        "source_type": HARD_NEGATIVE_OPEN,
        "error_type": HARD_NEGATIVE_ERROR_TYPE,
        "rule_ids": json.dumps(rule_ids, ensure_ascii=False),
        "edits": "[]",
        "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        "original_clean_source": text,
        "source_corpus": source_corpus,
        "source_subcorpus": source_subcorpus,
        "is_hard_negative": True,
        "is_real_pair": False,
        "template_id": "",
        "normalized_pair_hash": pair_hash,
        "error_types": json.dumps([HARD_NEGATIVE_ERROR_TYPE], ensure_ascii=False),
        "source_dataset": source_corpus or HARD_NEGATIVE_OPEN,
        "is_clean": True,
        "is_synthetic": source_kind == "fallback_template",
        "domain": domain,
        "rule_id": SERVICE_HARD_NEGATIVE_RULE_ID,
        "edit_operations": "[]",
        "dataset_contract": DATASET_CONTRACT,
        "dataset_layer": LAYER_ATOMIC_HARD_NEGATIVE,
        "is_atomic": True,
        "is_stress": False,
        "count_toward_rule_quota": False,
        "loss_weight": 1.0,
        "gold_edit_count": 0,
        "target_rule_id": target_rule_id,
        "candidate_source": candidate_source,
        "candidate_replacement": candidate_replacement,
        "candidate_start": candidate_start,
        "candidate_end": candidate_end,
        "verification_status": "",
        "rejection_reason": "",
    }


def _rejection_row(
    rule_id: str,
    source: str,
    reason: str,
    stage: str,
    *,
    candidate_rule_ids: list[str] | None = None,
    error: Exception | None = None,
) -> dict[str, Any]:
    payload = {
        "rule_id": rule_id,
        "source": source,
        "reason": reason,
        "stage": stage,
        "candidate_rule_ids": json.dumps(candidate_rule_ids or [], ensure_ascii=False),
        "error": "",
    }
    if error is not None:
        payload["error"] = f"{type(error).__name__}: {error}"
    return payload


def _coverage_status(generated_count: int, min_target: int, preferred_target: int) -> str:
    if generated_count >= preferred_target:
        return "complete"
    if generated_count >= min_target:
        return "below_preferred"
    return "under_min"
