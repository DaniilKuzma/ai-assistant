from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
import hashlib
import re
from typing import Any

from src.grammar_gen.safety import count_logical_token_edits
from src.schema import GeneratedExample


SPACE_RE = re.compile(r"\s+")
DASH_TRANSLATION = str.maketrans({"–": "—", "−": "—"})
QUOTE_TRANSLATION = str.maketrans({"“": '"', "”": '"', "«": '"', "»": '"'})


def normalize_diversity_text(text: str) -> str:
    normalized = text.strip().translate(DASH_TRANSLATION).translate(QUOTE_TRANSLATION)
    return SPACE_RE.sub(" ", normalized)


def source_target_pair_key(example: GeneratedExample) -> tuple[str, str]:
    return (
        normalize_diversity_text(example.source_text),
        normalize_diversity_text(example.target_text),
    )


def duplicate_pair_rate(examples: Iterable[GeneratedExample]) -> float:
    example_list = list(examples)
    if not example_list:
        return 0.0
    return _duplicate_rate(source_target_pair_key(example) for example in example_list)


def diversity_report(examples: Iterable[GeneratedExample]) -> dict[str, Any]:
    example_list = list(examples)
    count = len(example_list)
    source_keys = [normalize_diversity_text(example.source_text) for example in example_list]
    target_keys = [normalize_diversity_text(example.target_text) for example in example_list]
    pair_keys = [source_target_pair_key(example) for example in example_list]
    pair_counts = Counter(pair_keys)
    rule_distribution = Counter(example.primary_rule_id for example in example_list)
    sub_rule_distribution = Counter(_sub_rule_id(example) for example in example_list)
    template_distribution = Counter(_template_id(example) for example in example_list)
    token_edit_counts = Counter(_logical_token_edit_count(example) for example in example_list)
    gap_edit_counts = Counter(_logical_gap_edit_count(example) for example in example_list)

    duplicate_by_rule = _duplicate_rates_by_bucket(
        ((example.primary_rule_id, source_target_pair_key(example)) for example in example_list)
    )
    duplicate_by_sub_rule = _duplicate_rates_by_bucket(
        ((_sub_rule_id(example), source_target_pair_key(example)) for example in example_list)
    )
    top_duplicate_pairs = [
        {
            "source_text": source,
            "target_text": target,
            "count": pair_count,
        }
        for (source, target), pair_count in pair_counts.most_common(20)
        if pair_count > 1
    ]
    duplicate_diagnostics = {
        "top_duplicate_pairs": top_duplicate_pairs[:10],
        "top_duplicate_rule_ids": _top_duplicate_rates(duplicate_by_rule),
        "top_duplicate_sub_rule_ids": _top_duplicate_rates(duplicate_by_sub_rule),
    }

    return {
        "count": count,
        "unique_source_texts": len(set(source_keys)),
        "unique_target_texts": len(set(target_keys)),
        "unique_source_target_pairs": len(set(pair_keys)),
        "duplicate_source_rate": _duplicate_rate(source_keys),
        "duplicate_target_rate": _duplicate_rate(target_keys),
        "duplicate_pair_rate": _duplicate_rate(pair_keys),
        "top_duplicate_pairs": top_duplicate_pairs,
        "duplicate_rate_by_rule_id": duplicate_by_rule,
        "duplicate_rate_by_sub_rule_id": duplicate_by_sub_rule,
        "average_token_count": _average(len(example.source_tokens) for example in example_list),
        "token_edit_count_distribution": dict(sorted(token_edit_counts.items())),
        "gap_edit_count_distribution": dict(sorted(gap_edit_counts.items())),
        "template_distribution": dict(sorted(template_distribution.items())),
        "rule_distribution": dict(sorted(rule_distribution.items())),
        "sub_rule_distribution": dict(sorted(sub_rule_distribution.items())),
        "duplicate_diagnostics": duplicate_diagnostics,
    }


def sample_diverse_examples(
    generator: Any,
    *,
    count: int,
    stream_name: str = "batch",
    max_attempts: int = 8,
    max_duplicate_pair_rate: float | None = None,
) -> list[GeneratedExample]:
    if count < 0:
        raise ValueError("count must be non-negative.")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")

    examples: list[GeneratedExample] = []
    seen_pairs: set[tuple[str, str]] = set()
    accepted_duplicate_count = 0

    for index in range(count):
        last_candidate: GeneratedExample | None = None
        for attempt in range(max_attempts):
            candidate_index = _attempt_index(index, attempt, stream_name, max_attempts)
            candidate = generator.sample_by_index(candidate_index)
            last_candidate = candidate
            pair_key = source_target_pair_key(candidate)
            if pair_key not in seen_pairs:
                examples.append(candidate)
                seen_pairs.add(pair_key)
                break
        else:
            if last_candidate is None:
                raise RuntimeError("No generation attempt was made.")
            examples.append(last_candidate)
            pair_key = source_target_pair_key(last_candidate)
            if pair_key in seen_pairs:
                accepted_duplicate_count += 1
            seen_pairs.add(pair_key)

    if max_duplicate_pair_rate is not None:
        observed = duplicate_pair_rate(examples)
        if observed > max_duplicate_pair_rate:
            report = diversity_report(examples)
            raise RuntimeError(
                "Duplicate source-target pair rate is too high: "
                f"{observed:.4f} > {max_duplicate_pair_rate:.4f}; "
                f"accepted_duplicate_count={accepted_duplicate_count}; "
                f"diagnostics={report['duplicate_diagnostics']!r}"
            )
    return examples


def _attempt_index(index: int, attempt: int, stream_name: str, max_attempts: int) -> int:
    if attempt == 0:
        return index
    stream_offset = _stable_offset(stream_name)
    return stream_offset + index * max_attempts + attempt


def _stable_offset(stream_name: str) -> int:
    digest = hashlib.sha1(stream_name.encode("utf-8")).hexdigest()
    return 1_000_000 + (int(digest[:10], 16) % 100_000_000)


def _duplicate_rate(values: Iterable[Any]) -> float:
    value_list = list(values)
    if not value_list:
        return 0.0
    return 1.0 - (len(set(value_list)) / len(value_list))


def _duplicate_rates_by_bucket(items: Iterable[tuple[str, tuple[str, str]]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for bucket, pair_key in items:
        grouped[bucket or ""].append(pair_key)
    return {
        bucket: {
            "count": len(values),
            "unique_pairs": len(set(values)),
            "duplicate_pair_rate": _duplicate_rate(values),
        }
        for bucket, values in sorted(grouped.items())
    }


def _top_duplicate_rates(stats: dict[str, dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    rows = [
        {
            "id": key,
            "count": int(value["count"]),
            "unique_pairs": int(value["unique_pairs"]),
            "duplicate_pair_rate": float(value["duplicate_pair_rate"]),
        }
        for key, value in stats.items()
        if int(value["count"]) > 1 and float(value["duplicate_pair_rate"]) > 0.0
    ]
    rows.sort(key=lambda item: (item["duplicate_pair_rate"], item["count"]), reverse=True)
    return rows[:limit]


def _sub_rule_id(example: GeneratedExample) -> str:
    metadata = example.metadata
    for key in ("sub_rule_id", "case_id", "construction_id"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    return example.primary_rule_id


def _template_id(example: GeneratedExample) -> str:
    metadata = example.metadata
    for key in ("template_id", "construction_id", "case_id", "sub_rule_id"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    return example.primary_rule_id


def _logical_token_edit_count(example: GeneratedExample) -> int:
    try:
        return count_logical_token_edits(example.token_edit_labels)
    except ValueError:
        return sum(1 for label in example.token_edit_labels if label not in {"KEEP", "SKIP_MERGED"})


def _logical_gap_edit_count(example: GeneratedExample) -> int:
    raw = example.metadata.get("expected_gap_edit_count")
    if raw is not None and not isinstance(raw, bool):
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            pass
    return sum(1 for label in example.gap_labels if label != "NONE")


def _average(values: Sequence[int] | Iterable[int]) -> float:
    value_list = list(values)
    if not value_list:
        return 0.0
    return sum(value_list) / len(value_list)


__all__ = [
    "diversity_report",
    "duplicate_pair_rate",
    "normalize_diversity_text",
    "sample_diverse_examples",
    "source_target_pair_key",
]
