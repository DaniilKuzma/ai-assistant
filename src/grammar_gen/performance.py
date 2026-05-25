from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import time
from typing import Any

from src.grammar_gen import Lexicon, MorphologyEngine
from src.grammar_gen.audit import audit_example
from src.grammar_gen.generator import OnlineExampleGenerator, validate_generation_config
from src.grammar_gen.rules.registry import default_rule_registry


def benchmark_generation(config: Mapping[str, Any], count: int, seed: int) -> dict[str, Any]:
    if count < 0:
        raise ValueError("count must be non-negative.")

    validate_generation_config(config)
    generator = OnlineExampleGenerator(
        default_rule_registry(),
        Lexicon.default(),
        MorphologyEngine(use_pymorphy=False),
        config,
        seed=seed,
    )

    rule_distribution: Counter[str] = Counter()
    mode_distribution: Counter[str] = Counter()
    audit_failure_count = 0

    started = time.perf_counter()
    for index in range(count):
        example = generator.sample_by_index(index)
        rule_distribution[example.primary_rule_id] += 1
        mode_distribution[example.mode] += 1
        if audit_example(example):
            audit_failure_count += 1
    total_seconds = time.perf_counter() - started

    examples_per_second = count / total_seconds if total_seconds > 0 else 0.0
    avg_ms_per_example = (total_seconds * 1000.0 / count) if count > 0 else 0.0

    return {
        "count": count,
        "seed": int(seed),
        "total_seconds": total_seconds,
        "examples_per_second": examples_per_second,
        "avg_ms_per_example": avg_ms_per_example,
        "rule_distribution": dict(sorted(rule_distribution.items())),
        "mode_distribution": dict(sorted(mode_distribution.items())),
        "audit_failure_count": audit_failure_count,
    }


__all__ = ["benchmark_generation"]
