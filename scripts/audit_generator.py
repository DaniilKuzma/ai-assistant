from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.load_config import load_config
from src.grammar_gen.factory import online_generator_from_config
from src.grammar_gen.audit import audit_batch
from src.grammar_gen.diversity import sample_diverse_examples


def quality_gate_reasons(
    audit: dict,
    *,
    max_duplicate_pair_rate: float = 0.15,
    min_everyday_context_share: float = 0.20,
    max_editorial_official_context_share: float = 0.50,
) -> list[str]:
    diversity = audit.get("diversity", {}) if isinstance(audit, dict) else {}
    reasons: list[str] = []

    duplicate_pair_rate = float(diversity.get("duplicate_pair_rate", 0.0))
    if duplicate_pair_rate > max_duplicate_pair_rate:
        reasons.append(f"duplicate_pair_rate:{duplicate_pair_rate:.4f}>{max_duplicate_pair_rate:.4f}")

    style_distribution = diversity.get("context_style_bucket_distribution", {})
    if isinstance(style_distribution, dict):
        total = sum(int(value) for value in style_distribution.values())
        if total > 0:
            everyday_share = int(style_distribution.get("everyday", 0)) / total
            official_share = int(style_distribution.get("editorial_official", 0)) / total
            if everyday_share < min_everyday_context_share:
                reasons.append(f"everyday_context_share:{everyday_share:.4f}<{min_everyday_context_share:.4f}")
            if official_share > max_editorial_official_context_share:
                reasons.append(
                    "editorial_official_context_share:"
                    f"{official_share:.4f}>{max_editorial_official_context_share:.4f}"
                )
    return reasons


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit online grammar generation.")
    parser.add_argument("config")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument("--show-examples", action="store_true")
    parser.add_argument("--max-duplicate-pair-rate", type=float, default=0.15)
    parser.add_argument("--min-everyday-context-share", type=float, default=0.20)
    parser.add_argument("--max-editorial-official-context-share", type=float, default=0.50)
    args = parser.parse_args()

    if args.count < 0:
        parser.error("--count must be non-negative")

    config = load_config(args.config)
    generator = online_generator_from_config(config, seed=int(config.get("generation", {}).get("seed", 0)))

    started = time.perf_counter()
    if args.dedupe:
        examples = sample_diverse_examples(
            generator,
            count=args.count,
            stream_name="audit-generator",
            max_attempts=args.max_attempts,
        )
    else:
        examples = [generator.sample_by_index(index) for index in range(args.count)]
    elapsed = time.perf_counter() - started
    audit = audit_batch(examples)
    examples_per_sec = args.count / elapsed if elapsed > 0 else 0.0

    print(f"examples/sec: {examples_per_sec:.2f}")
    print("rule distribution:")
    print(json.dumps(audit["rule_distribution"], ensure_ascii=False, sort_keys=True))
    print("mode distribution:")
    print(json.dumps(audit["mode_distribution"], ensure_ascii=False, sort_keys=True))
    print("diversity:")
    print(json.dumps(audit["diversity"], ensure_ascii=False, sort_keys=True))
    print(f"audit failures: {audit['failed_examples_count']}")
    if audit["failure_reasons"]:
        print("audit failure reasons:")
        print(json.dumps(audit["failure_reasons"], ensure_ascii=False, sort_keys=True))
    gate_reasons = quality_gate_reasons(
        audit,
        max_duplicate_pair_rate=args.max_duplicate_pair_rate,
        min_everyday_context_share=args.min_everyday_context_share,
        max_editorial_official_context_share=args.max_editorial_official_context_share,
    )
    if gate_reasons:
        print("quality gate failures:")
        print(json.dumps(gate_reasons, ensure_ascii=False, sort_keys=True))

    if args.show_examples:
        print("examples:")
        for index, example in enumerate(examples[:10]):
            print(
                f"{index}: {example.primary_rule_id} {example.mode} "
                f"{example.source_text} => {example.target_text}"
            )

    return 1 if audit["failed_examples_count"] or gate_reasons else 0


if __name__ == "__main__":
    raise SystemExit(main())
