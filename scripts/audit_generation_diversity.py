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
from src.grammar_gen.audit import audit_batch
from src.grammar_gen.diversity import sample_diverse_examples
from src.grammar_gen.factory import online_generator_from_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit generation diversity for online grammar examples.")
    parser.add_argument("config")
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--max-duplicate-pair-rate", type=float, default=0.12)
    parser.add_argument("--max-attempts", type=int, default=16)
    parser.add_argument("--show-examples", action="store_true")
    args = parser.parse_args()

    if args.count < 0:
        parser.error("--count must be non-negative")
    if args.max_duplicate_pair_rate < 0 or args.max_duplicate_pair_rate > 1:
        parser.error("--max-duplicate-pair-rate must be between 0 and 1")

    config = load_config(args.config)
    seed = int(config.get("generation", {}).get("seed", 0))
    generator = online_generator_from_config(config, seed=seed)

    started = time.perf_counter()
    try:
        examples = sample_diverse_examples(
            generator,
            count=args.count,
            stream_name="audit-generation-diversity",
            max_attempts=args.max_attempts,
            max_duplicate_pair_rate=args.max_duplicate_pair_rate,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - started
    audit = audit_batch(examples)
    diversity = audit["diversity"]

    print(f"examples/sec: {args.count / elapsed if elapsed > 0 else 0.0:.2f}")
    print("rule distribution:")
    print(json.dumps(audit["rule_distribution"], ensure_ascii=False, sort_keys=True))
    print("mode distribution:")
    print(json.dumps(audit["mode_distribution"], ensure_ascii=False, sort_keys=True))
    print("diversity:")
    print(json.dumps(diversity, ensure_ascii=False, sort_keys=True))
    print(f"audit failures: {audit['failed_examples_count']}")
    print(f"duplicate pair rate: {diversity['duplicate_pair_rate']:.4f}")

    if args.show_examples:
        print("examples:")
        for index, example in enumerate(examples[:10]):
            print(
                f"{index}: {example.primary_rule_id} {example.mode} "
                f"{example.source_text} => {example.target_text}"
            )

    if audit["failed_examples_count"]:
        return 1
    return 0 if diversity["duplicate_pair_rate"] <= args.max_duplicate_pair_rate else 1


if __name__ == "__main__":
    raise SystemExit(main())
