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
from src.grammar_gen import Lexicon, MorphologyEngine
from src.grammar_gen.audit import audit_batch
from src.grammar_gen.generator import OnlineExampleGenerator
from src.grammar_gen.rules.registry import default_rule_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit online grammar generation.")
    parser.add_argument("config")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--show-examples", action="store_true")
    args = parser.parse_args()

    if args.count < 0:
        parser.error("--count must be non-negative")

    config = load_config(args.config)
    generator = OnlineExampleGenerator(
        default_rule_registry(),
        Lexicon.default(),
        MorphologyEngine(use_pymorphy=False),
        config,
        seed=int(config.get("generation", {}).get("seed", 0)),
    )

    started = time.perf_counter()
    examples = [generator.sample_by_index(index) for index in range(args.count)]
    elapsed = time.perf_counter() - started
    audit = audit_batch(examples)
    examples_per_sec = args.count / elapsed if elapsed > 0 else 0.0

    print(f"examples/sec: {examples_per_sec:.2f}")
    print("rule distribution:")
    print(json.dumps(audit["rule_distribution"], ensure_ascii=False, sort_keys=True))
    print("mode distribution:")
    print(json.dumps(audit["mode_distribution"], ensure_ascii=False, sort_keys=True))
    print(f"audit failures: {audit['failed_examples_count']}")
    if audit["failure_reasons"]:
        print("audit failure reasons:")
        print(json.dumps(audit["failure_reasons"], ensure_ascii=False, sort_keys=True))

    if args.show_examples:
        print("examples:")
        for index, example in enumerate(examples[:10]):
            print(
                f"{index}: {example.primary_rule_id} {example.mode} "
                f"{example.source_text} => {example.target_text}"
            )

    return 1 if audit["failed_examples_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
