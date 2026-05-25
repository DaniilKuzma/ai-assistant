from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.load_config import load_config
from src.grammar_gen.performance import benchmark_generation


DEFAULT_OUTPUT_PATH = Path("reports/generation_benchmark.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark AST-first online grammar generation.")
    parser.add_argument("config")
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    if args.count < 0:
        parser.error("--count must be non-negative")

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else _seed_from_config(config)
    report = benchmark_generation(config, count=args.count, seed=seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"count: {report['count']}")
    print(f"seed: {report['seed']}")
    print(f"total seconds: {report['total_seconds']:.3f}")
    print(f"examples/sec: {report['examples_per_second']:.2f}")
    print(f"avg ms/example: {report['avg_ms_per_example']:.3f}")
    print("rule distribution:")
    print(json.dumps(report["rule_distribution"], ensure_ascii=False, sort_keys=True))
    print("mode distribution:")
    print(json.dumps(report["mode_distribution"], ensure_ascii=False, sort_keys=True))
    print(f"audit failures: {report['audit_failure_count']}")
    print(f"wrote: {output_path}")

    return 1 if report["audit_failure_count"] else 0


def _seed_from_config(config: dict) -> int:
    generation = config.get("generation", {}) if isinstance(config, dict) else {}
    raw_seed = generation.get("seed", 0) if isinstance(generation, dict) else 0
    try:
        return int(raw_seed)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
