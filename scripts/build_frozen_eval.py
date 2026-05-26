from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.load_config import load_config
from src.grammar_gen.audit import audit_batch
from src.grammar_gen.factory import online_generator_from_config
from src.schema.serialization import write_jsonl_examples


SPLIT_SEED_OFFSETS = {
    "val": 10_000_000,
    "test": 20_000_000,
    "regression": 30_000_000,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build frozen generated evaluation JSONL.")
    parser.add_argument("config")
    parser.add_argument("--split", required=True, choices=tuple(SPLIT_SEED_OFFSETS))
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.count < 0:
        parser.error("--count must be non-negative")

    manifest, examples = build_frozen_eval(
        config_path=Path(args.config),
        split=args.split,
        count=args.count,
        output_path=Path(args.output),
    )
    manifest_path = _manifest_path(Path(args.output))
    _write_manifest(manifest_path, manifest)

    if manifest["audit_failures_count"] > 0:
        print(f"audit failures: {manifest['audit_failures_count']}", file=sys.stderr)
        print(f"manifest: {manifest_path}")
        return 1

    write_jsonl_examples(args.output, examples)
    print(f"wrote: {args.output}")
    print(f"manifest: {manifest_path}")
    return 0


def build_frozen_eval(
    *,
    config_path: Path,
    split: str,
    count: int,
    output_path: Path,
) -> tuple[dict[str, Any], list[Any]]:
    config = load_config(config_path)
    base_seed = int(config.get("generation", {}).get("seed", 0))
    seed_offset = SPLIT_SEED_OFFSETS[split]
    train_start, train_end = _train_seed_range(config, base_seed)
    train_max_index = train_end - train_start
    if seed_offset <= train_max_index + 1_000_000:
        raise RuntimeError(
            f"Frozen eval seed offset for {split!r} overlaps or is too close to training range: "
            f"offset={seed_offset}, train_max_index={train_max_index}."
        )
    effective_seed = base_seed + seed_offset
    split_seed_range_start = effective_seed
    split_seed_range_end = effective_seed + count - 1 if count > 0 else effective_seed - 1
    overlaps_train = _ranges_overlap(split_seed_range_start, split_seed_range_end, train_start, train_end)

    generator = online_generator_from_config(config, seed=effective_seed)
    examples = [generator.sample_by_index(index) for index in range(count)]
    audit = audit_batch(examples)

    manifest = {
        "output_path": str(output_path),
        "split": split,
        "count": count,
        "seed": effective_seed,
        "base_seed": base_seed,
        "seed_offset": seed_offset,
        "train_seed_range_start": train_start,
        "train_seed_range_end": train_end,
        "split_seed_range_start": split_seed_range_start,
        "split_seed_range_end": split_seed_range_end,
        "split_seed_overlap_with_train": overlaps_train,
        "rule_distribution": audit["rule_distribution"],
        "mode_distribution": audit["mode_distribution"],
        "audit_failures_count": audit["failed_examples_count"],
        "audit_failure_reasons": audit["failure_reasons"],
        "first_failed_examples": audit["first_failed_examples"],
    }
    return manifest, examples


def _manifest_path(output_path: Path) -> Path:
    return output_path.with_suffix(".manifest.json")


def _train_seed_range(config: dict[str, Any], base_seed: int) -> tuple[int, int]:
    generation = config.get("generation", {}) if isinstance(config, dict) else {}
    training = config.get("training", {}) if isinstance(config, dict) else {}
    samples_per_epoch = int(generation.get("samples_per_epoch", 0) or 0)
    epochs = int(training.get("epochs", 1) or 1)
    train_max_index = max(0, samples_per_epoch * epochs)
    return base_seed, base_seed + train_max_index


def _ranges_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    if left_end < left_start:
        return False
    return not (left_end < right_start or left_start > right_end)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
