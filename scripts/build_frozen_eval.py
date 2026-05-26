from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.load_config import load_config
from src.grammar_gen.audit import audit_batch
from src.grammar_gen.diversity import diversity_report
from src.grammar_gen.factory import online_generator_from_config
from src.schema import GeneratedExample
from src.schema.serialization import read_jsonl_examples, write_jsonl_examples


SPLIT_SEED_OFFSETS = {
    "val": 10_000_000,
    "test": 20_000_000,
    "regression": 30_000_000,
}
MAX_DEDUP_SCAN_MULTIPLIER = 20
MIN_DEDUP_SCAN_BUDGET = 10_000


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
    deduper, dedupe_index = _build_text_deduper(config, base_seed=base_seed, split=split, output_path=output_path)

    generator = online_generator_from_config(config, seed=effective_seed)
    examples, scan_count, skipped_count, skipped_reasons = _sample_unique_examples(
        generator,
        count=count,
        deduper=deduper,
    )
    split_seed_range_end = effective_seed + scan_count - 1 if scan_count > 0 else effective_seed - 1
    overlaps_train = _ranges_overlap(split_seed_range_start, split_seed_range_end, train_start, train_end)
    audit = audit_batch(examples)
    diversity = diversity_report(examples)

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
        "candidate_scan_count": scan_count,
        "dedupe_skipped_count": skipped_count,
        "dedupe_skipped_reasons": dict(sorted(skipped_reasons.items())),
        "dedupe_indexed_train_examples_count": dedupe_index["train_examples_count"],
        "dedupe_indexed_existing_split_examples_count": dedupe_index["existing_split_examples_count"],
        "dedupe_indexed_existing_split_paths": dedupe_index["existing_split_paths"],
        "rule_distribution": audit["rule_distribution"],
        "mode_distribution": audit["mode_distribution"],
        "duplicate_source_rate": diversity["duplicate_source_rate"],
        "duplicate_target_rate": diversity["duplicate_target_rate"],
        "duplicate_pair_rate": diversity["duplicate_pair_rate"],
        "unique_source_target_pairs": diversity["unique_source_target_pairs"],
        "top_duplicate_pairs": diversity["top_duplicate_pairs"],
        "duplicate_rate_by_rule_id": diversity["duplicate_rate_by_rule_id"],
        "duplicate_rate_by_sub_rule_id": diversity["duplicate_rate_by_sub_rule_id"],
        "average_token_count": diversity["average_token_count"],
        "token_edit_count_distribution": diversity["token_edit_count_distribution"],
        "gap_edit_count_distribution": diversity["gap_edit_count_distribution"],
        "template_distribution": diversity["template_distribution"],
        "audit_failures_count": audit["failed_examples_count"],
        "audit_failure_reasons": audit["failure_reasons"],
        "first_failed_examples": audit["first_failed_examples"],
    }
    return manifest, examples


def _manifest_path(output_path: Path) -> Path:
    return output_path.with_suffix(".manifest.json")


class _TextDeduper:
    def __init__(self) -> None:
        self._seen_texts: set[str] = set()

    def add(self, example: GeneratedExample) -> None:
        self._seen_texts.add(_text_key(example.source_text))
        self._seen_texts.add(_text_key(example.target_text))

    def duplicate_reasons(self, example: GeneratedExample) -> list[str]:
        reasons: list[str] = []
        if _text_key(example.source_text) in self._seen_texts:
            reasons.append("source_text")
        target_key = _text_key(example.target_text)
        if target_key in self._seen_texts:
            reasons.append("target_text")
        return reasons


def _build_text_deduper(
    config: dict[str, Any],
    *,
    base_seed: int,
    split: str,
    output_path: Path,
) -> tuple[_TextDeduper, dict[str, Any]]:
    deduper = _TextDeduper()
    train_examples_count = _train_example_count(config)
    if train_examples_count > 0:
        train_generator = online_generator_from_config(config, seed=base_seed)
        for index in range(train_examples_count):
            deduper.add(train_generator.sample_by_index(index))

    existing_split_paths = list(_existing_split_paths(output_path, split))
    existing_split_examples_count = 0
    for path in existing_split_paths:
        examples = read_jsonl_examples(path)
        existing_split_examples_count += len(examples)
        for example in examples:
            deduper.add(example)

    return deduper, {
        "train_examples_count": train_examples_count,
        "existing_split_examples_count": existing_split_examples_count,
        "existing_split_paths": [str(path) for path in existing_split_paths],
    }


def _sample_unique_examples(
    generator: Any,
    *,
    count: int,
    deduper: _TextDeduper,
) -> tuple[list[GeneratedExample], int, int, Counter[str]]:
    examples: list[GeneratedExample] = []
    skipped_reasons: Counter[str] = Counter()
    skipped_count = 0
    scan_limit = _dedup_scan_limit(count)
    candidate_index = 0

    while len(examples) < count:
        if candidate_index >= scan_limit:
            raise RuntimeError(
                "Unable to build frozen eval split with unique source/target texts: "
                f"accepted={len(examples)}, requested={count}, scanned={candidate_index}, "
                f"skipped={dict(sorted(skipped_reasons.items()))}."
            )
        example = generator.sample_by_index(candidate_index)
        candidate_index += 1
        reasons = deduper.duplicate_reasons(example)
        if reasons:
            skipped_count += 1
            skipped_reasons.update(reasons)
            continue
        deduper.add(example)
        examples.append(example)

    return examples, candidate_index, skipped_count, skipped_reasons


def _existing_split_paths(output_path: Path, split: str) -> list[Path]:
    current = output_path.resolve()
    paths: list[Path] = []
    for previous_split in _previous_splits(split):
        path = output_path.parent / f"{previous_split}.jsonl"
        if not path.exists():
            continue
        if path.resolve() == current:
            continue
        paths.append(path)
    return paths


def _previous_splits(split: str) -> tuple[str, ...]:
    order = tuple(SPLIT_SEED_OFFSETS)
    if split not in SPLIT_SEED_OFFSETS:
        raise ValueError(f"Unknown frozen eval split: {split!r}")
    return order[: order.index(split)]


def _dedup_scan_limit(count: int) -> int:
    return max(count * MAX_DEDUP_SCAN_MULTIPLIER, count + MIN_DEDUP_SCAN_BUDGET)


def _text_key(text: str) -> str:
    return text.strip()


def _train_example_count(config: dict[str, Any]) -> int:
    generation = config.get("generation", {}) if isinstance(config, dict) else {}
    training = config.get("training", {}) if isinstance(config, dict) else {}
    samples_per_epoch = int(generation.get("samples_per_epoch", 0) or 0)
    epochs = int(training.get("epochs", 1) or 1)
    return max(0, samples_per_epoch * epochs)


def _train_seed_range(config: dict[str, Any], base_seed: int) -> tuple[int, int]:
    train_max_index = _train_example_count(config)
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
