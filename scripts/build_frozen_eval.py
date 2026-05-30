from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.load_config import load_config
from src.grammar_gen.audit import audit_batch
from src.grammar_gen.diversity import diversity_report
from src.grammar_gen.factory import online_generator_from_config
from src.grammar_gen.rules.base import GenerationMode
from src.schema import GeneratedExample
from src.schema.serialization import read_jsonl_examples, write_jsonl_examples


_TRAIN_DEDUPER_CACHE: dict[str, frozenset[str]] = {}


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
    parser.add_argument("--split", choices=tuple(SPLIT_SEED_OFFSETS))
    parser.add_argument("--count", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()

    explicit_args = (args.split, args.count, args.output)
    if any(value is not None for value in explicit_args) and not all(value is not None for value in explicit_args):
        parser.error("--split, --count, and --output must be provided together")

    if args.count is not None and args.count < 0:
        parser.error("--count must be non-negative")

    if all(value is not None for value in explicit_args):
        return _build_and_write_split(
            config_path=Path(args.config),
            split=str(args.split),
            count=int(args.count),
            output_path=Path(str(args.output)),
        )

    config = load_config(args.config)
    outputs = _configured_split_outputs(config)
    failed = False
    for split, count, output_path in outputs:
        if count < 0:
            parser.error(f"configured {split}_examples must be non-negative")
        result = _build_and_write_split(
            config_path=Path(args.config),
            split=split,
            count=count,
            output_path=output_path,
        )
        failed = failed or result != 0
    return 1 if failed else 0


def _build_and_write_split(
    *,
    config_path: Path,
    split: str,
    count: int,
    output_path: Path,
) -> int:
    manifest, examples = build_frozen_eval(
        config_path=config_path,
        split=split,
        count=count,
        output_path=output_path,
    )
    manifest_path = _manifest_path(output_path)
    _write_manifest(manifest_path, manifest)

    if manifest["audit_failures_count"] > 0:
        print(f"audit failures: {manifest['audit_failures_count']}", file=sys.stderr)
        print(f"manifest: {manifest_path}")
        return 1

    write_jsonl_examples(output_path, examples)
    print(f"wrote: {output_path}")
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
    generator = online_generator_from_config(config, seed=effective_seed)
    coverage_deduper, coverage_index = _build_text_deduper(
        config,
        base_seed=base_seed,
        split=split,
        output_path=output_path,
        include_train=False,
    )
    coverage_examples, coverage_required, coverage_missing = _sample_coverage_examples(
        generator,
        count=count,
        deduper=coverage_deduper,
        seed_start=effective_seed + 500_000,
    )

    deduper, dedupe_index = _build_text_deduper(
        config,
        base_seed=base_seed,
        split=split,
        output_path=output_path,
    )
    for example in coverage_examples:
        deduper.add(example)
    remaining = count - len(coverage_examples)
    examples, scan_count, skipped_count, skipped_reasons = _sample_unique_examples(
        generator,
        count=remaining,
        deduper=deduper,
    )
    examples = [*coverage_examples, *examples]
    fill_seed_end = effective_seed + scan_count - 1 if scan_count > 0 else effective_seed - 1
    coverage_seed_end = effective_seed + 500_000 + len(coverage_examples) - 1 if coverage_examples else effective_seed - 1
    split_seed_range_end = max(fill_seed_end, coverage_seed_end)
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
        "coverage_required_rule_ids": coverage_required,
        "coverage_missing_rule_ids": coverage_missing,
        "coverage_examples_count": len(coverage_examples),
        "coverage_train_text_dedupe_bypassed": bool(coverage_examples and coverage_index["train_examples_count"] == 0),
        "rule_distribution": audit["rule_distribution"],
        "layer_distribution": audit["layer_distribution"],
        "family_distribution": audit["family_distribution"],
        "mode_distribution": audit["mode_distribution"],
        "duplicate_source_rate": diversity["duplicate_source_rate"],
        "duplicate_target_rate": diversity["duplicate_target_rate"],
        "duplicate_pair_rate": diversity["duplicate_pair_rate"],
        "unique_source_target_pairs": diversity["unique_source_target_pairs"],
        "top_duplicate_pairs": diversity["top_duplicate_pairs"],
        "duplicate_rate_by_layer": diversity["duplicate_rate_by_layer"],
        "duplicate_rate_by_rule_id": diversity["duplicate_rate_by_rule_id"],
        "duplicate_rate_by_sub_rule_id": diversity["duplicate_rate_by_sub_rule_id"],
        "sub_rule_distribution": diversity["sub_rule_distribution"],
        "average_token_count": diversity["average_token_count"],
        "token_edit_count_distribution": diversity["token_edit_count_distribution"],
        "gap_edit_count_distribution": diversity["gap_edit_count_distribution"],
        "template_distribution": diversity["template_distribution"],
        "context_style_bucket_distribution": diversity["context_style_bucket_distribution"],
        "context_style_bucket_shares": diversity["context_style_bucket_shares"],
        "audit_failures_count": audit["failed_examples_count"],
        "audit_failure_reasons": audit["failure_reasons"],
        "first_failed_examples": audit["first_failed_examples"],
    }
    return manifest, examples


def _configured_split_outputs(config: Mapping[str, Any]) -> list[tuple[str, int, Path]]:
    generation = config.get("generation", {}) if isinstance(config, Mapping) else {}
    frozen_eval = generation.get("frozen_eval", {}) if isinstance(generation, Mapping) else {}
    if not isinstance(frozen_eval, Mapping):
        frozen_eval = {}
    output_dir = _configured_output_dir(config, frozen_eval)
    return [
        ("val", int(frozen_eval.get("val_examples", 0) or 0), output_dir / "val.jsonl"),
        ("test", int(frozen_eval.get("test_examples", 0) or 0), output_dir / "test.jsonl"),
        (
            "regression",
            int(frozen_eval.get("regression_examples", 0) or 0),
            output_dir / "regression.jsonl",
        ),
    ]


def _configured_output_dir(config: Mapping[str, Any], frozen_eval: Mapping[str, Any]) -> Path:
    raw = frozen_eval.get("output_dir")
    if raw is None:
        paths = config.get("paths", {}) if isinstance(config, Mapping) else {}
        raw = paths.get("generated_eval_dir", "data/generated_eval") if isinstance(paths, Mapping) else "data/generated_eval"
    path = Path(str(raw))
    return path if path.is_absolute() else PROJECT_ROOT / path


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

    def add_texts(self, texts: frozenset[str]) -> None:
        self._seen_texts.update(texts)

    def snapshot(self) -> frozenset[str]:
        return frozenset(self._seen_texts)


def _build_text_deduper(
    config: dict[str, Any],
    *,
    base_seed: int,
    split: str,
    output_path: Path,
    include_train: bool = True,
) -> tuple[_TextDeduper, dict[str, Any]]:
    deduper = _TextDeduper()
    train_examples_count = _train_example_count(config) if include_train else 0
    if train_examples_count > 0:
        cache_key = _train_deduper_cache_key(config, base_seed, train_examples_count)
        cached_texts = _TRAIN_DEDUPER_CACHE.get(cache_key)
        if cached_texts is None:
            train_deduper = _TextDeduper()
            train_generator = online_generator_from_config(config, seed=base_seed)
            for index in range(train_examples_count):
                train_deduper.add(train_generator.sample_by_index(index))
            cached_texts = train_deduper.snapshot()
            _TRAIN_DEDUPER_CACHE[cache_key] = cached_texts
        deduper.add_texts(cached_texts)

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


def _train_deduper_cache_key(config: Mapping[str, Any], base_seed: int, train_examples_count: int) -> str:
    relevant_config = {
        "base_seed": base_seed,
        "train_examples_count": train_examples_count,
        "generation": config.get("generation", {}) if isinstance(config, Mapping) else {},
        "training_epochs": (config.get("training", {}) or {}).get("epochs") if isinstance(config, Mapping) else None,
    }
    return json.dumps(relevant_config, ensure_ascii=False, sort_keys=True, default=str)


def _sample_coverage_examples(
    generator: Any,
    *,
    count: int,
    deduper: _TextDeduper,
    seed_start: int,
) -> tuple[list[GeneratedExample], list[str], list[str]]:
    required = _coverage_rule_ids(generator)
    if count <= 0 or not required:
        return [], list(required), list(required)

    quota = _coverage_examples_per_rule(getattr(generator, "config", None), required_count=len(required), count=count)
    examples: list[GeneratedExample] = []
    missing: list[str] = []
    coverage_index = 0
    for rule_id in required:
        accepted_for_rule = 0
        while accepted_for_rule < quota and len(examples) < count:
            accepted = _sample_unique_rule_example(generator, rule_id, deduper)
            if accepted is None:
                break
            accepted = _with_coverage_metadata(
                accepted,
                generation_index=coverage_index,
                generation_seed=seed_start + coverage_index,
            )
            deduper.add(accepted)
            examples.append(accepted)
            accepted_for_rule += 1
            coverage_index += 1
        if accepted_for_rule == 0:
            missing.append(rule_id)
    return examples, list(required), missing


def _coverage_rule_ids(generator: Any) -> tuple[str, ...]:
    registry = getattr(generator, "registry", None)
    config = getattr(generator, "config", None)
    if registry is None or config is None or not hasattr(registry, "enabled_rules"):
        return ()
    excluded = _sampling_excluded_rule_ids(config)
    rule_ids = [
        rule.info.rule_id
        for rule in registry.enabled_rules(config)
        if rule.info.rule_id not in excluded and _coverage_modes(rule)
    ]
    return tuple(sorted(dict.fromkeys(rule_ids)))


def _sample_unique_rule_example(
    generator: Any,
    rule_id: str,
    deduper: _TextDeduper,
) -> GeneratedExample | None:
    rule = getattr(getattr(generator, "registry", None), "get_rule", lambda _rule_id: None)(rule_id)
    modes = _coverage_modes(rule)
    if not modes:
        return None
    for _ in range(100):
        for mode in modes:
            example = generator.sample(rule_id=rule_id, mode=mode)
            if deduper.duplicate_reasons(example):
                continue
            return example
    return None


def _coverage_modes(rule: Any) -> tuple[GenerationMode, ...]:
    if rule is None:
        return ()
    modes = tuple(
        mode
        for mode in (GenerationMode.POSITIVE, GenerationMode.HARD_NEGATIVE, GenerationMode.CLEAN_IDENTITY)
        if rule.can_generate(mode)
    )
    return modes


def _coverage_examples_per_rule(config: Any, *, required_count: int, count: int) -> int:
    generation = config.get("generation", {}) if isinstance(config, dict) else {}
    frozen_eval = generation.get("frozen_eval", {}) if isinstance(generation, dict) else {}
    raw_quota = frozen_eval.get("coverage_examples_per_rule", 3) if isinstance(frozen_eval, dict) else 3
    try:
        quota = int(raw_quota)
    except (TypeError, ValueError):
        quota = 3
    if required_count <= 0:
        return 0
    return max(1, min(max(1, quota), max(1, count // required_count)))


def _with_coverage_metadata(
    example: GeneratedExample,
    *,
    generation_index: int,
    generation_seed: int,
) -> GeneratedExample:
    data = example.to_dict()
    metadata = dict(data.get("metadata") or {})
    metadata["generation_index"] = generation_index
    metadata["generation_seed"] = generation_seed
    metadata["coverage_required_rule"] = True
    data["metadata"] = metadata
    return GeneratedExample.from_dict(data)


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


def _sampling_excluded_rule_ids(config: Any) -> frozenset[str]:
    generation = config.get("generation", {}) if isinstance(config, dict) else {}
    raw = generation.get("sampling_exclude_rule_ids", ()) if isinstance(generation, dict) else ()
    if raw is None:
        return frozenset()
    if isinstance(raw, str):
        return frozenset({raw})
    if isinstance(raw, (list, tuple, set)):
        return frozenset(str(item) for item in raw if str(item).strip())
    return frozenset()


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
