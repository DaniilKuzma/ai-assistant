from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any

from src.config.load_config import load_config
from src.training.train import _build_features_with_metadata, _load_rows_for_split


def main() -> int:
    parser = argparse.ArgumentParser(description="Build cached training features without running model training.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config.")
    parser.add_argument("--split", action="append", help="Split to build. Can be repeated.")
    parser.add_argument("--splits", help="Comma-separated splits to build.")
    parser.add_argument("--limit", type=int, help="Maximum rows per split for this cache key.")
    parser.add_argument("--force", action="store_true", help="Rebuild even when a valid cache exists.")
    parser.add_argument("--profile", action="store_true", help="Enable feature build profiling.")
    parser.add_argument("--no-syntax", action="store_true", help="Disable syntax parsing for feature build.")
    parser.add_argument("--output-dir", help="Override feature cache output directory.")
    args = parser.parse_args()

    config = load_config(args.config)
    splits = _requested_splits(args)
    for split in splits:
        split_config = _config_for_split(config, split=split, limit=args.limit, args=args)
        rows = _load_rows_for_split(split_config, split=split, fallback_rows=[])
        if args.limit is not None:
            rows = rows[: args.limit]
        started = time.perf_counter()
        result = _build_features_with_metadata(
            split_config,
            rows,
            split=split,
            force_cache=bool(args.force),
            limit=args.limit,
        )
        elapsed = time.perf_counter() - started
        rows_per_sec = len(result.features) / elapsed if elapsed > 0 else 0.0
        print(
            f"{split}: features={len(result.features)} "
            f"cache_hit={result.cache_result.hit} "
            f"rows/sec={rows_per_sec:.3f} "
            f"cache_path={result.cache_result.path}"
        )
        profile_path = _profile_output_path(split_config)
        if profile_path.exists():
            print(f"{split}: profile={profile_path}")
            print(f"{split}: profile_summary={profile_path.with_name('feature_build_profile_summary.md')}")
    return 0


def _requested_splits(args: argparse.Namespace) -> list[str]:
    splits: list[str] = []
    if args.splits:
        splits.extend(split.strip() for split in args.splits.split(",") if split.strip())
    if args.split:
        splits.extend(args.split)
    return splits or ["train"]


def _config_for_split(config: dict[str, Any], *, split: str, limit: int | None, args: argparse.Namespace) -> dict[str, Any]:
    cloned = copy.deepcopy(config)
    training = cloned.setdefault("training", {})
    feature_build = training.setdefault("feature_build", {})
    feature_cache = training.setdefault("feature_cache", {})
    feature_cache["enabled"] = True
    if args.output_dir:
        feature_cache["cache_dir"] = args.output_dir
    if args.profile:
        feature_build["profile"] = True
    if args.no_syntax:
        feature_build["enable_syntax"] = False
    if limit is not None:
        training[_limit_key_for_split(split)] = int(limit)
    return cloned


def _profile_output_path(config: dict[str, Any]) -> Path:
    feature_build = config.get("training", {}).get("feature_build", {}) or {}
    if feature_build.get("profile_output"):
        return Path(feature_build["profile_output"])
    return Path(config.get("paths", {}).get("reports_dir", "reports")) / "feature_build_profile.csv"


def _limit_key_for_split(split: str) -> str:
    return {
        "train": "max_train_examples",
        "val": "max_val_examples",
        "test": "max_test_examples",
    }.get(split, f"max_{split}_examples")


if __name__ == "__main__":
    raise SystemExit(main())
