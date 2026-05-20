from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import time
from typing import Any, Callable

from src.rules.registry import all_rules
from src.training.tensorization import TrainingFeature


CACHE_VERSION = 3


@dataclass(frozen=True)
class FeatureCacheResult:
    features: list[TrainingFeature]
    enabled: bool
    hit: bool
    path: Path
    build_time_sec: float
    rebuilt_from_corrupt: bool = False

    @property
    def miss(self) -> bool:
        return self.enabled and not self.hit

    def report_metrics(self) -> dict[str, Any]:
        return {
            "feature_cache_enabled": self.enabled,
            "feature_cache_hit": self.hit,
            "feature_build_skipped": self.enabled and self.hit,
            "feature_cache_path": str(self.path) if self.path else "",
            "feature_build_time_sec": round(self.build_time_sec, 3),
            "features_count": len(self.features),
        }


def build_or_load_features(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    split: str,
    builder: Callable[[], list[TrainingFeature]],
    force: bool = False,
    limit: int | None = None,
) -> FeatureCacheResult:
    started = time.perf_counter()
    cache_config = config.get("training", {}).get("feature_cache", {}) or {}
    enabled = bool(cache_config.get("enabled", False))
    if not enabled:
        return FeatureCacheResult(
            features=builder(),
            enabled=False,
            hit=False,
            path=Path(""),
            build_time_sec=time.perf_counter() - started,
        )

    path = feature_cache_path(config, rows, split=split, limit=limit)
    compression = bool(cache_config.get("compression", False))
    if path.exists() and not force:
        try:
            payload = _read_payload(path, compression=compression)
            if payload.get("version") == CACHE_VERSION and payload.get("cache_key") == feature_cache_key(
                config,
                rows,
                split=split,
                limit=limit,
            ):
                return FeatureCacheResult(
                    features=list(payload["features"]),
                    enabled=True,
                    hit=True,
                    path=path,
                    build_time_sec=time.perf_counter() - started,
                )
        except Exception:
            features = builder()
            _write_payload(path, _payload(config, rows, split, limit, features), compression=compression)
            return FeatureCacheResult(
                features=features,
                enabled=True,
                hit=False,
                path=path,
                build_time_sec=time.perf_counter() - started,
                rebuilt_from_corrupt=True,
            )

    features = builder()
    _write_payload(path, _payload(config, rows, split, limit, features), compression=compression)
    return FeatureCacheResult(
        features=features,
        enabled=True,
        hit=False,
        path=path,
        build_time_sec=time.perf_counter() - started,
    )


def feature_cache_path(config: dict[str, Any], rows: list[dict[str, Any]], *, split: str, limit: int | None = None) -> Path:
    cache_config = config.get("training", {}).get("feature_cache", {}) or {}
    cache_dir = Path(cache_config.get("cache_dir") or "data/processed/features_cache")
    model_config = config.get("model", {})
    max_length = int(model_config.get("max_sequence_length", 192))
    max_candidates = int(model_config.get("max_candidates", 32))
    digest = feature_cache_key(config, rows, split=split, limit=limit)[:16]
    suffix = ".pkl.gz" if bool(cache_config.get("compression", False)) else ".pkl"
    return cache_dir / f"{split}_msl{max_length}_mc{max_candidates}_{digest}{suffix}"


def feature_cache_key(config: dict[str, Any], rows: list[dict[str, Any]], *, split: str, limit: int | None = None) -> str:
    payload = _cache_identity(config, rows, split=split, limit=limit)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _payload(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    split: str,
    limit: int | None,
    features: list[TrainingFeature],
) -> dict[str, Any]:
    return {
        "version": CACHE_VERSION,
        "cache_key": feature_cache_key(config, rows, split=split, limit=limit),
        "features": features,
    }


def _cache_identity(config: dict[str, Any], rows: list[dict[str, Any]], *, split: str, limit: int | None) -> dict[str, Any]:
    training_config = config.get("training", {})
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    dictionary_config = config.get("dictionary", {})
    configured_limit = limit if limit is not None else int(training_config.get(_limit_key_for_split(split), len(rows)))
    row_limit = min(int(configured_limit), len(rows))
    dataset_path_value = str(data_config.get("processed_train_path") or "")
    dataset_path = Path(dataset_path_value) if dataset_path_value else None
    return {
        "version": CACHE_VERSION,
        "dataset": _dataset_identity(dataset_path, bool(training_config.get("feature_cache", {}).get("include_dataset_hash", True))),
        "rows_hash": "" if dataset_path is not None and dataset_path.exists() else _rows_hash(rows),
        "split": split,
        "row_limit": row_limit,
        "max_sequence_length": int(model_config.get("max_sequence_length", 192)),
        "max_candidates": int(model_config.get("max_candidates", 32)),
        "tokenizer": {
            "primary_encoder": model_config.get("primary_encoder", ""),
            "fallback_encoder": model_config.get("fallback_encoder", ""),
            "local_files_only": bool(model_config.get("local_files_only", False)),
            "kind": _tokenizer_kind(config),
            "debug_tokenizer": _tokenizer_kind(config) == "debug",
        },
        "labels": config.get("labels", {}),
        "candidate_generator": {
            "dictionary": dictionary_config,
            "feature_build": training_config.get("feature_build", {}),
            "rules_hash": _rules_hash(),
        },
        "config_hash": _config_hash(config)
        if bool(training_config.get("feature_cache", {}).get("include_config_hash", True))
        else "",
    }


def _dataset_identity(path: Path | None, include_hash: bool) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"path": str(path) if path is not None else "", "exists": False}
    stat = path.stat()
    identity: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
    }
    if include_hash:
        identity["sha256"] = _file_hash(path)
    return identity


def _config_hash(config: dict[str, Any]) -> str:
    relevant = {
        "model": config.get("model", {}),
        "labels": config.get("labels", {}),
        "dictionary": config.get("dictionary", {}),
        "evaluation_feature_tokenizer": config.get("evaluation", {}).get("feature_tokenizer", ""),
        "training_feature_build": config.get("training", {}).get("feature_build", {}),
    }
    encoded = json.dumps(relevant, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tokenizer_kind(config: dict[str, Any]) -> str:
    if str(config.get("evaluation", {}).get("feature_tokenizer", "")).lower() == "model":
        return "model"
    return "model" if bool(config.get("training", {}).get("run_model_training", False)) else "debug"


def _rules_hash() -> str:
    specs = [
        {
            "id": rule.spec.id,
            "group": rule.spec.group,
            "scope": rule.spec.scope,
            "edit_type": rule.spec.edit_type,
            "mode": rule.spec.mode,
            "requires": rule.spec.requires,
        }
        for rule in all_rules()
    ]
    rules_path = Path("configs/rules.yaml")
    config_text = rules_path.read_text(encoding="utf-8") if rules_path.exists() else ""
    encoded = json.dumps({"specs": specs, "rules_yaml": config_text}, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rows_hash(rows: list[dict[str, Any]]) -> str:
    encoded = json.dumps(rows, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_payload(path: Path, *, compression: bool) -> dict[str, Any]:
    opener = gzip.open if compression else open
    with opener(path, "rb") as handle:
        return pickle.load(handle)


def _write_payload(path: Path, payload: dict[str, Any], *, compression: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if compression else open
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=path.parent) as temp_handle:
        temp_path = Path(temp_handle.name)
    try:
        with opener(temp_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _limit_key_for_split(split: str) -> str:
    return {
        "train": "max_train_examples",
        "val": "max_val_examples",
        "test": "max_test_examples",
    }.get(split, f"max_{split}_examples")
