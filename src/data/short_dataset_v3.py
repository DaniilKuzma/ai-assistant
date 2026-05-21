from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from src.data.short_dataset_v2 import build_short_dataset_v2_from_config


V3_VERDICT_READY = "READY_FOR_SHORT_TRAINING_DATASET_V3"


def build_short_dataset_v3_from_config(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    """Build the canonical Wave 1 v3 dataset through the vetted v2 builder path."""

    prepared = _as_v2_compatible_config(config)
    result = build_short_dataset_v2_from_config(prepared, force=force)
    _upgrade_manifest_to_v3(prepared, result, fallback_used=False, fallback_reason="")
    if _should_try_fallback(prepared, result):
        fallback = _fallback_config(prepared)
        result = build_short_dataset_v2_from_config(fallback, force=True)
        _upgrade_manifest_to_v3(
            fallback,
            result,
            fallback_used=True,
            fallback_reason="100k requested dataset did not pass build/quality gates with available source constraints",
        )
    return _upgrade_result(result)


def _as_v2_compatible_config(config: dict[str, Any]) -> dict[str, Any]:
    cloned = copy.deepcopy(config)
    data = cloned.setdefault("data", {})
    v3 = dict(data.get("short_dataset_v3", {}) or {})
    if "short_dataset_v2" not in data:
        v2 = dict(v3)
        v2["enabled"] = True
        data["short_dataset_v2"] = v2
    data["config_path"] = data.get("config_path") or "configs/config.short_dataset_v3.yaml"
    return cloned


def _fallback_config(config: dict[str, Any]) -> dict[str, Any]:
    cloned = copy.deepcopy(config)
    data = cloned.setdefault("data", {})
    v3 = data.get("short_dataset_v3", {}) or {}
    split_sizes = dict(v3.get("fallback_split_sizes", {}) or {"train": 50000, "val": 5000, "test": 5000})
    total = sum(int(value) for value in split_sizes.values())
    data["target_total_examples"] = total
    data["total_examples"] = total
    data["train_examples"] = int(split_sizes.get("train", 0))
    data["val_examples"] = int(split_sizes.get("val", 0))
    data["test_examples"] = int(split_sizes.get("test", 0))
    data["exact_split_sizes"] = split_sizes
    data["short_dataset_v2"] = _scaled_v2_config(data.get("short_dataset_v2", {}), total=total, split_sizes=split_sizes)
    return cloned


def _scaled_v2_config(v2_config: dict[str, Any], *, total: int, split_sizes: dict[str, int]) -> dict[str, Any]:
    scaled = copy.deepcopy(v2_config)
    scale = total / max(1, int(scaled.get("expected_total") or 100000))
    scaled["expected_total"] = total
    source_targets = scaled.get("source_type_targets", {}) or {}
    scaled["source_type_targets"] = {key: int(round(int(value) * scale)) for key, value in source_targets.items()}
    split_targets = scaled.get("split_source_type_targets", {}) or {}
    if split_targets:
        new_targets: dict[str, dict[str, int]] = {}
        for split, size in split_sizes.items():
            source_total = sum(int(value) for value in source_targets.values()) or total
            ratio = int(size) / max(1, source_total)
            new_targets[split] = {key: int(round(int(value) * ratio)) for key, value in source_targets.items()}
        scaled["split_source_type_targets"] = new_targets
    return scaled


def _upgrade_manifest_to_v3(
    config: dict[str, Any],
    result: dict[str, Any],
    *,
    fallback_used: bool,
    fallback_reason: str,
) -> None:
    manifest_path_value = result.get("manifest_path") or config.get("data", {}).get("manifest_path")
    if not manifest_path_value:
        return
    manifest_path = Path(str(manifest_path_value))
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    data = config.get("data", {})
    manifest.update(
        {
            "dataset_version": "short_dataset_v3",
            "requested_total": int(data.get("target_total_examples", manifest.get("requested_total", 0) or 0)),
            "actual_total": int(result.get("total", manifest.get("total", 0) or 0)),
            "requested_split_sizes": dict(data.get("exact_split_sizes", {}) or {}),
            "actual_split_sizes": manifest.get("split_sizes", result.get("splits", {})),
            "fallback_used": bool(fallback_used),
            "fallback_reason": fallback_reason,
            "source_constraints": _source_constraints(config, manifest),
        }
    )
    if manifest.get("verdict") == "READY_FOR_SHORT_TRAINING_DATASET_V2":
        manifest["verdict"] = V3_VERDICT_READY
        manifest["final_verdict"] = V3_VERDICT_READY
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _source_constraints(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    v3 = config.get("data", {}).get("short_dataset_v3", {}) or {}
    return {
        "requested_total": v3.get("requested_total", config.get("data", {}).get("target_total_examples")),
        "fallback_total": v3.get("fallback_total"),
        "real_pair_shortage_reason": manifest.get("real_pair_shortage_reason", ""),
        "missing_external_sources": manifest.get("missing_external_sources", []),
    }


def _should_try_fallback(config: dict[str, Any], result: dict[str, Any]) -> bool:
    if bool(result.get("fallback_used")):
        return False
    data = config.get("data", {})
    v3 = data.get("short_dataset_v3", {}) or {}
    if not v3.get("fallback_split_sizes"):
        return False
    if result.get("status") == "blocked":
        return True
    return result.get("verdict") not in {"READY_FOR_SHORT_TRAINING_DATASET_V2", V3_VERDICT_READY}


def _upgrade_result(result: dict[str, Any]) -> dict[str, Any]:
    upgraded = dict(result)
    if upgraded.get("verdict") == "READY_FOR_SHORT_TRAINING_DATASET_V2":
        upgraded["verdict"] = V3_VERDICT_READY
    return upgraded
