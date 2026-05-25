from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.candidate_dataset_config import (
    CANONICAL_DATASET_CONFIG_PATH,
    candidate_dataset_audit,
    candidate_dataset_paths,
    candidate_dataset_rule_activation,
    candidate_dataset_rule_quota,
    candidate_dataset_totals,
    get_candidate_dataset_config,
    validate_candidate_dataset_config,
)
from src.config.load_config import load_config
from src.data.clean_sentence_pool import (
    clean_sentence_rejection_reasons,
    has_latin_confusable_inside_cyrillic_word,
    mixed_script_tokens,
)
from src.data.training_quality_audit import report_manifest_errors
from src.rules.capabilities import (
    activation_policy_from_config,
    capability_manifest_fields,
    expanded_activation_blocked_rows,
    load_rule_capabilities,
)


REQUIRED_CONTRACT_COLUMNS = {
    "dataset_contract",
    "dataset_layer",
    "gold_edit_count",
    "count_toward_rule_quota",
    "loss_weight",
    "target_rule_id",
    "verification_status",
}
REQUIRED_MANIFEST_FIELDS = {
    "dataset_hash",
    "config_hash",
    "layer_counts",
    "production_ready_rule_ids",
    "training_candidate_rule_ids",
    "active_rule_ids",
    "activation_stage_counts",
}
MIN_PRODUCTION_READY_RULE_COUNT = 12
MIN_TRAINING_CANDIDATE_RULE_COUNT = 43
MIN_FINAL_ACTIVE_RULE_COUNT = 43
TARGET_TRAINING_CANDIDATE_RULE_COUNT = 69
TARGET_FINAL_ACTIVE_RULE_COUNT = 69


def _activation_thresholds(activation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "expected_min_production_ready_rule_count": _activation_int(
            activation,
            "expected_min_production_ready_rule_count",
            MIN_PRODUCTION_READY_RULE_COUNT,
        ),
        "expected_min_training_candidate_rule_count": _activation_int(
            activation,
            "expected_min_training_candidate_rule_count",
            MIN_TRAINING_CANDIDATE_RULE_COUNT,
        ),
        "expected_min_final_active_rule_count": _activation_int(
            activation,
            "expected_min_final_active_rule_count",
            MIN_FINAL_ACTIVE_RULE_COUNT,
        ),
        "target_training_candidate_rule_count": _activation_int(
            activation,
            "target_training_candidate_rule_count",
            TARGET_TRAINING_CANDIDATE_RULE_COUNT,
        ),
        "target_final_active_rule_count": _activation_int(
            activation,
            "target_final_active_rule_count",
            TARGET_FINAL_ACTIVE_RULE_COUNT,
        ),
        "fail_below_min_training_candidate_rule_count": _activation_bool(
            activation,
            "fail_below_min_training_candidate_rule_count",
            True,
        ),
        "warn_below_target_training_candidate_rule_count": _activation_bool(
            activation,
            "warn_below_target_training_candidate_rule_count",
            True,
        ),
        "fail_below_final_active_rule_count": _activation_bool(
            activation,
            "fail_below_final_active_rule_count",
            True,
        ),
        "warn_below_target_final_active_rule_count": _activation_bool(
            activation,
            "warn_below_target_final_active_rule_count",
            True,
        ),
    }


@dataclass(frozen=True)
class PreflightOptions:
    config_path: Path | str = Path("configs/config.yaml")
    processed_dir: Path | str = Path("data/processed")
    reports_dir: Path | str = Path("reports/dataset_build")
    clean_stale_artifacts: bool = False
    scan_clean_pool_limit: int = 50_000
    full_clean_pool_scan: bool = False
    block_stale_artifacts: bool = True


def run_preflight(options: PreflightOptions) -> dict[str, Any]:
    config_path = Path(options.config_path)
    processed_dir = Path(options.processed_dir)
    reports_dir = Path(options.reports_dir)
    config = load_config(config_path)

    errors: list[str] = []
    warnings: list[str] = []
    cleaned_paths: list[str] = []

    config_validation_errors = validate_candidate_dataset_config(config)
    errors.extend(config_validation_errors)

    dependency_summary = check_dependencies()
    errors.extend(dependency_summary["errors"])

    config_summary = check_config(config)
    errors.extend(config_summary["errors"])
    warnings.extend(config_summary["warnings"])

    activation_summary = check_activation(config)
    errors.extend(activation_summary["errors"])
    warnings.extend(activation_summary["warnings"])

    stale_artifacts = check_stale_artifacts(processed_dir=processed_dir, reports_dir=reports_dir)
    if options.clean_stale_artifacts:
        cleaned_paths = clean_stale_artifacts(processed_dir=processed_dir, reports_dir=reports_dir)
        stale_artifacts = {
            **stale_artifacts,
            "cleaned": True,
            "after_clean": check_stale_artifacts(processed_dir=processed_dir, reports_dir=reports_dir),
        }
    elif stale_artifacts["stale"]:
        stale_message = "stale_generated_artifacts_detected"
        if options.block_stale_artifacts:
            errors.append(stale_message)
        else:
            warnings.append(stale_message)

    clean_pool_path = _clean_pool_path(config, processed_dir)
    clean_pool_summary = scan_clean_pool(
        clean_pool_path=clean_pool_path,
        config=config,
        row_limit=options.scan_clean_pool_limit,
        full_scan=options.full_clean_pool_scan,
    )
    errors.extend(clean_pool_summary["errors"])
    warnings.extend(clean_pool_summary["warnings"])

    errors = _dedupe(errors)
    warnings = _dedupe(warnings)
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "dependency_summary": dependency_summary,
        "config_summary": config_summary,
        "activation_summary": activation_summary,
        "stale_artifacts": stale_artifacts,
        "clean_pool_summary": clean_pool_summary,
        "cleaned_paths": cleaned_paths,
    }


def check_dependencies() -> dict[str, Any]:
    has_pymorphy3 = importlib.util.find_spec("pymorphy3") is not None
    has_pymorphy2 = importlib.util.find_spec("pymorphy2") is not None
    has_morphology = has_pymorphy3 or has_pymorphy2
    has_natasha = importlib.util.find_spec("natasha") is not None
    errors: list[str] = []
    messages: list[str] = []
    if not has_morphology:
        errors.append("missing_morphology_dependency")
        messages.append("missing_morphology_dependency: install project requirements with `pip install -r requirements.txt`.")
    if not has_natasha:
        errors.append("missing_syntax_dependency")
        messages.append("missing_syntax_dependency: install project requirements with `pip install -r requirements.txt`.")
    return {
        "ok": not errors,
        "errors": errors,
        "messages": messages,
        "pymorphy3": has_pymorphy3,
        "pymorphy2": has_pymorphy2,
        "morphology": has_morphology,
        "natasha": has_natasha,
    }


def check_config(config: Mapping[str, Any]) -> dict[str, Any]:
    candidate = get_candidate_dataset_config(config)
    clean_pool = dict(candidate.get("clean_pool", {}) or {})
    activation = candidate_dataset_rule_activation(config)
    quota = candidate_dataset_rule_quota(config)
    totals = candidate_dataset_totals(config)
    errors: list[str] = validate_candidate_dataset_config(config)
    warnings: list[str] = []

    _expect_equal(errors, "data.candidate_opportunity.contract", candidate.get("contract"), "candidate_opportunity")
    _expect_equal(errors, "data.candidate_opportunity.clean_pool.reject_mixed_script_tokens", clean_pool.get("reject_mixed_script_tokens"), True)
    _expect_equal(
        errors,
        "data.candidate_opportunity.clean_pool.reject_latin_confusable_inside_cyrillic_word",
        clean_pool.get("reject_latin_confusable_inside_cyrillic_word"),
        True,
    )
    _expect_equal(
        errors,
        "data.candidate_opportunity.clean_pool.reject_if_candidate_generator_finds_high_confidence_fix",
        clean_pool.get("reject_if_candidate_generator_finds_high_confidence_fix"),
        True,
    )
    _expect_float(
        errors,
        "data.candidate_opportunity.clean_pool.high_confidence_candidate_threshold",
        clean_pool.get("high_confidence_candidate_threshold"),
        0.95,
    )
    _expect_equal(errors, "data.candidate_opportunity.rule_activation.mode", activation.get("mode"), "expanded_safe")
    thresholds = _activation_thresholds(activation)
    _expect_min(
        errors,
        "data.candidate_opportunity.rule_activation.expected_min_production_ready_rule_count",
        thresholds["expected_min_production_ready_rule_count"],
        0,
    )
    _expect_min(
        errors,
        "data.candidate_opportunity.rule_activation.expected_min_training_candidate_rule_count",
        thresholds["expected_min_training_candidate_rule_count"],
        0,
    )
    _expect_min(
        errors,
        "data.candidate_opportunity.rule_activation.expected_min_final_active_rule_count",
        thresholds["expected_min_final_active_rule_count"],
        0,
    )
    _target_check(
        errors,
        warnings,
        activation,
        key="target_training_candidate_rule_count",
        fail_key="fail_below_target_training_candidate_rule_count",
        warn_key="warn_below_target_training_candidate_rule_count",
        minimum=thresholds["expected_min_training_candidate_rule_count"],
        default=TARGET_TRAINING_CANDIDATE_RULE_COUNT,
    )
    _target_check(
        errors,
        warnings,
        activation,
        key="target_final_active_rule_count",
        fail_key="fail_below_target_final_active_rule_count",
        warn_key="warn_below_target_final_active_rule_count",
        minimum=thresholds["expected_min_final_active_rule_count"],
        default=TARGET_FINAL_ACTIVE_RULE_COUNT,
    )
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "canonical_config_path": CANONICAL_DATASET_CONFIG_PATH,
        "dataset_contract": candidate.get("contract"),
        "totals": totals,
        "rule_quota": quota,
        "clean_pool": clean_pool,
        "rule_activation": activation,
        "rule_data_compiler": dict(candidate.get("rule_data_compiler", {}) or {}),
        "rule_lab": dict(candidate.get("rule_lab", {}) or {}),
    }


def check_activation(config: Mapping[str, Any]) -> dict[str, Any]:
    capabilities = load_rule_capabilities(ROOT_DIR / "configs" / "rules.yaml", config=config)
    policy = activation_policy_from_config(config)
    fields = capability_manifest_fields(capabilities, policy=policy)
    blocked_rows = expanded_activation_blocked_rows(capabilities, policy=policy, config=config)
    blocker_counts = Counter(str(row.get("blocker") or "unknown") for row in blocked_rows)
    errors = list(fields.get("activation_policy", {}).get("errors", []) or [])
    warnings = list(fields.get("activation_policy", {}).get("warnings", []) or [])
    activation = candidate_dataset_rule_activation(config)
    thresholds = _activation_thresholds(activation)

    production_count = int(fields.get("production_ready_rule_count", 0) or 0)
    training_count = int(fields.get("training_candidate_rule_count", 0) or 0)
    final_count = int(fields.get("final_active_rule_count", fields.get("active_rule_count", 0)) or 0)
    min_production = int(thresholds["expected_min_production_ready_rule_count"])
    min_training = int(thresholds["expected_min_training_candidate_rule_count"])
    target_training = int(thresholds["target_training_candidate_rule_count"])
    min_final = int(thresholds["expected_min_final_active_rule_count"])
    target_final = int(thresholds["target_final_active_rule_count"])
    if production_count < min_production:
        errors.append(f"production_ready_rule_count_below_min:{production_count}<{min_production}")
    if training_count < min_training and bool(thresholds["fail_below_min_training_candidate_rule_count"]):
        errors.append(f"training_candidate_rule_count_below_min:{training_count}<{min_training}")
    if (
        target_training > 0
        and training_count < target_training
        and bool(thresholds["warn_below_target_training_candidate_rule_count"])
    ):
        warnings.append(f"training_candidate_rule_count_below_target:{training_count}<{target_training}")
    if final_count < min_final and bool(thresholds["fail_below_final_active_rule_count"]):
        errors.append(f"final_active_rule_count_below_min:{final_count}<{min_final}")
    if target_final > 0 and final_count < target_final and bool(thresholds["warn_below_target_final_active_rule_count"]):
        warnings.append(f"final_active_rule_count_below_target:{final_count}<{target_final}")

    return {
        "ok": not errors,
        "errors": _dedupe(errors),
        "warnings": _dedupe(warnings),
        "production_ready_rule_count": production_count,
        "training_candidate_rule_count": training_count,
        "active_rule_count": int(fields.get("active_rule_count", 0) or 0),
        "final_active_rule_count": final_count,
        "production_ready_rule_ids": list(fields.get("production_ready_rule_ids", []) or []),
        "training_candidate_rule_ids": list(fields.get("training_candidate_rule_ids", []) or []),
        "active_rule_ids": list(fields.get("active_rule_ids", []) or []),
        "blockers": dict(sorted(blocker_counts.items())),
        "missing_module_counts": {
            "dictionary": int(fields.get("missing_dictionary_rule_count", 0) or 0),
            "syntax": int(fields.get("missing_syntax_rule_count", 0) or 0),
            "morphology": int(fields.get("missing_morphology_rule_count", 0) or 0),
            "ner": int(fields.get("missing_ner_rule_count", 0) or 0),
        },
        "activation_policy": fields.get("activation_policy", {}),
    }


def check_stale_artifacts(processed_dir: Path | str, reports_dir: Path | str) -> dict[str, Any]:
    processed = Path(processed_dir)
    reports = Path(reports_dir)
    dataset_path = processed / "correction_dataset.csv.gz"
    manifest_path = processed / "dataset_manifest.json"
    reasons: list[str] = []
    missing_contract_columns: list[str] = []
    report_errors: list[str] = []
    manifest: dict[str, Any] = {}

    if not dataset_path.exists() and not manifest_path.exists():
        return {
            "stale": False,
            "reasons": [],
            "dataset_path": str(dataset_path),
            "manifest_path": str(manifest_path),
            "report_errors": [],
            "missing_contract_columns": [],
        }

    if dataset_path.exists():
        try:
            header = pd.read_csv(dataset_path, nrows=0).columns
            missing_contract_columns = sorted(REQUIRED_CONTRACT_COLUMNS - set(str(column) for column in header))
            if missing_contract_columns:
                reasons.append("stale_dataset_missing_contract_columns")
        except Exception as exc:
            reasons.append(f"stale_dataset_header_unreadable:{type(exc).__name__}")

    if not manifest_path.exists():
        reasons.append("stale_manifest_missing")
    else:
        manifest = _read_json(manifest_path)
        if not manifest:
            reasons.append("stale_manifest_invalid")
        for field in sorted(REQUIRED_MANIFEST_FIELDS):
            if field not in manifest:
                reasons.append(f"stale_manifest_missing_{field}")
        if "source_type_counts" in manifest and "layer_counts" not in manifest:
            reasons.append("stale_manifest_source_type_only_counts")
        train_rule_counts = dict(dict(manifest.get("rule_id_counts_by_split", {}) or {}).get("train", {}) or {})
        flat_rule_counts = dict(manifest.get("rule_id_counts", {}) or {})
        if int(train_rule_counts.get("unknown", 0) or 0) > 0 or (
            not train_rule_counts and int(flat_rule_counts.get("unknown", 0) or 0) > 0
        ):
            reasons.append("stale_manifest_unknown_rule_in_train")

    dataset_hash = str(manifest.get("dataset_hash") or "")
    config_hash = str(manifest.get("config_hash") or "")
    if dataset_hash and config_hash:
        report_errors = report_manifest_errors(reports, dataset_hash=dataset_hash, config_hash=config_hash)
        reasons.extend(report_errors)
    elif dataset_path.exists() or manifest_path.exists():
        report_manifest = reports / "report_manifest.json"
        if not report_manifest.exists():
            reasons.append("stale_reports_manifest_missing")

    reasons = _dedupe(reasons)
    return {
        "stale": bool(reasons),
        "reasons": reasons,
        "dataset_path": str(dataset_path),
        "manifest_path": str(manifest_path),
        "report_errors": report_errors,
        "missing_contract_columns": missing_contract_columns,
    }


def clean_stale_artifacts(processed_dir: Path | str, reports_dir: Path | str) -> list[str]:
    processed = Path(processed_dir).resolve()
    reports = Path(reports_dir).resolve()
    targets: list[Path] = []
    for name in ("correction_dataset.csv.gz", "dataset_manifest.json", "train.csv", "val.csv", "test.csv"):
        targets.append(processed / name)
    for pattern in ("train_*.csv.gz", "val_*.csv.gz", "test_*.csv.gz"):
        targets.extend(processed.glob(pattern))
    features_cache = processed / "features_cache"
    if features_cache.exists() and features_cache.is_dir():
        targets.extend(features_cache.iterdir())
    if reports.exists() and reports.is_dir():
        targets.extend(reports.iterdir())

    cleaned: list[str] = []
    for target in sorted(set(targets)):
        resolved = target.resolve()
        if _is_under(resolved, processed) or _is_under(resolved, reports):
            if not resolved.exists():
                continue
            if resolved.is_dir():
                shutil.rmtree(resolved)
            else:
                resolved.unlink()
            cleaned.append(str(resolved))
    return cleaned


def scan_clean_pool(
    clean_pool_path: Path | str,
    config: Mapping[str, Any],
    *,
    row_limit: int = 50_000,
    full_scan: bool = False,
) -> dict[str, Any]:
    path = Path(clean_pool_path)
    if not path.exists():
        return {
            "exists": False,
            "path": str(path),
            "scanned_rows": 0,
            "full_scan": bool(full_scan),
            "mixed_script_token_count": 0,
            "latin_confusable_inside_cyrillic_word_count": 0,
            "high_confidence_autocorrection_candidate_count": 0,
            "high_confidence_check_enabled": False,
            "errors": [],
            "warnings": [],
            "message": "",
        }

    pool_config = dict(get_candidate_dataset_config(config).get("clean_pool", {}) or {})
    high_confidence_enabled = bool(full_scan and pool_config.get("reject_if_candidate_generator_finds_high_confidence_fix", False))
    candidate_generator = None
    if high_confidence_enabled:
        from src.candidates.candidate_generator import CandidateGenerator

        candidate_generator = CandidateGenerator.from_config(config)

    scanned = 0
    mixed_count = 0
    confusable_count = 0
    high_confidence_count = 0
    limit = None if full_scan else max(0, int(row_limit))
    for chunk in pd.read_csv(path, chunksize=10_000):
        if limit is not None and scanned >= limit:
            break
        if limit is not None:
            chunk = chunk.head(limit - scanned)
        for text in _text_values_from_clean_pool_chunk(chunk):
            scanned += 1
            if mixed_script_tokens(text):
                mixed_count += 1
            if has_latin_confusable_inside_cyrillic_word(text):
                confusable_count += 1
            if high_confidence_enabled:
                reasons = clean_sentence_rejection_reasons(
                    text,
                    {},
                    pool_config=pool_config,
                    candidate_generator=candidate_generator,
                )
                if "high_confidence_autocorrection_candidate" in reasons:
                    high_confidence_count += 1

    errors: list[str] = []
    warnings: list[str] = []
    message = ""
    if mixed_count > 0 or confusable_count > 0:
        code = "clean_pool_contamination_detected"
        message = "clean pool should be rebuilt via setup_data_sources"
        if bool(candidate_dataset_audit(config).get("fail_on_clean_pool_contamination", False)):
            errors.append(code)
        else:
            warnings.append(code)
    return {
        "exists": True,
        "path": str(path),
        "scanned_rows": scanned,
        "full_scan": bool(full_scan),
        "mixed_script_token_count": mixed_count,
        "latin_confusable_inside_cyrillic_word_count": confusable_count,
        "high_confidence_autocorrection_candidate_count": high_confidence_count,
        "high_confidence_check_enabled": high_confidence_enabled,
        "errors": errors,
        "warnings": warnings,
        "message": message,
    }


def print_preflight_summary(result: Mapping[str, Any]) -> None:
    activation = dict(result.get("activation_summary", {}) or {})
    stale = dict(result.get("stale_artifacts", {}) or {})
    clean = dict(result.get("clean_pool_summary", {}) or {})
    config_summary = dict(result.get("config_summary", {}) or {})
    totals = dict(config_summary.get("totals", {}) or {})
    quota = dict(config_summary.get("rule_quota", {}) or {})
    rule_activation = dict(config_summary.get("rule_activation", {}) or {})
    compiler = dict(config_summary.get("rule_data_compiler", {}) or {})
    rule_lab = dict(config_summary.get("rule_lab", {}) or {})
    summary = {
        "ok": bool(result.get("ok")),
        "canonical_config_path": config_summary.get("canonical_config_path", CANONICAL_DATASET_CONFIG_PATH),
        "error_count": len(result.get("errors", []) or []),
        "warning_count": len(result.get("warnings", []) or []),
        "dependencies_ok": bool(dict(result.get("dependency_summary", {}) or {}).get("ok")),
        "total_examples": totals.get("total_examples", 0),
        "train_examples": totals.get("train_examples", 0),
        "val_examples": totals.get("val_examples", 0),
        "test_examples": totals.get("test_examples", 0),
        "min_atomic_positives_per_active_rule": quota.get("min_atomic_positives_per_active_rule", 0),
        "preferred_atomic_positives_per_active_rule": quota.get("preferred_atomic_positives_per_active_rule", 0),
        "max_total_per_rule_id": quota.get("max_total_per_rule_id", 0),
        "min_hard_negatives_per_active_rule": quota.get("min_hard_negatives_per_active_rule", 0),
        "expected_min_final_active_rule_count": rule_activation.get("expected_min_final_active_rule_count", 0),
        "target_final_active_rule_count": rule_activation.get("target_final_active_rule_count", 0),
        "rule_data_compiler.enabled": bool(compiler.get("enabled", False)),
        "rule_lab.enabled": bool(rule_lab.get("enabled", False)),
        "production_ready_rule_count": activation.get("production_ready_rule_count", 0),
        "training_candidate_rule_count": activation.get("training_candidate_rule_count", 0),
        "final_active_rule_count": activation.get("final_active_rule_count", 0),
        "stale_artifacts": bool(stale.get("stale", False)),
        "clean_pool_scanned_rows": clean.get("scanned_rows", 0),
        "clean_pool_contamination": int(clean.get("mixed_script_token_count", 0) or 0)
        + int(clean.get("latin_confusable_inside_cyrillic_word_count", 0) or 0),
    }
    print("[candidate-preflight] " + json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    for message in dict(result.get("dependency_summary", {}) or {}).get("messages", []) or []:
        print(f"[candidate-preflight] {message}", flush=True)
    for warning in result.get("warnings", []) or []:
        print(f"[candidate-preflight] warning: {warning}", flush=True)
    for error in result.get("errors", []) or []:
        print(f"[candidate-preflight] error: {error}", flush=True)
    if stale.get("stale"):
        print(
            "[candidate-preflight] stale_reasons: "
            + json.dumps(stale.get("reasons", []), ensure_ascii=False, sort_keys=True),
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preflight checks for candidate_opportunity dataset generation.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--clean-stale-artifacts", action="store_true")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--reports-dir", default="reports/dataset_build")
    parser.add_argument("--scan-clean-pool-limit", type=int, default=50_000)
    parser.add_argument("--full-clean-pool-scan", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)

    result = run_preflight(
        PreflightOptions(
            config_path=args.config,
            processed_dir=args.processed_dir,
            reports_dir=args.reports_dir,
            clean_stale_artifacts=args.clean_stale_artifacts,
            scan_clean_pool_limit=args.scan_clean_pool_limit,
            full_clean_pool_scan=args.full_clean_pool_scan,
        )
    )
    if args.json_output:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print_preflight_summary(result)
    return 0 if result["ok"] else 1


def _clean_pool_path(config: Mapping[str, Any], processed_dir: Path) -> Path:
    raw_path = candidate_dataset_paths(config).get("clean_pool_path")
    if raw_path:
        return Path(str(raw_path))
    return processed_dir / "clean_sentence_pool.csv.gz"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _expect_equal(errors: list[str], path: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        errors.append(f"config_{path}_expected_{expected!r}_actual_{actual!r}")


def _expect_float(errors: list[str], path: str, actual: Any, expected: float) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError):
        errors.append(f"config_{path}_expected_{expected}_actual_{actual!r}")
        return
    if abs(value - expected) > 1e-12:
        errors.append(f"config_{path}_expected_{expected}_actual_{actual!r}")


def _expect_min(errors: list[str], path: str, actual: Any, minimum: int) -> None:
    try:
        value = int(actual)
    except (TypeError, ValueError):
        errors.append(f"config_{path}_below_min:{actual!r}<{minimum}")
        return
    if value < minimum:
        errors.append(f"config_{path}_below_min:{value}<{minimum}")


def _activation_int(activation: Mapping[str, Any], key: str, default: int) -> int:
    try:
        return max(0, int(activation.get(key, default) or 0))
    except (TypeError, ValueError):
        return max(0, int(default))


def _activation_bool(activation: Mapping[str, Any], key: str, default: bool) -> bool:
    raw = activation.get(key, default)
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"", "0", "false", "no", "off", "none", "null", "nan"}:
        return False
    if text in {"1", "true", "yes", "on"}:
        return True
    return bool(raw)


def _target_check(
    errors: list[str],
    warnings: list[str],
    activation: Mapping[str, Any],
    *,
    key: str,
    fail_key: str,
    warn_key: str,
    minimum: int,
    default: int,
) -> None:
    try:
        value = int(activation.get(key, default) or 0)
    except (TypeError, ValueError):
        errors.append(f"config_data.rule_activation.{key}_below_target:{activation.get(key)!r}<{minimum}")
        return
    if value >= minimum:
        return
    code = f"config_data.rule_activation.{key}_below_target:{value}<{minimum}"
    if bool(activation.get(fail_key, False)):
        errors.append(code)
    elif bool(activation.get(warn_key, True)):
        warnings.append(code)
    else:
        warnings.append(code)


def _text_values_from_clean_pool_chunk(chunk: pd.DataFrame) -> list[str]:
    if "text" in chunk.columns:
        values = chunk["text"]
    elif "source" in chunk.columns:
        values = chunk["source"]
    else:
        values = chunk.iloc[:, 0] if len(chunk.columns) else []
    return [str(value) for value in values.fillna("").tolist()]


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
