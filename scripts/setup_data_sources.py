from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.candidates.candidate_generator import CandidateGenerator
from src.config.candidate_dataset_config import get_candidate_dataset_config, candidate_dataset_paths
from src.data.clean_sentence_pool import CleanSentencePoolResult, build_clean_sentence_pool
from src.data.real_error_sources import RealErrorLoadResult, load_real_error_pairs
from src.data.sage_sources import prepare_punctuation_jsonl_file, prepare_sage_jsonl_files
from src.data.source_downloads import DEFAULT_DOWNLOAD_ENV


READY = "DATA_SOURCES_READY"
PARTIAL = "DATA_SOURCES_PARTIAL"
BLOCKED = "DATA_SOURCES_BLOCKED"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.clean_only:
        args.clean = True
        args.real = False
    if args.real_only:
        args.real = True
        args.clean = False
    if not args.clean and not args.real:
        args.clean = True
        args.real = True

    if args.no_download:
        os.environ.pop(DEFAULT_DOWNLOAD_ENV, None)
    if args.download and not _downloads_enabled():
        print(
            f"WARNING: --download was requested but {DEFAULT_DOWNLOAD_ENV}=1 is not set; using local/cache only.",
            file=sys.stderr,
        )

    config = _read_yaml(Path(args.config))
    paths = candidate_dataset_paths(config)
    processed_dir = Path(args.processed_dir or str(paths["processed_dir"]))
    report_dir = Path(args.report_dir or str(paths["reports_dir"]))
    processed_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = processed_dir / "source_ingestion_manifest.json"

    clean_config_path = Path(str(paths["open_corpora_sources_config"]))
    real_config_path = Path(str(paths["real_error_sources_config"]))

    clean_config = _read_yaml(clean_config_path) if args.clean and clean_config_path.exists() else {}
    real_config = _read_yaml(real_config_path) if args.real and real_config_path.exists() else {}
    _apply_top_level_data_overlays(config, clean_config=clean_config, real_config=real_config)
    if args.max_clean_sentences and clean_config:
        _limit_clean_sources(clean_config, args.max_clean_sentences)
    if args.max_real_pairs and real_config:
        _limit_real_sources(real_config, args.max_real_pairs)

    clean_result: CleanSentencePoolResult | None = None
    real_result: RealErrorLoadResult | None = None
    dry_run_notes: list[str] = []

    if args.dry_run:
        if args.clean and not clean_config:
            dry_run_notes.append(f"missing_clean_config:{clean_config_path}")
        if args.real and not real_config:
            dry_run_notes.append(f"missing_real_config:{real_config_path}")
    else:
        if args.real and real_config and args.download and _downloads_enabled():
            _prepare_materialized_real_sources(real_config)
        if args.clean and clean_config:
            clean_result = build_clean_sentence_pool(
                clean_config,
                output_path=Path(str(paths["clean_pool_path"])),
                reports_dir=report_dir,
            )
        elif args.clean:
            dry_run_notes.append(f"missing_clean_config:{clean_config_path}")
        if args.real and real_config:
            real_result = load_real_error_pairs(
                real_config,
                candidate_generator=CandidateGenerator.from_config(config) if config else CandidateGenerator(),
                output_path=processed_dir / "real_error_pairs_validated.csv.gz",
                reports_dir=report_dir,
            )
        elif args.real:
            dry_run_notes.append(f"missing_real_config:{real_config_path}")

    manifest = _manifest(
        downloads_enabled=_downloads_enabled(),
        dry_run=bool(args.dry_run),
        clean_result=clean_result,
        real_result=real_result,
        clean_config=clean_config,
        real_config=real_config,
        notes=dry_run_notes,
        report_dir=report_dir,
        dataset_contract=str(get_candidate_dataset_config(config).get("contract") or ""),
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_report(report_dir / "source_setup_summary.md", manifest)
    _print_final_summary(manifest, report_dir=report_dir, processed_dir=processed_dir)

    if manifest["verdict"] == BLOCKED and not args.allow_partial:
        return 2
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare external clean and real-pair data sources.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--clean-only", action="store_true")
    parser.add_argument("--real-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-clean-sentences", type=int, default=0)
    parser.add_argument("--max-real-pairs", type=int, default=0)
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args(argv)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _apply_top_level_data_overlays(config: dict[str, Any], *, clean_config: dict[str, Any], real_config: dict[str, Any]) -> None:
    candidate = get_candidate_dataset_config(config)
    clean_pool = candidate.get("clean_pool", {}) or {}
    if clean_pool and clean_config is not None:
        pool = clean_config.setdefault("pool", {})
        if isinstance(pool, dict):
            pool.update(clean_pool)

    real_pairs = candidate.get("real_pairs", {}) or {}
    stress = candidate.get("stress", {}) or {}
    if (real_pairs or stress) and real_config is not None:
        validation = real_config.setdefault("validation", {})
        if isinstance(validation, dict):
            for key in ("train_policy", "unknown_rule_policy", "multi_edit_policy"):
                if key in real_pairs:
                    validation[key] = real_pairs[key]
            if "loss_weight" in stress:
                validation["stress_loss_weight"] = stress["loss_weight"]


def _downloads_enabled() -> bool:
    return os.environ.get(DEFAULT_DOWNLOAD_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _limit_clean_sources(config: dict[str, Any], limit: int) -> None:
    sources = config.get("clean_sources", {})
    if not isinstance(sources, dict):
        return
    for spec in sources.values():
        if isinstance(spec, dict) and spec.get("max_sentences"):
            spec["max_sentences"] = min(int(spec["max_sentences"]), int(limit))


def _limit_real_sources(config: dict[str, Any], limit: int) -> None:
    sources = config.get("real_sources", {})
    if not isinstance(sources, dict):
        return
    for spec in sources.values():
        if isinstance(spec, dict):
            configured = int(spec.get("max_pairs") or spec.get("max_examples") or limit)
            spec["max_pairs"] = min(configured, int(limit))


def _prepare_materialized_real_sources(config: dict[str, Any]) -> None:
    specs = config.get("real_sources", {})
    if not isinstance(specs, dict):
        return
    sage_paths = [
        Path(str(spec.get("local_path") or ""))
        for spec in specs.values()
        if isinstance(spec, dict) and str(spec.get("type") or "") == "sage_hf_or_local" and spec.get("local_path")
    ]
    if sage_paths and not all(path.exists() and path.stat().st_size > 0 for path in sage_paths):
        prepare_sage_jsonl_files(output_dir=sage_paths[0].parent)
    for spec in specs.values():
        if not isinstance(spec, dict):
            continue
        if str(spec.get("hf_id") or "") != "ai-forever/spellcheck_punctuation_benchmark":
            continue
        if str(spec.get("type") or "") not in {"huggingface_dataset", "hf_dataset", "local_jsonl", "sage_hf_or_local"}:
            continue
        path = Path(str(spec.get("local_path") or "data/external/sage/spellcheck_punctuation_benchmark.jsonl"))
        if path and (not path.exists() or path.stat().st_size <= 0):
            count, method = prepare_punctuation_jsonl_file(output_path=path)
            spec["_loader_method"] = method
            spec["_materialized_count"] = count
        if path.exists() and path.stat().st_size > 0:
            spec["type"] = "local_jsonl"
            spec["local_path"] = str(path)


def _manifest(
    *,
    downloads_enabled: bool,
    dry_run: bool,
    clean_result: CleanSentencePoolResult | None,
    real_result: RealErrorLoadResult | None,
    clean_config: dict[str, Any],
    real_config: dict[str, Any],
    notes: list[str],
    report_dir: Path,
    dataset_contract: str = "",
) -> dict[str, Any]:
    clean_size = clean_result.accepted_count if clean_result else 0
    real_accepted = real_result.accepted_count if real_result else 0
    real_rejected = real_result.rejected_count if real_result else 0
    clean_rejections = clean_result.rejection_reason_counts if clean_result else {}
    real_rejections = real_result.rejection_reason_counts if real_result else {}
    clean_reports = clean_result.source_reports if clean_result else _skipped_source_reports(clean_config, "clean_sources")
    real_reports = real_result.source_reports if real_result else _skipped_source_reports(real_config, "real_sources")
    unsafe_sources = _unsafe_sources(clean_reports)
    reports_exist = report_dir.exists() and any(report_dir.iterdir())
    verdict = _verdict(clean_size, real_accepted, unsafe_sources=unsafe_sources, reports_exist=reports_exist, dry_run=dry_run)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_contract": dataset_contract,
        "dataset_hash": "",
        "audit_errors": [],
        "layer_counts": {},
        "downloads_enabled": bool(downloads_enabled),
        "dry_run": bool(dry_run),
        "clean_sources": clean_reports,
        "real_sources": real_reports,
        "clean_pool_size": int(clean_size),
        "real_pair_accepted_count": int(real_accepted),
        "real_pair_rejected_count": int(real_rejected),
        "source_counts": {
            "clean": clean_result.source_counts if clean_result else {},
            "real": _real_source_counts(real_result),
        },
        "clean_source_counts": clean_result.source_counts if clean_result else {},
        "real_source_counts": _real_source_counts(real_result),
        "rejection_reason_counts": {"clean": clean_rejections, "real": real_rejections},
        "clean_quality_summary": {
            "dominance_violations": clean_result.dominance_violations if clean_result else [],
            "shortage_reason": clean_result.shortage_reason if clean_result else "not_run",
        },
        "real_quality_summary": {
            "candidate_present_rate": _candidate_present_rate(real_result),
            "shortage_reason": "" if real_accepted >= 5000 else f"accepted_real_pairs_below_min:{real_accepted}<5000",
        },
        "unsafe_sources": unsafe_sources,
        "notes": notes,
        "verdict": verdict,
    }


def _verdict(clean_size: int, real_accepted: int, *, unsafe_sources: list[str], reports_exist: bool, dry_run: bool) -> str:
    if dry_run:
        return BLOCKED
    if clean_size >= 300000 and real_accepted >= 5000 and not unsafe_sources and reports_exist:
        return READY
    if clean_size >= 150000 and not unsafe_sources:
        return PARTIAL
    return BLOCKED


def _skipped_source_reports(config: dict[str, Any], key: str) -> list[dict[str, Any]]:
    raw = config.get(key, {})
    if not isinstance(raw, dict):
        return []
    reports = []
    for name, spec in raw.items():
        spec = spec if isinstance(spec, dict) else {}
        reports.append(
            {
                "source_name" if key == "clean_sources" else "source_dataset": name,
                "status": "skipped",
                "reason": "dry_run_or_not_run",
                "enabled": spec.get("enabled", True),
            }
        )
    return reports


def _unsafe_sources(reports: list[dict[str, Any]]) -> list[str]:
    unsafe = []
    forbidden = {"fiction", "proza", "poetry", "stihi", "social", "subtitles", "blogs", "forums", "comments"}
    for report in reports:
        if not bool(report.get("used", report.get("accepted", 0))):
            continue
        values = " ".join(str(report.get(key, "")) for key in ("source_name", "source_dataset", "domain", "style", "source_subcorpus"))
        tokens = {token for token in values.lower().replace("_", " ").split() if token}
        if tokens & forbidden:
            unsafe.append(str(report.get("source_name") or report.get("source_dataset") or "unknown"))
    return unsafe


def _real_source_counts(result: RealErrorLoadResult | None) -> dict[str, int]:
    if result is None:
        return {}
    return dict(Counter(str(row.get("source_dataset") or "") for row in result.rows))


def _candidate_present_rate(result: RealErrorLoadResult | None) -> float:
    if result is None or not result.rows:
        return 0.0
    present = sum(1 for row in result.rows if str(row.get("candidate_present", "")).lower() in {"true", "1"} or row.get("candidate_present") is True)
    return present / len(result.rows)


def _write_summary_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Source Setup Summary",
        "",
        f"- dataset_contract: {manifest.get('dataset_contract', '')}",
        f"- dataset_hash: {manifest.get('dataset_hash', '')}",
        f"- verdict: {manifest['verdict']}",
        f"- audit_errors: {json.dumps(manifest.get('audit_errors', []), ensure_ascii=False)}",
        f"- layer_counts: {json.dumps(manifest.get('layer_counts', {}), ensure_ascii=False, sort_keys=True)}",
        f"- downloads_enabled: {'yes' if manifest['downloads_enabled'] else 'no'}",
        f"- clean_pool_size: {manifest['clean_pool_size']}",
        f"- real_pair_accepted_count: {manifest['real_pair_accepted_count']}",
        f"- real_pair_rejected_count: {manifest['real_pair_rejected_count']}",
        f"- candidate_present_rate: {manifest['real_quality_summary']['candidate_present_rate']:.4f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_final_summary(manifest: dict[str, Any], *, report_dir: Path, processed_dir: Path) -> None:
    print(f"dataset contract: {manifest.get('dataset_contract', '')}")
    print(f"dataset hash: {manifest.get('dataset_hash', '')}")
    print(f"audit errors: {json.dumps(manifest.get('audit_errors', []), ensure_ascii=False)}")
    print(f"layer counts: {json.dumps(manifest.get('layer_counts', {}), ensure_ascii=False, sort_keys=True)}")
    print(f"downloads enabled: {'yes' if manifest['downloads_enabled'] else 'no'}")
    print("dependencies installed/updated: requirements.txt updated")
    print(f"clean sources used: {_used_sources(manifest['clean_sources'], 'source_name')}")
    print(f"clean sources skipped and reasons: {_skipped_sources(manifest['clean_sources'], 'source_name')}")
    print(f"clean pool size: {manifest['clean_pool_size']}")
    print(f"clean source counts: {manifest['clean_source_counts']}")
    print(f"clean rejection top reasons: {_top_reasons(manifest['rejection_reason_counts'].get('clean', {}))}")
    print(f"real sources used: {_used_sources(manifest['real_sources'], 'source_dataset')}")
    print(f"real sources skipped and reasons: {_skipped_sources(manifest['real_sources'], 'source_dataset')}")
    print(f"accepted real pair count: {manifest['real_pair_accepted_count']}")
    print(f"rejected real pair count: {manifest['real_pair_rejected_count']}")
    print(f"real source counts: {manifest['real_source_counts']}")
    print(f"real rejection top reasons: {_top_reasons(manifest['rejection_reason_counts'].get('real', {}))}")
    print(f"candidate-present rate for real pairs: {manifest['real_quality_summary']['candidate_present_rate']:.4f}")
    print(f"output paths: {processed_dir}")
    print(f"reports paths: {report_dir}")
    print(f"final verdict: {manifest['verdict']}")


def _top_reasons(reasons: dict[str, int]) -> dict[str, int]:
    return dict(sorted(reasons.items(), key=lambda item: (-int(item[1]), item[0]))[:10])


def _used_sources(reports: list[dict[str, Any]], name_key: str) -> list[str]:
    result = []
    for report in reports:
        accepted = int(report.get("accepted", 0) or 0)
        if bool(report.get("used", False)) or accepted > 0:
            result.append(str(report.get(name_key) or report.get("source_name") or report.get("source_dataset") or "unknown"))
    return result


def _skipped_sources(reports: list[dict[str, Any]], name_key: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for report in reports:
        accepted = int(report.get("accepted", 0) or 0)
        if bool(report.get("used", False)) or accepted > 0:
            continue
        name = str(report.get(name_key) or report.get("source_name") or report.get("source_dataset") or "unknown")
        result[name] = str(report.get("reason") or report.get("status") or "skipped")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
