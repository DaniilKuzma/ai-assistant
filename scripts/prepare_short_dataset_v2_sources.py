from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from src.candidates.candidate_generator import CandidateGenerator
from src.data.clean_sentence_pool import build_clean_sentence_pool
from src.data.real_error_sources import load_real_error_pairs
from src.data.sage_sources import (
    MANUAL_DOWNLOAD_COMMANDS,
    NerusCheck,
    SageMaterializeResult,
    downloads_enabled,
    prepare_punctuation_jsonl_file,
    prepare_sage_jsonl_files,
    verify_nerus_conllu,
)


OPEN_SOURCES_CONFIG = Path("configs/open_corpora_sources.yaml")
REAL_SOURCES_CONFIG = Path("configs/real_error_sources.yaml")
SHORT_V2_CONFIG = Path("configs/config.short_dataset_v2.yaml")
REPORTS_DIR = Path("reports/short_dataset_v2")
SAGE_DIR = Path("data/external/sage")
PUNCTUATION_JSONL = SAGE_DIR / "spellcheck_punctuation_benchmark.jsonl"
ROOT_REAL_PAIRS = Path("data/processed/real_error_pairs_validated.csv.gz")
SHORT_V2_REAL_PAIRS = Path("data/processed/short_dataset_v2/real_error_pairs_validated.csv.gz")
ROOT_CLEAN_POOL = Path("data/processed/clean_sentence_pool.csv.gz")


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SAGE_DIR.mkdir(parents=True, exist_ok=True)
    ensure_gitignore_entries()

    open_config = _source_prep_open_config(_read_yaml(OPEN_SOURCES_CONFIG))
    real_config = _source_prep_real_config(_read_yaml(REAL_SOURCES_CONFIG))
    env_name = str(
        open_config.get("sources", {})
        .get("download_policy", {})
        .get("allow_downloads_env", "RUSSIAN_CORRECTOR_ALLOW_SOURCE_DOWNLOADS")
    )
    enabled = downloads_enabled(env_name)

    clean_result = build_clean_sentence_pool(
        open_config,
        output_path=ROOT_CLEAN_POOL,
        reports_dir=REPORTS_DIR,
    )
    nerus_path = Path(open_config["clean_sources"]["nerus_news"]["local_path"])
    nerus_check = verify_nerus_conllu(nerus_path)
    append_nerus_check_to_source_report(REPORTS_DIR / "source_ingestion_report.md", nerus_check)

    sage_result = prepare_sage_jsonl_files(output_dir=SAGE_DIR, env_name=env_name)
    punctuation_count, punctuation_method = prepare_punctuation_jsonl_file(output_path=PUNCTUATION_JSONL, env_name=env_name)
    if punctuation_count > 0:
        real_config["real_sources"]["spellcheck_punctuation_benchmark"].update(
            {
                "type": "local_jsonl",
                "local_path": str(PUNCTUATION_JSONL),
                "max_raw_pairs": 8000,
                "max_pairs": 800,
                "cap_share": 0.30,
            }
        )
    real_config["_loader_method"] = sage_result.loader_method
    if punctuation_count > 0:
        real_config["_loader_method"] += f"+punctuation:{punctuation_method}"
    real_result = load_real_error_pairs(
        real_config,
        candidate_generator=CandidateGenerator(syntax_provider=lambda _text: ()),
        output_path=ROOT_REAL_PAIRS,
        reports_dir=REPORTS_DIR,
    )
    SHORT_V2_REAL_PAIRS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT_REAL_PAIRS, SHORT_V2_REAL_PAIRS)

    blocked_reasons = blocked_source_reasons(nerus_check, sage_result, real_result.accepted_count)
    if blocked_reasons:
        append_blocked_commands(REPORTS_DIR / "real_error_source_report.md", blocked_reasons)

    print_summary(
        downloads_are_enabled=enabled,
        nerus_check=nerus_check,
        sage_result=sage_result,
        accepted_real_pairs=real_result.accepted_count,
        rejected_real_pairs=real_result.rejected_count,
        blocked_reasons=blocked_reasons,
        clean_shortage_reason=clean_result.shortage_reason,
        punctuation_count=punctuation_count,
    )
    return 2 if blocked_reasons else 0


def ensure_gitignore_entries() -> None:
    path = Path(".gitignore")
    required = [
        "data/external/",
        "data/external/**",
        "data/processed/clean_sentence_pool*.csv.gz",
        "data/processed/real_error_pairs_validated*.csv.gz",
    ]
    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    missing = [line for line in required if line not in existing]
    if not missing:
        return
    with path.open("a", encoding="utf-8") as handle:
        if existing and existing[-1].strip():
            handle.write("\n")
        for line in missing:
            handle.write(line + "\n")


def append_nerus_check_to_source_report(path: Path, check: NerusCheck) -> None:
    lines = path.read_text(encoding="utf-8").rstrip().splitlines() if path.exists() else ["# Source Ingestion Report"]
    lines.extend(
        [
            "",
            "## Nerus Verification",
            "",
            f"- status: {check.status}",
            f"- reason: {check.reason}" if check.reason else "- reason:",
            f"- path: {check.path}",
            f"- size_bytes: {check.size_bytes}",
            f"- gzip_check: {'ok' if check.gzip_ok else 'failed'}",
            f"- text_comment_count: {check.text_comment_count}",
            f"- extracted_sample_sentence_count: {check.extracted_sample_sentence_count}",
            f"- accepted_sample_count: {check.accepted_sample_count}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def blocked_source_reasons(nerus_check: NerusCheck, sage_result: SageMaterializeResult, accepted_real_pairs: int) -> list[str]:
    reasons: list[str] = []
    if nerus_check.status != "ready":
        reasons.append(f"nerus:{nerus_check.reason or nerus_check.status}")
    if sage_result.missing_datasets:
        reasons.append(f"sage_missing:{','.join(sage_result.missing_datasets)}")
    if accepted_real_pairs < 1000:
        reasons.append(f"accepted_real_pairs_below_min:{accepted_real_pairs}<1000")
    return reasons


def append_blocked_commands(path: Path, reasons: list[str]) -> None:
    lines = path.read_text(encoding="utf-8").rstrip().splitlines() if path.exists() else ["# Real Error Source Report"]
    lines.extend(
        [
            "",
            "## Blocked Manual Recovery",
            "",
            f"- blocked_status: BLOCKED_BY_NO_INTERNET",
            f"- blocked_reasons: {json.dumps(reasons, ensure_ascii=False)}",
            "",
        ]
    )
    for command in MANUAL_DOWNLOAD_COMMANDS:
        lines.extend(["```bash", command, "```", ""])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def print_summary(
    *,
    downloads_are_enabled: bool,
    nerus_check: NerusCheck,
    sage_result: SageMaterializeResult,
    accepted_real_pairs: int,
    rejected_real_pairs: int,
    blocked_reasons: list[str],
    clean_shortage_reason: str,
    punctuation_count: int,
) -> None:
    print("downloads_enabled:", "yes" if downloads_are_enabled else "no")
    print("nerus_status:", nerus_check.status, nerus_check.reason)
    print("sage_status:", "ready" if sage_result.ready else "blocked")
    print("sage_loader_method:", sage_result.loader_method)
    print("sage_counts:", json.dumps(sage_result.written_counts, ensure_ascii=False, sort_keys=True))
    print("punctuation_pairs:", punctuation_count)
    print("accepted_real_pairs:", accepted_real_pairs)
    print("rejected_real_pairs:", rejected_real_pairs)
    print("clean_shortage_reason:", clean_shortage_reason)
    print("paths_written:")
    for path in [
        SAGE_DIR,
        ROOT_REAL_PAIRS,
        SHORT_V2_REAL_PAIRS,
        REPORTS_DIR / "source_ingestion_report.md",
        REPORTS_DIR / "real_error_source_report.md",
        REPORTS_DIR / "real_pair_filter_report.csv",
        REPORTS_DIR / "rejected_real_pairs.csv",
    ]:
        print("-", path)
    if blocked_reasons:
        print("final_verdict: BLOCKED")
        print("BLOCKED_BY_NO_INTERNET")
        print("blocked_reasons:", json.dumps(blocked_reasons, ensure_ascii=False))
    else:
        print("final_verdict: READY_FOR_REBUILD_SOURCES")


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _source_prep_open_config(open_config: dict[str, Any]) -> dict[str, Any]:
    config = dict(open_config)
    if not SHORT_V2_CONFIG.exists():
        return config
    short_config = _read_yaml(SHORT_V2_CONFIG)
    v2_pool = (
        short_config.get("data", {})
        .get("short_dataset_v2", {})
        .get("pool", {})
    )
    if v2_pool:
        pool = dict(config.get("pool", {}) or {})
        pool.update(v2_pool)
        config["pool"] = pool
    return config


def _source_prep_real_config(real_config: dict[str, Any]) -> dict[str, Any]:
    config = dict(real_config)
    real_sources = {name: dict(spec or {}) for name, spec in (config.get("real_sources", {}) or {}).items()}
    raw_limits = {
        "sage_ruspellru": 4000,
        "sage_multidomain_gold": 6000,
        "sage_medspellchecker": 1200,
        "sage_github_typo_ru": 1000,
    }
    accepted_limits = {
        "sage_ruspellru": 1200,
        "sage_multidomain_gold": 1200,
        "sage_medspellchecker": 200,
        "sage_github_typo_ru": 200,
    }
    for source_name, limit in raw_limits.items():
        if source_name in real_sources:
            real_sources[source_name]["max_raw_pairs"] = limit
    for source_name, limit in accepted_limits.items():
        if source_name in real_sources:
            real_sources[source_name]["max_pairs"] = limit
    config["real_sources"] = real_sources
    return config


if __name__ == "__main__":
    raise SystemExit(main())
