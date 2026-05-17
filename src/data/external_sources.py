from __future__ import annotations

import csv
from dataclasses import asdict
import gzip
from pathlib import Path
import json
import re
from typing import Any

from src.preprocessing.protected_spans import find_protected_spans
from src.validation.diff_analyzer import DiffAnalyzer, Edit
from src.validation.edit_classifier import coarse_error_type, is_allowed_edit_type, is_context_dependent_pair


DEFAULT_HF_JSONL_SOURCES = [
    {
        "repo": "RussianNLP/RuSpellGold",
        "files": [
            "data/complete_test/test.json",
        ],
    },
    {
        "repo": "ai-forever/spellcheck_punctuation_benchmark",
        "files": [
            "data/RUSpellRU/train.json",
            "data/RUSpellRU/test.json",
            "data/MultidomainGold/train.json",
            "data/MultidomainGold/test.json",
        ],
    },
]

SOURCE_COLUMNS = ("source", "input", "input_text", "incorrect", "erroneous", "original")
TARGET_COLUMNS = ("target", "target_text", "correction", "correct", "corrected", "output")
RUSSIAN_RE = re.compile(r"[А-Яа-яЁё]")


def load_hf_jsonl_pairs(
    source_specs: list[dict[str, Any]] | None = None,
    limit: int | None = None,
    *,
    local_files_only: bool = False,
) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    rows: list[dict[str, Any]] = []
    specs = source_specs or DEFAULT_HF_JSONL_SOURCES
    for spec in specs:
        repo = spec["repo"]
        for filename in spec.get("files", []):
            path = hf_hub_download(repo, filename, repo_type="dataset", local_files_only=local_files_only)
            rows.extend(
                load_local_jsonl_pairs(
                    path,
                    source_dataset=f"{repo}:{filename}",
                    limit=None if limit is None else limit - len(rows),
                )
            )
            if limit is not None and len(rows) >= limit:
                return rows[:limit]
    return rows


def load_external_pair_sources(
    source_specs: list[dict[str, Any]],
    *,
    limit: int | None = None,
    local_files_only: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for spec in source_specs:
        if not spec.get("enabled", True):
            continue
        remaining = None if limit is None else max(0, limit - len(rows))
        if remaining == 0:
            break
        try:
            loaded_rows = _load_external_pair_source(
                spec,
                limit=None if remaining is None else max(remaining * 2, remaining + 500),
                local_files_only=local_files_only,
            )
        except Exception:
            continue
        for row in loaded_rows:
            key = (row["source"], row["target"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                return rows
    return rows[:limit]


def load_local_jsonl_pairs(path: str | Path, source_dataset: str, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            source = str(item.get("source") or "").strip()
            target = str(item.get("correction") or item.get("target") or "").strip()
            row = _row_from_pair(
                source,
                target,
                source_dataset=source_dataset,
                domain=str(item.get("domain") or "external"),
            )
            if row is None:
                continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_local_table_pairs(
    path: str | Path,
    source_dataset: str,
    limit: int | None = None,
    *,
    trusted_word_pairs: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    delimiter = "\t" if str(path).endswith((".tsv", ".tab")) else ","
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for item in reader:
            source = _first_existing_value(item, SOURCE_COLUMNS)
            target = _first_existing_value(item, TARGET_COLUMNS)
            if trusted_word_pairs:
                row = _trusted_word_pair_row(source, target, source_dataset)
            else:
                row = _row_from_pair(source, target, source_dataset=source_dataset, domain="external")
            if row is None:
                continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_local_m2_pairs(path: str | Path, source_dataset: str, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, target in _m2_pairs(path):
        row = _row_from_pair(source, target, source_dataset=source_dataset, domain="m2")
        if row is None:
            continue
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def load_github_typo_corpus(path: str | Path, source_dataset: str, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            for edit in item.get("edits", []):
                source_info = edit.get("src", {})
                target_info = edit.get("tgt", {})
                source = str(source_info.get("text") or "").strip()
                target = str(target_info.get("text") or "").strip()
                language = str(source_info.get("lang") or target_info.get("lang") or "")
                if language and language != "rus":
                    continue
                if not RUSSIAN_RE.search(source + target):
                    continue
                row = _row_from_pair(source, target, source_dataset=source_dataset, domain="github_typo_corpus")
                if row is None:
                    continue
                rows.append(row)
                if limit is not None and len(rows) >= limit:
                    return rows
    return rows


def _load_external_pair_source(
    spec: dict[str, Any],
    *,
    limit: int | None,
    local_files_only: bool,
) -> list[dict[str, Any]]:
    source_type = str(spec.get("type", "jsonl"))
    name = str(spec.get("name") or spec.get("source_dataset") or source_type)
    source_dataset = str(spec.get("source_dataset") or name)
    source_limit = _source_limit(spec, limit)

    if source_type == "hf_jsonl":
        hf_specs = [{"repo": spec["repo"], "files": spec.get("files", [])}]
        return load_hf_jsonl_pairs(hf_specs, limit=source_limit, local_files_only=local_files_only)

    path = _resolve_source_path(spec, local_files_only=local_files_only)
    if path is None:
        return []

    if source_type in {"jsonl", "local_jsonl"}:
        return load_local_jsonl_pairs(path, source_dataset=source_dataset, limit=source_limit)
    if source_type in {"csv", "tsv", "table"}:
        return load_local_table_pairs(
            path,
            source_dataset=source_dataset,
            limit=source_limit,
            trusted_word_pairs=bool(spec.get("trusted_word_pairs", False)),
        )
    if source_type == "m2":
        return load_local_m2_pairs(path, source_dataset=source_dataset, limit=source_limit)
    if source_type in {"github_typo_jsonl", "github_typo_jsonl_gz"}:
        return load_github_typo_corpus(path, source_dataset=source_dataset, limit=source_limit)
    return []


def _resolve_source_path(spec: dict[str, Any], *, local_files_only: bool) -> Path | None:
    raw_path = spec.get("path")
    if raw_path:
        path = Path(raw_path)
        if path.exists():
            return path
    if local_files_only:
        return None
    url = spec.get("url")
    if not url:
        return None
    path = Path(raw_path or Path("data/raw/external") / _download_filename(str(url)))
    if path.exists():
        return path
    if _download_url(str(url), path):
        return path
    return None


def _download_url(url: str, output_path: Path) -> bool:
    import subprocess

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "4",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "15",
        "--max-time",
        "180",
        "-o",
        str(output_path),
        url,
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        if output_path.exists() and output_path.stat().st_size == 0:
            output_path.unlink()
        return False
    return output_path.exists() and output_path.stat().st_size > 0


def _download_filename(url: str) -> str:
    candidate = url.rstrip("/").rsplit("/", 1)[-1]
    if not candidate or candidate == "file_downloaded":
        return "external_downloaded_source"
    return candidate


def _source_limit(spec: dict[str, Any], global_limit: int | None) -> int | None:
    configured = spec.get("max_examples")
    if configured is None:
        return global_limit
    value = int(configured)
    return value if global_limit is None else min(value, global_limit)


def _row_from_pair(source: str, target: str, *, source_dataset: str, domain: str) -> dict[str, Any] | None:
    source = str(source or "").strip()
    target = str(target or "").strip()
    if not source or not target or source == target:
        return None
    if not RUSSIAN_RE.search(source + target):
        return None
    if len(source) > 350 or len(target) > 350:
        return None

    diff_analyzer = DiffAnalyzer()
    analyzed_edits = diff_analyzer.analyze(source, target)
    supported_edits = [edit for edit in analyzed_edits if _is_supported_external_edit(source, edit)]
    if not supported_edits:
        return None

    strict_target = _apply_edits_to_source(source, supported_edits)
    if strict_target == source:
        return None
    strict_edits = [edit for edit in diff_analyzer.analyze(source, strict_target) if _is_supported_external_edit(source, edit)]
    if not strict_edits:
        return None
    if _apply_edits_to_source(source, strict_edits) != strict_target:
        return None
    return _real_row(source, strict_target, strict_edits, source_dataset, domain)


def _trusted_word_pair_row(source: str, target: str, source_dataset: str) -> dict[str, Any] | None:
    source = str(source or "").strip()
    target = str(target or "").strip()
    if not source or not target or source == target:
        return None
    if not _is_single_russian_word(source) or not _is_single_russian_word(target):
        return None
    if "ё" in source.lower() or "ё" in target.lower():
        return None
    edit = Edit(source, target, "spelling_replace", 0, len(source), confidence=0.85)
    return _real_row(source, target, [edit], source_dataset, "external_word_pair")


def _is_supported_external_edit(source_text: str, edit: Edit) -> bool:
    if not is_allowed_edit_type(edit.edit_type):
        return False
    if is_context_dependent_pair(edit.source, edit.replacement):
        return False
    if edit.start < 0 or edit.end < edit.start:
        return False
    if _edit_touches_protected_span(source_text, edit):
        return False
    return True


def _edit_touches_protected_span(source_text: str, edit: Edit) -> bool:
    if edit.start < 0:
        return True
    start = edit.start
    end = edit.end if edit.end > edit.start else edit.start + 1
    for span in find_protected_spans(source_text):
        if start < span.end and span.start < end:
            return True
    return False


def _apply_edits_to_source(source: str, edits: list[Edit]) -> str:
    result = source
    for edit in sorted(edits, key=lambda item: (item.start, item.end), reverse=True):
        if edit.start < 0 or edit.end < edit.start or edit.start > len(result):
            continue
        result = result[: edit.start] + edit.replacement + result[edit.end :]
    return result


def _m2_pairs(path: str | Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    current: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            current.append(line)
            continue
        if current:
            pair = _m2_block_pair(current)
            if pair:
                pairs.append(pair)
            current = []
    if current:
        pair = _m2_block_pair(current)
        if pair:
            pairs.append(pair)
    return pairs


def _m2_block_pair(block: list[str]) -> tuple[str, str] | None:
    source_line = next((line for line in block if line.startswith("S ")), "")
    if not source_line:
        return None
    source_tokens = source_line[2:].split()
    edits: list[tuple[int, int, list[str]]] = []
    for line in block:
        if not line.startswith("A "):
            continue
        fields = line[2:].split("|||")
        if len(fields) < 3:
            continue
        start_end = fields[0].split()
        if len(start_end) != 2:
            continue
        correction = fields[2].strip()
        replacement_tokens = [] if correction in {"", "-NONE-"} else correction.split()
        edits.append((int(start_end[0]), int(start_end[1]), replacement_tokens))
    if not edits:
        return None
    target_tokens = list(source_tokens)
    for start, end, replacement_tokens in sorted(edits, reverse=True):
        target_tokens[start:end] = replacement_tokens
    source = _detokenize(source_tokens)
    target = _detokenize(target_tokens)
    return (source, target) if source != target else None


def _detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    replacements = {
        " ,": ",",
        " .": ".",
        " !": "!",
        " ?": "?",
        " :": ":",
        " ;": ";",
        "( ": "(",
        " )": ")",
        " « ": " «",
        " »": "»",
        " - ": "-",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    return re.sub(r"\s+", " ", text).strip()


def _first_existing_value(item: dict[str, Any], names: tuple[str, ...]) -> str:
    lowered = {str(key).lower(): value for key, value in item.items()}
    for name in names:
        value = lowered.get(name)
        if value is not None:
            return str(value).strip()
    return ""


def _is_single_russian_word(value: str) -> bool:
    return bool(re.fullmatch(r"[А-Яа-яЁё-]+", value))


def _deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["source"], row["target"])
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _real_row(source: str, target: str, edits: list[Edit], source_dataset: str, domain: str) -> dict[str, Any]:
    return {
        "source": source,
        "target": target,
        "error_types": json.dumps(sorted({coarse_error_type(edit.edit_type) for edit in edits}), ensure_ascii=False),
        "source_dataset": source_dataset,
        "is_clean": False,
        "is_synthetic": False,
        "split": "train",
        "domain": domain,
        "edit_operations": json.dumps([asdict(edit) for edit in edits], ensure_ascii=False),
    }
