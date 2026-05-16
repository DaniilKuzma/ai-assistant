from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from src.validation.diff_analyzer import DiffAnalyzer, Edit
from src.validation.edit_classifier import coarse_error_type, is_allowed_edit_type


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
            rows.extend(load_local_jsonl_pairs(path, source_dataset=f"{repo}:{filename}", limit=None if limit is None else limit - len(rows)))
            if limit is not None and len(rows) >= limit:
                return rows[:limit]
    return rows


def load_local_jsonl_pairs(path: str | Path, source_dataset: str, limit: int | None = None) -> list[dict[str, Any]]:
    diff_analyzer = DiffAnalyzer()
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            source = str(item.get("source") or "").strip()
            target = str(item.get("correction") or item.get("target") or "").strip()
            if not source or not target or source == target:
                continue
            edits = [edit for edit in diff_analyzer.analyze(source, target) if is_allowed_edit_type(edit.edit_type)]
            if not edits:
                continue
            if any(not is_allowed_edit_type(edit.edit_type) for edit in diff_analyzer.analyze(source, target)):
                continue
            rows.append(_real_row(source, target, edits, source_dataset, str(item.get("domain") or "external")))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _real_row(source: str, target: str, edits: list[Edit], source_dataset: str, domain: str) -> dict[str, Any]:
    from dataclasses import asdict

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
