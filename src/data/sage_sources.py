from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import json
import os
from pathlib import Path
from typing import Any, Iterable

from src.data.clean_sentence_pool import clean_sentence_acceptance_reason
from src.data.source_downloads import DEFAULT_DOWNLOAD_ENV


SOURCE_COLUMNS = ("source", "input", "input_text", "incorrect", "erroneous", "original", "error_text")
TARGET_COLUMNS = ("target", "target_text", "correction", "correct", "corrected", "output", "correct_text")
SAGE_DATASET_ALIASES = {
    "RUSpellRU": ("RUSpellRU",),
    "MultidomainGold": ("MultidomainGold",),
    "MedSpellChecker": ("MedSpellChecker", "MedSpellchecker", "MedSpellcheck"),
    "GitHubTypoCorpusRu": ("GitHubTypoCorpusRu",),
}
HF_CONFIG_NAMES = {
    "RUSpellRU": "RUSpellRU",
    "MultidomainGold": "MultidomainGold",
    "MedSpellChecker": "MedSpellchecker",
    "GitHubTypoCorpusRu": "GitHubTypoCorpusRu",
}
MANUAL_DOWNLOAD_COMMANDS = (
    "export RUSSIAN_CORRECTOR_ALLOW_SOURCE_DOWNLOADS=1\n"
    "mkdir -p data/external data/external/sage data/external/hf\n"
    "curl -L --fail --retry 4 -C - -o data/external/nerus_lenta.conllu.gz "
    "https://storage.yandexcloud.net/natasha-nerus/data/nerus_lenta.conllu.gz",
    ".venv/bin/python - <<'PY'\n"
    "from huggingface_hub import snapshot_download\n\n"
    "for repo in [\n"
    '    "ai-forever/spellcheck_benchmark",\n'
    '    "ai-forever/spellcheck_punctuation_benchmark",\n'
    "]:\n"
    "    snapshot_download(\n"
    "        repo_id=repo,\n"
    '        repo_type="dataset",\n'
    "        local_dir=f\"data/external/hf/{repo.replace('/', '__')}\",\n"
    '        allow_patterns=["README.md", "*.py", "data/**/*.json", "data/**/*.jsonl", "data/**/*.gz", "data/**/*.zip"],\n'
    "    )\n"
    "PY",
    "git clone --depth 1 https://github.com/ai-forever/sage data/external/sage_repo",
)


@dataclass(frozen=True)
class NerusCheck:
    exists: bool
    path: str
    size_bytes: int = 0
    gzip_ok: bool = False
    text_comment_count: int = 0
    extracted_sample_sentence_count: int = 0
    accepted_sample_count: int = 0
    status: str = "missing"
    reason: str = ""


@dataclass
class SageMaterializeResult:
    output_dir: str
    loader_method: str
    written_counts: dict[str, int] = field(default_factory=dict)
    source_paths: dict[str, list[str]] = field(default_factory=dict)
    skipped_reasons: dict[str, str] = field(default_factory=dict)

    @property
    def missing_datasets(self) -> list[str]:
        return [dataset for dataset in SAGE_DATASET_ALIASES if self.written_counts.get(dataset, 0) <= 0]

    @property
    def ready(self) -> bool:
        return not self.missing_datasets


def downloads_enabled(env_name: str = DEFAULT_DOWNLOAD_ENV) -> bool:
    return os.environ.get(env_name, "").strip().lower() in {"1", "true", "yes", "on"}


def verify_nerus_conllu(
    path: str | Path,
    *,
    min_size_bytes: int = 500 * 1024 * 1024,
    min_sample_sentences: int = 100,
) -> NerusCheck:
    nerus_path = Path(path)
    if not nerus_path.exists():
        return NerusCheck(False, str(nerus_path), status="missing", reason="missing")
    size = nerus_path.stat().st_size
    if size < min_size_bytes:
        return NerusCheck(
            True,
            str(nerus_path),
            size_bytes=size,
            status="invalid",
            reason=f"size_below_min:{size}<{min_size_bytes}",
        )

    text_count = 0
    samples: list[str] = []
    accepted_count = 0
    try:
        with gzip.open(nerus_path, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.startswith("# text = "):
                    continue
                text_count += 1
                if len(samples) < min_sample_sentences:
                    sentence = line.removeprefix("# text = ").strip()
                    samples.append(sentence)
                    accepted, _, _ = clean_sentence_acceptance_reason(
                        sentence,
                        {"source_name": "nerus_news", "domain": "news", "style": "neutral", "source_subcorpus": "nerus_lenta"},
                    )
                    if accepted:
                        accepted_count += 1
                if len(samples) >= min_sample_sentences and text_count >= min_sample_sentences:
                    break
    except OSError as exc:
        return NerusCheck(
            True,
            str(nerus_path),
            size_bytes=size,
            gzip_ok=False,
            status="invalid",
            reason=f"gzip_error:{exc.__class__.__name__}",
        )

    if text_count <= 0:
        status = "invalid"
        reason = "missing_text_comments"
    elif len(samples) < min_sample_sentences:
        status = "invalid"
        reason = f"sample_sentence_shortage:{len(samples)}<{min_sample_sentences}"
    else:
        status = "ready"
        reason = ""
    return NerusCheck(
        True,
        str(nerus_path),
        size_bytes=size,
        gzip_ok=True,
        text_comment_count=text_count,
        extracted_sample_sentence_count=len(samples),
        accepted_sample_count=accepted_count,
        status=status,
        reason=reason,
    )


def canonicalize_sage_row(
    item: dict[str, Any],
    *,
    dataset: str,
    split: str = "",
    raw_id: str = "",
    source_dataset: str | None = None,
) -> dict[str, Any] | None:
    source = _first_value(item, SOURCE_COLUMNS)
    target = _first_value(item, TARGET_COLUMNS)
    if not source or not target:
        return None
    if source == target:
        return None
    canonical_dataset = canonical_dataset_name(dataset)
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    metadata = dict(metadata)
    if split:
        metadata.setdefault("split", split)
    metadata.setdefault("loader_dataset", dataset)
    row = {
        "source": source,
        "target": target,
        "dataset": canonical_dataset,
        "source_dataset": source_dataset or canonical_dataset,
        "metadata": metadata,
    }
    domain = item.get("domain")
    if domain:
        row["domain"] = str(domain)
    raw_identifier = item.get("id") or item.get("raw_id") or raw_id
    if raw_identifier != "":
        row["raw_id"] = str(raw_identifier)
    return row


def canonical_dataset_name(name: str) -> str:
    lowered = str(name).lower()
    for canonical, aliases in SAGE_DATASET_ALIASES.items():
        if lowered == canonical.lower() or lowered in {alias.lower() for alias in aliases}:
            return canonical
    return str(name)


def materialize_sage_jsonl_from_directory(
    root: str | Path,
    output_dir: str | Path,
    *,
    loader_method: str = "local_directory",
    overwrite: bool = True,
) -> SageMaterializeResult:
    root_path = Path(root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows_by_dataset: dict[str, list[dict[str, Any]]] = {dataset: [] for dataset in SAGE_DATASET_ALIASES}
    source_paths: dict[str, list[str]] = {dataset: [] for dataset in SAGE_DATASET_ALIASES}

    for canonical, aliases in SAGE_DATASET_ALIASES.items():
        for data_path in _dataset_files(root_path, aliases):
            split = _split_name(data_path)
            source_paths[canonical].append(str(data_path))
            for raw_index, item in enumerate(_iter_json_records(data_path)):
                row = canonicalize_sage_row(
                    item,
                    dataset=canonical,
                    split=split,
                    raw_id=f"{data_path.name}:{raw_index}",
                )
                if row is not None:
                    rows_by_dataset[canonical].append(row)

    written_counts: dict[str, int] = {}
    skipped_reasons: dict[str, str] = {}
    for dataset, rows in rows_by_dataset.items():
        if not rows:
            skipped_reasons[dataset] = "missing_dataset_files" if not source_paths[dataset] else "no_valid_rows"
            continue
        rows = _dedupe_canonical_rows(rows)
        output_file = output_path / f"{dataset}.jsonl"
        mode = "w" if overwrite else "a"
        with output_file.open(mode, encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        written_counts[dataset] = len(rows)

    return SageMaterializeResult(
        output_dir=str(output_path),
        loader_method=loader_method,
        written_counts=written_counts,
        source_paths={key: value for key, value in source_paths.items() if value},
        skipped_reasons=skipped_reasons,
    )


def materialize_punctuation_jsonl_from_directory(
    root: str | Path,
    output_path: str | Path,
    *,
    source_dataset: str = "spellcheck_punctuation_benchmark",
) -> int:
    root_path = Path(root)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for canonical, aliases in SAGE_DATASET_ALIASES.items():
        for data_path in _dataset_files(root_path, aliases):
            split = _split_name(data_path)
            for raw_index, item in enumerate(_iter_json_records(data_path)):
                row = canonicalize_sage_row(
                    item,
                    dataset=canonical,
                    split=split,
                    raw_id=f"{data_path.name}:{raw_index}",
                    source_dataset=source_dataset,
                )
                if row is None:
                    continue
                metadata = dict(row.get("metadata") or {})
                metadata["source_benchmark"] = source_dataset
                metadata["original_dataset"] = canonical
                row["metadata"] = metadata
                rows.append(row)
    rows = _dedupe_canonical_rows(rows)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def prepare_punctuation_jsonl_file(
    *,
    output_path: str | Path = "data/external/sage/spellcheck_punctuation_benchmark.jsonl",
    punctuation_repo: str = "ai-forever/spellcheck_punctuation_benchmark",
    local_hf_dir: str | Path = "data/external/hf",
    env_name: str = DEFAULT_DOWNLOAD_ENV,
) -> tuple[int, str]:
    output = Path(output_path)
    if output.exists() and output.stat().st_size > 0:
        count, invalid = _jsonl_count_and_invalid_count(output)
        if invalid == 0 and count > 0:
            return count, "existing_local_jsonl"
        output.unlink(missing_ok=True)
    for root, method in _local_candidate_roots(punctuation_repo, punctuation_repo, local_hf_dir):
        count = materialize_punctuation_jsonl_from_directory(root, output)
        if count > 0:
            return count, method
    if downloads_enabled(env_name):
        result = _snapshot_and_materialize(punctuation_repo, output.parent, local_hf_dir)
        for root, method in _local_candidate_roots(punctuation_repo, punctuation_repo, local_hf_dir):
            count = materialize_punctuation_jsonl_from_directory(root, output)
            if count > 0:
                return count, result.loader_method or method
    return 0, "missing"


def prepare_sage_jsonl_files(
    *,
    output_dir: str | Path = "data/external/sage",
    benchmark_repo: str = "ai-forever/spellcheck_benchmark",
    punctuation_repo: str = "ai-forever/spellcheck_punctuation_benchmark",
    local_hf_dir: str | Path = "data/external/hf",
    env_name: str = DEFAULT_DOWNLOAD_ENV,
) -> SageMaterializeResult:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    existing = _existing_sage_result(output_path)
    if existing.ready:
        return existing
    _remove_incomplete_sage_jsonl(output_path)

    attempts: list[SageMaterializeResult] = []
    for root, method in _local_candidate_roots(benchmark_repo, punctuation_repo, local_hf_dir):
        result = materialize_sage_jsonl_from_directory(root, output_path, loader_method=method, overwrite=False)
        attempts.append(result)
        merged = _existing_sage_result(output_path, loader_method=method)
        if merged.ready:
            return merged

    if downloads_enabled(env_name):
        result = _materialize_with_datasets(benchmark_repo, output_path)
        attempts.append(result)
        merged = _existing_sage_result(output_path, loader_method=result.loader_method)
        if merged.ready:
            return merged

        result = _snapshot_and_materialize(benchmark_repo, output_path, local_hf_dir)
        attempts.append(result)
        merged = _existing_sage_result(output_path, loader_method=result.loader_method)
        if merged.ready:
            return merged

        result = _snapshot_and_materialize(punctuation_repo, output_path, local_hf_dir)
        attempts.append(result)

    return _merge_attempt_results(output_path, attempts)


def _existing_sage_result(output_dir: Path, *, loader_method: str = "existing_local_jsonl") -> SageMaterializeResult:
    counts: dict[str, int] = {}
    paths: dict[str, list[str]] = {}
    skipped: dict[str, str] = {}
    for dataset in SAGE_DATASET_ALIASES:
        path = output_dir / f"{dataset}.jsonl"
        if not path.exists():
            skipped[dataset] = "missing_local_jsonl"
            continue
        count, invalid = _jsonl_count_and_invalid_count(path)
        if invalid:
            skipped[dataset] = f"invalid_local_jsonl:{invalid}"
            continue
        if count > 0:
            counts[dataset] = count
            paths[dataset] = [str(path)]
        else:
            skipped[dataset] = "empty_local_jsonl"
    return SageMaterializeResult(str(output_dir), loader_method, counts, paths, skipped)


def _remove_incomplete_sage_jsonl(output_dir: Path) -> None:
    for dataset in SAGE_DATASET_ALIASES:
        path = output_dir / f"{dataset}.jsonl"
        if not path.exists():
            continue
        count, invalid = _jsonl_count_and_invalid_count(path)
        if invalid or count <= 0:
            path.unlink(missing_ok=True)


def _jsonl_count_and_invalid_count(path: Path) -> tuple[int, int]:
    count = 0
    invalid = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            count += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            source = str(item.get("source") or "")
            target = str(item.get("target") or "")
            if not source or not target or source == target:
                invalid += 1
    return count, invalid


def _materialize_with_datasets(repo_id: str, output_dir: Path) -> SageMaterializeResult:
    rows_by_dataset: dict[str, list[dict[str, Any]]] = {dataset: [] for dataset in SAGE_DATASET_ALIASES}
    skipped: dict[str, str] = {}
    try:
        from datasets import load_dataset
    except Exception as exc:
        return SageMaterializeResult(str(output_dir), "datasets_unavailable", skipped_reasons={"all": exc.__class__.__name__})

    for dataset, config_name in HF_CONFIG_NAMES.items():
        split_loaded = False
        for split in ("train", "test"):
            try:
                try:
                    loaded = load_dataset(repo_id, name=config_name, split=split, trust_remote_code=True)
                except TypeError:
                    loaded = load_dataset(repo_id, name=config_name, split=split)
            except Exception as exc:
                skipped[f"{dataset}:{split}"] = f"{exc.__class__.__name__}"
                continue
            split_loaded = True
            for raw_index, item in enumerate(loaded):
                row = canonicalize_sage_row(dict(item), dataset=dataset, split=split, raw_id=f"{split}:{raw_index}")
                if row is not None:
                    rows_by_dataset[dataset].append(row)
        if not split_loaded:
            skipped.setdefault(dataset, "no_splits_loaded")

    counts = _write_rows_by_dataset(rows_by_dataset, output_dir, loader_method="datasets.load_dataset")
    return SageMaterializeResult(str(output_dir), "datasets.load_dataset", counts, skipped_reasons=skipped)


def _snapshot_and_materialize(repo_id: str, output_dir: Path, local_hf_dir: str | Path) -> SageMaterializeResult:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        return SageMaterializeResult(str(output_dir), "snapshot_download_unavailable", skipped_reasons={"all": exc.__class__.__name__})

    local_dir = Path(local_hf_dir) / repo_id.replace("/", "__")
    try:
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            allow_patterns=["README.md", "*.py", "data/**/*.json", "data/**/*.jsonl", "data/**/*.gz", "data/**/*.zip"],
        )
    except Exception as exc:
        return SageMaterializeResult(str(output_dir), "snapshot_download_failed", skipped_reasons={"all": exc.__class__.__name__})
    return materialize_sage_jsonl_from_directory(snapshot_path, output_dir, loader_method=f"snapshot_download:{repo_id}", overwrite=False)


def _write_rows_by_dataset(rows_by_dataset: dict[str, list[dict[str, Any]]], output_dir: Path, *, loader_method: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset, rows in rows_by_dataset.items():
        if not rows:
            continue
        rows = _dedupe_canonical_rows(rows)
        with (output_dir / f"{dataset}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                metadata = dict(row.get("metadata") or {})
                metadata.setdefault("loader_method", loader_method)
                row = dict(row)
                row["metadata"] = metadata
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        counts[dataset] = len(rows)
    return counts


def _local_candidate_roots(benchmark_repo: str, punctuation_repo: str, local_hf_dir: str | Path) -> Iterable[tuple[Path, str]]:
    for repo_id in (benchmark_repo, punctuation_repo):
        local_dir = Path(local_hf_dir) / repo_id.replace("/", "__")
        if local_dir.exists():
            yield local_dir, f"local_dir:{repo_id}"
        for snapshot in _hf_hub_snapshots(repo_id):
            yield snapshot, f"hf_hub_cache:{repo_id}"
    sage_repo = Path("data/external/sage_repo")
    if sage_repo.exists():
        yield sage_repo, "local_sage_repo"


def _hf_hub_snapshots(repo_id: str) -> list[Path]:
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    repo_cache = cache_root / f"datasets--{repo_id.replace('/', '--')}"
    snapshots = repo_cache / "snapshots"
    if not snapshots.exists():
        return []
    return [path for path in sorted(snapshots.iterdir()) if path.is_dir()]


def _dataset_files(root: Path, aliases: tuple[str, ...]) -> list[Path]:
    result: list[Path] = []
    search_roots: list[Path] = []
    for alias in aliases:
        search_roots.extend([root / "data" / alias, root / alias])
    for search_root in search_roots:
        if search_root.exists():
            result.extend(_json_files(search_root))
    if result:
        return sorted(dict.fromkeys(result))
    lowered_aliases = tuple(alias.lower() for alias in aliases)
    for path in _json_files(root):
        lowered_parts = {part.lower() for part in path.parts}
        if any(alias.lower() in lowered_parts or alias.lower() in path.name.lower() for alias in lowered_aliases):
            result.append(path)
    return sorted(dict.fromkeys(result))


def _json_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if _is_json_path(root) else []
    return [path for path in root.rglob("*") if path.is_file() and _is_json_path(path)]


def _is_json_path(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".json", ".jsonl", ".json.gz", ".jsonl.gz"))


def _iter_json_records(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    stripped = text.strip()
    if not stripped:
        return
    if stripped.startswith("["):
        payload = json.loads(stripped)
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    for line in stripped.splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            yield item


def _split_name(path: Path) -> str:
    stem = path.name
    for suffix in (".jsonl.gz", ".json.gz", ".jsonl", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def _dedupe_canonical_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row.get("dataset", "")), str(row.get("source", "")), str(row.get("target", "")))
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _merge_attempt_results(output_dir: Path, attempts: list[SageMaterializeResult]) -> SageMaterializeResult:
    existing = _existing_sage_result(output_dir, loader_method="partial_local_jsonl")
    if not attempts:
        return existing
    skipped: dict[str, str] = {}
    source_paths: dict[str, list[str]] = dict(existing.source_paths)
    for attempt in attempts:
        for key, value in attempt.skipped_reasons.items():
            skipped[f"{attempt.loader_method}:{key}"] = value
        for dataset, paths in attempt.source_paths.items():
            source_paths.setdefault(dataset, []).extend(paths)
    existing.skipped_reasons.update(skipped)
    existing.source_paths = source_paths
    existing.loader_method = "partial_local_jsonl"
    return existing


def _first_value(item: dict[str, Any], names: tuple[str, ...]) -> str:
    lowered = {str(key).lower(): value for key, value in item.items()}
    for name in names:
        value = lowered.get(name)
        if value is not None:
            return str(value).strip()
    return ""
