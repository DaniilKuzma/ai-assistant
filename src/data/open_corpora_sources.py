from __future__ import annotations

import bz2
import csv
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import re
from typing import Any, Iterable
import zipfile

import yaml

from src.data.clean_corpus_sources import (
    _iter_opencorpora_xml_sentences,
    _strip_wiki_markup,
    normalize_sentence,
    parse_ud_conllu_texts,
    split_text_to_sentences,
)
from src.data.source_downloads import DownloadBudget, SourceDownloadResult, download_if_allowed


@dataclass(frozen=True)
class OpenCorpusSentence:
    text: str
    source_name: str
    source_subcorpus: str = ""
    domain: str = ""
    style: str = ""
    license_status: str = ""
    source_doc_id: str = ""
    sentence_id: str = ""
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class OpenCorporaLoadResult:
    records: list[OpenCorpusSentence]
    source_reports: list[dict[str, Any]]


def load_open_corpora_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_open_corpora_sentences(config: dict[str, Any] | str | Path) -> OpenCorporaLoadResult:
    """Load configured clean-source sentence candidates without treating absence as fatal."""

    if isinstance(config, str | Path):
        config = load_open_corpora_config(config)
    policy = _download_policy(config)
    source_specs = _source_specs(config)
    budget = DownloadBudget()
    records: list[OpenCorpusSentence] = []
    reports: list[dict[str, Any]] = []

    for source_name, spec in source_specs:
        if spec.get("enabled") is False:
            reports.append(_source_report(source_name, spec, status="skipped", reason="disabled"))
            continue
        source_type = str(spec.get("type") or "local_text")
        max_sentences = int(spec.get("max_sentences", 100_000))
        download = download_if_allowed(source_name, spec, policy, budget)
        path = Path(download.path) if download.path else None
        if not download.used and source_type not in {"hf_dataset", "huggingface_dataset"}:
            reports.append(_source_report(source_name, spec, status="skipped", reason=download.reason or download.mode, download=download))
            continue
        if source_type in {"hf_dataset", "huggingface_dataset"} and not download.used:
            reports.append(_source_report(source_name, spec, status="skipped", reason=download.reason or download.mode, download=download))
            continue

        source_records: list[OpenCorpusSentence] = []
        total_seen = 0
        try:
            for sentence, metadata in _iter_source_sentences(source_type, path, spec):
                total_seen += 1
                sentence = normalize_sentence(sentence)
                if not _looks_like_sentence_candidate(sentence):
                    continue
                source_records.append(
                    OpenCorpusSentence(
                        text=sentence,
                        source_name=source_name,
                        source_subcorpus=str(
                            metadata.get("source_subcorpus")
                            or spec.get("source_subcorpus")
                            or spec.get("subcorpus")
                            or ""
                        ),
                        domain=str(metadata.get("domain") or spec.get("domain") or ""),
                        style=str(metadata.get("style") or spec.get("style") or ""),
                        license_status=str(
                            metadata.get("license_status")
                            or spec.get("license_status")
                            or spec.get("license/status")
                            or spec.get("license")
                            or spec.get("license_note")
                            or ""
                        ),
                        source_doc_id=str(metadata.get("source_doc_id") or ""),
                        sentence_id=str(metadata.get("sentence_id") or f"{len(source_records)}"),
                        metadata={key: value for key, value in metadata.items() if key not in {"text"}},
                    )
                )
                if len(source_records) >= max_sentences:
                    break
        except ModuleNotFoundError as exc:
            reports.append(
                _source_report(
                    source_name,
                    spec,
                    status="skipped",
                    reason=f"missing_optional_dependency:{exc.name}",
                    mode=download.mode,
                    local_path=str(path or spec.get("local_path", "")),
                    download=download,
                )
            )
            continue
        except Exception as exc:
            reports.append(
                _source_report(
                    source_name,
                    spec,
                    status="skipped",
                    reason=f"loader_error:{exc.__class__.__name__}",
                    mode=download.mode,
                    local_path=str(path or spec.get("local_path", "")),
                    download=download,
                )
            )
            continue

        records.extend(source_records)
        reports.append(
            _source_report(
                source_name,
                spec,
                status="loaded",
                reason="",
                mode=download.mode,
                local_path=str(path or spec.get("local_path", "")),
                total_seen=total_seen,
                accepted=len(source_records),
                download=download,
            )
        )

    return OpenCorporaLoadResult(records=records, source_reports=reports)


def _download_policy(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("download_policy", {}) or config.get("sources", {}).get("download_policy", {}) or {})


def _source_specs(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    raw = config.get("clean_sources", config.get("sources", []))
    if isinstance(raw, dict):
        return [(str(name), dict(spec or {})) for name, spec in raw.items()]
    if isinstance(raw, list):
        result: list[tuple[str, dict[str, Any]]] = []
        for index, spec in enumerate(raw):
            if not isinstance(spec, dict):
                continue
            name = str(spec.get("name") or spec.get("id") or f"source_{index}")
            result.append((name, dict(spec)))
        return result
    return []


def _iter_source_sentences(
    source_type: str,
    path: Path | None,
    spec: dict[str, Any],
) -> Iterable[tuple[str, dict[str, Any]]]:
    if source_type in {"local_text", "text", "plain_text"}:
        assert path is not None
        yield from _iter_local_text(path, spec)
    elif source_type in {"local_jsonl", "jsonl"}:
        assert path is not None
        yield from _iter_local_jsonl(path, spec)
    elif source_type in {"local_csv", "csv", "tsv"}:
        assert path is not None
        yield from _iter_local_table(path, spec)
    elif source_type in {"ud_conllu", "conllu"}:
        assert path is not None
        yield from _iter_ud_conllu(path, spec)
    elif source_type in {"corus_lenta", "corus_lenta2"}:
        assert path is not None
        yield from _iter_corus_lenta(path, spec)
    elif source_type in {"nerus", "nerus_conllu"}:
        assert path is not None
        yield from _iter_nerus(path, spec)
    elif source_type == "taiga":
        assert path is not None
        yield from _iter_taiga(path, spec)
    elif source_type in {"opencorpora", "opencorpora_xml"}:
        assert path is not None
        yield from _iter_opencorpora(path, spec)
    elif source_type == "wikipedia_dump":
        assert path is not None
        yield from _iter_wikipedia(path, spec)
    elif source_type == "hf_dataset":
        yield from _iter_hf_dataset(spec)


def _iter_local_text(path: Path, spec: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    for line_index, line in enumerate(_read_text_lines(path)):
        for sentence in split_text_to_sentences(line):
            yield sentence, {"sentence_id": str(line_index)}


def _iter_local_jsonl(path: Path, spec: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    fields = tuple(spec.get("text_fields") or ("text", "sentence", "content"))
    with _open_text(path) as handle:
        for line_index, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            for field in fields:
                value = item.get(field)
                if not value:
                    continue
                for sentence in split_text_to_sentences(str(value)):
                    yield sentence, {
                        "source_doc_id": str(item.get("id") or item.get("doc_id") or line_index),
                        "source_subcorpus": str(item.get("source_subcorpus") or item.get("subcorpus") or ""),
                    }


def _iter_local_table(path: Path, spec: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    delimiter = "\t" if str(path).endswith((".tsv", ".tab")) else ","
    fields = tuple(spec.get("text_fields") or ("text", "sentence", "content"))
    with _open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row_index, row in enumerate(reader):
            value = next((row.get(field) for field in fields if row.get(field)), "")
            for sentence in split_text_to_sentences(str(value)):
                yield sentence, {"source_doc_id": str(row.get("id") or row_index)}


def _iter_ud_conllu(path: Path, spec: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    content = "\n".join(_read_text_lines(path))
    for index, sentence in enumerate(parse_ud_conllu_texts(content)):
        yield sentence, {"sentence_id": str(index), "source_subcorpus": str(spec.get("source_subcorpus") or "ud")}


def _iter_corus_lenta(path: Path, spec: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    try:
        from corus import load_lenta, load_lenta2
    except ModuleNotFoundError:
        yield from _iter_lenta_csv(path, spec)
        return

    loader = load_lenta2 if str(path).endswith(".bz2") else load_lenta
    for index, record in enumerate(loader(str(path))):
        text = "\n".join(str(value or "") for value in (getattr(record, "title", ""), getattr(record, "text", ""))).strip()
        for sentence_index, sentence in enumerate(split_text_to_sentences(text)):
            yield sentence, {
                "source_doc_id": str(getattr(record, "url", "") or index),
                "sentence_id": f"{index}:{sentence_index}",
                "source_subcorpus": str(getattr(record, "topic", "") or spec.get("source_subcorpus") or "lenta"),
                "domain": "news",
                "style": "neutral",
            }


def _iter_lenta_csv(path: Path, spec: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    with _open_text(path) as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            doc_id = str(row.get("url") or row.get("id") or index)
            topic = str(row.get("topic") or spec.get("source_subcorpus") or "lenta")
            text = "\n".join(str(row.get(field) or "") for field in ("title", "text"))
            for sentence_index, sentence in enumerate(split_text_to_sentences(text)):
                yield sentence, {
                    "source_doc_id": doc_id,
                    "sentence_id": f"{index}:{sentence_index}",
                    "source_subcorpus": topic,
                    "domain": "news",
                    "style": "neutral",
                }


def _iter_nerus(path: Path, spec: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    with _open_text(path) as handle:
        current_doc_id = ""
        sentence_index = 0
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("# newdoc id = ") or line.startswith("# doc_id = "):
                current_doc_id = line.split("=", 1)[1].strip()
            if not line.startswith("# text = "):
                continue
            sentence = normalize_sentence(line.removeprefix("# text = "))
            yield sentence, {
                "source_doc_id": current_doc_id,
                "sentence_id": str(sentence_index),
                "source_subcorpus": str(spec.get("source_subcorpus") or "nerus_lenta"),
                "domain": "news",
                "style": "neutral",
            }
            sentence_index += 1


def _iter_taiga(path: Path, spec: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    forbidden = {str(item).lower() for item in spec.get("forbidden_subcorpora", [])}
    allowed = {str(item).lower() for item in spec.get("allowed_subcorpora", [])}
    for file_path in _iter_files(path):
        subcorpus = _subcorpus_for_path(file_path, path)
        lowered = subcorpus.lower()
        if any(item and item in lowered for item in forbidden):
            continue
        if allowed and not any(item and item in lowered for item in allowed):
            continue
        file_spec = dict(spec)
        file_spec["source_subcorpus"] = subcorpus
        yield from _iter_local_text(file_path, file_spec)


def _iter_opencorpora(path: Path, spec: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.endswith(".xml"):
                    continue
                with archive.open(name) as xml_file:
                    for index, sentence in enumerate(_iter_opencorpora_xml_sentences(xml_file)):
                        yield sentence, {"sentence_id": str(index), "source_subcorpus": "opencorpora"}
        return
    for file_path in _iter_files(path):
        if not str(file_path).endswith(".xml"):
            continue
        with file_path.open("rb") as xml_file:
            for index, sentence in enumerate(_iter_opencorpora_xml_sentences(xml_file)):
                yield sentence, {"sentence_id": str(index), "source_subcorpus": "opencorpora"}


def _iter_wikipedia(path: Path, spec: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    opener = bz2.open if str(path).endswith(".bz2") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            text = _strip_wiki_markup(line)
            for sentence in split_text_to_sentences(text):
                yield sentence, {"sentence_id": str(index), "source_subcorpus": "ruwiki"}


def _iter_hf_dataset(spec: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    from datasets import load_dataset

    repo = spec.get("repo") or spec.get("hf_id")
    fields = tuple(spec.get("text_fields") or ("text",))
    for split in spec.get("splits", ("train",)):
        dataset = load_dataset(repo, split=split, streaming=bool(spec.get("streaming", True)))
        for row_index, row in enumerate(dataset):
            for field in fields:
                value = row.get(field)
                if not value:
                    continue
                for sentence in split_text_to_sentences(str(value)):
                    yield sentence, {
                        "source_doc_id": str(row.get("id") or row_index),
                        "source_subcorpus": str(split),
                    }


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file() and not file_path.name.startswith("."):
            yield file_path


def _subcorpus_for_path(file_path: Path, root: Path) -> str:
    try:
        relative = file_path.relative_to(root)
    except ValueError:
        return file_path.stem
    return relative.parts[0] if len(relative.parts) > 1 else file_path.stem


def _read_text_lines(path: Path) -> Iterable[str]:
    with _open_text(path) as handle:
        for line in handle:
            line = normalize_sentence(line)
            if line:
                yield line


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    if str(path).endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _looks_like_sentence_candidate(text: str) -> bool:
    if len(text) < 25:
        return False
    if not re.search(r"[А-Яа-яЁё]", text):
        return False
    if not text.endswith((".", "!", "?", "…")):
        return False
    return True


def _source_report(
    source_name: str,
    spec: dict[str, Any],
    *,
    status: str,
    reason: str,
    mode: str = "skipped",
    local_path: str = "",
    total_seen: int = 0,
    accepted: int = 0,
    download: SourceDownloadResult | None = None,
) -> dict[str, Any]:
    path = local_path or str(spec.get("local_path") or spec.get("path") or "")
    size = Path(path).stat().st_size if path and Path(path).exists() else 0
    if download is not None:
        path = download.path or path
        size = download.downloaded_size_bytes or size
    return {
        "source_name": source_name,
        "type": str(spec.get("type") or ""),
        "status": status,
        "mode": download.mode if download is not None else mode,
        "reason": reason,
        "local_path": path,
        "url": str(spec.get("url") or ""),
        "hf_id": str(spec.get("hf_id") or spec.get("repo") or ""),
        "downloaded_size_bytes": size if (download.mode if download is not None else mode) in {"downloaded", "cached", "local"} else 0,
        "total_seen": total_seen,
        "accepted": accepted,
        "rejected": max(0, total_seen - accepted),
        "domain": str(spec.get("domain") or ""),
        "style": str(spec.get("style") or ""),
        "license_status": str(spec.get("license_status") or spec.get("license/status") or spec.get("license") or spec.get("license_note") or ""),
        "required_domains": ", ".join(download.required_domains) if download is not None else "",
        "required_commands": " || ".join(download.required_commands) if download is not None else "",
        "used": bool(accepted),
    }
