from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.clean_corpus_sources import load_clean_corpus_sentences
from src.data.synthetic_generator import SyntheticGenerator
from src.data.external_sources import load_hf_jsonl_pairs
from src.validation.diff_analyzer import DiffAnalyzer, Edit
from src.validation.edit_classifier import coarse_error_type, is_allowed_edit_type

SPLIT_STRATEGY = "normalized_target_v2"


@dataclass(frozen=True)
class DatasetBuildConfig:
    target_total_examples: int = 450_000
    clean_identity_ratio: float = 0.10
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    seed: int = 13
    domain: str = "synthetic_general"
    external_rows: tuple[dict[str, Any], ...] = ()
    clean_texts: tuple[str, ...] = ()


def build_dataset_rows(config: DatasetBuildConfig) -> list[dict[str, Any]]:
    if config.target_total_examples <= 0:
        return []

    randomizer = random.Random(config.seed)
    diff_analyzer = DiffAnalyzer()
    generator = SyntheticGenerator(seed=config.seed, max_errors_per_sentence=3)
    clean_count = int(round(config.target_total_examples * config.clean_identity_ratio))
    real_rows = list(config.external_rows)
    real_rows = real_rows[: max(0, config.target_total_examples - clean_count)]
    dirty_count = config.target_total_examples - clean_count - len(real_rows)

    rows: list[dict[str, Any]] = []
    rows.extend(real_rows)
    sentence_factory = CleanSentenceFactory()
    clean_texts = list(config.clean_texts)

    synthetic_rows: list[dict[str, Any]] = []
    if clean_texts:
        synthetic_rows.extend(
            _build_synthetic_rows_from_clean_corpus(
                clean_texts,
                dirty_count=dirty_count,
                generator=generator,
                diff_analyzer=diff_analyzer,
                domain=config.domain,
            )
        )
    else:
        index = 0
        while len(synthetic_rows) < dirty_count:
            target = sentence_factory.make(index)
            example = generator.generate_from_clean(target)
            edits = _supported_edits(diff_analyzer.analyze(example.source, example.target))
            if example.source != example.target and edits:
                synthetic_rows.append(
                    _row(
                        source=example.source,
                        target=example.target,
                        edits=edits,
                        source_dataset="synthetic_rules",
                        is_clean=False,
                        is_synthetic=True,
                        domain=config.domain,
                    )
                )
            index += 1
    rows.extend(synthetic_rows)

    for clean_index in range(clean_count):
        target = _clean_target(clean_index, clean_texts, sentence_factory)
        rows.append(
            _row(
                source=target,
                target=target,
                edits=[],
                source_dataset="clean_identity",
                is_clean=True,
                is_synthetic=False,
                domain=config.domain,
            )
        )

    randomizer.shuffle(rows)
    _assign_splits(rows, val_ratio=config.val_ratio, test_ratio=config.test_ratio)
    return rows


def write_dataset(rows: list[dict[str, Any]], output_path: str | Path, manifest_path: str | Path | None = None) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    if manifest_path is not None:
        manifest = {
            "total": len(rows),
            "composition": dataset_composition(rows),
            "splits": _split_counts(rows),
            "split_strategy": SPLIT_STRATEGY,
            "columns": list(rows[0].keys()) if rows else [],
        }
        manifest_output = Path(manifest_path)
        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def build_dataset_from_config(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    data_config = config.get("data", {})
    output_path = Path(data_config.get("processed_train_path") or "data/processed/correction_dataset.csv.gz")
    manifest_path = Path(data_config.get("manifest_path") or "reports/dataset_manifest.json")
    target_total = int(data_config.get("target_total_examples", 450_000))
    env_limit = _env_int("RUSSIAN_CORRECTOR_DATASET_LIMIT")
    if env_limit is not None:
        target_total = env_limit

    if output_path.exists() and not force and _existing_row_count(output_path) >= target_total and _has_current_split_strategy(manifest_path):
        return {
            "status": "exists",
            "path": str(output_path),
            "manifest_path": str(manifest_path),
            "total": _existing_row_count(output_path),
        }

    clean_corpus = load_clean_corpus_sentences(data_config, needed_count=target_total)
    build_config = DatasetBuildConfig(
        target_total_examples=target_total,
        clean_identity_ratio=float(data_config.get("clean_identity_ratio", 0.10)),
        val_ratio=float(data_config.get("val_ratio", 0.05)),
        test_ratio=float(data_config.get("test_ratio", 0.05)),
        seed=int(data_config.get("synthetic_seed", 13)),
        domain=str(data_config.get("domain", "synthetic_general")),
        external_rows=tuple(_load_external_rows(data_config)),
        clean_texts=tuple(clean_corpus.sentences),
    )
    rows = build_dataset_rows(build_config)
    write_dataset(rows, output_path, manifest_path)
    return {
        "status": "built",
        "path": str(output_path),
        "manifest_path": str(manifest_path),
        "total": len(rows),
        "composition": dataset_composition(rows),
        "splits": _split_counts(rows),
        "clean_corpus": {
            "count": len(clean_corpus.sentences),
            "cache_path": clean_corpus.cache_path,
            "source_counts": clean_corpus.source_counts,
        },
    }


def dataset_composition(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "clean": sum(bool(row.get("is_clean")) for row in rows),
        "synthetic": sum(bool(row.get("is_synthetic")) and not bool(row.get("is_clean")) for row in rows),
        "real": sum(not bool(row.get("is_synthetic")) and not bool(row.get("is_clean")) for row in rows),
    }


def _load_external_rows(data_config: dict[str, Any]) -> list[dict[str, Any]]:
    if not bool(data_config.get("use_external_sources", False)):
        return []
    max_external = int(data_config.get("max_external_examples", 10_000))
    try:
        return load_hf_jsonl_pairs(limit=max_external)
    except Exception:
        return []


class CleanSentenceFactory:
    def __init__(self) -> None:
        self.nouns = [
            "проект",
            "текст",
            "отчет",
            "пример",
            "раздел",
            "документ",
            "вывод",
            "ответ",
            "план",
            "модуль",
            "абзац",
            "файл",
            "результат",
            "корпус",
            "словарь",
            "алгоритм",
        ]
        self.adjectives = [
            "важный",
            "новый",
            "полезный",
            "точный",
            "понятный",
            "сложный",
            "краткий",
            "рабочий",
            "учебный",
            "итоговый",
        ]
        self.actions = [
            "проверяю",
            "сохраняю",
            "открываю",
            "сравниваю",
            "исправляю",
            "обновляю",
            "читаю",
            "запускаю",
        ]
        self.templates = [
            "Я не знаю, что делать с {noun} {number}.",
            "Во-первых, это {adjective} {noun} {number}.",
            "В общем, это {adjective} пример для раздела {number}.",
            "Вряд ли это {adjective} результат для документа {number}.",
            "Сегодня {adjective} день, потому что готов {noun} {number}.",
            "Мы проверяем что-то важное в разделе {number}.",
            "Автор пишет по-русски, когда готовит {noun} {number}.",
            "Я периодически {action} {noun} {number}, потому что это важно.",
            "Кто-нибудь проверит {noun} {number}, если будет время.",
            "Кое-как работает {noun} {number}, но результат важен.",
        ]

    def make(self, index: int) -> str:
        noun = self.nouns[index % len(self.nouns)]
        adjective = self.adjectives[(index // len(self.nouns)) % len(self.adjectives)]
        action = self.actions[(index // (len(self.nouns) * len(self.adjectives))) % len(self.actions)]
        template = self.templates[index % len(self.templates)]
        return template.format(noun=noun, adjective=adjective, action=action, number=index)


def _clean_target(index: int, clean_texts: list[str], sentence_factory: CleanSentenceFactory) -> str:
    if 0 <= index < len(clean_texts):
        return clean_texts[index]
    return sentence_factory.make(index - len(clean_texts))


def _build_synthetic_rows_from_clean_corpus(
    clean_texts: list[str],
    *,
    dirty_count: int,
    generator: SyntheticGenerator,
    diff_analyzer: DiffAnalyzer,
    domain: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for target in clean_texts:
        for example in generator.generate_variants_from_clean(target):
            if len(rows) >= dirty_count:
                return rows
            key = (example.source, example.target)
            if key in seen_pairs:
                continue
            edits = _supported_edits(diff_analyzer.analyze(example.source, example.target))
            if example.source != example.target and edits:
                seen_pairs.add(key)
                rows.append(
                    _row(
                        source=example.source,
                        target=example.target,
                        edits=edits,
                        source_dataset="synthetic_rules",
                        is_clean=False,
                        is_synthetic=True,
                        domain=domain,
                    )
                )

    if len(rows) < dirty_count:
        raise ValueError(
            "Not enough clean-corpus sentences/variants to build synthetic dataset without template fallback: "
            f"needed {dirty_count}, built {len(rows)}"
        )
    return rows


def _supported_edits(edits: list[Edit]) -> list[Edit]:
    return [edit for edit in edits if is_allowed_edit_type(edit.edit_type)]


def _row(
    *,
    source: str,
    target: str,
    edits: list[Edit],
    source_dataset: str,
    is_clean: bool,
    is_synthetic: bool,
    domain: str,
) -> dict[str, Any]:
    return {
        "source": source,
        "target": target,
        "error_types": json.dumps(sorted({coarse_error_type(edit.edit_type) for edit in edits}), ensure_ascii=False),
        "source_dataset": source_dataset,
        "is_clean": is_clean,
        "is_synthetic": is_synthetic,
        "split": "train",
        "domain": domain,
        "edit_operations": json.dumps([asdict(edit) for edit in edits], ensure_ascii=False),
    }


def _assign_splits(rows: list[dict[str, Any]], *, val_ratio: float, test_ratio: float) -> None:
    total = len(rows)
    test_target = int(round(total * test_ratio))
    val_target = int(round(total * val_ratio))
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_split_group_key(row), []).append(row)

    test_count = 0
    val_count = 0
    for group in groups.values():
        if test_count < test_target:
            split = "test"
            test_count += len(group)
        elif val_count < val_target:
            split = "val"
            val_count += len(group)
        else:
            split = "train"
        for row in group:
            row["split"] = split


def _split_group_key(row: dict[str, Any]) -> tuple[str, str]:
    return ("target", _normalize_for_split(row.get("target", "")))


def _normalize_for_split(value: Any) -> str:
    text = str(value).lower().strip()
    text = re.sub(r"\d+", "<NUM>", text)
    text = re.sub(r"[^\w\s<>]+", " ", text, flags=re.U)
    return re.sub(r"\s+", " ", text)


def _split_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        split = str(row.get("split", "train"))
        counts[split] = counts.get(split, 0) + 1
    return counts


def _existing_row_count(path: Path) -> int:
    return int(sum(len(chunk) for chunk in pd.read_csv(path, chunksize=50_000)))


def _has_current_split_strategy(manifest_path: Path) -> bool:
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("split_strategy") == SPLIT_STRATEGY


def _env_int(name: str) -> int | None:
    import os

    value = os.environ.get(name)
    if not value:
        return None
    return int(value)
