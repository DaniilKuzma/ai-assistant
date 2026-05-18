from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, replace
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.candidates.frequent_errors import HYPHEN_WHITELIST, WRONG_TO_CORRECT
from src.candidates.morphology import morph_analyzer
from src.data.clean_corpus_sources import load_clean_corpus_sentences
from src.data.dataset_stats import dataset_stats
from src.data.external_sources import load_external_pair_sources, load_hf_jsonl_pairs
from src.data.synthetic_generator import SyntheticGenerator
from src.evaluation.reports import write_dataset_report
from src.rules.orthography import orthography_rules
from src.rules.rule_ids import normalize_rule_id
from src.rules.synthetic import (
    DEFAULT_ORTHOGRAPHY_BALANCE,
    DEFAULT_PUNCTUATION_BALANCE,
    DICTIONARY_FUZZY_SYNTHETIC_ERRORS,
    ORTHOGRAPHY_BALANCE_GROUPS,
    PUNCTUATION_BALANCE_GROUPS,
)
from src.validation.diff_analyzer import DiffAnalyzer, Edit
from src.validation.edit_classifier import coarse_error_type, is_allowed_edit_type, is_context_dependent_pair

SPLIT_STRATEGY = "normalized_target_v2"
DATASET_COLUMNS = [
    "source",
    "target",
    "error_types",
    "source_dataset",
    "is_clean",
    "is_synthetic",
    "split",
    "domain",
    "edit_operations",
]

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
    min_spelling_examples: int = 60_000
    min_split_join_examples: int = 20_000
    min_hyphen_examples: int = 20_000
    punctuation_hard_negative_clean_ratio: float = 0.25
    orthography_balance: tuple[tuple[str, int], ...] = tuple(DEFAULT_ORTHOGRAPHY_BALANCE.items())
    punctuation_balance: tuple[tuple[str, int], ...] = tuple(DEFAULT_PUNCTUATION_BALANCE.items())


def build_dataset_rows(config: DatasetBuildConfig) -> list[dict[str, Any]]:
    if config.target_total_examples <= 0:
        return []

    randomizer = random.Random(config.seed)
    diff_analyzer = DiffAnalyzer()
    generator = SyntheticGenerator(seed=config.seed, max_errors_per_sentence=3)
    clean_count = int(round(config.target_total_examples * config.clean_identity_ratio))
    if config.target_total_examples > 0 and config.clean_identity_ratio > 0:
        clean_count = max(1, min(clean_count, config.target_total_examples))
    real_rows = [_normalize_dataset_row(row, default_domain=config.domain) for row in config.external_rows]
    min_synthetic_count = 1 if config.target_total_examples - clean_count > 0 else 0
    real_rows = real_rows[: max(0, config.target_total_examples - clean_count - min_synthetic_count)]
    dirty_count = config.target_total_examples - clean_count - len(real_rows)

    rows: list[dict[str, Any]] = []
    rows.extend(real_rows)
    sentence_factory = CleanSentenceFactory()
    clean_texts = list(config.clean_texts)

    synthetic_rows: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    lexical_targets, orthography_targets, punctuation_targets = _scaled_synthetic_targets(config, dirty_count)

    for error_type, required_count in lexical_targets.items():
        remaining = dirty_count - len(synthetic_rows)
        if remaining <= 0:
            break
        synthetic_rows.extend(
            _build_targeted_lexical_rows(
                error_type,
                required_count=min(required_count, remaining),
                diff_analyzer=diff_analyzer,
                domain=config.domain,
                seen_pairs=seen_pairs,
            )
        )

    for group, required_count in orthography_targets.items():
        remaining = dirty_count - len(synthetic_rows)
        if remaining <= 0:
            break
        synthetic_rows.extend(
            _build_targeted_orthography_rows(
                group,
                required_count=min(required_count, remaining),
                diff_analyzer=diff_analyzer,
                domain=config.domain,
                seen_pairs=seen_pairs,
            )
        )

    for group, required_count in punctuation_targets.items():
        remaining = dirty_count - len(synthetic_rows)
        if remaining <= 0:
            break
        synthetic_rows.extend(
            _build_targeted_punctuation_rows(
                group,
                required_count=min(required_count, remaining),
                diff_analyzer=diff_analyzer,
                domain=config.domain,
                seen_pairs=seen_pairs,
            )
        )

    remaining_dirty_count = dirty_count - len(synthetic_rows)
    if clean_texts:
        synthetic_rows.extend(
            _build_synthetic_rows_from_clean_corpus(
                clean_texts,
                dirty_count=remaining_dirty_count,
                generator=generator,
                diff_analyzer=diff_analyzer,
                domain=config.domain,
                seen_pairs=seen_pairs,
            )
        )
    remaining_dirty_count = dirty_count - len(synthetic_rows)
    if remaining_dirty_count > 0:
        synthetic_rows.extend(
            _build_template_synthetic_rows(
                dirty_count=remaining_dirty_count,
                generator=generator,
                diff_analyzer=diff_analyzer,
                domain=config.domain,
                sentence_factory=sentence_factory,
                seen_pairs=seen_pairs,
            )
        )
    rows.extend(synthetic_rows)

    hard_negative_count = _hard_negative_count(clean_count, config.punctuation_hard_negative_clean_ratio)
    for clean_index in range(clean_count):
        if clean_index < hard_negative_count:
            target = _hard_negative_target(clean_index)
            source_dataset = "clean_identity_hard_negative"
        else:
            clean_target_index = clean_index - hard_negative_count
            target = _clean_target(clean_target_index, clean_texts, sentence_factory)
            source_dataset = "clean_identity_open_corpus" if clean_target_index < len(clean_texts) else "clean_identity_template"
        rows.append(
            _row(
                source=target,
                target=target,
                edits=[],
                source_dataset=source_dataset,
                is_clean=True,
                is_synthetic=False,
                domain=config.domain,
            )
        )

    rows = [_normalize_dataset_row(row, default_domain=config.domain) for row in rows]
    randomizer.shuffle(rows)
    _assign_splits(rows, val_ratio=config.val_ratio, test_ratio=config.test_ratio)
    return rows


def write_dataset(
    rows: list[dict[str, Any]],
    output_path: str | Path,
    manifest_path: str | Path | None = None,
    *,
    manifest_metadata: dict[str, Any] | None = None,
    report_path: str | Path | None = None,
) -> None:
    normalized_rows = [_normalize_dataset_row(row) for row in rows]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(normalized_rows, columns=DATASET_COLUMNS).to_csv(output, index=False)
    if manifest_path is not None:
        manifest = _dataset_manifest(normalized_rows, manifest_metadata=manifest_metadata)
        manifest_output = Path(manifest_path)
        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if report_path is not None:
        report_output = Path(report_path)
        report_output.parent.mkdir(parents=True, exist_ok=True)
        write_dataset_report(dataset_stats(pd.DataFrame(normalized_rows, columns=DATASET_COLUMNS)), report_output)


def build_dataset_from_config(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    data_config = config.get("data", {})
    output_path = Path(data_config.get("processed_train_path") or "data/processed/correction_dataset.csv.gz")
    manifest_path = Path(data_config.get("manifest_path") or "reports/dataset_manifest.json")
    configured_target_total = int(data_config.get("target_total_examples", 450_000))
    target_total = configured_target_total
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
    clean_texts = list(clean_corpus.sentences)
    if not clean_texts:
        clean_texts = [str(text) for text in data_config.get("debug_clean_texts", []) if str(text).strip()]

    external_rows = _load_external_rows(data_config)
    external_budget = _external_allocated_budget(
        data_config,
        target_total=target_total,
        configured_target_total=configured_target_total,
    )
    selected_external_rows, external_cap_reason = _select_external_rows(
        external_rows,
        budget=external_budget,
        seed=int(data_config.get("synthetic_seed", 13)),
    )

    build_config = DatasetBuildConfig(
        target_total_examples=target_total,
        clean_identity_ratio=float(data_config.get("clean_identity_ratio", 0.10)),
        val_ratio=float(data_config.get("val_ratio", 0.05)),
        test_ratio=float(data_config.get("test_ratio", 0.05)),
        seed=int(data_config.get("synthetic_seed", 13)),
        domain=str(data_config.get("domain", "synthetic_general")),
        external_rows=tuple(selected_external_rows),
        clean_texts=tuple(clean_texts),
        min_spelling_examples=int(data_config.get("synthetic_balance", {}).get("spelling_min_examples", 60_000)),
        min_split_join_examples=int(data_config.get("synthetic_balance", {}).get("split_join_min_examples", 20_000)),
        min_hyphen_examples=int(data_config.get("synthetic_balance", {}).get("hyphen_min_examples", 20_000)),
        punctuation_hard_negative_clean_ratio=float(data_config.get("punctuation_hard_negative_clean_ratio", 0.0)),
        orthography_balance=_orthography_balance_from_config(data_config),
        punctuation_balance=_punctuation_balance_from_config(data_config),
    )
    rows = build_dataset_rows(build_config)
    composition = dataset_composition(rows)
    build_metadata = {
        "requested_limit": target_total,
        "final_total_rows": len(rows),
        "external_available_rows": len(external_rows),
        "external_allocated_budget": external_budget,
        "external_used_rows": composition["real"],
        "synthetic_rows": composition["synthetic"],
        "clean_rows": composition["clean"],
    }
    if external_cap_reason:
        build_metadata["external_cap_reason"] = external_cap_reason
    report_path = _dataset_report_path(config, manifest_path)
    write_dataset(rows, output_path, manifest_path, manifest_metadata=build_metadata, report_path=report_path)
    return {
        "status": "built",
        "path": str(output_path),
        "manifest_path": str(manifest_path),
        "dataset_report_path": str(report_path),
        "total": len(rows),
        "composition": composition,
        "splits": _split_counts(rows),
        **build_metadata,
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


def error_type_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for error_type in _row_error_types(row):
            counts[error_type] = counts.get(error_type, 0) + 1
    return dict(sorted(counts.items()))


def rule_id_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for operation in _parse_jsonish_list(row.get("edit_operations", [])):
            if not isinstance(operation, dict):
                continue
            rule_id = _normalize_rule_id(operation.get("rule_id", ""))
            if rule_id == "unknown":
                continue
            counts[rule_id] = counts.get(rule_id, 0) + 1
    return dict(sorted(counts.items()))


def hard_negative_count(rows: list[dict[str, Any]]) -> int:
    return sum(row.get("source_dataset") == "clean_identity_hard_negative" for row in rows)


def _dataset_manifest(rows: list[dict[str, Any]], *, manifest_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    composition = dataset_composition(rows)
    manifest = {
        "total": len(rows),
        "final_total_rows": len(rows),
        "composition": composition,
        "error_type_counts": error_type_counts(rows),
        "rule_id_counts": rule_id_counts(rows),
        "hard_negative_count": hard_negative_count(rows),
        "splits": _split_counts(rows),
        "split_strategy": SPLIT_STRATEGY,
        "columns": DATASET_COLUMNS,
        "synthetic_rows": composition["synthetic"],
        "clean_rows": composition["clean"],
    }
    if manifest_metadata:
        manifest.update(manifest_metadata)
        manifest["total"] = len(rows)
        manifest["final_total_rows"] = len(rows)
        manifest["composition"] = composition
        manifest["error_type_counts"] = error_type_counts(rows)
        manifest["rule_id_counts"] = rule_id_counts(rows)
        manifest["hard_negative_count"] = hard_negative_count(rows)
        manifest["splits"] = _split_counts(rows)
        manifest["columns"] = DATASET_COLUMNS
        manifest["synthetic_rows"] = composition["synthetic"]
        manifest["clean_rows"] = composition["clean"]
    return manifest


def _dataset_report_path(config: dict[str, Any], manifest_path: Path) -> Path:
    reports_dir = config.get("paths", {}).get("reports_dir")
    return Path(reports_dir or manifest_path.parent) / "dataset_report.md"


def _load_external_rows(data_config: dict[str, Any]) -> list[dict[str, Any]]:
    if not bool(data_config.get("use_external_sources", False)):
        return []
    if _external_sources_disabled():
        return []
    max_external = int(data_config.get("max_external_examples", 10_000))
    local_files_only = bool(data_config.get("external_local_files_only", False)) or _hf_offline_mode()
    source_specs = data_config.get("external_sources", [])
    try:
        if source_specs:
            return load_external_pair_sources(source_specs, limit=max_external, local_files_only=local_files_only)
        return load_hf_jsonl_pairs(limit=max_external, local_files_only=local_files_only)
    except Exception:
        return []


def _external_allocated_budget(
    data_config: dict[str, Any],
    *,
    target_total: int,
    configured_target_total: int,
) -> int:
    if not bool(data_config.get("use_external_sources", False)) or target_total <= 0:
        return 0
    max_external = max(0, int(data_config.get("max_external_examples", 10_000)))
    if max_external <= 0:
        return 0
    baseline_total = max(1, configured_target_total)
    scaled_budget = int(round(max_external * target_total / baseline_total))
    if target_total < baseline_total and max_external > 0:
        scaled_budget = max(1, scaled_budget)
    return min(max_external, max(0, scaled_budget))


def _select_external_rows(rows: list[dict[str, Any]], *, budget: int, seed: int) -> tuple[list[dict[str, Any]], str]:
    if budget <= 0 or not rows:
        reason = "external_rows_scaled_to_zero" if rows else ""
        return [], reason
    normalized_rows = [_normalize_dataset_row(row) for row in rows]
    if len(normalized_rows) <= budget:
        return normalized_rows, ""
    selected = random.Random(seed).sample(normalized_rows, budget)
    return selected, "external_rows_capped_to_scaled_budget"


def _external_sources_disabled() -> bool:
    return _env_flag("RUSSIAN_CORRECTOR_DISABLE_EXTERNAL_SOURCES") or _env_flag("RUSSIAN_CORRECTOR_OFFLINE")


def _hf_offline_mode() -> bool:
    return _env_flag("HF_HUB_OFFLINE") or _env_flag("TRANSFORMERS_OFFLINE")


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


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


def _clean_target(index: int, clean_texts: list[str], sentence_factory: CleanSentenceFactory | None = None) -> str:
    if 0 <= index < len(clean_texts):
        return clean_texts[index]
    return (sentence_factory or CleanSentenceFactory()).make(index)


def _hard_negative_count(clean_count: int, ratio: float) -> int:
    if clean_count <= 0:
        return 0
    clamped = _clamp_ratio(ratio)
    if clamped <= 0:
        return 0
    return max(1, min(clean_count, int(round(clean_count * clamped))))


def _hard_negative_target(index: int) -> str:
    places = ["Йоркшире", "Новосибирске", "Казани", "Перми", "Владивостоке", "Самаре"]
    months = ["январь", "март", "июнь", "сентябрь", "ноябрь", "декабрь"]
    nouns = ["отчет", "план", "раздел", "документ", "пример", "модуль"]
    marker = _hard_negative_marker(index)
    place = places[index % len(places)]
    month = months[(index // len(places)) % len(months)]
    noun = nouns[(index // (len(places) * len(months))) % len(nouns)]
    year = 2020 + index % 7
    decimal = f"{10 + index % 17},{index % 10}"
    small = 1 + index % 9
    templates = [
        "В отчете группы {marker} указано 40,16% роста и 5% снижения.",
        "Сайт https://example.com работает, а почта test@example.com указана верно для группы {marker}.",
        "Они могут появиться завтра, а он учится каждый день в группе {marker}.",
        "Кто-то пришел, кое-где были ошибки, и он говорит по-русски для группы {marker}.",
        "Он сказал: «Проект готов» (это важно) для группы {marker}.",
        "Конечно, проект сложный, но команда готова для группы {marker}.",
        "Он работает как инженер в группе {marker}.",
        "Мы проверяем что-то важное для группы {marker}.",
        "69-летний эксперт из США, РФ и НББ согласовал документ для группы {marker}.",
        "Он родился 31 июля {year} года в {place} на севере страны в группе {marker}.",
        "Встреча прошла в субботу в историческом зале мэрии для группы {marker}.",
        "Команда обсудила {noun} за {month} {year} года в группе {marker}.",
        "Получается это решение подходит группе {marker}, но требует проверки.",
        "Значит и следующий вариант остается рабочим для группы {marker}.",
        "Проект демократизации системы сложнее, чем проект обновления группы {marker}.",
        "В отчете группы {marker} указано {decimal}% роста и {small}% снижения.",
        "Песня Lautar с активными девушками бэк-вокалистками вошла в программу группы {marker}.",
        "Компания открыла офис в Москве на Тверской улице для группы {marker}.",
        "Состояние у группы {marker} удовлетворительное и никто в психологической помощи не нуждается.",
    ]
    template = templates[index % len(templates)]
    return template.format(
        place=place,
        month=month,
        noun=noun,
        year=year,
        decimal=decimal,
        small=small,
        marker=marker,
    )


def _punctuation_hard_negative_target(index: int) -> str:
    return _hard_negative_target(index)


def _hard_negative_marker(index: int) -> str:
    adjectives = [
        "алой",
        "белой",
        "быстрой",
        "важной",
        "гибкой",
        "дальней",
        "единой",
        "живой",
        "зимней",
        "краткой",
        "левой",
        "мягкой",
        "новой",
        "общей",
        "первой",
        "ровной",
        "сильной",
        "точной",
        "умной",
        "ясной",
    ]
    nouns = [
        "анкеты",
        "базы",
        "версии",
        "группы",
        "детали",
        "записи",
        "карты",
        "линии",
        "модели",
        "нормы",
        "опции",
        "папки",
        "рамки",
        "схемы",
        "таблицы",
        "формы",
        "цепочки",
        "шкалы",
        "этапа",
        "ячейки",
    ]
    suffixes = [
        "альфа",
        "бета",
        "гамма",
        "дельта",
        "зета",
        "каппа",
        "лямбда",
        "омега",
        "сигма",
        "тау",
        "вектор",
        "контур",
        "профиль",
        "сектор",
        "уровень",
        "фактор",
        "шаблон",
        "элемент",
        "маркер",
        "индекс",
        "поток",
        "режим",
        "сигнал",
        "узел",
        "фрагмент",
    ]
    adjective = adjectives[index % len(adjectives)]
    noun = nouns[(index // len(adjectives)) % len(nouns)]
    suffix = suffixes[(index // (len(adjectives) * len(nouns))) % len(suffixes)]
    return f"{adjective} {noun} {suffix}"


def _clamp_ratio(value: float) -> float:
    return max(0.0, min(1.0, value))


def _build_synthetic_rows_from_clean_corpus(
    clean_texts: list[str],
    *,
    dirty_count: int,
    generator: SyntheticGenerator,
    diff_analyzer: DiffAnalyzer,
    domain: str,
    seen_pairs: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_pairs = seen_pairs if seen_pairs is not None else set()

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
                        source_dataset=example.source_dataset,
                        is_clean=False,
                        is_synthetic=True,
                        domain=domain,
                        rule_ids=example.rule_ids,
                    )
                )

    return rows


def _build_template_synthetic_rows(
    *,
    dirty_count: int,
    generator: SyntheticGenerator,
    diff_analyzer: DiffAnalyzer,
    domain: str,
    sentence_factory: CleanSentenceFactory,
    seen_pairs: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_pairs = seen_pairs if seen_pairs is not None else set()
    index = 0
    while len(rows) < dirty_count:
        target = sentence_factory.make(index)
        example = generator.generate_from_clean(target)
        key = (example.source, example.target)
        edits = _supported_edits(diff_analyzer.analyze(example.source, example.target))
        if example.source != example.target and edits and key not in seen_pairs:
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
                    rule_ids=example.rule_ids,
                )
            )
        index += 1
    return rows


def _build_targeted_lexical_rows(
    error_type: str,
    *,
    required_count: int,
    diff_analyzer: DiffAnalyzer,
    domain: str,
    seen_pairs: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    if required_count <= 0:
        return []
    entries = _lexical_balance_entries(error_type)
    if not entries:
        return []

    rows: list[dict[str, Any]] = []
    seen_pairs = seen_pairs if seen_pairs is not None else set()
    index = 0
    while len(rows) < required_count:
        wrong, correct = entries[index % len(entries)]
        target = _lexical_balance_target(correct, index)
        source = target.replace(correct, wrong, 1)
        key = (source, target)
        edits = _supported_edits(diff_analyzer.analyze(source, target))
        if source != target and key not in seen_pairs and _edits_include_error_type(edits, error_type):
            seen_pairs.add(key)
            rows.append(
                _row(
                    source=source,
                    target=target,
                    edits=edits,
                    source_dataset="synthetic_balanced_rules",
                    is_clean=False,
                    is_synthetic=True,
                    domain=domain,
                    rule_ids=[_lexical_rule_id(error_type)],
                )
            )
        index += 1
    return rows


def _build_targeted_orthography_rows(
    group: str,
    *,
    required_count: int,
    diff_analyzer: DiffAnalyzer,
    domain: str,
    seen_pairs: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    if required_count <= 0 or group not in ORTHOGRAPHY_BALANCE_GROUPS:
        return []
    entries = _orthography_balance_entries(group)
    if not entries:
        return []

    rows: list[dict[str, Any]] = []
    seen_pairs = seen_pairs if seen_pairs is not None else set()
    index = 0
    source_dataset = f"synthetic_balanced_orthography_{group}"
    while len(rows) < required_count:
        wrong, correct, rule_id = entries[index % len(entries)]
        target = _orthography_balance_target(correct, group, index)
        source = target.replace(correct, wrong, 1)
        key = (source, target)
        edits = _orthography_edits_for_pair(
            source,
            target,
            wrong=wrong,
            correct=correct,
            rule_id=rule_id,
            group=group,
            diff_analyzer=diff_analyzer,
        )
        if source != target and key not in seen_pairs and edits:
            seen_pairs.add(key)
            rows.append(
                _row(
                    source=source,
                    target=target,
                    edits=edits,
                    source_dataset=source_dataset,
                    is_clean=False,
                    is_synthetic=True,
                    domain=domain,
                    rule_ids=[rule_id],
                    )
                )
        index += 1
    return rows


def _orthography_edits_for_pair(
    source: str,
    target: str,
    *,
    wrong: str,
    correct: str,
    rule_id: str,
    group: str,
    diff_analyzer: DiffAnalyzer,
) -> list[Edit]:
    edits = _supported_edits(diff_analyzer.analyze(source, target))
    if edits:
        return edits
    if group != "dictionary_fuzzy":
        return []
    start = source.lower().find(wrong.lower())
    if start < 0:
        return []
    return [
        Edit(
            source[start : start + len(wrong)],
            correct,
            "spelling_replace",
            start,
            start + len(wrong),
            confidence=0.85,
            rule_id=rule_id,
        )
    ]


def _build_targeted_punctuation_rows(
    group: str,
    *,
    required_count: int,
    diff_analyzer: DiffAnalyzer,
    domain: str,
    seen_pairs: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    if required_count <= 0 or group not in PUNCTUATION_BALANCE_GROUPS:
        return []

    rows: list[dict[str, Any]] = []
    seen_pairs = seen_pairs if seen_pairs is not None else set()
    index = 0
    source_dataset = f"synthetic_balanced_punctuation_{group}"
    while len(rows) < required_count:
        source, target = _punctuation_balance_pair(group, index)
        key = (source, target)
        edits = _supported_edits(diff_analyzer.analyze(source, target))
        if source != target and key not in seen_pairs and _edits_include_punctuation(edits):
            seen_pairs.add(key)
            rows.append(
                _row(
                    source=source,
                    target=target,
                    edits=edits,
                    source_dataset=source_dataset,
                    is_clean=False,
                    is_synthetic=True,
                    domain=domain,
                    rule_ids=_punctuation_rule_ids(group, source, target),
                )
            )
        index += 1
    return rows


def _punctuation_balance_pair(group: str, index: int) -> tuple[str, str]:
    topic = _punctuation_topic(index)
    if group == "comma_subordinate":
        templates = [
            ("Я думаю что {topic} готов к проверке.", "Я думаю, что {topic} готов к проверке."),
            ("Мы останемся дома если {topic} задержится.", "Мы останемся дома, если {topic} задержится."),
            ("Когда {topic} будет готов мы начнем проверку.", "Когда {topic} будет готов, мы начнем проверку."),
            ("Он пришел чтобы {topic} стал понятнее.", "Он пришел, чтобы {topic} стал понятнее."),
        ]
    elif group == "comma_conjunction":
        templates = [
            ("{topic} готов но требует проверки.", "{topic} готов, но требует проверки."),
            ("{topic} небольшой а результат важный.", "{topic} небольшой, а результат важный."),
            ("Мы начали проверку но {topic} еще сырой.", "Мы начали проверку, но {topic} еще сырой."),
            ("Автор сохранил текст а редактор проверил {topic}.", "Автор сохранил текст, а редактор проверил {topic}."),
        ]
    elif group == "introductory":
        templates = [
            ("Конечно {topic} требует внимания.", "Конечно, {topic} требует внимания."),
            ("Например {topic} можно проверить отдельно.", "Например, {topic} можно проверить отдельно."),
            ("Однако {topic} остается рабочим.", "Однако, {topic} остается рабочим."),
            ("Во-первых {topic} уже готов.", "Во-первых, {topic} уже готов."),
        ]
    elif group == "address_comma":
        templates = [
            ("Коллеги проверьте {topic}.", "Коллеги, проверьте {topic}."),
            ("Иван открой {topic}.", "Иван, открой {topic}."),
            ("Мария посмотри {topic}.", "Мария, посмотри {topic}."),
            ("Коллеги исправим {topic}.", "Коллеги, исправим {topic}."),
        ]
    elif group == "homogeneous_members":
        templates = [
            ("Мы проверили и файл и {topic}.", "Мы проверили и файл, и {topic}."),
            ("Автор сохранил ни план ни {topic}.", "Автор сохранил ни план, ни {topic}."),
            ("В архиве есть и отчет и {topic}.", "В архиве есть и отчет, и {topic}."),
            ("Редактор смотрит и текст и {topic}.", "Редактор смотрит и текст, и {topic}."),
        ]
    elif group == "detached_members":
        templates = [
            ("Закончив работу мы проверили {topic}.", "Закончив работу, мы проверили {topic}."),
            ("Сделав правки автор сохранил {topic}.", "Сделав правки, автор сохранил {topic}."),
            ("Прочитав отчет редактор открыл {topic}.", "Прочитав отчет, редактор открыл {topic}."),
            ("Закончив проверку команда приняла {topic}.", "Закончив проверку, команда приняла {topic}."),
        ]
    elif group == "colon":
        templates = [
            ("Он сказал «{topic} готов».", "Он сказал: «{topic} готов»."),
            ("Нужно проверить следующее {topic}, отчет и план.", "Нужно проверить следующее: {topic}, отчет и план."),
            ("Автор отметил главное {topic} важен.", "Автор отметил главное: {topic} важен."),
            ("В списке три пункта {topic}, файл и отчет.", "В списке три пункта: {topic}, файл и отчет."),
        ]
    elif group == "dash":
        templates = [
            ("{topic} это важный результат.", "{topic} — это важный результат."),
            ("{topic} часть общего плана.", "{topic} — часть общего плана."),
            ("Главная задача это проверить текст.", "Главная задача — это проверить текст."),
            ("Итоговый вывод рабочий вариант.", "Итоговый вывод — рабочий вариант."),
        ]
    elif group == "subject_predicate_dash":
        templates = [
            ("Москва это столица.", "Москва — это столица."),
            ("Главная задача это проверить {topic}.", "Главная задача — это проверить {topic}."),
            ("Итоговый вывод это рабочий вариант.", "Итоговый вывод — это рабочий вариант."),
            ("Документ это важный результат.", "Документ — это важный результат."),
        ]
    elif group == "direct_speech":
        templates = [
            ("Он сказал {topic} готов.", "Он сказал: «{topic} готов»."),
            ("Она ответила {topic} принят.", "Она ответила: «{topic} принят»."),
            ("Редактор спросил {topic} готов?", "Редактор спросил: «{topic} готов?»"),
            ("«{topic} готов» сказал автор.", "«{topic} готов» — сказал автор."),
        ]
    elif group == "semicolon":
        templates = [
            ("Первая часть готова, вторая требует проверки {topic}.", "Первая часть готова; вторая требует проверки {topic}."),
            ("Документ сохранен, отчет еще открыт для {topic}.", "Документ сохранен; отчет еще открыт для {topic}."),
            ("Текст короткий, пример остается понятным для {topic}.", "Текст короткий; пример остается понятным для {topic}."),
            ("План принят, правки будут завтра по теме {topic}.", "План принят; правки будут завтра по теме {topic}."),
        ]
    elif group == "quotes_brackets":
        templates = [
            ("Он сказал: {topic} готов это важно.", "Он сказал: «{topic} готов» (это важно)."),
            ("Автор назвал это {topic} в отчете смотри приложение.", "Автор назвал это «{topic}» в отчете (смотри приложение)."),
            ("Нужно проверить {topic} сегодня это важно.", "Нужно проверить «{topic}» сегодня (это важно)."),
            ("Комментарий {topic} остался в тексте версия рабочая.", "Комментарий «{topic}» остался в тексте (версия рабочая)."),
        ]
    elif group == "final_punctuation":
        templates = [
            ("Как проверить {topic}", "Как проверить {topic}?"),
            ("Проверь {topic}", "Проверь {topic}!"),
            ("{topic} готов", "{topic} готов."),
            ("Мы ждали {topic}.", "Мы ждали {topic}…"),
        ]
    elif group == "delete_replace":
        templates = [
            ("Я думаю:: что {topic} готов.", "Я думаю, что {topic} готов."),
            ("{topic},, готов к проверке.", "{topic} готов к проверке."),
            ("Он сказал,, {topic} готов.", "Он сказал: {topic} готов."),
            ("Первая часть готова:: вторая ждет {topic}.", "Первая часть готова; вторая ждет {topic}."),
        ]
    elif group == "punctuation_noise":
        templates = [
            ("Я думаю,, что {topic} готов.", "Я думаю, что {topic} готов."),
            ("{topic} готов!!", "{topic} готов!"),
            ("Он сказал:: {topic} готов.", "Он сказал: {topic} готов."),
            ("Первая часть готова;; вторая ждет {topic}.", "Первая часть готова; вторая ждет {topic}."),
        ]
    else:
        templates = [("{topic} готов", "{topic} готов.")]
    source_template, target_template = templates[index % len(templates)]
    return source_template.format(topic=topic), target_template.format(topic=topic)


def _punctuation_topic(index: int) -> str:
    nouns = [
        "проект",
        "отчет",
        "раздел",
        "документ",
        "пример",
        "модуль",
        "абзац",
        "файл",
        "вывод",
        "план",
    ]
    adjective = ["рабочий", "важный", "точный", "новый", "итоговый"][index % 5]
    noun = nouns[(index // 5) % len(nouns)]
    return f"{adjective} {noun} {index}"


def _scaled_synthetic_targets(
    config: DatasetBuildConfig,
    dirty_count: int,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    lexical_requested = {
        "spelling": max(0, config.min_spelling_examples),
        "split_join": max(0, config.min_split_join_examples),
        "hyphen": max(0, config.min_hyphen_examples),
    }
    orthography_requested = {
        group: max(0, count)
        for group, count in config.orthography_balance
        if group in ORTHOGRAPHY_BALANCE_GROUPS
    }
    punctuation_requested = {
        group: max(0, count)
        for group, count in config.punctuation_balance
        if group in PUNCTUATION_BALANCE_GROUPS
    }
    entries: list[tuple[str, str, int]] = [
        *[("lexical", key, count) for key, count in lexical_requested.items() if count > 0],
        *[("orthography", key, count) for key, count in orthography_requested.items() if count > 0],
        *[("punctuation", key, count) for key, count in punctuation_requested.items() if count > 0],
    ]
    allocations = _scale_target_entries(entries, dirty_count)
    return (
        {name: count for (kind, name), count in allocations.items() if kind == "lexical"},
        {name: count for (kind, name), count in allocations.items() if kind == "orthography"},
        {name: count for (kind, name), count in allocations.items() if kind == "punctuation"},
    )


def _scaled_balance_targets(config: DatasetBuildConfig, dirty_count: int) -> dict[str, int]:
    lexical_targets, _orthography_targets, _punctuation_targets = _scaled_synthetic_targets(config, dirty_count)
    return lexical_targets


def _scale_target_entries(entries: list[tuple[str, str, int]], dirty_count: int) -> dict[tuple[str, str], int]:
    if dirty_count <= 0 or not entries:
        return {}
    requested_total = sum(count for _kind, _name, count in entries)
    if requested_total <= dirty_count:
        return {(kind, name): count for kind, name, count in entries}

    if dirty_count < len(entries):
        ranked = sorted(entries, key=lambda item: (-item[2], item[0], item[1]))
        selected = {(kind, name): 1 for kind, name, _count in ranked[:dirty_count]}
        return {(kind, name): selected[(kind, name)] for kind, name, _count in entries if (kind, name) in selected}

    raw_allocations: list[tuple[tuple[str, str], float, int]] = []
    allocations: dict[tuple[str, str], int] = {}
    assigned = 0
    for kind, name, count in entries:
        key = (kind, name)
        raw_value = dirty_count * count / requested_total
        value = max(1, int(raw_value))
        raw_allocations.append((key, raw_value, count))
        allocations[key] = value
        assigned += value

    while assigned > dirty_count:
        reducible = [item for item in raw_allocations if allocations[item[0]] > 1]
        key, _raw_value, _count = min(
            reducible,
            key=lambda item: (item[1] - int(item[1]), -allocations[item[0]], item[0][0], item[0][1]),
        )
        allocations[key] -= 1
        assigned -= 1

    while assigned < dirty_count:
        key, _raw_value, _count = max(
            raw_allocations,
            key=lambda item: (item[1] - int(item[1]), item[2], item[0][0], item[0][1]),
        )
        allocations[key] += 1
        assigned += 1

    return allocations


def _punctuation_balance_from_config(data_config: dict[str, Any]) -> tuple[tuple[str, int], ...]:
    synthetic_balance = data_config.get("synthetic_balance", {})
    configured = synthetic_balance.get("punctuation_groups", {})
    return tuple(
        (
            group,
            int(configured.get(f"{group}_min_examples", configured.get(group, DEFAULT_PUNCTUATION_BALANCE[group]))),
        )
        for group in PUNCTUATION_BALANCE_GROUPS
    )


def _orthography_balance_from_config(data_config: dict[str, Any]) -> tuple[tuple[str, int], ...]:
    synthetic_balance = data_config.get("synthetic_balance", {})
    configured = synthetic_balance.get("orthography_groups", {})
    return tuple(
        (
            group,
            int(configured.get(f"{group}_min_examples", configured.get(group, DEFAULT_ORTHOGRAPHY_BALANCE[group]))),
        )
        for group in ORTHOGRAPHY_BALANCE_GROUPS
    )


def _lexical_balance_entries(error_type: str) -> list[tuple[str, str]]:
    if error_type == "spelling":
        return [
            (wrong, correct)
            for wrong, correct in WRONG_TO_CORRECT.items()
            if " " not in correct and "-" not in correct
        ]
    if error_type == "split_join":
        return [
            (wrong, correct)
            for wrong, correct in WRONG_TO_CORRECT.items()
            if " " in wrong or " " in correct
        ]
    if error_type == "hyphen":
        return [(wrong, correct) for wrong, correct in HYPHEN_WHITELIST.items() if wrong != correct]
    return []


def _orthography_balance_entries(group: str) -> list[tuple[str, str, str]]:
    if group == "ne_verb":
        forms = _known_forms(
            [
                "думать",
                "знать",
                "работать",
                "понимать",
                "хотеть",
                "делать",
                "читать",
                "писать",
                "говорить",
                "видеть",
                "слышать",
                "помнить",
                "любить",
                "играть",
                "спать",
                "идти",
                "ехать",
                "решать",
                "смотреть",
                "отвечать",
            ],
            poses={"VERB", "INFN"},
            limit=320,
        )
        return _rule_backed_orthography_entries(group, [f"не {form}" for form in forms])
    if group == "ne_pos":
        return _rule_backed_orthography_entries(
            group,
            [
                "некрасивый",
                "неинтересный",
                "непонятный",
                "непрочитанный",
                "непроверенный",
                "недолго",
                "небыстро",
                "несложно",
            ],
        )
    if group == "tsya":
        return _rule_backed_orthography_entries(
            group,
            _known_forms(
                [
                    "учиться",
                    "стараться",
                    "смеяться",
                    "бояться",
                    "заниматься",
                    "готовиться",
                    "получаться",
                    "казаться",
                    "улыбаться",
                    "делаться",
                ],
                poses={"VERB", "INFN"},
                limit=320,
            ),
        )
    if group == "combo":
        return _rule_backed_orthography_entries(
            group,
            _known_forms(
                [
                    "жизнь",
                    "живой",
                    "животное",
                    "широкий",
                    "ширина",
                    "машина",
                    "частый",
                    "часто",
                    "защита",
                    "чаща",
                    "чудо",
                    "чувство",
                    "щука",
                    "искать",
                    "писать",
                    "держать",
                    "сказать",
                    "хотеть",
                    "молчать",
                    "тащить",
                ],
                limit=520,
            ),
        )
    if group == "hard_sign":
        return _rule_backed_orthography_entries(group, _hard_sign_terms(limit=420))
    if group == "prefix_z_s":
        return _rule_backed_orthography_entries(group, _prefix_z_s_terms(limit=520))
    if group == "prefix_pre_pri":
        return _rule_backed_orthography_entries(
            group,
            _known_forms(
                [
                    "превосходный",
                    "приблизительный",
                    "преувеличивать",
                    "приоритет",
                    "привычный",
                    "прекрасный",
                    "препятствие",
                    "прибрежный",
                ],
                limit=420,
            ),
        )
    if group == "ci":
        return _rule_backed_orthography_entries(
            group,
            _known_forms(
                [
                    "цифра",
                    "цирк",
                    "цитата",
                    "цивилизация",
                    "цикл",
                    "циркуль",
                    "цистерна",
                    "цилиндр",
                    "циничный",
                    "цинга",
                    "циновка",
                    "цифровой",
                    "медицина",
                    "акация",
                    "станция",
                    "операция",
                    "лекция",
                    "традиция",
                    "полиция",
                    "нация",
                ],
                limit=520,
            ),
        )
    if group == "hissing_o_e":
        return _rule_backed_orthography_entries(
            group,
            _known_forms(
                [
                    "шел",
                    "пришел",
                    "нашел",
                    "желтый",
                    "черный",
                    "дешевый",
                    "печеный",
                    "тушеный",
                    "сгущенный",
                    "жесткий",
                    "шелковый",
                ],
                limit=520,
            ),
        )
    if group == "n_nn":
        return _rule_backed_orthography_entries(
            group,
            _known_forms(
                [
                    "длинный",
                    "раненный",
                    "раненый",
                    "жареный",
                    "жаренный",
                    "прочитан",
                    "искусственный",
                    "деревянный",
                    "ветреный",
                    "сделанный",
                ],
                limit=520,
            ),
        )
    if group == "context_pairs":
        return _rule_backed_orthography_entries(
            group,
            [
                "так же",
                "также",
                "то же",
                "тоже",
                "что бы",
                "чтобы",
                "за то",
                "зато",
                "в следствие",
                "вследствие",
                "не смотря",
                "несмотря",
                "не смотря на",
                "несмотря на",
            ],
        )
    if group == "dictionary_fuzzy":
        return [
            (dirty, clean, "dictionary_fuzzy")
            for clean, dirty in DICTIONARY_FUZZY_SYNTHETIC_ERRORS.items()
        ]
    return []


def _orthography_balance_target(term: str, group: str, index: int) -> str:
    templates = [
        "В проверочном примере форма «{term}» остается важной для правила {group} в серии {index}.",
        "Редактор видит форму «{term}» и сохраняет обычный контекст правила {group} в серии {index}.",
        "Для обучения модели используется форма «{term}», потому что правило {group} должно быть заметным в серии {index}.",
        "В корпусе встретилась форма «{term}», и это помогает проверить правило {group} в серии {index}.",
    ]
    template = templates[index % len(templates)]
    return template.format(term=term, group=group.replace("_", "-"), index=index)


def _known_forms(lemmas: list[str], poses: set[str] | None = None, limit: int = 300) -> list[str]:
    forms: list[str] = []
    seen: set[str] = set()
    for lemma in lemmas:
        for parsed in morph_analyzer().parse(lemma)[0].lexeme:
            word = parsed.word.replace("ё", "е")
            if word in seen or not re.fullmatch(r"[а-я]+", word):
                continue
            if poses and parsed.tag.POS not in poses:
                continue
            seen.add(word)
            forms.append(word)
            if len(forms) >= limit:
                return forms
    return forms


def _rule_backed_orthography_entries(group: str, clean_terms: list[str]) -> list[tuple[str, str, str]]:
    entries: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    rules = orthography_rules()
    for clean in clean_terms:
        for rule in rules:
            if not hasattr(rule, "generate_corruptions"):
                continue
            for corruption in rule.generate_corruptions(clean):
                if corruption.group != group:
                    continue
                dirty = corruption.apply(clean)
                key = (dirty, clean, corruption.rule_id)
                if dirty == clean or key in seen:
                    continue
                seen.add(key)
                entries.append(key)
    return entries


def _hard_sign_terms(limit: int) -> list[str]:
    return [
        form
        for form in _known_forms(
            [
                "подъезд",
                "объект",
                "объявление",
                "съезд",
                "въезд",
                "изъян",
                "разъяснение",
                "объяснение",
                "предъявление",
                "съемка",
                "адъютант",
                "конъюнктура",
                "субъект",
                "инъекция",
                "объятие",
                "объединение",
            ],
            limit=420,
        )
        if "ъ" in form
    ][:limit]


def _prefix_z_s_terms(limit: int) -> list[str]:
    return _known_forms(
        [
            "бесполезный",
            "бесплатный",
            "беспокойный",
            "бесконечный",
            "бесшумный",
            "безвкусный",
            "безграмотный",
            "бездарный",
            "безбрежный",
            "разбить",
            "рассказать",
            "расписать",
            "исписать",
            "избить",
            "воспитать",
            "возвратить",
            "вспомнить",
            "взбить",
        ],
        limit=max(limit, 700),
    )[:limit]


def _lexical_balance_target(term: str, index: int) -> str:
    nouns = [
        "тексте",
        "отчете",
        "разделе",
        "документе",
        "примере",
        "модуле",
        "словаре",
        "корпусе",
        "абзаце",
        "файле",
        "выводе",
        "плане",
    ]
    adjectives = [
        "важный",
        "точный",
        "рабочий",
        "учебный",
        "итоговый",
        "понятный",
        "полезный",
        "краткий",
    ]
    actions = [
        "проверяет",
        "сравнивает",
        "записывает",
        "обновляет",
        "читает",
        "сохраняет",
        "разбирает",
        "отмечает",
    ]
    templates = [
        "В рабочем {noun} встречается форма «{term}», потому что это {adjective} пример.",
        "Редактор {action} выражение «{term}», когда готовит {adjective} {noun}.",
        "Для проверки правила используется форма «{term}», и этот {noun} остается {adjective}.",
        "В учебном {noun} есть вариант «{term}», который помогает проверить {adjective} случай.",
        "Автор {action} строку с формой «{term}», чтобы сохранить {adjective} контекст.",
    ]
    noun = nouns[index % len(nouns)]
    adjective = adjectives[(index // len(nouns)) % len(adjectives)]
    action = actions[(index // (len(nouns) * len(adjectives))) % len(actions)]
    template = templates[(index // (len(nouns) * len(adjectives) * len(actions))) % len(templates)]
    sentence = template.format(noun=noun, adjective=adjective, action=action, term=term)
    if sentence.endswith("."):
        return f"{sentence[:-1]} в серии {index}."
    return f"{sentence} в серии {index}."


def _count_rows_with_error_type(rows: list[dict[str, Any]], error_type: str) -> int:
    return sum(error_type in _row_error_types(row) for row in rows)


def _count_rows_from_source_dataset(rows: list[dict[str, Any]], source_dataset: str) -> int:
    return sum(row.get("source_dataset") == source_dataset for row in rows)


def _edits_include_error_type(edits: list[Edit], error_type: str) -> bool:
    return any(coarse_error_type(edit.edit_type) == error_type for edit in edits)


def _edits_include_punctuation(edits: list[Edit]) -> bool:
    return any(coarse_error_type(edit.edit_type) in {"punctuation", "final_punctuation"} for edit in edits)


def _row_error_types(row: dict[str, Any]) -> set[str]:
    value = row.get("error_types", [])
    if isinstance(value, list):
        return {str(item) for item in value}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {item.strip() for item in value.split(",") if item.strip()}
        if isinstance(parsed, list):
            return {str(item) for item in parsed}
        return {str(parsed)} if str(parsed) else set()
    return set()


def _supported_edits(edits: list[Edit]) -> list[Edit]:
    if any(not is_allowed_edit_type(edit.edit_type) for edit in edits):
        return []
    if any(
        is_context_dependent_pair(edit.source, edit.replacement)
        and not str(edit.rule_id).startswith("context_")
        for edit in edits
    ):
        return []
    return list(edits)


def _lexical_rule_id(error_type: str) -> str:
    if error_type == "hyphen":
        return "hyphen_whitelist"
    return "frequent_error_exact"


def _orthography_rule_id(group: str, wrong: str, correct: str) -> str:
    if group == "ne_verb":
        return "ne_verb"
    if group == "tsya":
        if wrong.endswith("тся") and correct.endswith("ться"):
            return "tsya_soft_insert"
        return "tsya_soft_delete"
    if group == "hard_sign":
        if "ь" in wrong and "ъ" in correct:
            return "soft_to_hard_sign"
        return "missing_hard_sign"
    if group == "prefix_z_s":
        return "prefix_z_to_s" if wrong[:3] in {"без", "раз", "воз", "низ"} else "prefix_s_to_z"
    return group


def _punctuation_rule_id(group: str) -> str:
    return {
        "comma_subordinate": "comma_subordinate",
        "comma_conjunction": "comma_conjunction",
        "introductory": "introductory_comma",
        "address_comma": "address_comma",
        "homogeneous_members": "homogeneous_comma",
        "detached_members": "detached_adverbial_comma",
        "colon": "enumeration_colon",
        "dash": "subject_predicate_dash",
        "subject_predicate_dash": "subject_predicate_dash",
        "direct_speech": "direct_speech_colon",
        "semicolon": "semicolon",
        "quotes_brackets": "quote_pair_balance",
        "final_punctuation": "final_punctuation_default",
        "delete_replace": "punctuation_delete_replace",
        "punctuation_noise": "punctuation_delete_replace",
    }.get(group, group)


def _punctuation_rule_ids(group: str, source: str, target: str) -> list[str]:
    if group == "direct_speech":
        if "—" in target and "—" not in source:
            return ["direct_speech_dash"]
        return ["direct_speech_colon", "direct_speech_quotes", "direct_speech_quotes"]
    if group == "quotes_brackets":
        rule_ids: list[str] = []
        if "«" in target and "«" not in source:
            rule_ids.append("quote_open")
        if "»" in target and "»" not in source:
            rule_ids.append("quote_close")
        if "(" in target and "(" not in source:
            rule_ids.append("bracket_pair_balance")
        if ")" in target and ")" not in source:
            rule_ids.append("bracket_pair_balance")
        return rule_ids or ["quote_pair_balance"]
    return [_punctuation_rule_id(group)]


def _normalize_dataset_row(row: dict[str, Any], *, default_domain: str = "synthetic_general") -> dict[str, Any]:
    edit_operations = _normalize_edit_operations(row.get("edit_operations", []))
    return {
        "source": str(row.get("source", "")),
        "target": str(row.get("target", "")),
        "error_types": _normalize_error_types(row.get("error_types", []), edit_operations),
        "source_dataset": str(row.get("source_dataset", "unknown")),
        "is_clean": _as_bool(row.get("is_clean", False)),
        "is_synthetic": _as_bool(row.get("is_synthetic", False)),
        "split": str(row.get("split", "train") or "train"),
        "domain": str(row.get("domain", default_domain) or default_domain),
        "edit_operations": edit_operations,
    }


def _normalize_error_types(value: Any, edit_operations_json: str) -> str:
    error_types = [str(item) for item in _parse_jsonish_list(value) if str(item)]
    if not error_types:
        for operation in _parse_jsonish_list(edit_operations_json):
            if not isinstance(operation, dict):
                continue
            error_type = coarse_error_type(str(operation.get("edit_type", "")))
            if error_type != "unknown":
                error_types.append(error_type)
    return json.dumps(sorted(set(error_types)), ensure_ascii=False)


def _normalize_edit_operations(value: Any) -> str:
    operations: list[dict[str, Any]] = []
    for operation in _parse_jsonish_list(value):
        if isinstance(operation, Edit):
            record = asdict(operation)
        elif isinstance(operation, dict):
            record = dict(operation)
        else:
            continue
        operations.append(
            {
                "source": str(record.get("source", "")),
                "replacement": str(record.get("replacement", "")),
                "edit_type": str(record.get("edit_type", "unknown") or "unknown"),
                "start": _as_int(record.get("start", -1), -1),
                "end": _as_int(record.get("end", -1), -1),
                "status": str(record.get("status", "proposed") or "proposed"),
                "reason": str(record.get("reason", "")),
                "confidence": _as_float(record.get("confidence", 1.0), 1.0),
                "rule_id": _normalize_rule_id(record.get("rule_id", "")),
            }
        )
    return json.dumps(operations, ensure_ascii=False)


def _parse_jsonish_list(value: Any) -> list[Any]:
    if _is_missing_value(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple | set):
        return list(value)
    if isinstance(value, Edit):
        return [value]
    if isinstance(value, dict):
        return [value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(stripped)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, tuple | set):
                return list(parsed)
            if isinstance(parsed, dict):
                return [parsed]
            if parsed is None:
                return []
            return [parsed]
        return [item.strip() for item in stripped.split(",") if item.strip()]
    return [value]


def _normalize_rule_id(value: Any) -> str:
    return normalize_rule_id(value)


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "none", "nan", "null"}:
        return True
    return False


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_int(value: Any, default: int) -> int:
    try:
        if _is_missing_value(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        if _is_missing_value(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _row(
    *,
    source: str,
    target: str,
    edits: list[Edit],
    source_dataset: str,
    is_clean: bool,
    is_synthetic: bool,
    domain: str,
    rule_ids: list[str] | None = None,
) -> dict[str, Any]:
    edits = _edits_with_rule_ids(edits, rule_ids or [])
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


def _edits_with_rule_ids(edits: list[Edit], rule_ids: list[str]) -> list[Edit]:
    if not rule_ids:
        return edits
    result: list[Edit] = []
    fallback_index = 0
    for edit in edits:
        if edit.rule_id:
            result.append(edit)
            continue
        rule_id = rule_ids[min(fallback_index, len(rule_ids) - 1)]
        fallback_index += 1
        result.append(replace(edit, rule_id=rule_id))
    return result


def _assign_splits(rows: list[dict[str, Any]], *, val_ratio: float, test_ratio: float) -> None:
    global_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        global_groups.setdefault(_split_group_key(row), []).append(row)

    buckets: dict[str, list[list[dict[str, Any]]]] = {}
    for group in global_groups.values():
        buckets.setdefault(_split_bucket_for_group(group), []).append(group)

    for groups in buckets.values():
        bucket_total = sum(len(group) for group in groups)
        test_target = int(round(bucket_total * test_ratio))
        val_target = int(round(bucket_total * val_ratio))
        test_count = 0
        val_count = 0
        for group in groups:
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


def _split_bucket_for_group(group: list[dict[str, Any]]) -> str:
    for row in group:
        if not bool(row.get("is_synthetic")) and not bool(row.get("is_clean")):
            return f"real:{row.get('source_dataset', 'external')}"
    for row in group:
        if bool(row.get("is_synthetic")) and not bool(row.get("is_clean")):
            return f"synthetic:{row.get('source_dataset', 'synthetic')}"
    row = group[0]
    if bool(row.get("is_clean")):
        return "clean"
    return "other"


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
