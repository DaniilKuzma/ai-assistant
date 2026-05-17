from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.candidates.morphology import morph_analyzer
from src.candidates.frequent_errors import HYPHEN_WHITELIST, WRONG_TO_CORRECT
from src.data.clean_corpus_sources import load_clean_corpus_sentences
from src.data.external_sources import load_external_pair_sources, load_hf_jsonl_pairs
from src.data.synthetic_generator import SyntheticGenerator
from src.validation.diff_analyzer import DiffAnalyzer, Edit
from src.validation.edit_classifier import coarse_error_type, is_allowed_edit_type, is_context_dependent_pair

SPLIT_STRATEGY = "normalized_target_v2"
PUNCTUATION_BALANCE_GROUPS = (
    "comma_subordinate",
    "comma_conjunction",
    "introductory",
    "colon",
    "dash",
    "semicolon",
    "quotes_brackets",
    "final_punctuation",
    "delete_replace",
)
DEFAULT_PUNCTUATION_BALANCE = {
    "comma_subordinate": 20_000,
    "comma_conjunction": 15_000,
    "introductory": 10_000,
    "colon": 8_000,
    "dash": 8_000,
    "semicolon": 5_000,
    "quotes_brackets": 8_000,
    "final_punctuation": 20_000,
    "delete_replace": 15_000,
}
ORTHOGRAPHY_BALANCE_GROUPS = (
    "ne_verb",
    "tsya",
    "combo",
    "hard_sign",
    "prefix_z_s",
    "ci",
    "hissing_o_e",
)
DEFAULT_ORTHOGRAPHY_BALANCE = {
    "ne_verb": 8_000,
    "tsya": 6_000,
    "combo": 8_000,
    "hard_sign": 6_000,
    "prefix_z_s": 6_000,
    "ci": 4_000,
    "hissing_o_e": 5_000,
}


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
    punctuation_hard_negative_clean_ratio: float = 0.20
    orthography_balance: tuple[tuple[str, int], ...] = tuple(DEFAULT_ORTHOGRAPHY_BALANCE.items())
    punctuation_balance: tuple[tuple[str, int], ...] = tuple(DEFAULT_PUNCTUATION_BALANCE.items())


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
    seen_pairs: set[tuple[str, str]] = set()
    balance_targets = _scaled_balance_targets(config, dirty_count)
    for error_type, minimum_count in balance_targets.items():
        missing_count = max(0, minimum_count - _count_rows_with_error_type([*rows, *synthetic_rows], error_type))
        if missing_count <= 0:
            continue
        synthetic_rows.extend(
            _build_targeted_lexical_rows(
                error_type,
                required_count=min(missing_count, dirty_count - len(synthetic_rows)),
                diff_analyzer=diff_analyzer,
                domain=config.domain,
                seen_pairs=seen_pairs,
            )
        )

    for group, minimum_count in config.orthography_balance:
        missing_count = max(
            0,
            minimum_count
            - _count_rows_from_source_dataset([*rows, *synthetic_rows], f"synthetic_balanced_orthography_{group}"),
        )
        if missing_count <= 0:
            continue
        synthetic_rows.extend(
            _build_targeted_orthography_rows(
                group,
                required_count=min(missing_count, dirty_count - len(synthetic_rows)),
                diff_analyzer=diff_analyzer,
                domain=config.domain,
                seen_pairs=seen_pairs,
            )
        )

    for group, minimum_count in config.punctuation_balance:
        missing_count = max(0, minimum_count - _count_rows_from_source_dataset([*rows, *synthetic_rows], f"synthetic_balanced_punctuation_{group}"))
        if missing_count <= 0:
            continue
        synthetic_rows.extend(
            _build_targeted_punctuation_rows(
                group,
                required_count=min(missing_count, dirty_count - len(synthetic_rows)),
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
        if len(synthetic_rows) < dirty_count:
            synthetic_rows.extend(
                _build_template_synthetic_rows(
                    dirty_count=dirty_count - len(synthetic_rows),
                    generator=generator,
                    diff_analyzer=diff_analyzer,
                    domain=config.domain,
                    sentence_factory=sentence_factory,
                    seen_pairs=seen_pairs,
                )
            )
    else:
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

    hard_negative_count = int(round(clean_count * _clamp_ratio(config.punctuation_hard_negative_clean_ratio)))
    for clean_index in range(clean_count):
        if clean_index < hard_negative_count:
            target = _punctuation_hard_negative_target(clean_index)
            source_dataset = "clean_identity_punctuation_hard_negative"
        else:
            target = _clean_target(clean_index - hard_negative_count, clean_texts, sentence_factory)
            source_dataset = "clean_identity"
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
            "error_type_counts": error_type_counts(rows),
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
        min_spelling_examples=int(data_config.get("synthetic_balance", {}).get("spelling_min_examples", 60_000)),
        min_split_join_examples=int(data_config.get("synthetic_balance", {}).get("split_join_min_examples", 20_000)),
        min_hyphen_examples=int(data_config.get("synthetic_balance", {}).get("hyphen_min_examples", 20_000)),
        punctuation_hard_negative_clean_ratio=float(data_config.get("punctuation_hard_negative_clean_ratio", 0.20)),
        orthography_balance=_orthography_balance_from_config(data_config),
        punctuation_balance=_punctuation_balance_from_config(data_config),
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


def error_type_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for error_type in _row_error_types(row):
            counts[error_type] = counts.get(error_type, 0) + 1
    return dict(sorted(counts.items()))


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


def _clean_target(index: int, clean_texts: list[str], sentence_factory: CleanSentenceFactory) -> str:
    if 0 <= index < len(clean_texts):
        return clean_texts[index]
    return sentence_factory.make(index - len(clean_texts))


def _punctuation_hard_negative_target(index: int) -> str:
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
                        source_dataset="synthetic_rules",
                        is_clean=False,
                        is_synthetic=True,
                        domain=domain,
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
        wrong, correct = entries[index % len(entries)]
        target = _orthography_balance_target(correct, group, index)
        source = target.replace(correct, wrong, 1)
        key = (source, target)
        edits = _supported_edits(diff_analyzer.analyze(source, target))
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
                )
            )
        index += 1
    return rows


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


def _scaled_balance_targets(config: DatasetBuildConfig, dirty_count: int) -> dict[str, int]:
    requested = {
        "spelling": max(0, config.min_spelling_examples),
        "split_join": max(0, config.min_split_join_examples),
        "hyphen": max(0, config.min_hyphen_examples),
    }
    requested_total = sum(requested.values())
    if dirty_count <= 0 or requested_total <= dirty_count:
        return requested

    scaled: dict[str, int] = {}
    assigned = 0
    for error_type, count in requested.items():
        value = int(dirty_count * count / requested_total)
        if count > 0 and value == 0:
            value = 1
        scaled[error_type] = value
        assigned += value

    while assigned > dirty_count:
        largest = max(scaled, key=lambda key: scaled[key])
        scaled[largest] -= 1
        assigned -= 1
    while assigned < dirty_count:
        largest = max(requested, key=lambda key: requested[key])
        scaled[largest] += 1
        assigned += 1
    return scaled


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


def _orthography_balance_entries(group: str) -> list[tuple[str, str]]:
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
        return [(f"не{form}", f"не {form}") for form in forms]
    if group == "tsya":
        return [
            (wrong, correct)
            for correct in _known_forms(
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
            )
            for wrong in [_toggle_tsya(correct)]
            if wrong
        ]
    if group == "combo":
        return _corrupt_forms_with_replacements(
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
            {"жи": "жы", "ши": "шы", "ча": "чя", "ща": "щя", "чу": "чю", "щу": "щю"},
            limit=520,
        )
    if group == "hard_sign":
        return _hard_sign_cases(limit=420)
    if group == "prefix_z_s":
        return _prefix_z_s_cases(limit=520)
    if group == "ci":
        return _corrupt_forms_with_replacements(
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
            {"ци": "цы"},
            limit=520,
        )
    if group == "hissing_o_e":
        return _corrupt_forms_with_replacements(
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
            {"же": "жо", "ше": "шо", "че": "чо", "ще": "що"},
            limit=520,
        )
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


def _toggle_tsya(word: str) -> str | None:
    if word.endswith("ться"):
        return word[: -len("ться")] + "тся"
    if word.endswith("тся"):
        return word[: -len("тся")] + "ться"
    return None


def _hard_sign_cases(limit: int) -> list[tuple[str, str]]:
    correct_forms = _known_forms(
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
    cases: list[tuple[str, str]] = []
    for correct in correct_forms:
        if "ъ" not in correct:
            continue
        cases.append((correct.replace("ъ", ""), correct))
        cases.append((correct.replace("ъ", "ь"), correct))
        if len(cases) >= limit:
            return cases
    return cases


def _prefix_z_s_cases(limit: int) -> list[tuple[str, str]]:
    correct_forms = _known_forms(
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
        limit=700,
    )
    cases: list[tuple[str, str]] = []
    for correct in correct_forms:
        wrong = _toggle_prefix_z_s(correct)
        if wrong and wrong != correct:
            cases.append((wrong, correct))
            if len(cases) >= limit:
                return cases
    return cases


def _toggle_prefix_z_s(word: str) -> str | None:
    for z_prefix, s_prefix in [
        ("без", "бес"),
        ("раз", "рас"),
        ("из", "ис"),
        ("воз", "вос"),
        ("вз", "вс"),
    ]:
        if word.startswith(z_prefix):
            return s_prefix + word[len(z_prefix) :]
        if word.startswith(s_prefix):
            return z_prefix + word[len(s_prefix) :]
    return None


def _corrupt_forms_with_replacements(
    correct_forms: list[str],
    correct_to_wrong: dict[str, str],
    limit: int,
) -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []
    for correct in correct_forms:
        for correct_part, wrong_part in correct_to_wrong.items():
            if correct_part not in correct:
                continue
            wrong = correct.replace(correct_part, wrong_part, 1)
            cases.append((wrong, correct))
            break
        if len(cases) >= limit:
            return cases
    return cases


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
    if any(is_context_dependent_pair(edit.source, edit.replacement) for edit in edits):
        return []
    return list(edits)


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
