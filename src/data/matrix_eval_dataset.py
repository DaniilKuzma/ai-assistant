from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.candidates.candidate_generator import Candidate, CandidateGenerator
from src.candidates.matching import candidate_edit_type_for_labels, candidate_matches_edit
from src.config.load_config import load_config
from src.data.short_dataset_v2 import detect_hard_negative_traps
from src.data.synthetic_generator import TargetedBackfillGenerator
from src.evaluation.candidate_recall import build_candidate_recall_reports
from src.rules.coverage_matrix import iter_coverage_entries, load_rules_coverage
from src.rules.registry import rule_by_id
from src.rules.rule_ids import UNKNOWN_RULE_ID, normalize_rule_id
from src.validation.diff_analyzer import DiffAnalyzer, Edit
from src.validation.edit_classifier import coarse_error_type, is_allowed_edit_type


MATRIX_EVAL_COLUMNS = [
    "source",
    "target",
    "rule_id",
    "rule_ids",
    "error_type",
    "error_types",
    "matrix_group",
    "source_type",
    "candidate_present",
    "candidate_rule_ids",
    "gap_label_present",
    "split",
    "metadata",
    "template_id",
    "normalized_pair_hash",
    "edit_operations",
    "edits",
    "source_dataset",
    "is_clean",
    "is_synthetic",
    "is_hard_negative",
    "domain",
]

DEFAULT_MIN_EXAMPLES_PER_EXECUTABLE_RULE = 50
DEFAULT_PREFERRED_EXAMPLES_PER_EXECUTABLE_RULE = 100
DEFAULT_MAX_EXAMPLES_PER_RULE = 300
RISKY_RULE_IDS = {
    "dictionary_fuzzy",
    "double_consonant_candidate",
    "keyboard_typo_candidate",
    "swapped_letters_candidate",
    "missing_letter_candidate",
    "extra_letter_candidate",
    "yo_e_candidate",
    "capitalization_ner",
    "abbreviation_case_protection",
    "capitalization_sentence_start",
}
DICTIONARY_VALIDATION_RULE_IDS = {
    "dictionary_fuzzy",
    "double_consonant_candidate",
    "keyboard_typo_candidate",
    "swapped_letters_candidate",
    "missing_letter_candidate",
    "extra_letter_candidate",
    "yo_e_candidate",
}


@dataclass(frozen=True)
class MatrixEvalBuildResult:
    frame: pd.DataFrame
    by_rule: pd.DataFrame
    manifest: dict[str, Any]
    rejected: list[dict[str, Any]]


STATIC_PROBES: dict[str, tuple[tuple[str, str], ...]] = {
    "final_punctuation_default": (
        ("Документ готов", "Документ готов."),
        ("Отчет принят", "Отчет принят."),
        ("Решение согласовано", "Решение согласовано."),
    ),
    "comma_subordinate": (
        ("Я думаю что отчет готов.", "Я думаю, что отчет готов."),
        ("Он сказал что проверка завершена.", "Он сказал, что проверка завершена."),
        ("Мы знаем что архив открыт.", "Мы знаем, что архив открыт."),
    ),
    "comma_conjunction": (
        ("Отчет готов но подписи нет.", "Отчет готов, но подписи нет."),
        ("Файл открыт а запись закрыта.", "Файл открыт, а запись закрыта."),
    ),
    "introductory_comma": (
        ("Конечно отчет готов.", "Конечно, отчет готов."),
        ("Во-первых список обновлен.", "Во-первых, список обновлен."),
    ),
    "address_comma": (
        ("Иван подпишите отчет.", "Иван, подпишите отчет."),
        ("Коллеги проверьте список.", "Коллеги, проверьте список."),
    ),
    "homogeneous_comma": (
        ("В папке лежали отчеты письма и акты.", "В папке лежали отчеты, письма и акты."),
    ),
    "detached_adverbial_comma": (
        ("Проверив отчет редактор закрыл файл.", "Проверив отчет, редактор закрыл файл."),
    ),
    "comparative_turnover_comma": (
        ("Он работал как опытный редактор.", "Он работал, как опытный редактор."),
    ),
    "direct_speech_quotes": (
        ("Редактор сказал: отчет готов.", "Редактор сказал: «отчет готов»."),
    ),
    "direct_speech_colon": (
        ("Редактор сказал отчет готов.", "Редактор сказал: отчет готов."),
    ),
    "direct_speech_dash": (
        ("Отчет готов сказал редактор.", "Отчет готов, - сказал редактор."),
    ),
    "enumeration_colon": (
        ("В папке три файла отчет акт письмо.", "В папке три файла: отчет, акт, письмо."),
    ),
    "explanation_colon": (
        ("Решение простое файл нужно закрыть.", "Решение простое: файл нужно закрыть."),
    ),
    "asyndetic_dash": (
        ("Солнце село архив закрыли.", "Солнце село - архив закрыли."),
    ),
    "consequence_dash": (
        ("Файл поврежден отчет не приняли.", "Файл поврежден - отчет не приняли."),
    ),
    "semicolon": (
        ("Первая группа проверила отчет вторая закрыла архив.", "Первая группа проверила отчет; вторая закрыла архив."),
    ),
    "punctuation_delete_replace": (
        ("Документ,, готов.", "Документ, готов."),
        ("Отчет готов..", "Отчет готов."),
        ("Да;; нет::", "Да; нет:"),
    ),
    "capitalization_sentence_start": (
        ("отчет готов. проверка завершена.", "отчет готов. Проверка завершена."),
    ),
    "abbreviation_case_protection": (
        ("в сша подготовили отчет.", "в США подготовили отчет."),
        ("нбб опубликовал решение.", "НББ опубликовал решение."),
    ),
    "quote_open": (
        ("Редактор сказал: отчет готов».", "Редактор сказал: «отчет готов»."),
    ),
    "quote_close": (
        ("Редактор сказал: «отчет готов.", "Редактор сказал: «отчет готов»."),
    ),
    "quote_pair_balance": (
        ("Редактор сказал: отчет готов».", "Редактор сказал: «отчет готов»."),
        ("Редактор сказал: «отчет готов.", "Редактор сказал: «отчет готов»."),
    ),
    "bracket_pair_balance": (
        ("Проверь документ (черновик.", "Проверь документ (черновик)."),
        ("Проверь документ черновик).", "Проверь документ (черновик)."),
    ),
    "frequent_error_exact": (
        ("В течении дня отчет обновили.", "В течение дня отчет обновили."),
        ("На встречу пошли двое сотрудников.", "Навстречу пошли двое сотрудников."),
    ),
    "tsya_soft_delete": (
        ("Нужно учиться быстро.", "Нужно учится быстро."),
    ),
    "tsya_soft_insert": (
        ("Он учится быстро.", "Он учиться быстро."),
    ),
}

DEFAULT_CLEAN_SENTENCES = [
    "Документ готов.",
    "Отчет принят.",
    "Я думаю, что отчет готов.",
    "Он сказал, что проверка завершена.",
    "В папке лежали отчеты, письма и акты.",
    "Коллеги, проверьте список.",
    "Во-первых, список обновлен.",
    "Файл открыт, а запись закрыта.",
    "Проверив отчет, редактор закрыл файл.",
    "Решение согласовано.",
    "ООО «Вектор» подписало акт.",
    "Он учится быстро.",
    "Нужно учиться быстро.",
    "В течение дня отчет обновили.",
    "Список состоит из трех пунктов: отчет, акт, письмо.",
]


def build_matrix_eval_dataset(
    *,
    output_dir: str | Path = "data/processed/matrix_eval",
    reports_dir: str | Path = "reports/matrix_eval",
    rules_config_path: str | Path = "configs/rules.yaml",
    config_path: str | Path = "configs/config.yaml",
    selected_rule_ids: Iterable[str] | None = None,
    min_examples_per_rule: int = DEFAULT_MIN_EXAMPLES_PER_EXECUTABLE_RULE,
    preferred_examples_per_rule: int = DEFAULT_PREFERRED_EXAMPLES_PER_EXECUTABLE_RULE,
    max_examples_per_rule: int = DEFAULT_MAX_EXAMPLES_PER_RULE,
    hard_negative_count: int = 100,
    current_dataset_path: str | Path = "data/processed/short_dataset_v2/correction_dataset.csv.gz",
    clean_pool_path: str | Path = "data/processed/short_dataset_v2/clean_sentence_pool.csv.gz",
    extended_backfill_rule_ids: Iterable[str] | None = None,
) -> MatrixEvalBuildResult:
    del output_dir, reports_dir
    config = _load_optional_config(config_path)
    candidate_generator = _candidate_generator(config)
    fast_candidate_generator = CandidateGenerator(dictionary_lexicon=(), dictionary_limit=0, syntax_provider=lambda _text: ())
    dictionary_candidate_generator = _dictionary_only_candidate_generator(candidate_generator)
    matrix_groups = _matrix_groups(rules_config_path)
    rule_ids = list(selected_rule_ids) if selected_rule_ids is not None else _executable_matrix_rule_ids(matrix_groups)
    rule_ids = [normalize_rule_id(rule_id) for rule_id in rule_ids if normalize_rule_id(rule_id) != UNKNOWN_RULE_ID]
    analyzer = DiffAnalyzer()
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str, str]] = set()
    existing_rows = _read_existing_rows(current_dataset_path)
    existing_rows_by_rule = _existing_rows_by_rule(existing_rows)
    extended_backfill = {
        normalize_rule_id(rule_id)
        for rule_id in (extended_backfill_rule_ids or [])
        if normalize_rule_id(rule_id) != UNKNOWN_RULE_ID
    }

    for rule_id in rule_ids:
        quota = _quota_for_rule(rule_id, min_examples_per_rule, preferred_examples_per_rule)
        target_examples = min(max_examples_per_rule, quota)
        accepted_for_rule = 0
        for source, target, source_type, template_id, source_dataset in _candidate_sources_for_rule(
            rule_id,
            current_rows=existing_rows_by_rule.get(rule_id, []),
            candidate_generator=candidate_generator,
            clean_pool_path=clean_pool_path,
            quota=target_examples,
            extended_backfill=rule_id in extended_backfill,
        ):
            if accepted_for_rule >= target_examples:
                break
            key = (source, target, rule_id)
            if key in seen_pairs:
                continue
            row = _candidate_backed_row(
                source=source,
                target=target,
                rule_id=rule_id,
                matrix_group=matrix_groups.get(rule_id, ""),
                source_type=source_type,
                template_id=template_id,
                source_dataset=source_dataset,
                candidate_generator=_validation_candidate_generator(
                    rule_id,
                    default_generator=candidate_generator,
                    fast_generator=fast_candidate_generator,
                    dictionary_generator=dictionary_candidate_generator,
                ),
                analyzer=analyzer,
            )
            if row is None:
                rejected.append(
                    {
                        "rule_id": rule_id,
                        "source": source,
                        "target": target,
                        "reason": "no_matching_candidate",
                        "source_type": source_type,
                    }
                )
                continue
            rows.append(row)
            seen_pairs.add(key)
            accepted_for_rule += 1

    rows.extend(
        _hard_negative_rows(
            clean_pool_path=clean_pool_path,
            count=hard_negative_count,
            existing_count=len(rows),
        )
    )
    frame = pd.DataFrame(rows, columns=MATRIX_EVAL_COLUMNS)
    by_rule = _by_rule(frame, rule_ids, min_examples_per_rule=min_examples_per_rule)
    manifest = _manifest(frame, by_rule, rejected, rule_ids, min_examples_per_rule, preferred_examples_per_rule, max_examples_per_rule)
    return MatrixEvalBuildResult(frame=frame, by_rule=by_rule, manifest=manifest, rejected=rejected)


def write_matrix_eval_dataset(
    *,
    output_dir: str | Path = "data/processed/matrix_eval",
    reports_dir: str | Path = "reports/matrix_eval",
    rules_config_path: str | Path = "configs/rules.yaml",
    config_path: str | Path = "configs/config.yaml",
    selected_rule_ids: Iterable[str] | None = None,
    min_examples_per_rule: int = DEFAULT_MIN_EXAMPLES_PER_EXECUTABLE_RULE,
    preferred_examples_per_rule: int = DEFAULT_PREFERRED_EXAMPLES_PER_EXECUTABLE_RULE,
    max_examples_per_rule: int = DEFAULT_MAX_EXAMPLES_PER_RULE,
    hard_negative_count: int = 100,
    current_dataset_path: str | Path = "data/processed/short_dataset_v2/correction_dataset.csv.gz",
    clean_pool_path: str | Path = "data/processed/short_dataset_v2/clean_sentence_pool.csv.gz",
    extended_backfill_rule_ids: Iterable[str] | None = None,
) -> dict[str, str]:
    output = Path(output_dir)
    reports = Path(reports_dir)
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    result = build_matrix_eval_dataset(
        output_dir=output,
        reports_dir=reports,
        rules_config_path=rules_config_path,
        config_path=config_path,
        selected_rule_ids=selected_rule_ids,
        min_examples_per_rule=min_examples_per_rule,
        preferred_examples_per_rule=preferred_examples_per_rule,
        max_examples_per_rule=max_examples_per_rule,
        hard_negative_count=hard_negative_count,
        current_dataset_path=current_dataset_path,
        clean_pool_path=clean_pool_path,
        extended_backfill_rule_ids=extended_backfill_rule_ids,
    )
    dataset_path = output / "matrix_eval.csv.gz"
    by_rule_path = output / "matrix_eval_by_rule.csv"
    manifest_path = output / "matrix_eval_manifest.json"
    result.frame.to_csv(dataset_path, index=False)
    result.by_rule.to_csv(by_rule_path, index=False)
    manifest_path.write_text(json.dumps(result.manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = reports / "matrix_eval_dataset_report.md"
    report_path.write_text(_dataset_report(result), encoding="utf-8")
    config = _load_optional_config(config_path)
    recall_reports = build_candidate_recall_reports(
        result.frame.to_dict("records"),
        candidate_generator=_candidate_generator(config),
        rules_config_path=rules_config_path,
    )
    recall_reports["candidate_recall_by_rule"].to_csv(reports / "matrix_eval_candidate_recall_by_rule.csv", index=False)
    recall_reports["gap_label_coverage_by_rule"].to_csv(reports / "matrix_eval_gap_coverage_by_rule.csv", index=False)
    return {
        "dataset_path": str(dataset_path),
        "by_rule_path": str(by_rule_path),
        "manifest_path": str(manifest_path),
        "dataset_report_path": str(report_path),
        "candidate_recall_path": str(reports / "matrix_eval_candidate_recall_by_rule.csv"),
        "gap_coverage_path": str(reports / "matrix_eval_gap_coverage_by_rule.csv"),
    }


def _candidate_sources_for_rule(
    rule_id: str,
    *,
    current_rows: list[dict[str, Any]],
    candidate_generator: CandidateGenerator,
    clean_pool_path: str | Path,
    quota: int,
    extended_backfill: bool = False,
) -> Iterable[tuple[str, str, str, str, str]]:
    existing_attempts = 0
    max_existing_attempts = quota
    yielded = 0
    for row in current_rows:
        if existing_attempts >= max_existing_attempts:
            break
        if rule_id not in _row_rule_ids(row):
            continue
        source = str(row.get("source", ""))
        target = str(row.get("target", ""))
        if source and target and source != target:
            existing_attempts += 1
            yielded += 1
            yield source, target, "matrix_existing_dataset", str(row.get("template_id", "")), "short_dataset_v2"
    for index, (source, target) in enumerate(STATIC_PROBES.get(rule_id, ())):
        if not extended_backfill and yielded >= quota:
            break
        yielded += 1
        yield source, target, "matrix_synthetic_eval", f"static_probe:{rule_id}:{index}", "static_candidate_probe"

    if extended_backfill or yielded < quota:
        backfill_required = quota if extended_backfill else max(0, quota - yielded)
        backfill = TargetedBackfillGenerator(candidate_generator, seed=23).generate_for_rule(rule_id, backfill_required)
        for index, example in enumerate(backfill.examples):
            if not extended_backfill and yielded >= quota:
                break
            yielded += 1
            yield example.source, example.target, "matrix_synthetic_eval", f"targeted_backfill:{rule_id}:{index}", example.source_dataset

    if extended_backfill or yielded < quota:
        for index, (source, target) in enumerate(_rule_corruption_pairs(rule_id, clean_pool_path=clean_pool_path)):
            if index >= quota:
                break
            if not extended_backfill and yielded >= quota:
                break
            yielded += 1
            yield source, target, "matrix_synthetic_eval", f"rule_corruption:{rule_id}:{index}", "rule_generate_corruptions"


def _candidate_backed_row(
    *,
    source: str,
    target: str,
    rule_id: str,
    matrix_group: str,
    source_type: str,
    template_id: str,
    source_dataset: str,
    candidate_generator: CandidateGenerator,
    analyzer: DiffAnalyzer,
) -> dict[str, Any] | None:
    candidates = candidate_generator.generate(source)
    backed_edits = _candidate_backed_edits(source, target, candidates, analyzer)
    if not any(edit.rule_id == rule_id for edit in backed_edits):
        return None
    selected_edits = [edit for edit in backed_edits if edit.rule_id == rule_id]
    candidate_rule_ids = sorted({candidate.rule_id for candidate in candidates if candidate.rule_id})
    edit_payload = [_edit_payload(edit) for edit in selected_edits]
    error_types = sorted({coarse_error_type(edit.edit_type) for edit in selected_edits})
    metadata = {
        "candidate_present": True,
        "target_family": rule_id,
        "candidate_rule_ids": candidate_rule_ids,
        "source_type": source_type,
        "template_id": template_id,
    }
    return {
        "source": source,
        "target": target,
        "rule_id": rule_id,
        "rule_ids": _json([rule_id]),
        "error_type": error_types[0] if error_types else "unknown",
        "error_types": _json(error_types),
        "matrix_group": matrix_group,
        "source_type": source_type,
        "candidate_present": True,
        "candidate_rule_ids": _json(candidate_rule_ids),
        "gap_label_present": any(candidate.rule_id == rule_id and candidate.gap_index is not None for candidate in candidates),
        "split": "matrix_eval",
        "metadata": _json(metadata),
        "template_id": template_id,
        "normalized_pair_hash": _pair_hash(source, target),
        "edit_operations": _json(edit_payload),
        "edits": _json(edit_payload),
        "source_dataset": source_dataset,
        "is_clean": False,
        "is_synthetic": source_type != "matrix_real_eval",
        "is_hard_negative": False,
        "domain": "matrix_eval",
    }


def _candidate_backed_edits(
    source: str,
    target: str,
    candidates: list[Candidate],
    analyzer: DiffAnalyzer,
) -> list[Edit]:
    result: list[Edit] = []
    seen: set[tuple[int, int, str, str, str]] = set()
    for edit in analyzer.analyze(source, target, candidates=candidates):
        if not is_allowed_edit_type(edit.edit_type):
            continue
        matches = [candidate for candidate in candidates if candidate.rule_id and candidate_matches_edit(candidate, edit)]
        if not matches:
            continue
        candidate = matches[0]
        backed = Edit(
            source=edit.source,
            replacement=edit.replacement,
            edit_type=edit.edit_type,
            start=edit.start,
            end=edit.end,
            status=edit.status,
            reason=edit.reason,
            confidence=candidate.confidence,
            rule_id=candidate.rule_id,
        )
        key = (backed.start, backed.end, backed.replacement, backed.edit_type, backed.rule_id)
        if key not in seen:
            seen.add(key)
            result.append(backed)
    for candidate in candidates:
        if not candidate.rule_id or candidate.edit_type == "keep":
            continue
        if _apply_candidate(source, candidate) != target:
            continue
        edit_type = candidate_edit_type_for_labels(candidate.edit_type)
        if not is_allowed_edit_type(edit_type):
            continue
        backed = Edit(
            source=candidate.source,
            replacement=candidate.replacement,
            edit_type=edit_type,
            start=candidate.start,
            end=candidate.end,
            confidence=candidate.confidence,
            rule_id=candidate.rule_id,
        )
        key = (backed.start, backed.end, backed.replacement, backed.edit_type, backed.rule_id)
        if key not in seen:
            seen.add(key)
            result.append(backed)
    return result


def _rule_corruption_pairs(rule_id: str, *, clean_pool_path: str | Path) -> list[tuple[str, str]]:
    rule = rule_by_id(rule_id)
    if rule is None or not hasattr(rule, "generate_corruptions"):
        return []
    pairs: list[tuple[str, str]] = []
    for target in _clean_sentences(clean_pool_path):
        try:
            corruptions = rule.generate_corruptions(target)
        except Exception:
            corruptions = []
        for corruption in corruptions:
            if normalize_rule_id(getattr(corruption, "rule_id", "")) != rule_id:
                continue
            source = corruption.apply(target)
            if source != target:
                pairs.append((source, target))
    return pairs


def _hard_negative_rows(*, clean_pool_path: str | Path, count: int, existing_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if count <= 0:
        return rows
    clean = _clean_sentences(clean_pool_path)
    for index in range(count):
        text = clean[index % len(clean)]
        trap_types = detect_hard_negative_traps(text) or ["clean_identity"]
        metadata = {"trap_types": trap_types, "candidate_present": False, "target_family": "hard_negative"}
        rows.append(
            {
                "source": text,
                "target": text,
                "rule_id": "clean_identity_hard_negative",
                "rule_ids": _json(["clean_identity_hard_negative"]),
                "error_type": "clean_identity",
                "error_types": _json([]),
                "matrix_group": "hard_negative",
                "source_type": "matrix_hard_negative",
                "candidate_present": False,
                "candidate_rule_ids": _json([]),
                "gap_label_present": False,
                "split": "matrix_eval",
                "metadata": _json(metadata),
                "template_id": f"hard_negative:{existing_count + index}",
                "normalized_pair_hash": _pair_hash(text, text),
                "edit_operations": _json([]),
                "edits": _json([]),
                "source_dataset": "clean_sentence_pool",
                "is_clean": True,
                "is_synthetic": False,
                "is_hard_negative": True,
                "domain": "matrix_eval",
            }
        )
    return rows


def _by_rule(frame: pd.DataFrame, requested_rule_ids: list[str], *, min_examples_per_rule: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    positives = frame[frame["source_type"] != "matrix_hard_negative"] if not frame.empty else pd.DataFrame()
    for rule_id in sorted(set(requested_rule_ids)):
        current = positives[positives["rule_id"] == rule_id] if not positives.empty else pd.DataFrame()
        rows.append(
            {
                "rule_id": rule_id,
                "eval_examples": int(len(current)),
                "candidate_backed_examples": int(current["candidate_present"].astype(bool).sum()) if not current.empty else 0,
                "min_quota": int(min_examples_per_rule),
                "quota_met": int(len(current)) >= int(min_examples_per_rule),
            }
        )
    return pd.DataFrame(rows, columns=["rule_id", "eval_examples", "candidate_backed_examples", "min_quota", "quota_met"])


def _manifest(
    frame: pd.DataFrame,
    by_rule: pd.DataFrame,
    rejected: list[dict[str, Any]],
    requested_rule_ids: list[str],
    min_examples_per_rule: int,
    preferred_examples_per_rule: int,
    max_examples_per_rule: int,
) -> dict[str, Any]:
    positives = frame[frame["source_type"] != "matrix_hard_negative"] if not frame.empty else pd.DataFrame()
    underfilled = by_rule[~by_rule["quota_met"]]["rule_id"].tolist() if not by_rule.empty else requested_rule_ids
    no_candidate = [rule_id for rule_id in requested_rule_ids if rule_id not in set(positives.get("rule_id", []))]
    return {
        "verdict": "MATRIX_EVAL_DATASET_READY" if not underfilled else "MATRIX_EVAL_DATASET_PARTIAL",
        "total_rows": int(len(frame)),
        "positive_rows": int(len(positives)),
        "hard_negative_rows": int((frame["source_type"] == "matrix_hard_negative").sum()) if not frame.empty else 0,
        "rule_id_counts": {str(key): int(value) for key, value in Counter(positives.get("rule_id", [])).items()},
        "requested_rule_ids": requested_rule_ids,
        "underfilled_rule_ids": underfilled,
        "no_candidate_path_rule_ids": no_candidate,
        "rejected_count": len(rejected),
        "quota": {
            "min_eval_examples_per_executable_rule": int(min_examples_per_rule),
            "preferred_eval_examples_per_executable_rule": int(preferred_examples_per_rule),
            "max_eval_examples_per_rule": int(max_examples_per_rule),
            "risky_min_examples": 30,
        },
        "taxonomy_reference": {
            "orthography": "https://orfogrammka.ru/орфография/",
            "punctuation": "https://orfogrammka.ru/пунктуация/",
        },
    }


def _dataset_report(result: MatrixEvalBuildResult) -> str:
    underfilled = result.manifest.get("underfilled_rule_ids", [])
    lines = [
        "# Matrix Eval Dataset Report",
        "",
        f"- verdict: {result.manifest['verdict']}",
        f"- total rows: {result.manifest['total_rows']}",
        f"- positive rows: {result.manifest['positive_rows']}",
        f"- hard negative rows: {result.manifest['hard_negative_rows']}",
        f"- requested rule ids: {len(result.manifest.get('requested_rule_ids', []))}",
        f"- underfilled rule ids: {len(underfilled)}",
        f"- rejected candidate pairs: {result.manifest['rejected_count']}",
        "",
        "This corpus is eval/audit-only and is not a training set.",
        "",
    ]
    if underfilled:
        lines.extend(["## Underfilled Rules", "", *[f"- {rule_id}" for rule_id in underfilled], ""])
    return "\n".join(lines)


def _matrix_groups(rules_config_path: str | Path) -> dict[str, str]:
    groups: dict[str, str] = {}
    coverage = load_rules_coverage(rules_config_path)
    for _section, group, entry in iter_coverage_entries(coverage):
        for raw_rule_id in entry.get("rules", []):
            rule_id = normalize_rule_id(raw_rule_id)
            if rule_id != UNKNOWN_RULE_ID:
                groups.setdefault(rule_id, group)
    return groups


def _executable_matrix_rule_ids(matrix_groups: dict[str, str]) -> list[str]:
    return sorted(rule_id for rule_id in matrix_groups if rule_by_id(rule_id) is not None)


def _quota_for_rule(rule_id: str, min_examples: int, preferred_examples: int) -> int:
    if rule_id in RISKY_RULE_IDS:
        return max(min_examples, min(30, preferred_examples))
    return max(min_examples, preferred_examples)


def _candidate_generator(config: dict[str, Any]) -> CandidateGenerator:
    try:
        if config:
            return CandidateGenerator.from_config(config)
    except Exception:
        pass
    return CandidateGenerator(dictionary_lexicon=(), dictionary_limit=0, syntax_provider=lambda _text: ())


def _dictionary_only_candidate_generator(candidate_generator: CandidateGenerator) -> CandidateGenerator:
    return CandidateGenerator(
        dictionary_provider=getattr(candidate_generator, "dictionary_provider", None),
        dictionary_limit=getattr(candidate_generator, "dictionary_limit", 2),
        dictionary_min_score=getattr(candidate_generator, "dictionary_min_score", 85),
        dictionary_yo_e_enabled=getattr(candidate_generator, "dictionary_yo_e_enabled", False),
        syntax_provider=lambda _text: (),
        dictionary_token_cache_enabled=getattr(candidate_generator, "dictionary_token_cache_enabled", True),
        dictionary_token_cache_max_size=getattr(candidate_generator, "dictionary_token_cache_max_size", 200_000),
        dictionary_max_choices_per_token=getattr(candidate_generator, "dictionary_max_choices_per_token", 50_000),
        dictionary_use_second_letter_index=getattr(candidate_generator, "dictionary_use_second_letter_index", False),
    )


def _validation_candidate_generator(
    rule_id: str,
    *,
    default_generator: CandidateGenerator,
    fast_generator: CandidateGenerator,
    dictionary_generator: CandidateGenerator,
) -> CandidateGenerator:
    if rule_id in DICTIONARY_VALIDATION_RULE_IDS:
        return dictionary_generator
    if rule_id in {"capitalization_ner", "abbreviation_case_protection"}:
        return default_generator
    return fast_generator


def _load_optional_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    try:
        return load_config(config_path)
    except Exception:
        return {}


def _read_existing_rows(path: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        return []
    try:
        frame = pd.read_csv(dataset_path)
    except Exception:
        return []
    return frame.to_dict("records")


def _existing_rows_by_rule(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for rule_id in _row_rule_ids(row):
            by_rule[rule_id].append(row)
    return by_rule


def _clean_sentences(clean_pool_path: str | Path) -> list[str]:
    path = Path(clean_pool_path)
    if path.exists():
        try:
            frame = pd.read_csv(path)
            for column in ("text", "sentence", "target", "source"):
                if column in frame.columns:
                    values = [str(value).strip() for value in frame[column].dropna().head(500).tolist()]
                    values = [value for value in values if value]
                    if values:
                        return values
        except Exception:
            pass
    return DEFAULT_CLEAN_SENTENCES


def _row_rule_ids(row: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    raw_rule_id = row.get("rule_id")
    if raw_rule_id is not None and not pd.isna(raw_rule_id):
        ids.append(normalize_rule_id(raw_rule_id))
    raw_rule_ids = row.get("rule_ids")
    if isinstance(raw_rule_ids, str) and raw_rule_ids.strip():
        try:
            parsed = json.loads(raw_rule_ids)
        except json.JSONDecodeError:
            parsed = [part.strip() for part in raw_rule_ids.split(",")]
        if isinstance(parsed, list):
            ids.extend(normalize_rule_id(item) for item in parsed)
    elif isinstance(raw_rule_ids, list):
        ids.extend(normalize_rule_id(item) for item in raw_rule_ids)
    return sorted({rule_id for rule_id in ids if rule_id != UNKNOWN_RULE_ID})


def _edit_payload(edit: Edit) -> dict[str, Any]:
    return {
        "source": edit.source,
        "replacement": edit.replacement,
        "edit_type": edit.edit_type,
        "start": int(edit.start),
        "end": int(edit.end),
        "status": edit.status,
        "reason": edit.reason,
        "confidence": float(edit.confidence),
        "rule_id": edit.rule_id,
    }


def _apply_candidate(source: str, candidate: Candidate) -> str:
    start = int(getattr(candidate, "start", -1))
    end = int(getattr(candidate, "end", -1))
    if start < 0 or end < start:
        return source
    return source[:start] + str(getattr(candidate, "replacement", "")) + source[end:]


def _pair_hash(source: str, target: str) -> str:
    return hashlib.sha256(f"{source}\0{target}".encode("utf-8")).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)
