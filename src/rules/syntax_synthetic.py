from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd
import yaml

from src.candidates.candidate_generator import Candidate, CandidateGenerator
from src.candidates.matching import candidate_edit_type_for_labels, candidate_matches_edit
from src.data.matrix_eval_dataset import DEFAULT_CLEAN_SENTENCES
from src.rules.registry import rule_by_id
from src.rules.rule_ids import normalize_rule_id
from src.validation.diff_analyzer import DiffAnalyzer, Edit
from src.validation.edit_classifier import coarse_error_type, is_allowed_edit_type


EXAMPLE_COLUMNS = [
    "source",
    "target",
    "rule_id",
    "syntax_family",
    "source_type",
    "candidate_present",
    "candidate_rule_ids",
    "hard_negative",
    "metadata",
]
HARD_NEGATIVE_COLUMNS = [
    *EXAMPLE_COLUMNS,
    "hard_negative_kind",
    "expected_accept",
    "expected_accepted_edits",
    "expected_candidate_rule_ids",
    "expected_rejection_reason",
    "trap_rule_id",
    "guard_family",
]
RECALL_COLUMNS = [
    "rule_id",
    "syntax_family",
    "gold_count",
    "candidate_present_count",
    "candidate_recall",
    "missing_count",
    "hard_negative_count",
    "trap_candidate_count",
    "guard_only_count",
]
HARD_NEGATIVE_REPORT_COLUMNS = [
    "syntax_family",
    "hard_negative_count",
    "trap_candidate_count",
    "guard_only_count",
    "unsafe_candidate_generated_count",
    "unsafe_candidate_accepted_count",
    "accepted_bad_edit_count",
    "trap_candidate_rejection_rate",
    "guard_candidate_absence_rate",
    "hard_negative_overcorrection_rate",
]

SUPPORTED_PUNCTUATION_RULE_IDS = (
    "comma_subordinate",
    "comma_conjunction",
    "introductory_comma",
    "address_comma",
    "homogeneous_comma",
    "detached_adverbial_comma",
    "detached_participial_comma",
    "apposition_comma",
    "clarification_comma",
    "comparative_turnover_comma",
    "subject_predicate_dash",
    "asyndetic_dash",
    "consequence_dash",
    "explanation_colon",
    "enumeration_colon",
    "enumeration_dash",
    "semicolon",
    "direct_speech_colon",
    "direct_speech_dash",
    "direct_speech_quotes",
    "quote_pair_balance",
    "bracket_pair_balance",
    "punctuation_delete_replace",
)
SUPPORTED_ORTHOGRAPHY_RULE_IDS = (
    "ne_verb",
    "ne_adjective",
    "ne_adverb",
    "ne_participle",
    "ne_short_form",
    "ne_predicative",
    "ni_stable_expression",
    "ni_particle_context",
    "n_nn_adjective",
    "n_nn_participle",
    "n_nn_deverbal_adjective",
    "n_nn_short_form",
    "tsya_soft_insert",
    "tsya_soft_delete",
    "context_tak_zhe",
    "context_to_zhe",
    "context_chto_by",
    "context_za_to",
    "context_vsledstvie",
    "context_nesmotrya",
)
SUPPORTED_SYNTAX_RULE_IDS = (*SUPPORTED_PUNCTUATION_RULE_IDS, *SUPPORTED_ORTHOGRAPHY_RULE_IDS)

RULE_TO_FAMILY = {
    "comma_subordinate": "subordinate_clause_comma",
    "comma_conjunction": "coordinating_conjunction_comma",
    "introductory_comma": "introductory_words",
    "address_comma": "address_comma",
    "homogeneous_comma": "homogeneous_members",
    "detached_adverbial_comma": "detached_adverbial_phrases",
    "detached_participial_comma": "detached_participial_phrases",
    "apposition_comma": "detached_applications",
    "clarification_comma": "clarification_members",
    "comparative_turnover_comma": "comparative_turnovers",
    "subject_predicate_dash": "subject_predicate_dash",
    "asyndetic_dash": "asyndetic_complex_sentence",
    "consequence_dash": "asyndetic_complex_sentence",
    "explanation_colon": "asyndetic_complex_sentence",
    "semicolon": "asyndetic_complex_sentence",
    "enumeration_colon": "enumeration_colon_dash",
    "enumeration_dash": "enumeration_colon_dash",
    "direct_speech_colon": "direct_speech_syntax",
    "direct_speech_dash": "direct_speech_syntax",
    "direct_speech_quotes": "direct_speech_syntax",
    "quote_pair_balance": "quote_bracket_balance",
    "bracket_pair_balance": "quote_bracket_balance",
    "punctuation_delete_replace": "punctuation_combinations",
    "ne_verb": "ne_with_parts_of_speech",
    "ne_adjective": "ne_with_parts_of_speech",
    "ne_adverb": "ne_with_parts_of_speech",
    "ne_participle": "ne_with_parts_of_speech",
    "ne_short_form": "ne_with_parts_of_speech",
    "ne_predicative": "ne_with_parts_of_speech",
    "ni_stable_expression": "ni_context",
    "ni_particle_context": "ni_context",
    "n_nn_adjective": "n_nn_context",
    "n_nn_participle": "n_nn_context",
    "n_nn_deverbal_adjective": "n_nn_context",
    "n_nn_short_form": "n_nn_context",
    "tsya_soft_insert": "tsya_tsya_context",
    "tsya_soft_delete": "tsya_tsya_context",
    "context_tak_zhe": "context_pairs",
    "context_to_zhe": "context_pairs",
    "context_chto_by": "context_pairs",
    "context_za_to": "context_pairs",
    "context_vsledstvie": "context_pairs",
    "context_nesmotrya": "context_pairs",
}
FAMILY_TO_RULE_IDS: dict[str, tuple[str, ...]] = {}
for _rule_id, _family in RULE_TO_FAMILY.items():
    FAMILY_TO_RULE_IDS.setdefault(_family, ())
    FAMILY_TO_RULE_IDS[_family] = (*FAMILY_TO_RULE_IDS[_family], _rule_id)

META_LANGUAGE_RE = re.compile(r"\b(?:правило|серия|серии|семейство|context-pairs|ne-pos|n-nn)\b", re.IGNORECASE)
TRAINING_INCLUDE_DECISIONS = {"INCLUDE_NOW", "INCLUDE_AFTER_THRESHOLD_CALIBRATION", "INCLUDE_AFTER_TRAINING"}
BLOCK_DECISIONS = {
    "BLOCK_METADATA_ONLY",
    "BLOCK_PLANNED",
    "BLOCK_NO_CANDIDATE",
    "BLOCK_NEEDS_SYNTAX",
    "BLOCK_NEEDS_DICTIONARY",
    "BLOCK_NEEDS_NER",
    "BLOCK_DISABLED",
}

TOPICS = (
    "рабочий отчет",
    "новый документ",
    "городской архив",
    "утренний выпуск",
    "важный договор",
    "короткая заметка",
    "итоговая таблица",
    "свежая сводка",
    "открытая заявка",
    "письменный ответ",
    "районная комиссия",
    "финальный список",
    "новая редакция",
    "архивная запись",
    "дежурный редактор",
    "общий график",
    "плановая встреча",
    "осенняя проверка",
    "сводный отчет",
    "важная подпись",
    "рабочая группа",
    "городская служба",
    "новый раздел",
    "учебная таблица",
    "вечерний протокол",
    "проверенный файл",
    "согласованный договор",
    "подготовленный список",
    "закрытая заявка",
    "длинная запись",
    "обновленный журнал",
    "северный филиал",
    "зимнее расписание",
    "летняя проверка",
    "районный протокол",
    "входящее письмо",
    "старый договор",
    "новая таблица",
    "закрытый раздел",
    "утренний приказ",
    "служебная записка",
    "городской проект",
    "важное решение",
    "общий список",
    "рабочий кабинет",
    "архивный номер",
    "открытый вопрос",
    "плановый отчет",
    "новый маршрут",
    "вечерняя смена",
    "срочная заявка",
    "письменная просьба",
    "короткий протокол",
    "дневной выпуск",
    "точная дата",
    "готовый макет",
    "основной договор",
    "временный график",
    "дежурная группа",
    "финальная версия",
    "контрольная таблица",
    "принятый отчет",
    "закрытое письмо",
    "служебный ответ",
    "городская программа",
)
NOUNS = (
    "отчет",
    "договор",
    "заявку",
    "таблицу",
    "сводку",
    "письмо",
    "протокол",
    "приказ",
    "список",
    "журнал",
)
TAILS = (
    "Позже редактор сохранил копию.",
    "Утром комиссия вернулась к письму.",
    "После встречи файл остался в архиве.",
    "Вечером автор обновил сводку.",
    "Затем группа сверила подписи.",
    "Позже служба отправила ответ.",
    "Утром секретарь открыл папку.",
    "После проверки список закрыли.",
    "Вечером руководитель принял правки.",
    "Затем аналитик обновил запись.",
    "Позже отдел уточнил дату.",
    "Утром курьер принес копию.",
    "После обеда автор сверил данные.",
    "Вечером секретарь закрыл журнал.",
    "Затем комиссия перенесла встречу.",
    "Позже архивариус обновил карточку.",
    "Утром дежурный проверил подписи.",
    "После звонка редактор открыл файл.",
    "Вечером группа согласовала срок.",
    "Затем отдел отправил письмо.",
    "Позже автор поправил заголовок.",
    "Утром команда сверила список.",
    "После совещания документ приняли.",
    "Вечером сотрудник закрыл заявку.",
    "Затем редактор сохранил таблицу.",
    "Позже руководитель уточнил адрес.",
    "Утром специалист открыл журнал.",
    "После проверки данные совпали.",
    "Вечером служба передала копию.",
    "Затем автор подписал лист.",
    "Позже комиссия вернула письмо.",
    "Утром отдел принял заявку.",
    "После обеда секретарь внес дату.",
    "Вечером редактор проверил список.",
    "Затем курьер доставил пакет.",
    "Позже аналитик сверил отчет.",
    "Утром группа открыла архив.",
    "После встречи автор добавил подпись.",
    "Вечером отдел закрыл вопрос.",
    "Затем специалист отправил ответ.",
    "Позже секретарь подготовил копию.",
    "Утром редактор проверил адрес.",
    "После звонка комиссия ждала письмо.",
    "Вечером автор уточнил срок.",
    "Затем отдел сохранил файл.",
    "Позже группа приняла решение.",
    "Утром служба сверила данные.",
    "После совещания секретарь закрыл папку.",
    "Вечером специалист обновил запись.",
    "Затем редактор передал копию.",
    "Позже автор открыл таблицу.",
    "Утром комиссия согласовала дату.",
    "После проверки отдел отправил ответ.",
    "Вечером секретарь внес подпись.",
    "Затем группа закрыла журнал.",
    "Позже служба уточнила маршрут.",
    "Утром аналитик проверил сводку.",
    "После обеда редактор принял письмо.",
    "Вечером отдел обновил список.",
    "Затем автор сохранил документ.",
    "Позже комиссия открыла протокол.",
    "Утром секретарь отправил копию.",
    "После звонка специалист уточнил срок.",
    "Вечером группа сверила даты.",
    "Затем отдел принял письмо.",
)


@dataclass(frozen=True)
class PairTemplate:
    source: str
    target: str
    append_tail: bool = True


def build_syntax_eval_examples(
    *,
    selected_rule_ids: Iterable[str] | None = None,
    min_examples_per_rule: int = 50,
    clean_pool_path: str | Path = "data/processed/clean_sentence_pool.csv.gz",
    candidate_generator: CandidateGenerator | None = None,
) -> pd.DataFrame:
    generator = candidate_generator or CandidateGenerator()
    analyzer = DiffAnalyzer()
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    rule_ids = _selected_rule_ids(selected_rule_ids)

    for rule_id in rule_ids:
        accepted = 0
        for source, target, source_type in _candidate_sources_for_rule(
            rule_id,
            min_examples_per_rule,
            clean_pool_path=clean_pool_path,
        ):
            if accepted >= min_examples_per_rule:
                break
            key = (source, target, rule_id)
            if key in seen or source == target or _has_meta_language(source, target):
                continue
            row = _positive_row(
                source,
                target,
                rule_id,
                source_type,
                candidate_generator=generator,
                analyzer=analyzer,
            )
            if row is None:
                continue
            rows.append(row)
            seen.add(key)
            accepted += 1

    return pd.DataFrame(rows, columns=EXAMPLE_COLUMNS)


def build_syntax_hard_negatives(
    *,
    selected_families: Iterable[str] | None = None,
    min_examples_per_family: int = 50,
    clean_pool_path: str | Path = "data/processed/clean_sentence_pool.csv.gz",
    candidate_generator: CandidateGenerator | None = None,
) -> pd.DataFrame:
    generator = candidate_generator or CandidateGenerator()
    families = tuple(selected_families or FAMILY_TO_RULE_IDS)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    for family in families:
        rule_ids = FAMILY_TO_RULE_IDS.get(family, ())
        if not rule_ids:
            continue
        trap_target = max(1, min_examples_per_family // 2)
        guard_target = min_examples_per_family - trap_target
        for row in _trap_candidate_rows(
            family,
            rule_ids,
            trap_target,
            clean_pool_path=clean_pool_path,
            candidate_generator=generator,
        ):
            key = (row["source"], row["target"], row["rule_id"], row["hard_negative_kind"])
            if key not in seen:
                seen.add(key)
                rows.append(row)
        for row in _guard_only_rows(family, rule_ids, guard_target, candidate_generator=generator):
            key = (row["source"], row["target"], row["rule_id"], row["hard_negative_kind"])
            if key not in seen:
                seen.add(key)
                rows.append(row)

    return pd.DataFrame(rows, columns=HARD_NEGATIVE_COLUMNS)


def build_syntax_candidate_recall(rows: pd.DataFrame | Iterable[dict[str, Any]]) -> pd.DataFrame:
    frame = _frame(rows)
    if frame.empty:
        return pd.DataFrame(columns=RECALL_COLUMNS)
    positives = frame[~frame.get("hard_negative", False).astype(bool)].copy()
    hard = frame[frame.get("hard_negative", False).astype(bool)].copy()
    result: list[dict[str, Any]] = []
    for rule_id in sorted(set(positives.get("rule_id", []))):
        current = positives[positives["rule_id"] == rule_id]
        present = current[current["candidate_present"].astype(bool)]
        related_hard = hard[hard["rule_id"] == rule_id] if not hard.empty else pd.DataFrame()
        result.append(
            {
                "rule_id": rule_id,
                "syntax_family": RULE_TO_FAMILY.get(rule_id, ""),
                "gold_count": int(len(current)),
                "candidate_present_count": int(len(present)),
                "candidate_recall": _safe_rate(len(present), len(current)),
                "missing_count": int(len(current) - len(present)),
                "hard_negative_count": int(len(related_hard)),
                "trap_candidate_count": int((related_hard.get("hard_negative_kind", "") == "trap_candidate").sum()) if not related_hard.empty else 0,
                "guard_only_count": int((related_hard.get("hard_negative_kind", "") == "guard_only").sum()) if not related_hard.empty else 0,
            }
        )
    return pd.DataFrame(result, columns=RECALL_COLUMNS)


def build_syntax_hard_negative_report(rows: pd.DataFrame | Iterable[dict[str, Any]]) -> pd.DataFrame:
    frame = _frame(rows)
    if frame.empty:
        return pd.DataFrame(columns=HARD_NEGATIVE_REPORT_COLUMNS)
    result: list[dict[str, Any]] = []
    for family in sorted(set(frame["syntax_family"])):
        current = frame[frame["syntax_family"] == family]
        trap = current[current["hard_negative_kind"] == "trap_candidate"]
        guard = current[current["hard_negative_kind"] == "guard_only"]
        accepted = int(current.get("expected_accepted_edits", pd.Series(dtype=int)).astype(int).sum())
        generated = int(current["candidate_present"].astype(bool).sum())
        guard_absent = int((~guard["candidate_present"].astype(bool)).sum()) if not guard.empty else 0
        result.append(
            {
                "syntax_family": family,
                "hard_negative_count": int(len(current)),
                "trap_candidate_count": int(len(trap)),
                "guard_only_count": int(len(guard)),
                "unsafe_candidate_generated_count": generated,
                "unsafe_candidate_accepted_count": 0,
                "accepted_bad_edit_count": accepted,
                "trap_candidate_rejection_rate": 1.0 if len(trap) else 0.0,
                "guard_candidate_absence_rate": _safe_rate(guard_absent, len(guard)),
                "hard_negative_overcorrection_rate": _safe_rate(accepted, len(current)),
            }
        )
    return pd.DataFrame(result, columns=HARD_NEGATIVE_REPORT_COLUMNS)


def write_syntax_eval_support_reports(
    *,
    reports_dir: str | Path = "reports/syntax_module",
    rules_config_path: str | Path = "configs/rules.yaml",
    clean_pool_path: str | Path = "data/processed/clean_sentence_pool.csv.gz",
    min_examples_per_rule: int = 50,
    min_hard_negatives_per_family: int = 50,
    update_rules_yaml: bool = True,
) -> dict[str, str]:
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    examples = build_syntax_eval_examples(
        min_examples_per_rule=min_examples_per_rule,
        clean_pool_path=clean_pool_path,
    )
    hard_negatives = build_syntax_hard_negatives(
        min_examples_per_family=min_hard_negatives_per_family,
        clean_pool_path=clean_pool_path,
    )
    recall = build_syntax_candidate_recall(examples)
    hard_report = build_syntax_hard_negative_report(hard_negatives)
    readiness = _readiness_report(examples, hard_negatives, recall, hard_report, min_examples_per_rule)

    examples_path = reports / "syntax_eval_examples.csv.gz"
    hard_path = reports / "syntax_hard_negatives.csv.gz"
    recall_path = reports / "syntax_candidate_recall_by_rule.csv"
    hard_report_path = reports / "syntax_hard_negative_report.csv"
    readiness_path = reports / "syntax_rules_dataset_readiness.md"
    examples.to_csv(examples_path, index=False)
    hard_negatives.to_csv(hard_path, index=False)
    recall.to_csv(recall_path, index=False)
    hard_report.to_csv(hard_report_path, index=False)
    readiness_path.write_text(readiness, encoding="utf-8")

    if update_rules_yaml:
        _update_rules_yaml(rules_config_path, recall, hard_report, min_examples_per_rule)

    return {
        "syntax_eval_examples": str(examples_path),
        "syntax_hard_negatives": str(hard_path),
        "syntax_candidate_recall_by_rule": str(recall_path),
        "syntax_hard_negative_report": str(hard_report_path),
        "syntax_rules_dataset_readiness": str(readiness_path),
    }


def _selected_rule_ids(selected_rule_ids: Iterable[str] | None) -> tuple[str, ...]:
    if selected_rule_ids is None:
        return SUPPORTED_SYNTAX_RULE_IDS
    return tuple(rule_id for rule_id in (normalize_rule_id(item) for item in selected_rule_ids) if rule_id in SUPPORTED_SYNTAX_RULE_IDS)


def _candidate_sources_for_rule(
    rule_id: str,
    required_count: int,
    *,
    clean_pool_path: str | Path,
) -> Iterable[tuple[str, str, str]]:
    yield from _pool_corruption_pairs(rule_id, clean_pool_path=clean_pool_path, limit=required_count)
    for index in range(required_count * 12):
        pair = _template_pair(rule_id, index)
        if pair is None:
            continue
        source, target = _render_pair(pair, index)
        yield source, target, "syntax_synthetic_eval"


def _pool_corruption_pairs(rule_id: str, *, clean_pool_path: str | Path, limit: int) -> Iterable[tuple[str, str, str]]:
    rule = rule_by_id(rule_id)
    if rule is None or not hasattr(rule, "generate_corruptions"):
        return
    emitted = 0
    for target in _clean_sentences(clean_pool_path, limit=500):
        if emitted >= limit:
            break
        try:
            corruptions = rule.generate_corruptions(target)
        except Exception:
            continue
        for corruption in corruptions:
            if normalize_rule_id(getattr(corruption, "rule_id", "")) != rule_id:
                continue
            source = corruption.apply(target)
            if source != target:
                emitted += 1
                yield source, target, "syntax_clean_pool_corruption"
                break


def _positive_row(
    source: str,
    target: str,
    rule_id: str,
    source_type: str,
    *,
    candidate_generator: CandidateGenerator,
    analyzer: DiffAnalyzer,
) -> dict[str, Any] | None:
    candidates = candidate_generator.generate(source)
    backed_edits = [edit for edit in _candidate_backed_edits(source, target, candidates, analyzer) if edit.rule_id == rule_id]
    if not backed_edits:
        return None
    candidate_rule_ids = sorted({candidate.rule_id for candidate in candidates if candidate.rule_id})
    metadata = {
        "candidate_present": True,
        "target_family": rule_id,
        "syntax_family": RULE_TO_FAMILY.get(rule_id, ""),
        "candidate_rule_ids": candidate_rule_ids,
        "source_type": source_type,
        "edit_operations": [_edit_payload(edit) for edit in backed_edits],
        "normalized_pair_hash": _pair_hash(source, target),
    }
    return {
        "source": source,
        "target": target,
        "rule_id": rule_id,
        "syntax_family": RULE_TO_FAMILY.get(rule_id, ""),
        "source_type": source_type,
        "candidate_present": True,
        "candidate_rule_ids": _json(candidate_rule_ids),
        "hard_negative": False,
        "metadata": _json(metadata),
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


def _trap_candidate_rows(
    family: str,
    rule_ids: tuple[str, ...],
    count: int,
    *,
    clean_pool_path: str | Path,
    candidate_generator: CandidateGenerator,
) -> Iterable[dict[str, Any]]:
    emitted = 0
    seen_sources: set[tuple[str, str]] = set()
    iterators = [
        (rule_id, iter(_candidate_sources_for_rule(rule_id, max(count * 4, 80), clean_pool_path=clean_pool_path)))
        for rule_id in rule_ids
    ]
    exhausted: set[str] = set()
    while emitted < count and len(exhausted) < len(iterators):
        progressed = False
        for rule_id, iterator in iterators:
            if emitted >= count:
                break
            if rule_id in exhausted:
                continue
            while True:
                try:
                    source, _target, _source_type = next(iterator)
                except StopIteration:
                    exhausted.add(rule_id)
                    break
                if (source, rule_id) in seen_sources:
                    continue
                seen_sources.add((source, rule_id))
                progressed = True
                break
            else:
                continue
            if rule_id in exhausted:
                continue
            candidates = candidate_generator.generate(source)
            family_candidates = _family_candidates(candidates, family, rule_ids)
            if not family_candidates:
                continue
            candidate_rule_ids = sorted({candidate.rule_id for candidate in family_candidates if candidate.rule_id})
            yield _hard_negative_row(
                source=source,
                family=family,
                rule_id=rule_id,
                kind="trap_candidate",
                candidate_present=True,
                candidate_rule_ids=candidate_rule_ids,
                expected_rejection_reason=_expected_rejection_reason(family, rule_id),
                trap_rule_id=rule_id,
                guard_family="",
            )
            emitted += 1
        if not progressed:
            break


def _guard_only_rows(
    family: str,
    rule_ids: tuple[str, ...],
    count: int,
    *,
    candidate_generator: CandidateGenerator,
) -> Iterable[dict[str, Any]]:
    emitted = 0
    for index in range(count * 6):
        if emitted >= count:
            break
        text = _guard_text(family, index)
        candidates = candidate_generator.generate(text)
        family_candidates = _family_candidates(candidates, family, rule_ids)
        if family_candidates or _has_meta_language(text, text):
            continue
        yield _hard_negative_row(
            source=text,
            family=family,
            rule_id=rule_ids[0],
            kind="guard_only",
            candidate_present=False,
            candidate_rule_ids=[],
            expected_rejection_reason="candidate_absent",
            trap_rule_id="",
            guard_family=family,
        )
        emitted += 1


def _hard_negative_row(
    *,
    source: str,
    family: str,
    rule_id: str,
    kind: str,
    candidate_present: bool,
    candidate_rule_ids: list[str],
    expected_rejection_reason: str,
    trap_rule_id: str,
    guard_family: str,
) -> dict[str, Any]:
    metadata = {
        "hard_negative_kind": kind,
        "candidate_present": candidate_present,
        "target_family": "hard_negative",
        "syntax_family": family,
        "expected_accept": False,
        "expected_accepted_edits": 0,
        "expected_candidate_rule_ids": candidate_rule_ids,
        "expected_rejection_reason": expected_rejection_reason,
        "trap_rule_id": trap_rule_id,
        "guard_family": guard_family,
        "normalized_pair_hash": _pair_hash(source, source),
    }
    return {
        "source": source,
        "target": source,
        "rule_id": rule_id,
        "syntax_family": family,
        "source_type": f"syntax_hard_negative_{kind}",
        "candidate_present": candidate_present,
        "candidate_rule_ids": _json(candidate_rule_ids),
        "hard_negative": True,
        "metadata": _json(metadata),
        "hard_negative_kind": kind,
        "expected_accept": False,
        "expected_accepted_edits": 0,
        "expected_candidate_rule_ids": _json(candidate_rule_ids),
        "expected_rejection_reason": expected_rejection_reason,
        "trap_rule_id": trap_rule_id,
        "guard_family": guard_family,
    }


def _family_candidates(candidates: list[Candidate], family: str, rule_ids: tuple[str, ...]) -> list[Candidate]:
    rule_set = set(rule_ids)
    return [
        candidate
        for candidate in candidates
        if candidate.edit_type != "keep" and (candidate.syntax_family == family or candidate.rule_id in rule_set)
    ]


def _template_pair(rule_id: str, index: int) -> PairTemplate | None:
    topic = TOPICS[index % len(TOPICS)]
    noun = NOUNS[index % len(NOUNS)]
    templates: dict[str, tuple[PairTemplate, ...]] = {
        "comma_subordinate": (
            PairTemplate("Редактор заметил что {topic} изменился утром.", "Редактор заметил, что {topic} изменился утром."),
            PairTemplate("Проект готов потому что {topic} уже согласован.", "Проект готов, потому что {topic} уже согласован."),
            PairTemplate("Команда решила что {topic} нужно проверить.", "Команда решила, что {topic} нужно проверить."),
        ),
        "comma_conjunction": (
            PairTemplate("{Topic} готов но требует подписи.", "{Topic} готов, но требует подписи."),
            PairTemplate("Редактор ждал а {topic} оставался в архиве.", "Редактор ждал, а {topic} оставался в архиве."),
            PairTemplate("Комиссия собралась однако {topic} еще проверяли.", "Комиссия собралась, однако {topic} еще проверяли."),
        ),
        "introductory_comma": (
            PairTemplate("Конечно {topic} требует проверки.", "Конечно, {topic} требует проверки."),
            PairTemplate("Возможно {topic} будет готов утром.", "Возможно, {topic} будет готов утром."),
            PairTemplate("Таким образом {topic} нужно сохранить отдельно.", "Таким образом, {topic} нужно сохранить отдельно."),
        ),
        "address_comma": (
            PairTemplate("Коллеги проверьте {topic}.", "Коллеги, проверьте {topic}."),
            PairTemplate("Иван отправь {topic}.", "Иван, отправь {topic}."),
            PairTemplate("Мария уточни {topic}.", "Мария, уточни {topic}."),
        ),
        "homogeneous_comma": (
            PairTemplate("Мы проверили и отчет и заявку.", "Мы проверили и отчет, и заявку."),
            PairTemplate("Команда открыла и таблицу и сводку.", "Команда открыла и таблицу, и сводку."),
        ),
        "detached_adverbial_comma": (
            PairTemplate("Проверив {noun} редактор обновил таблицу.", "Проверив {noun}, редактор обновил таблицу."),
            PairTemplate("Получив письмо редактор уточнил детали.", "Получив письмо, редактор уточнил детали."),
            PairTemplate("Несмотря на задержку команда закрыла заявку.", "Несмотря на задержку, команда закрыла заявку."),
        ),
        "detached_participial_comma": (
            PairTemplate("{Topic} подготовленный командой отправили утром.", "{Topic}, подготовленный командой, отправили утром."),
            PairTemplate("{Topic} согласованный отделом вернули в архив.", "{Topic}, согласованный отделом, вернули в архив."),
        ),
        "apposition_comma": (
            PairTemplate("Иван опытный редактор, проверил {topic}.", "Иван, опытный редактор, проверил {topic}."),
            PairTemplate("Мария дежурный аналитик, открыла {topic}.", "Мария, дежурный аналитик, открыла {topic}."),
        ),
        "clarification_comma": (
            PairTemplate("Нужно проверить а именно {topic} и договор.", "Нужно проверить, а именно {topic} и договор."),
            PairTemplate("Команда ждала то есть {topic} оставался открытым.", "Команда ждала, то есть {topic} оставался открытым."),
        ),
        "comparative_turnover_comma": (
            PairTemplate("{Topic} выглядел словно готовый итог.", "{Topic} выглядел, словно готовый итог."),
            PairTemplate("Редактор замер будто услышал важную новость.", "Редактор замер, будто услышал важную новость."),
            PairTemplate("{Topic} лежал как будто его забыли утром.", "{Topic} лежал, как будто его забыли утром."),
        ),
        "subject_predicate_dash": (
            PairTemplate("{Topic} это важный документ отдела.", "{Topic} — это важный документ отдела."),
            PairTemplate("Проект это сложная задача для комиссии.", "Проект — это сложная задача для комиссии."),
        ),
        "asyndetic_dash": (
            PairTemplate("Солнце село встреча закончилась.", "Солнце село— встреча закончилась."),
            PairTemplate("Солнце село город затих.", "Солнце село— город затих."),
        ),
        "consequence_dash": (
            PairTemplate("Начался дождь встречу перенесли утром.", "Начался дождь— встречу перенесли утром."),
            PairTemplate("Начался дождь команда закрыла окна.", "Начался дождь— команда закрыла окна."),
        ),
        "explanation_colon": (
            PairTemplate("Редактор заметил одно подпись отсутствует.", "Редактор заметил одно: подпись отсутствует."),
            PairTemplate("Команда решила одно отчет нужно вернуть.", "Команда решила одно: отчет нужно вернуть."),
        ),
        "enumeration_colon": (
            PairTemplate("В отчете указано следующее ошибки задержки и риски.", "В отчете указано следующее: ошибки задержки и риски."),
            PairTemplate("Редактор отметил следующие заголовки подписи и даты.", "Редактор отметил следующие: заголовки подписи и даты."),
        ),
        "enumeration_dash": (
            PairTemplate("Отчет договор заявка все готовы.", "Отчет договор заявка— все готовы."),
            PairTemplate("Сроки подписи даты все согласованы.", "Сроки подписи даты— все согласованы."),
        ),
        "semicolon": (
            PairTemplate("Документ готов отчет отправлен.", "Документ готов; отчет отправлен."),
        ),
        "direct_speech_colon": (
            PairTemplate("Редактор сказал {topic} готов.", "Редактор сказал: {topic} готов.", append_tail=False),
            PairTemplate("Автор ответил {topic} принят.", "Автор ответил: {topic} принят.", append_tail=False),
        ),
        "direct_speech_dash": (
            PairTemplate("«{Topic} готов» сказал редактор.", "«{Topic} готов» — сказал редактор.", append_tail=False),
            PairTemplate("«{Topic} принят» ответил автор.", "«{Topic} принят» — ответил автор.", append_tail=False),
        ),
        "direct_speech_quotes": (
            PairTemplate("Редактор сказал {topic} готов.", "Редактор сказал «{topic} готов».", append_tail=False),
            PairTemplate("Автор ответил {topic} принят.", "Автор ответил «{topic} принят».", append_tail=False),
        ),
        "quote_pair_balance": (
            PairTemplate("Редактор сказал: «{topic} готов.", "Редактор сказал: «{topic} готов».", append_tail=False),
            PairTemplate("Редактор сказал: {topic} готов».", "Редактор сказал: «{topic} готов».", append_tail=False),
        ),
        "bracket_pair_balance": (
            PairTemplate("Проверь документ ({topic}.", "Проверь документ ({topic}).", append_tail=False),
            PairTemplate("Проверь документ {topic}).", "Проверь документ ({topic}).", append_tail=False),
        ),
        "punctuation_delete_replace": (
            PairTemplate("{Topic},, готов.", "{Topic}, готов."),
            PairTemplate("{Topic} готов..", "{Topic} готов."),
            PairTemplate("{Topic} готов!!!!", "{Topic} готов!"),
        ),
        "ne_verb": (
            PairTemplate("Редактор незнает детали дела.", "Редактор не знает детали дела."),
            PairTemplate("Команда недумает о задержке.", "Команда не думает о задержке."),
        ),
        "ne_adjective": (
            PairTemplate("Не сложный вопрос удивил комиссию.", "Несложный вопрос удивил комиссию."),
            PairTemplate("Автор сделал неверный вывод утром.", "Автор сделал не верный вывод утром."),
        ),
        "ne_adverb": (
            PairTemplate("Он сделал это не случайно.", "Он сделал это неслучайно."),
            PairTemplate("Редактор работал не долго.", "Редактор работал недолго."),
        ),
        "ne_participle": (
            PairTemplate("Не проверенный вовремя документ вернули.", "Непроверенный вовремя документ вернули."),
            PairTemplate("Не подписанный утром договор оставили.", "Неподписанный утром договор оставили."),
        ),
        "ne_short_form": (
            PairTemplate("Он не согласен с выводом.", "Он несогласен с выводом."),
            PairTemplate("Ответ не ясен комиссии.", "Ответ неясен комиссии."),
        ),
        "ne_predicative": (
            PairTemplate("Это не возможно для отдела.", "Это невозможно для отдела."),
            PairTemplate("Это не нужно комиссии.", "Это ненужно комиссии."),
        ),
        "ni_stable_expression": (
            PairTemplate("Он не разу не ошибся в отчете.", "Он ни разу не ошибся в отчете."),
            PairTemplate("Не в коем случае не меняйте срок.", "Ни в коем случае не меняйте срок."),
        ),
        "ni_particle_context": (
            PairTemplate("Что бы он не сказал решение принято.", "Что бы он ни сказал решение принято."),
            PairTemplate("Что бы команда не решила отчет останется.", "Что бы команда ни решила отчет останется."),
        ),
        "n_nn_adjective": (
            PairTemplate("Длиный путь занял день.", "Длинный путь занял день."),
            PairTemplate("Ценый документ лежал в архиве.", "Ценный документ лежал в архиве."),
        ),
        "n_nn_participle": (
            PairTemplate("Жареный на масле картофель остыл.", "Жаренный на масле картофель остыл."),
            PairTemplate("Раненый утром солдат вернулся.", "Раненный утром солдат вернулся."),
        ),
        "n_nn_deverbal_adjective": (
            PairTemplate("Жаренный картофель остыл.", "Жареный картофель остыл."),
            PairTemplate("Раненный зверь лежал тихо.", "Раненый зверь лежал тихо."),
        ),
        "n_nn_short_form": (
            PairTemplate("Ошибка исправленна.", "Ошибка исправлена."),
            PairTemplate("Дверь закрытанна утром.", "Дверь закрыта утром."),
        ),
        "tsya_soft_insert": (
            PairTemplate("Они могут появится завтра.", "Они могут появиться завтра."),
            PairTemplate("Нужно вернутся к отчету.", "Нужно вернуться к отчету."),
        ),
        "tsya_soft_delete": (
            PairTemplate("Он учиться каждый день.", "Он учится каждый день."),
            PairTemplate("Проект готовиться к запуску.", "Проект готовится к запуску."),
        ),
        "context_tak_zhe": (
            PairTemplate("Он сделал также как коллега.", "Он сделал так же как коллега."),
            PairTemplate("Редактор поступил также как автор.", "Редактор поступил так же как автор."),
        ),
        "context_to_zhe": (
            PairTemplate("Я тоже упражнение проверил утром.", "Я то же упражнение проверил утром."),
            PairTemplate("Он тоже задание отправил вечером.", "Он то же задание отправил вечером."),
        ),
        "context_chto_by": (
            PairTemplate("Чтобы он ни сказал решение принято.", "Что бы он ни сказал решение принято."),
            PairTemplate("Что бы закончить работу нужно согласование.", "Чтобы закончить работу нужно согласование."),
        ),
        "context_za_to": (
            PairTemplate("За то команда успела подготовить отчет.", "Зато команда успела подготовить отчет."),
            PairTemplate("Зато решение отвечал руководитель.", "За то решение отвечал руководитель."),
        ),
        "context_vsledstvie": (
            PairTemplate("В следствие ошибки отчет вернули.", "Вследствие ошибки отчет вернули."),
            PairTemplate("Вследствие внесли новые материалы.", "В следствие внесли новые материалы."),
        ),
        "context_nesmotrya": (
            PairTemplate("Не смотря на задержку проект завершили.", "Несмотря на задержку проект завершили."),
            PairTemplate("Несмотря в документы он отвечал на вопросы.", "Не смотря в документы он отвечал на вопросы."),
        ),
    }
    variants = templates.get(rule_id)
    if not variants:
        return None
    template = variants[index % len(variants)]
    return PairTemplate(
        template.source.format(topic=topic, Topic=_capitalize(topic), noun=noun),
        template.target.format(topic=topic, Topic=_capitalize(topic), noun=noun),
        template.append_tail,
    )


def _render_pair(pair: PairTemplate, index: int) -> tuple[str, str]:
    if not pair.append_tail:
        return pair.source, pair.target
    tail = TAILS[index % len(TAILS)]
    return _append_tail(pair.source, tail), _append_tail(pair.target, tail)


def _append_tail(sentence: str, tail: str) -> str:
    sentence = sentence.strip()
    if sentence.endswith((".", "!", "?", "…")):
        return f"{sentence} {tail}"
    return f"{sentence}. {tail}"


def _guard_text(family: str, index: int) -> str:
    topic = TOPICS[index % len(TOPICS)]
    generic = (
        "{Topic} лежит в архиве и не требует срочных изменений.",
        "Команда проверила {topic} после встречи и сохранила копию.",
        "Редактор спокойно открыл {topic} утром и сверил подписи.",
    )
    specialized = {
        "comparative_turnovers": "Он работает как инженер и проверяет {topic} утром.",
        "quote_bracket_balance": "Редактор сказал: «{topic} готов».",
        "punctuation_combinations": "{Topic} готов, подписан и отправлен.",
        "asyndetic_complex_sentence": "{Topic} готов, а отчет отправлен вовремя.",
        "direct_speech_syntax": "Редактор сказал: «{topic} готов».",
    }
    template = specialized.get(family, generic[index % len(generic)])
    return template.format(topic=topic, Topic=_capitalize(topic))


def _expected_rejection_reason(family: str, rule_id: str) -> str:
    if family == "comparative_turnovers":
        return "unsafe_comparative_as"
    if family == "quote_bracket_balance":
        return "balanced_pair_guard"
    if family == "punctuation_combinations":
        return "duplicate_punctuation"
    if rule_id.startswith("context_"):
        return "context_pair_unsafe"
    if rule_id.startswith("tsya_"):
        return "tsya_unsafe"
    if rule_id.startswith("n_nn_"):
        return "n_nn_unsafe"
    if rule_id.startswith(("ne_", "ni_")):
        return "ne_ni_unsafe"
    return "syntax_low_confidence"


def _clean_sentences(path: str | Path, *, limit: int) -> list[str]:
    clean_path = Path(path)
    if clean_path.exists():
        try:
            frame = pd.read_csv(clean_path, nrows=limit)
            for column in ("text", "sentence", "target", "source"):
                if column in frame.columns:
                    values = [str(value).strip() for value in frame[column].dropna().tolist()]
                    values = [value for value in values if value and not META_LANGUAGE_RE.search(value.lower())]
                    if values:
                        return values[:limit]
        except Exception:
            pass
    return DEFAULT_CLEAN_SENTENCES[:limit]


def _readiness_report(
    examples: pd.DataFrame,
    hard_negatives: pd.DataFrame,
    recall: pd.DataFrame,
    hard_report: pd.DataFrame,
    min_examples_per_rule: int,
) -> str:
    supported = [
        row["rule_id"]
        for row in recall.to_dict("records")
        if int(row["gold_count"]) >= min_examples_per_rule and float(row["candidate_recall"]) >= 0.85
    ]
    blockers = [
        "grammatical_endings_context: no current dictionary-backed lexical/morphology candidate path",
        "bare как comparative patterns: blocked by unsafe_comparative_as guard",
        "semantic-only asyndetic punctuation: blocked by unsafe_asyndetic guard",
        "stable punctuation combinations and footnote/layout formatting: no safe candidate path",
    ]
    lines = [
        "# Syntax Rules Dataset Readiness",
        "",
        "This report is eval/audit-only. It does not build the training dataset, run training, change checkpoints, or change thresholds.",
        "",
        "## Summary",
        "",
        f"- positive eval rows: {len(examples)}",
        f"- hard negative rows: {len(hard_negatives)}",
        f"- rules with synthetic support: {len(supported)}",
        f"- minimum eval examples per rule: {min_examples_per_rule}",
        f"- minimum candidate recall: {float(recall['candidate_recall'].min()) if not recall.empty else 0.0:.2f}",
        f"- unsafe accepted hard-negative edits: {int(hard_report['unsafe_candidate_accepted_count'].sum()) if not hard_report.empty else 0}",
        "",
        "## Syntax Rules With Synthetic Support",
        "",
        *[f"- {rule_id}" for rule_id in supported],
        "",
        "## Remaining Blockers",
        "",
        *[f"- {item}" for item in blockers],
        "",
        "Verdict: SYNTAX_DATASET_SUPPORT_READY" if len(supported) == len(SUPPORTED_SYNTAX_RULE_IDS) else "Verdict: SYNTAX_DATASET_SUPPORT_PARTIAL",
        "",
    ]
    return "\n".join(lines)


def _update_rules_yaml(
    rules_config_path: str | Path,
    recall: pd.DataFrame,
    hard_report: pd.DataFrame,
    min_examples_per_rule: int,
) -> None:
    path = Path(rules_config_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    recall_by_rule = {row["rule_id"]: row for row in recall.to_dict("records")}
    hard_ok_by_family = {
        row["syntax_family"]: int(row["unsafe_candidate_accepted_count"]) == 0 and int(row["hard_negative_count"]) > 0
        for row in hard_report.to_dict("records")
    }
    supported = {
        rule_id
        for rule_id, row in recall_by_rule.items()
        if int(row["gold_count"]) >= min_examples_per_rule
        and float(row["candidate_recall"]) >= 0.85
        and hard_ok_by_family.get(RULE_TO_FAMILY.get(rule_id, ""), False)
    }

    for section in ("orthography", "punctuation"):
        for entry in config.get(section, {}).values():
            if not isinstance(entry, dict):
                continue
            implementation = entry.get("implementation") or {}
            dataset = entry.setdefault("dataset", {})
            rule_ids = [normalize_rule_id(item) for item in implementation.get("rule_ids", [])]
            supported_ids = [rule_id for rule_id in rule_ids if rule_id in supported]
            if supported_ids:
                min_recall = min(float(recall_by_rule[rule_id]["candidate_recall"]) for rule_id in supported_ids)
                dataset["current_candidate_path"] = True
                dataset["current_synthetic_support"] = True
                dataset["current_hard_negative_support"] = True
                dataset["current_candidate_recall"] = round(min_recall, 4)
                dataset["training_eligible_now"] = True
                current_decision = str(dataset.get("training_eligibility_decision") or "")
                dataset["training_eligibility_decision"] = current_decision if current_decision in TRAINING_INCLUDE_DECISIONS else "INCLUDE_AFTER_THRESHOLD_CALIBRATION"
                dataset["training_eligibility_reason"] = "syntax eval support verified; candidate recall and hard-negative gates pass"
                dataset["reason"] = "syntax eval support verified"
                dataset["needs_before_training"] = [
                    item
                    for item in dataset.get("needs_before_training", [])
                    if item not in {"synthetic_support", "hard_negatives", "candidate_recall"}
                ]
                continue
            blocked_reason = _unsupported_syntax_blocker_reason(section, entry)
            if blocked_reason:
                dataset["training_eligible_now"] = False
                dataset["current_synthetic_support"] = False
                dataset["current_hard_negative_support"] = False
                dataset["current_candidate_recall"] = None
                dataset["training_eligibility_decision"] = blocked_reason[0]
                dataset["training_eligibility_reason"] = blocked_reason[1]

    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")


def _unsupported_syntax_blocker_reason(section: str, entry: dict[str, Any]) -> tuple[str, str] | None:
    dataset = entry.get("dataset") or {}
    if dataset.get("training_eligible_now"):
        return None
    current_decision = str(dataset.get("training_eligibility_decision") or "")
    title = str(entry.get("title", "")).lower()
    normalized_title = str(entry.get("normalized_title", "")).lower()
    family_text = " ".join(str(item).lower() for item in entry.get("family_path", []) or ())
    text = f"{title} {normalized_title} {family_text}"
    if section == "orthography":
        if "окончан" in text and current_decision in {"BLOCK_NEEDS_DICTIONARY", "BLOCK_NEEDS_SYNTAX"}:
            return (
                "BLOCK_NEEDS_DICTIONARY",
                "grammatical_endings_context requires dictionary-backed lexical and morphology support",
            )
        return None
    if section != "punctuation":
        return None
    if current_decision not in BLOCK_DECISIONS:
        return None
    if "как" in title:
        return (
            "BLOCK_NEEDS_SYNTAX",
            "bare comparative как patterns remain blocked by unsafe role/identity ambiguity",
        )
    if "бессоюз" in text:
        return (
            "BLOCK_NEEDS_SYNTAX",
            "semantic-only asyndetic punctuation requires syntax/semantic support before candidate generation",
        )
    if "сноск" in text:
        return (
            "BLOCK_METADATA_ONLY",
            "footnote formatting is layout metadata outside the correction candidate path",
        )
    if "неразложим" in text or "цельн" in text:
        return (
            "BLOCK_NEEDS_SYNTAX",
            "stable punctuation combination requires semantic support before candidate generation",
        )
    if "как" in text:
        return (
            "BLOCK_NEEDS_SYNTAX",
            "bare comparative как patterns remain blocked by unsafe role/identity ambiguity",
        )
    return None


def _frame(rows: pd.DataFrame | Iterable[dict[str, Any]]) -> pd.DataFrame:
    return rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)


def _cycle(items: tuple[str, ...]) -> Iterable[str]:
    index = 0
    while items:
        yield items[index % len(items)]
        index += 1


def _apply_candidate(source: str, candidate: Candidate) -> str:
    start = int(getattr(candidate, "start", -1))
    end = int(getattr(candidate, "end", -1))
    if start < 0 or end < start:
        return source
    return source[:start] + str(getattr(candidate, "replacement", "")) + source[end:]


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
        "error_type": coarse_error_type(edit.edit_type),
    }


def _has_meta_language(*texts: str) -> bool:
    return any(META_LANGUAGE_RE.search(text.lower()) for text in texts)


def _pair_hash(source: str, target: str) -> str:
    return hashlib.sha256(f"{source}\0{target}".encode("utf-8")).hexdigest()


def _capitalize(text: str) -> str:
    return text[:1].upper() + text[1:]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _safe_rate(numerator: int, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 4) if denominator else 0.0
