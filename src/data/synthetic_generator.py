from __future__ import annotations

from dataclasses import asdict, dataclass
import random
import re
from typing import Any

from src.candidates.candidate_generator import CandidateGenerator
from src.candidates.matching import candidate_edit_type_for_labels, candidate_matches_edit
from src.preprocessing.protected_spans import find_protected_spans
from src.rules.registry import rule_by_id
from src.rules.synthetic import (
    DICTIONARY_FUZZY_SYNTHETIC_ERRORS,
    SyntheticTransformation,
    TARGETED_BACKFILL_PAIR_TEMPLATES,
    TARGETED_BACKFILL_TERM_BANK,
    TARGETED_BACKFILL_TOPICS,
    TARGETED_BACKFILL_UNSUPPORTED_PROBES,
    apply_transformations,
    source_dataset_for_groups,
    synthetic_transformations,
    synthetic_variant_sources,
)
from src.validation.diff_analyzer import DiffAnalyzer, Edit
from src.validation.edit_classifier import is_allowed_edit_type

TARGETED_BACKFILL_DICTIONARY_RULE_IDS = frozenset(
    {
        "dictionary_fuzzy",
        "double_consonant_candidate",
        "keyboard_typo_candidate",
        "swapped_letters_candidate",
        "missing_letter_candidate",
        "extra_letter_candidate",
    }
)


@dataclass(frozen=True)
class SyntheticExample:
    source: str
    target: str
    error_types: list[str]
    source_dataset: str = "synthetic"
    is_clean: bool = False
    is_synthetic: bool = True
    split: str = "train"
    domain: str = "general"
    rule_ids: list[str] | None = None


@dataclass(frozen=True)
class TargetedBackfillExample:
    source: str
    target: str
    rule_ids: list[str]
    edits: list[dict[str, Any]]
    source_dataset: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RejectedBackfillTemplate:
    rule_id: str
    proposed_source: str
    proposed_target: str
    reason: str
    candidate_count: int
    matching_candidate_found: bool
    generated_candidate_rule_ids: list[str]


@dataclass(frozen=True)
class TargetedBackfillResult:
    examples: list[TargetedBackfillExample]
    rejected: list[RejectedBackfillTemplate]
    excluded_reason: str = ""


class SyntheticGenerator:
    def __init__(self, seed: int = 13, max_errors_per_sentence: int = 3) -> None:
        self.random = random.Random(seed)
        self.max_errors_per_sentence = max_errors_per_sentence

    def generate_from_clean(self, text: str) -> SyntheticExample:
        transformations = self._available_transformations(text)
        self.random.shuffle(transformations)
        selected: list[SyntheticTransformation] = []
        for transformation in transformations:
            if len(selected) >= self.max_errors_per_sentence:
                break
            if _overlaps_any(transformation, selected):
                continue
            selected.append(transformation)

        source = apply_transformations(text, selected)
        error_types = [transformation.error_type for transformation in selected]
        rule_ids = [transformation.rule_id for transformation in selected if transformation.rule_id]

        if not error_types:
            source = self._remove_final_punctuation(source)
            if source != text:
                error_types.append("final_punctuation")
                rule_ids.append("final_punctuation_default")

        return SyntheticExample(
            source=source,
            target=text,
            error_types=error_types or ["punctuation"],
            rule_ids=rule_ids,
        )

    def generate_variants_from_clean(self, text: str, max_variants: int = 20) -> list[SyntheticExample]:
        variants: list[SyntheticExample] = []
        seen_sources: set[str] = set()

        for source, error_types, groups, rule_ids in synthetic_variant_sources(text):
            if source == text or source in seen_sources:
                continue
            seen_sources.add(source)
            variants.append(
                SyntheticExample(
                    source=source,
                    target=text,
                    error_types=error_types,
                    source_dataset=source_dataset_for_groups(groups),
                    rule_ids=[rule_id for rule_id in rule_ids if rule_id],
                )
            )
            if len(variants) >= max_variants:
                break

        return variants or [self.generate_from_clean(text)]

    def add_identity_examples(self, texts: list[str], source_dataset: str = "clean") -> list[SyntheticExample]:
        return [
            SyntheticExample(
                source=text,
                target=text,
                error_types=[],
                source_dataset=source_dataset,
                is_clean=True,
                is_synthetic=False,
                rule_ids=[],
            )
            for text in texts
        ]

    def _available_transformations(self, text: str) -> list[SyntheticTransformation]:
        return synthetic_transformations(text)

    def _remove_final_punctuation(self, text: str) -> str:
        stripped = text.rstrip()
        if stripped and stripped[-1] in ".!?":
            return stripped[:-1]
        return text


class TargetedBackfillGenerator:
    def __init__(self, candidate_generator: CandidateGenerator, seed: int = 13) -> None:
        self.candidate_generator = candidate_generator
        self.dictionary_candidate_generator = CandidateGenerator(
            dictionary_provider=getattr(candidate_generator, "dictionary_provider", None),
            dictionary_limit=getattr(candidate_generator, "dictionary_limit", 2),
            dictionary_min_score=getattr(candidate_generator, "dictionary_min_score", 85),
            dictionary_yo_e_enabled=getattr(candidate_generator, "dictionary_yo_e_enabled", False),
            syntax_provider=lambda _text: (),
        )
        self.fast_candidate_generator = CandidateGenerator(dictionary_lexicon=(), syntax_provider=lambda _text: ())
        self.random = random.Random(seed)
        self.analyzer = DiffAnalyzer()

    def generate_for_rule(
        self,
        rule_id: str,
        required_count: int,
        *,
        seen_pairs: set[tuple[str, str]] | None = None,
    ) -> TargetedBackfillResult:
        if required_count <= 0:
            return TargetedBackfillResult([], [])
        if rule_id not in TARGETED_BACKFILL_PAIR_TEMPLATES and rule_id not in TARGETED_BACKFILL_TERM_BANK:
            return TargetedBackfillResult([], [], "no_existing_candidate_backed_pattern")
        seen_pairs = seen_pairs if seen_pairs is not None else set()
        examples: list[TargetedBackfillExample] = []
        rejected: list[RejectedBackfillTemplate] = []

        for source, target in TARGETED_BACKFILL_UNSUPPORTED_PROBES.get(rule_id, ()):
            rejected.append(self._rejection(rule_id, source, target, "no_existing_candidate_backed_pattern"))

        attempt_limit = _attempt_limit_for_rule(rule_id, required_count)
        for index in range(attempt_limit):
            if len(examples) >= required_count:
                break
            if not examples and index >= max(80, min(required_count, 120)):
                break
            for source, target in self._candidate_pairs(rule_id, index):
                if len(examples) >= required_count:
                    break
                if (source, target) in seen_pairs:
                    continue
                validation = self._validate_pair(rule_id, source, target)
                if isinstance(validation, RejectedBackfillTemplate):
                    if len(rejected) < 500:
                        rejected.append(validation)
                    continue
                seen_pairs.add((source, target))
                examples.append(validation)

        reason = "" if len(examples) >= required_count else "no_existing_candidate_backed_pattern"
        return TargetedBackfillResult(examples, rejected, reason)

    def _candidate_pairs(self, rule_id: str, index: int) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        templates = TARGETED_BACKFILL_PAIR_TEMPLATES.get(rule_id, ())
        if templates:
            source_template, target_template = templates[index % len(templates)]
            topic = TARGETED_BACKFILL_TOPICS[(index // max(1, len(templates))) % len(TARGETED_BACKFILL_TOPICS)]
            source = source_template.format(topic=topic)
            target = target_template.format(topic=topic)
            pairs.append(_with_natural_tail(source, target, index))
        pairs.extend(self._term_pairs(rule_id, index))
        return pairs

    def _term_pairs(self, rule_id: str, index: int) -> list[tuple[str, str]]:
        terms = TARGETED_BACKFILL_TERM_BANK.get(rule_id, ())
        if not terms:
            return []
        term = terms[index % len(terms)]
        topic = TARGETED_BACKFILL_TOPICS[(index // max(1, len(terms))) % len(TARGETED_BACKFILL_TOPICS)]
        target = _term_target_sentence(term, topic, index)
        dirty = _dirty_term_for_rule(rule_id, term)
        if dirty:
            return [(target.replace(term, dirty, 1), target)]
        rule = rule_by_id(rule_id)
        if rule is None or not hasattr(rule, "generate_corruptions"):
            return []
        result: list[tuple[str, str]] = []
        for corruption in rule.generate_corruptions(target):
            if corruption.rule_id == rule_id:
                result.append((corruption.apply(target), target))
        return result

    def _validate_pair(self, rule_id: str, source: str, target: str) -> TargetedBackfillExample | RejectedBackfillTemplate:
        if source == target:
            return self._rejection(rule_id, source, target, "identity_pair")
        if _contains_meta_language(source) or _contains_meta_language(target):
            return self._rejection(rule_id, source, target, "meta_language")
        if find_protected_spans(source) or find_protected_spans(target):
            return self._rejection(rule_id, source, target, "protected_span")
        candidates = self._generate_candidates(rule_id, source)
        candidate_rule_ids = sorted({candidate.rule_id for candidate in candidates if candidate.rule_id})
        edits = self._candidate_backed_edits(source, target, candidates)
        if not edits:
            return RejectedBackfillTemplate(rule_id, source, target, "no_supported_edit", len(candidates), False, candidate_rule_ids)
        if not any(edit.rule_id == rule_id for edit in edits):
            return RejectedBackfillTemplate(
                rule_id,
                source,
                target,
                "no_existing_candidate_backed_pattern",
                len(candidates),
                False,
                candidate_rule_ids,
            )
        return TargetedBackfillExample(
            source=source,
            target=target,
            rule_ids=_ordered_rule_ids(edits),
            edits=[asdict(edit) for edit in edits],
            source_dataset=f"synthetic_targeted_{rule_id}",
            metadata={"target_family": rule_id, "candidate_present": True, "source_type": "synthetic_augmented_from_open_clean"},
        )

    def _candidate_backed_edits(self, source: str, target: str, candidates: list[Any]) -> list[Edit]:
        result: list[Edit] = []
        seen: set[tuple[int, int, str, str, str]] = set()
        for edit in self.analyzer.analyze(source, target, candidates=candidates):
            if not is_allowed_edit_type(edit.edit_type):
                continue
            matches = [candidate for candidate in candidates if candidate_matches_edit(candidate, edit) and candidate.rule_id]
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

    def _rejection(self, rule_id: str, source: str, target: str, reason: str) -> RejectedBackfillTemplate:
        candidates = self._generate_candidates(rule_id, source)
        edits = self.analyzer.analyze(source, target, candidates=candidates)
        matching = any(candidate_matches_edit(candidate, edit) for candidate in candidates for edit in edits)
        return RejectedBackfillTemplate(
            rule_id=rule_id,
            proposed_source=source,
            proposed_target=target,
            reason=reason,
            candidate_count=len(candidates),
            matching_candidate_found=matching,
            generated_candidate_rule_ids=sorted({candidate.rule_id for candidate in candidates if candidate.rule_id}),
        )

    def _generate_candidates(self, rule_id: str, source: str) -> list[Any]:
        generator = self.dictionary_candidate_generator if rule_id in TARGETED_BACKFILL_DICTIONARY_RULE_IDS else self.fast_candidate_generator
        return generator.generate(source)


def _overlaps_any(transformation: SyntheticTransformation, selected: list[SyntheticTransformation]) -> bool:
    return any(_overlaps(transformation, item) for item in selected)


def _overlaps(left: SyntheticTransformation, right: SyntheticTransformation) -> bool:
    if left.start == left.end or right.start == right.end:
        return left.start == right.start
    return max(left.start, right.start) < min(left.end, right.end)


def _attempt_limit_for_rule(rule_id: str, required_count: int) -> int:
    template_count = len(TARGETED_BACKFILL_PAIR_TEMPLATES.get(rule_id, ()))
    term_count = len(TARGETED_BACKFILL_TERM_BANK.get(rule_id, ()))
    estimated_unique = template_count * len(TARGETED_BACKFILL_TOPICS)
    estimated_unique += term_count * len(TARGETED_BACKFILL_TOPICS) * 6
    return max(required_count * 4, estimated_unique * 4, 64)


def _apply_candidate(source: str, candidate: Any) -> str:
    start = int(getattr(candidate, "start", -1))
    end = int(getattr(candidate, "end", -1))
    if start < 0 or end < start:
        return source
    return source[:start] + str(getattr(candidate, "replacement", "")) + source[end:]


def _with_natural_tail(source: str, target: str, index: int) -> tuple[str, str]:
    tails = (
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
        "Позже юрист согласовал договор.",
        "Утром редакция выпустила заметку.",
        "После совещания команда сверила сроки.",
        "Вечером архивариус нашел копию.",
        "Затем отдел подготовил справку.",
        "Позже координатор закрыл заявку.",
        "Утром эксперт прочитал письмо.",
        "После обеда комиссия уточнила вывод.",
        "Вечером автор проверил таблицу.",
        "Затем редактор отправил файл.",
        "Позже группа обсудила договор.",
        "Утром служба приняла заявку.",
        "После проверки руководитель подписал лист.",
        "Вечером секретарь обновил график.",
        "Затем эксперт сверил данные.",
        "Позже редакция открыла архив.",
        "Утром автор исправил дату.",
        "После встречи отдел сохранил запись.",
        "Вечером комиссия закрыла вопрос.",
        "Затем координатор отправил сводку.",
        "Позже аналитик проверил договор.",
        "Утром редактор принял отчет.",
        "После обеда автор открыл журнал.",
        "Вечером отдел сверил ведомость.",
        "Затем юрист обновил папку.",
        "Позже комиссия приняла справку.",
        "Утром группа закрыла протокол.",
        "После встречи редактор сохранил таблицу.",
        "Вечером эксперт уточнил маршрут.",
        "Затем секретарь проверил журнал.",
        "Позже служба согласовала заявку.",
        "Утром автор отправил письмо.",
        "После проверки отдел обновил график.",
        "Вечером координатор принял ответ.",
        "Затем комиссия открыла раздел.",
        "Позже редактор сверил список.",
        "Утром аналитик сохранил отчет.",
        "После обеда юрист проверил подпись.",
        "Вечером группа обновила запись.",
        "Затем служба закрыла вопрос.",
        "Позже эксперт согласовал таблицу.",
        "Утром отдел отправил договор.",
        "После совещания автор уточнил дату.",
        "Вечером редактор открыл протокол.",
        "Затем координатор сверил журнал.",
        "Позже комиссия сохранила решение.",
        "Утром секретарь проверил папку.",
        "После проверки аналитик принял сводку.",
        "Вечером служба обновила список.",
        "Затем юрист отправил письмо.",
        "Позже отдел согласовал график.",
        "Утром редакция сверила выпуск.",
        "После встречи эксперт закрыл заявку.",
        "Вечером автор сохранил раздел.",
        "Затем группа проверила договор.",
        "Позже секретарь уточнил таблицу.",
        "Утром комиссия открыла журнал.",
        "После обеда координатор принял протокол.",
        "Вечером аналитик сверил письмо.",
        "Затем редактор обновил дату.",
        "Позже служба проверила запись.",
        "Утром юрист сохранил справку.",
        "После проверки автор отправил отчет.",
        "Вечером отдел открыл папку.",
        "Затем эксперт согласовал список.",
        "Позже координатор обновил ведомость.",
        "Утром группа приняла решение.",
        "После встречи редакция проверила маршрут.",
        "Вечером комиссия сохранила таблицу.",
        "Затем секретарь закрыл раздел.",
        "Позже аналитик отправил сводку.",
        "Утром служба уточнила график.",
        "После обеда редактор принял письмо.",
        "Вечером юрист сверил договор.",
        "Затем отдел сохранил журнал.",
        "Позже автор проверил подпись.",
        "Утром эксперт обновил протокол.",
        "После проверки комиссия отправила ответ.",
        "Вечером координатор открыл список.",
        "Затем редакция согласовала выпуск.",
        "Позже группа сохранила папку.",
        "Утром аналитик проверил заявку.",
        "После встречи служба приняла справку.",
        "Вечером секретарь отправил письмо.",
        "Затем юрист обновил решение.",
        "Позже отдел сверил таблицу.",
        "Утром автор закрыл вопрос.",
    )
    tail = tails[index % len(tails)]
    return _append_tail(source, tail), _append_tail(target, tail)


def _append_tail(sentence: str, tail: str) -> str:
    sentence = sentence.strip()
    if sentence.endswith("."):
        return f"{sentence} {tail}"
    return f"{sentence}. {tail}"


def _term_target_sentence(term: str, topic: str, index: int) -> str:
    templates = (
        "Редактор проверил {term}, когда готовил {topic}.",
        "Команда внесла {term} в {topic} перед встречей.",
        "Автор увидел {term} и открыл {topic} утром.",
        "Секретарь сохранил {term}, потому что {topic} был важен.",
        "Эксперт отметил {term}, когда читал {topic}.",
        "Аналитик записал {term} и обновил {topic} вечером.",
    )
    return templates[index % len(templates)].format(term=term, topic=topic)


def _dirty_term_for_rule(rule_id: str, term: str) -> str:
    if rule_id == "dictionary_fuzzy":
        return DICTIONARY_FUZZY_SYNTHETIC_ERRORS.get(term, "")
    pairs = {
        "double_consonant_candidate": {
            "грамматика": "граматика",
            "территория": "територия",
            "профессия": "проффесия",
            "комиссия": "комисия",
        },
        "keyboard_typo_candidate": {
            "молоко": "молокл",
            "корова": "клрова",
            "грамматика": "грсмматика",
            "собака": "слбака",
        },
        "swapped_letters_candidate": {
            "корова": "коорва",
            "библиотека": "бибилотека",
            "молоко": "молкоо",
            "собака": "соабка",
        },
        "missing_letter_candidate": {
            "молоко": "млоко",
            "корова": "корва",
            "библиотека": "библотека",
            "собака": "сбака",
        },
        "extra_letter_candidate": {
            "собака": "собакаа",
            "молоко": "молокоо",
            "корова": "коорова",
            "библиотека": "библиотекаа",
        },
    }
    return pairs.get(rule_id, {}).get(term, "")


def _contains_meta_language(text: str) -> bool:
    return bool(re.search(r"\b(?:правило|серия|серии|семейство|context-pairs|ne-pos|n-nn)\b", text.lower()))


def _ordered_rule_ids(edits: list[Edit]) -> list[str]:
    result: list[str] = []
    for edit in edits:
        if edit.rule_id and edit.rule_id not in result:
            result.append(edit.rule_id)
    return result
