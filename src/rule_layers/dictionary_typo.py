from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import csv
import random
import re
from typing import Any

import yaml

from src.orthography_gen.lexeme_cards import load_lexeme_cards
from src.rule_layers.base import LayerDirectCase, LayerOperation, LayerRuleSpec
from src.rule_layers.spec_loader import DEFAULT_LAYERS_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAYER = "dictionary_typo"
FAMILY = "dictionary_typo"
DICTIONARY_TYPO_DIR = "dictionary_typo"
WORD_SPEC_FILES = (
    "normative_words.yaml",
    "borrowed_words.yaml",
    "domain_terms.yaml",
    "common_misspellings.yaml",
)
RUSSIAN_ALPHABET = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
INSERT_REPLACE_ALPHABET = "оеаинтсрлвкмпудя"
MIN_TYPO_WORD_LENGTH = 4
TOKEN_RE = re.compile(r"[А-Яа-яЁё]+")
URL_RE = re.compile(r"^(?:https?://|www\.)", re.IGNORECASE)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
CODE_RE = re.compile(r"^(?=.*[A-Za-z])(?=.*(?:\d|-|_))[A-Za-z0-9_-]+$")

DEFAULT_POSITIVE_TEMPLATES = (
    "В отчёте указано слово «{source}».",
    "Студент исправил пример «{source}».",
    "Система проверяет написание «{source}».",
    "В документе встретилось «{source}».",
    "Редактор отметил форму «{source}».",
    "Пользователь загрузил строку «{source}».",
    "Эксперт сравнил вариант «{source}».",
)
DEFAULT_HARD_NEGATIVE_TEMPLATES = (
    "В отчёте указано слово «{target}».",
    "Система проверяет термин «{target}».",
    "Эксперт сравнил ИИ и вариант «{target}».",
)
DEFAULT_CLEAN_TEMPLATES = (
    "В документе встретилось слово «{target}».",
    "Редактор отметил нормативную форму «{target}».",
    "Пользователь загрузил пример «{target}».",
)


RUSSIAN_KEYBOARD_ROWS = ("йцукенгшщзхъ", "фывапролджэ", "ячсмитьбю")


@dataclass(frozen=True)
class CompiledCorrection:
    source: str
    target: str
    rule_id: str
    sub_rule_id: str
    operation: str
    explanation_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source = self.source.strip()
        target = self.target.strip()
        rule_id = self.rule_id.strip()
        operation = _runtime_operation(self.operation)
        if not source or not target or source == target:
            raise ValueError("CompiledCorrection requires non-empty distinct source and target.")
        if not rule_id:
            raise ValueError("CompiledCorrection rule_id must not be empty.")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "rule_id", rule_id)
        object.__setattr__(self, "sub_rule_id", self.sub_rule_id.strip() or rule_id)
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "explanation_id", self.explanation_id.strip() or rule_id)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class _WordEntry:
    rule_id: str
    sub_rule_id: str
    correct: str
    variants: tuple[str, ...] = ()
    wrong_variants: tuple[str, ...] = ()
    typo_ops: tuple[str, ...] = ()
    allow_short: bool = False
    positive_templates: tuple[str, ...] = ()
    hard_negative_templates: tuple[str, ...] = ()
    clean_templates: tuple[str, ...] = ()


def load_dictionary_typo_specs(
    root: str | Path = DEFAULT_LAYERS_DIR,
    seed: int | None = None,
) -> tuple[LayerRuleSpec, ...]:
    base = _dictionary_typo_dir(root)
    entries = _load_word_entries(base)
    corrections = load_dictionary_typo_corrections(root, seed=seed)

    grouped: dict[str, list[LayerDirectCase]] = defaultdict(list)
    for correction in corrections:
        grouped[correction.rule_id].extend(_positive_cases(correction))

    for entry in entries:
        grouped[entry.rule_id].extend(_identity_cases_for_entry(entry, mode="hard_negative"))
        grouped[entry.rule_id].extend(_identity_cases_for_entry(entry, mode="clean_identity"))

    for rule_id in ("typo_character_noise", "typo_keyboard_neighbor"):
        for entry in entries:
            if entry.rule_id == "dictionary_common_misspellings":
                continue
            grouped[rule_id].extend(_identity_cases_for_entry(entry, rule_id=rule_id, mode="hard_negative", limit=1))
            grouped[rule_id].extend(_identity_cases_for_entry(entry, rule_id=rule_id, mode="clean_identity", limit=1))

    grouped["typo_space_noise"].extend(_space_identity_cases())

    return tuple(
        LayerRuleSpec(
            layer=LAYER,
            rule_id=rule_id,
            family=FAMILY,
            cases=tuple(cases),
            description=_description_for(rule_id),
            explanation=rule_id,
            enabled=True,
            weight=1.0,
            metadata={"source": "lexicon/layers/dictionary_typo"},
        )
        for rule_id, cases in sorted(grouped.items())
        if cases
    )


def load_dictionary_typo_corrections(
    root: str | Path = DEFAULT_LAYERS_DIR,
    seed: int | None = None,
) -> tuple[CompiledCorrection, ...]:
    base = _dictionary_typo_dir(root)
    entries = _load_word_entries(base)
    protected = _protected_lexicon(base, entries)
    explicit = _load_explicit_corrections(base)
    from_entries = _corrections_from_wrong_variants(entries)
    generated = _generated_corrections(base, entries, protected, seed=seed)
    return _dedupe_and_drop_ambiguous((*explicit, *from_entries, *generated))


def generate_typos_for_word(
    word: str,
    *,
    operations: Sequence[str] | None = None,
    seed: int | None = None,
    protected_lexicon: Iterable[str] | None = None,
    allow_short: bool = False,
    max_count: int | None = None,
) -> tuple[str, ...]:
    normalized = word.strip()
    if _skip_typo_source(normalized, allow_short=allow_short):
        return ()

    protected = {_normalize_token(item) for item in protected_lexicon or () if str(item).strip()}
    ops = tuple(operations or ("delete", "insert", "replace", "transpose", "duplicate", "keyboard_neighbor"))
    candidates: list[str] = []
    for operation in ops:
        candidates.extend(_typos_for_operation(normalized, operation))

    filtered = [
        typo
        for typo in dict.fromkeys(candidates)
        if typo != normalized
        and _normalize_token(typo) not in protected
        and not is_protected_token(typo)
    ]
    rng = random.Random(seed)
    rng.shuffle(filtered)
    if max_count is not None:
        filtered = filtered[: max(0, int(max_count))]
    return tuple(filtered)


def generate_keyboard_neighbor_typos(word: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_keyboard_neighbor_typos(word)))


def generate_char_delete_typos(word: str) -> tuple[str, ...]:
    return _typos_for_operation(word.strip(), "delete")


def generate_char_insert_typos(word: str) -> tuple[str, ...]:
    return _typos_for_operation(word.strip(), "insert")


def generate_char_replace_typos(word: str) -> tuple[str, ...]:
    return _typos_for_operation(word.strip(), "replace")


def generate_char_transpose_typos(word: str) -> tuple[str, ...]:
    return _typos_for_operation(word.strip(), "transpose")


def generate_char_duplicate_typos(word: str) -> tuple[str, ...]:
    return _typos_for_operation(word.strip(), "duplicate")


def generate_word_split_typos(
    word: str,
    *,
    seed: int | None = None,
    protected_lexicon: Iterable[str] | None = None,
    allow_short: bool = False,
    max_count: int | None = None,
) -> tuple[str, ...]:
    return generate_typos_for_word(
        word,
        operations=("split_space",),
        seed=seed,
        protected_lexicon=protected_lexicon,
        allow_short=allow_short,
        max_count=max_count,
    )


def generate_glued_phrase_typos(phrase: str) -> tuple[str, ...]:
    value = phrase.strip()
    if not re.fullmatch(r"[А-Яа-яЁё]+(?:\s+[А-Яа-яЁё]+)+", value):
        return ()
    if any(is_protected_token(part) for part in value.split()):
        return ()

    typos: list[str] = []
    for match in re.finditer(r"\s+", value):
        typos.append(value[: match.start()] + value[match.end() :])
    return tuple(dict.fromkeys(typos))


def is_protected_token(token: str) -> bool:
    value = token.strip()
    if not value:
        return True
    if URL_RE.search(value) or EMAIL_RE.search(value):
        return True
    if any("A" <= char <= "Z" or "a" <= char <= "z" for char in value):
        return True
    if any(char.isdigit() for char in value):
        return True
    letters = "".join(TOKEN_RE.findall(value))
    if letters and letters.upper() == letters and len(letters) <= 5:
        return True
    return False


def _positive_cases(correction: CompiledCorrection) -> tuple[LayerDirectCase, ...]:
    templates = tuple(correction.metadata.get("positive_templates") or DEFAULT_POSITIVE_TEMPLATES)
    label = "DICT_REPLACE" if correction.operation == "dict_replace" else "SPAN_REPLACE_BY_LEXICON"
    operation_kind = "token" if label == "DICT_REPLACE" else "token_span"
    cases: list[LayerDirectCase] = []
    for template in templates:
        source_text = _render_template(template, source=correction.source, target=correction.target)
        target_text = _render_template(template, source=correction.target, target=correction.target)
        metadata = _case_metadata(
            correction.rule_id,
            correction.sub_rule_id,
            source=correction.source,
            target=correction.target,
            operation=str(correction.metadata.get("operation") or correction.operation),
            runtime_operation=correction.operation,
            extra=correction.metadata,
        )
        cases.append(
            LayerDirectCase(
                rule_id=correction.rule_id,
                family=FAMILY,
                sub_rule_id=correction.sub_rule_id,
                mode="positive",
                source_text=source_text,
                target_text=target_text,
                token_operations=(
                    LayerOperation(
                        kind=operation_kind,
                        label=label,
                        source_pattern=correction.source,
                        target_pattern=correction.target,
                        metadata={"operation": correction.operation},
                    ),
                ),
                expected_token_edit_count=1,
                expected_gap_edit_count=0,
                metadata=metadata,
            )
        )
    return tuple(cases)


def _identity_cases_for_entry(
    entry: _WordEntry,
    *,
    rule_id: str | None = None,
    mode: str,
    limit: int | None = None,
) -> tuple[LayerDirectCase, ...]:
    resolved_rule_id = rule_id or entry.rule_id
    templates = entry.hard_negative_templates if mode == "hard_negative" else entry.clean_templates
    if not templates:
        templates = DEFAULT_HARD_NEGATIVE_TEMPLATES if mode == "hard_negative" else DEFAULT_CLEAN_TEMPLATES
    cases: list[LayerDirectCase] = []
    for template in tuple(templates)[:limit]:
        text = _render_template(template, source=entry.correct, target=entry.correct)
        cases.append(
            LayerDirectCase(
                rule_id=resolved_rule_id,
                family=FAMILY,
                sub_rule_id=entry.sub_rule_id,
                mode=mode,
                source_text=text,
                target_text=text,
                expected_token_edit_count=0,
                expected_gap_edit_count=0,
                metadata=_case_metadata(
                    resolved_rule_id,
                    entry.sub_rule_id,
                    source=entry.correct,
                    target=entry.correct,
                    operation=mode,
                    runtime_operation="identity",
                    extra={"correct": entry.correct},
                ),
            )
        )
    return tuple(cases)


def _space_identity_cases() -> tuple[LayerDirectCase, ...]:
    cases: list[LayerDirectCase] = []
    for text in (
        "В документе встретилось слово «орфография».",
        "Редактор отметил нормативную форму «в отчёте».",
        "Эксперт сравнил ИИ и слово «пунктуация».",
    ):
        cases.append(
            LayerDirectCase(
                rule_id="typo_space_noise",
                family=FAMILY,
                sub_rule_id="space_identity",
                mode="clean_identity" if "ИИ" not in text else "hard_negative",
                source_text=text,
                target_text=text,
                expected_token_edit_count=0,
                expected_gap_edit_count=0,
                metadata=_case_metadata(
                    "typo_space_noise",
                    "space_identity",
                    source=text,
                    target=text,
                    operation="identity",
                    runtime_operation="identity",
                ),
            )
        )
    return tuple(cases)


def _case_metadata(
    rule_id: str,
    sub_rule_id: str,
    *,
    source: str,
    target: str,
    operation: str,
    runtime_operation: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = dict(extra or {})
    metadata.update(
        {
            "family": FAMILY,
            "rule_id": rule_id,
            "sub_rule_id": sub_rule_id,
            "source": source,
            "target": target,
            "correct": target,
            "operation": operation,
            "runtime_operation": runtime_operation,
        }
    )
    return metadata


def _generated_corrections(
    base: Path,
    entries: Sequence[_WordEntry],
    protected: set[str],
    *,
    seed: int | None,
) -> tuple[CompiledCorrection, ...]:
    config = _read_yaml(base / "typo_generation.yaml")
    generation = config.get("generation", {}) if isinstance(config, Mapping) else {}
    if not isinstance(generation, Mapping) or not bool(generation.get("enabled", True)):
        return ()

    base_seed = _int_value(generation.get("seed"), default=0) if seed is None else int(seed)
    default_max = _int_value(generation.get("max_per_word"), default=2)
    rules = generation.get("rules", [])
    if not isinstance(rules, list):
        raise ValueError(f"typo_generation.yaml generation.rules must be a list: {base}")

    by_source_rule = {(entry.rule_id, entry.sub_rule_id): entry for entry in entries}
    corrections: list[CompiledCorrection] = []
    serial = 0
    for raw_rule in rules:
        if not isinstance(raw_rule, Mapping):
            raise ValueError("typo_generation rule must be a mapping.")
        rule_id = _required_str(raw_rule, "rule_id", base / "typo_generation.yaml")
        operations = _str_tuple(raw_rule.get("operations"))
        include_rule_ids = set(_str_tuple(raw_rule.get("include_rule_ids")))
        max_per_word = _int_value(raw_rule.get("max_per_word"), default=default_max)
        for entry in entries:
            if include_rule_ids and entry.rule_id not in include_rule_ids:
                continue
            allowed_ops = operations or entry.typo_ops or ("delete", "insert", "replace", "transpose", "duplicate")
            typo_seed = base_seed + serial
            serial += 1
            for typo in generate_typos_for_word(
                entry.correct,
                operations=allowed_ops,
                seed=typo_seed,
                protected_lexicon=protected,
                allow_short=entry.allow_short,
                max_count=max_per_word,
            ):
                operation_name = _first_operation_that_generates(entry.correct, typo, allowed_ops)
                runtime_operation = "split_join" if operation_name == "split_space" else "dict_replace"
                corrections.append(
                    CompiledCorrection(
                        source=typo,
                        target=entry.correct,
                        rule_id=rule_id,
                        sub_rule_id=f"{entry.sub_rule_id}_{operation_name}",
                        operation=runtime_operation,
                        metadata={
                            "generated": True,
                            "operation": operation_name,
                            "source_rule_id": entry.rule_id,
                            "source_sub_rule_id": entry.sub_rule_id,
                        },
                    )
                )
    return tuple(corrections)


def _corrections_from_wrong_variants(entries: Sequence[_WordEntry]) -> tuple[CompiledCorrection, ...]:
    corrections: list[CompiledCorrection] = []
    for entry in entries:
        for wrong in entry.wrong_variants:
            operation = "split_join" if " " in wrong or " " in entry.correct else "dict_replace"
            corrections.append(
                CompiledCorrection(
                    source=wrong,
                    target=entry.correct,
                    rule_id=entry.rule_id,
                    sub_rule_id=entry.sub_rule_id,
                    operation=operation,
                    metadata={"operation": "explicit_misspelling"},
                )
            )
    return tuple(corrections)


def _load_word_entries(base: Path) -> tuple[_WordEntry, ...]:
    entries: list[_WordEntry] = []
    for name in WORD_SPEC_FILES:
        path = base / name
        raw = _read_yaml(path)
        for item in _entry_list(raw, path):
            rule_id = _required_str(item, "rule_id", path)
            correct = _required_str(item, "correct", path)
            entries.append(
                _WordEntry(
                    rule_id=rule_id,
                    sub_rule_id=str(item.get("sub_rule_id") or _slug(correct)).strip(),
                    correct=correct,
                    variants=_str_tuple(item.get("variants")),
                    wrong_variants=_str_tuple(item.get("wrong_variants")),
                    typo_ops=_str_tuple(item.get("typo_ops")),
                    allow_short=bool(item.get("allow_short", False)),
                    positive_templates=_str_tuple(item.get("positive_templates")),
                    hard_negative_templates=_str_tuple(item.get("hard_negative_templates")),
                    clean_templates=_str_tuple(item.get("clean_templates")),
                )
            )
    return tuple(entries)


def _load_explicit_corrections(base: Path) -> tuple[CompiledCorrection, ...]:
    path = base / "corrections.yaml"
    raw = _read_yaml(path)
    raw_corrections = raw.get("corrections", []) if isinstance(raw, Mapping) else []
    if not isinstance(raw_corrections, list):
        raise ValueError(f"corrections must be a list: {path}")
    corrections: list[CompiledCorrection] = []
    for item in raw_corrections:
        if not isinstance(item, Mapping):
            raise ValueError(f"correction entry must be a mapping: {path}")
        corrections.append(
            CompiledCorrection(
                source=_required_str(item, "source", path),
                target=_required_str(item, "target", path),
                rule_id=_required_str(item, "rule_id", path),
                sub_rule_id=str(item.get("sub_rule_id") or "").strip(),
                operation=_required_str(item, "operation", path),
                explanation_id=str(item.get("explanation_id") or ""),
                metadata={
                    "operation": str(item.get("subtype") or item.get("operation") or ""),
                    "explicit": True,
                },
            )
        )
    return tuple(corrections)


def _dedupe_and_drop_ambiguous(corrections: Sequence[CompiledCorrection]) -> tuple[CompiledCorrection, ...]:
    unique: dict[tuple[str, str, str, str], CompiledCorrection] = {}
    for correction in corrections:
        key = (
            _normalize_token(correction.source),
            correction.target,
            correction.rule_id,
            correction.operation,
        )
        unique.setdefault(key, correction)

    grouped: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for correction in unique.values():
        grouped[(_normalize_token(correction.source), correction.rule_id, correction.operation)].add(correction.target)

    return tuple(
        correction
        for correction in unique.values()
        if len(grouped[(_normalize_token(correction.source), correction.rule_id, correction.operation)]) == 1
    )


def _protected_lexicon(base: Path, entries: Sequence[_WordEntry]) -> set[str]:
    lexicon_root = _lexicon_root(base)
    protected: set[str] = set()
    for entry in entries:
        protected.add(_normalize_token(entry.correct))
        protected.update(_normalize_token(variant) for variant in entry.variants)

    for csv_path in sorted(lexicon_root.glob("*.csv")):
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    for value in row.values():
                        for token in TOKEN_RE.findall(str(value or "")):
                            protected.add(_normalize_token(token))
        except (OSError, csv.Error):
            continue

    orthography_dir = lexicon_root / "orthography"
    if orthography_dir.exists():
        for card in load_lexeme_cards(orthography_dir):
            protected.add(_normalize_token(card.correct_lemma))
            for forms in card.forms.values():
                correct = str(forms.get("correct") or "")
                if correct:
                    protected.add(_normalize_token(correct))

    protected.update(_normalize_token(item) for item in ("США", "РФ", "ИИ", "ООО", "АО", "ИП", "Анна", "Борис"))
    return {item for item in protected if item}


def _typos_for_operation(word: str, operation: str) -> tuple[str, ...]:
    normalized = operation.strip().lower()
    if normalized == "delete":
        return tuple(word[:index] + word[index + 1 :] for index in range(len(word)))
    if normalized == "insert":
        return tuple(
            word[:index] + char + word[index:]
            for index in range(len(word) + 1)
            for char in INSERT_REPLACE_ALPHABET
        )
    if normalized == "replace":
        return tuple(
            word[:index] + char + word[index + 1 :]
            for index, original in enumerate(word)
            for char in INSERT_REPLACE_ALPHABET
            if char != original
        )
    if normalized == "transpose":
        return tuple(
            word[:index] + word[index + 1] + word[index] + word[index + 2 :]
            for index in range(len(word) - 1)
            if word[index] != word[index + 1]
        )
    if normalized == "duplicate":
        return tuple(word[:index] + word[index] + word[index:] for index in range(len(word)))
    if normalized == "keyboard_neighbor":
        return tuple(_keyboard_neighbor_typos(word))
    if normalized == "split_space":
        return tuple(
            word[:index] + " " + word[index:]
            for index in range(2, len(word) - 1)
            if word[index - 1].lower() in RUSSIAN_ALPHABET and word[index].lower() in RUSSIAN_ALPHABET
        )
    return ()


def _keyboard_neighbor_typos(word: str) -> list[str]:
    typos: list[str] = []
    for index, char in enumerate(word):
        lower = char.lower()
        for replacement in RUSSIAN_KEYBOARD_NEIGHBORS.get(lower, ()):
            rendered = replacement.upper() if char.isupper() else replacement
            typos.append(word[:index] + rendered + word[index + 1 :])
    return typos


def _first_operation_that_generates(word: str, typo: str, operations: Sequence[str]) -> str:
    for operation in operations:
        if typo in _typos_for_operation(word, operation):
            return operation
    return "typo"


def _build_keyboard_neighbors(rows: Sequence[str]) -> dict[str, tuple[str, ...]]:
    positions: dict[str, tuple[int, int]] = {}
    for row_index, row in enumerate(rows):
        for col_index, char in enumerate(row):
            positions[char] = (row_index, col_index)

    neighbors: dict[str, list[str]] = {char: [] for char in positions}
    for char, (row_index, col_index) in positions.items():
        for other, (other_row, other_col) in positions.items():
            if other == char:
                continue
            if abs(row_index - other_row) <= 1 and abs(col_index - other_col) <= 1:
                neighbors[char].append(other)
    return {char: tuple(values) for char, values in neighbors.items()}


RUSSIAN_KEYBOARD_NEIGHBORS = _build_keyboard_neighbors(RUSSIAN_KEYBOARD_ROWS)


def _skip_typo_source(word: str, *, allow_short: bool) -> bool:
    if is_protected_token(word):
        return True
    if not allow_short and len(word) < MIN_TYPO_WORD_LENGTH:
        return True
    return not re.fullmatch(r"[А-Яа-яЁё-]+", word)


def _dictionary_typo_dir(root: str | Path) -> Path:
    path = Path(root)
    if path.name == DICTIONARY_TYPO_DIR:
        return path
    return path / DICTIONARY_TYPO_DIR


def _lexicon_root(base: Path) -> Path:
    if base.name == DICTIONARY_TYPO_DIR and base.parent.name == "layers":
        return base.parent.parent
    if base.name == "layers":
        return base.parent
    return PROJECT_ROOT / "lexicon"


def _read_yaml(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Dictionary typo YAML must contain a mapping: {path}")
    return data


def _entry_list(raw: Mapping[str, Any], path: Path) -> tuple[Mapping[str, Any], ...]:
    entries = raw.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(f"entries must be a list: {path}")
    result: list[Mapping[str, Any]] = []
    for item in entries:
        if not isinstance(item, Mapping):
            raise ValueError(f"entry must be a mapping: {path}")
        result.append(item)
    return tuple(result)


def _required_str(data: Mapping[str, Any], key: str, path: Path) -> str:
    value = str(data.get(key) or "").strip()
    if not value:
        raise ValueError(f"Dictionary typo entry is missing {key!r}: {path}")
    return value


def _str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _int_value(value: Any, *, default: int) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _runtime_operation(operation: str) -> str:
    normalized = operation.strip().lower()
    if normalized in {"replace", "dict_replace", "dictionary"}:
        return "dict_replace"
    if normalized in {"split", "merge", "join", "split_join", "split_word", "glue_words", "glue"}:
        return "split_join"
    return normalized


def _render_template(template: str, *, source: str, target: str) -> str:
    return (
        template.replace("{source}", source)
        .replace("{target}", target)
        .replace("{Source}", _capitalize(source))
        .replace("{Target}", _capitalize(target))
    )


def _capitalize(value: str) -> str:
    if not value:
        return value
    return value[:1].upper() + value[1:]


def _normalize_token(value: str) -> str:
    return value.strip().casefold().replace("ё", "е")


def _slug(value: str) -> str:
    normalized = _normalize_token(value).replace(" ", "_").replace("-", "_")
    return re.sub(r"[^а-яa-z0-9_]+", "", normalized) or "entry"


def _description_for(rule_id: str) -> str:
    return {
        "dictionary_normative_words": "Controlled dictionary spelling examples for frequent Russian words.",
        "dictionary_borrowed_words": "Controlled spelling examples for borrowed Russian words.",
        "dictionary_domain_terms": "Controlled spelling examples for IT, educational, and business terms.",
        "dictionary_common_misspellings": "Trusted exact common misspelling examples.",
        "typo_character_noise": "Bounded generated character typo examples with trusted targets.",
        "typo_keyboard_neighbor": "Russian keyboard-neighbor typo examples with trusted targets.",
        "typo_space_noise": "Trusted split-word and glued-word typo examples.",
    }.get(rule_id, f"Dictionary typo examples for {rule_id}.")


__all__ = [
    "CompiledCorrection",
    "RUSSIAN_KEYBOARD_NEIGHBORS",
    "generate_char_delete_typos",
    "generate_char_duplicate_typos",
    "generate_char_insert_typos",
    "generate_char_replace_typos",
    "generate_char_transpose_typos",
    "generate_glued_phrase_typos",
    "generate_keyboard_neighbor_typos",
    "generate_typos_for_word",
    "generate_word_split_typos",
    "is_protected_token",
    "load_dictionary_typo_corrections",
    "load_dictionary_typo_specs",
]
