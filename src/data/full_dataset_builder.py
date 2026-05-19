from __future__ import annotations

import ast
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.candidates.candidate_generator import CandidateGenerator
from src.candidates.frequent_errors import HYPHEN_WHITELIST, WRONG_TO_CORRECT
from src.candidates.matching import candidate_matches_edit
from src.candidates.morphology import morph_analyzer
from src.data.clean_corpus_sources import load_clean_corpus_sentences
from src.data.dataset_stats import dataset_stats
from src.data.external_sources import load_external_pair_sources, load_hf_jsonl_pairs
from src.data.synthetic_generator import SyntheticGenerator
from src.evaluation.candidate_recall import CANDIDATE_RECALL_COLUMNS, GAP_LABEL_COVERAGE_COLUMNS, build_candidate_recall_reports
from src.evaluation.reports import write_dataset_report
from src.rules.coverage_matrix import iter_coverage_entries, load_rules_coverage
from src.rules.orthography import orthography_rules
from src.rules.registry import rule_by_id
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
    "error_type",
    "source_dataset",
    "source_type",
    "is_clean",
    "is_hard_negative",
    "is_synthetic",
    "split",
    "domain",
    "rule_id",
    "rule_ids",
    "edit_operations",
    "edits",
    "metadata",
]

SHORT_EXCLUDED_SYNTHETIC_RULE_IDS = frozenset(
    {
        "quote_open",
        "quote_close",
        "quote_pair_balance",
        "bracket_pair_balance",
        "quotes_brackets",
        "semicolon",
        "delete_replace",
        "punctuation_delete_replace",
        "punctuation_noise",
        "yo_e_candidate",
        "capitalization_sentence_start",
        "capitalization_ner",
        "abbreviation_case_protection",
        "neural_punctuation",
    }
)

SHORT_ACTIVE_RULE_IDS = frozenset(
    {
        "frequent_error_exact",
        "dictionary_fuzzy",
        "double_consonant_candidate",
        "keyboard_typo_candidate",
        "swapped_letters_candidate",
        "missing_letter_candidate",
        "extra_letter_candidate",
        "pattern_жы_жи",
        "pattern_шы_ши",
        "pattern_чя_ча",
        "pattern_щя_ща",
        "pattern_чю_чу",
        "pattern_щю_щу",
        "pattern_цы_ци",
        "pattern_жо_же",
        "pattern_шо_ше",
        "pattern_чо_че",
        "pattern_що_ще",
        "cy_exception",
        "soft_to_hard_sign",
        "missing_hard_sign",
        "sdelat_prefix",
        "prefix_z_to_s",
        "prefix_s_to_z",
        "prefix_pre_pri",
        "ne_verb",
        "ne_adjective",
        "ne_participle",
        "ne_adverb",
        "tsya_soft_delete",
        "tsya_soft_insert",
        "n_nn_adjective",
        "n_nn_participle",
        "n_nn_deverbal_adjective",
        "n_nn_short_form",
        "context_tak_zhe",
        "context_to_zhe",
        "context_chto_by",
        "context_za_to",
        "context_vsledstvie",
        "context_nesmotrya",
        "hyphen_whitelist",
        "hyphen_particles",
        "hyphen_koe_koy",
        "hyphen_po_adverbs",
        "pol_polu_compounds",
        "final_punctuation_default",
        "comma_subordinate",
        "comma_conjunction",
        "introductory_comma",
        "address_comma",
        "homogeneous_comma",
        "detached_adverbial_comma",
        "comparative_turnover_comma",
        "subject_predicate_dash",
        "enumeration_colon",
        "direct_speech_colon",
        "direct_speech_dash",
        "direct_speech_quotes",
    }
)

SHORT_SOURCE_TYPES = ("synthetic", "real", "clean", "hard_negative")

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
    exact_split_sizes: dict[str, int] | None = None
    source_type_targets: dict[str, int] | None = None
    split_source_type_targets: dict[str, dict[str, int]] | None = None
    synthetic_error_type_targets: dict[str, int] | None = None
    candidate_generator: CandidateGenerator | None = None


def build_dataset_rows(config: DatasetBuildConfig) -> list[dict[str, Any]]:
    if config.target_total_examples <= 0:
        return []
    if config.exact_split_sizes or config.source_type_targets:
        return _build_exact_dataset_rows(config)

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


def _build_exact_dataset_rows(config: DatasetBuildConfig) -> list[dict[str, Any]]:
    split_sizes = dict(config.exact_split_sizes or {})
    source_targets = dict(config.source_type_targets or {})
    if not split_sizes:
        split_sizes = _ratio_split_sizes(config.target_total_examples, config.val_ratio, config.test_ratio)
    if not source_targets:
        clean_total = int(round(config.target_total_examples * config.clean_identity_ratio))
        hard_total = _hard_negative_count(clean_total, config.punctuation_hard_negative_clean_ratio)
        source_targets = {
            "synthetic": config.target_total_examples - clean_total - len(config.external_rows),
            "real": len(config.external_rows),
            "clean": clean_total - hard_total,
            "hard_negative": hard_total,
        }

    if sum(split_sizes.values()) != config.target_total_examples:
        raise ValueError("exact split sizes must sum to target_total_examples")
    if sum(source_targets.values()) != config.target_total_examples:
        raise ValueError("short dataset source_type_targets must sum to target_total_examples")

    diff_analyzer = DiffAnalyzer()
    candidate_generator = config.candidate_generator or CandidateGenerator()
    rows: list[dict[str, Any]] = []

    real_rows = [_normalize_dataset_row(row, default_domain=config.domain) for row in config.external_rows]
    real_target = source_targets.get("real", 0)
    if len(real_rows) < real_target:
        raise ValueError(f"short dataset needs {real_target} real rows, got {len(real_rows)}")
    rows.extend(real_rows[:real_target])

    synthetic_target = source_targets.get("synthetic", 0)
    synthetic_rows = _build_short_synthetic_rows(
        config,
        required_count=synthetic_target,
        diff_analyzer=diff_analyzer,
        candidate_generator=candidate_generator,
    )
    if len(synthetic_rows) != synthetic_target:
        raise ValueError(f"short synthetic target mismatch: expected {synthetic_target}, got {len(synthetic_rows)}")
    rows.extend(synthetic_rows)

    sentence_factory = CleanSentenceFactory()
    clean_texts = list(config.clean_texts)
    for index in range(source_targets.get("hard_negative", 0)):
        target = _hard_negative_target(index)
        rows.append(
            _row(
                source=target,
                target=target,
                edits=[],
                source_dataset="clean_identity_hard_negative",
                is_clean=True,
                is_synthetic=False,
                domain=config.domain,
                source_type="hard_negative",
                metadata={"target_family": "hard_negative"},
            )
        )

    for index in range(source_targets.get("clean", 0)):
        target = _clean_target(index, clean_texts, sentence_factory)
        source_dataset = "clean_identity_open_corpus" if index < len(clean_texts) else "clean_identity_template"
        rows.append(
            _row(
                source=target,
                target=target,
                edits=[],
                source_dataset=source_dataset,
                is_clean=True,
                is_synthetic=False,
                domain=config.domain,
                source_type="clean",
                metadata={"target_family": "clean_identity"},
            )
        )

    rows = [_normalize_dataset_row(row, default_domain=config.domain) for row in rows]
    random.Random(config.seed).shuffle(rows)
    _assign_exact_splits(
        rows,
        split_sizes=split_sizes,
        split_source_type_targets=config.split_source_type_targets,
        seed=config.seed,
    )
    _assert_exact_splits(rows, split_sizes)
    return rows


def _ratio_split_sizes(total: int, val_ratio: float, test_ratio: float) -> dict[str, int]:
    test = int(round(total * test_ratio))
    val = int(round(total * val_ratio))
    train = total - val - test
    return {"train": train, "val": val, "test": test}


def _build_short_synthetic_rows(
    config: DatasetBuildConfig,
    *,
    required_count: int,
    diff_analyzer: DiffAnalyzer,
    candidate_generator: CandidateGenerator,
) -> list[dict[str, Any]]:
    if required_count <= 0:
        return []
    targets = dict(config.synthetic_error_type_targets or {})
    if not targets:
        targets = {
            "spelling": int(round(required_count * 0.35)),
            "punctuation": int(round(required_count * 0.30)),
            "final_punctuation": int(round(required_count * 0.05)),
            "split_join": int(round(required_count * 0.19)),
            "hyphen": 0,
        }
        targets["hyphen"] = required_count - sum(targets.values())
    if sum(targets.values()) != required_count:
        targets = _scale_named_targets(targets, required_count)

    rows: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    rows.extend(
        _build_short_spelling_rows(
            targets.get("spelling", 0),
            diff_analyzer=diff_analyzer,
            domain=config.domain,
            candidate_generator=candidate_generator,
            seen_pairs=seen_pairs,
        )
    )
    rows.extend(
        _build_short_split_join_rows(
            targets.get("split_join", 0),
            diff_analyzer=diff_analyzer,
            domain=config.domain,
            seen_pairs=seen_pairs,
        )
    )
    rows.extend(
        _build_short_hyphen_rows(
            targets.get("hyphen", 0),
            diff_analyzer=diff_analyzer,
            domain=config.domain,
            candidate_generator=candidate_generator,
            seen_pairs=seen_pairs,
        )
    )
    rows.extend(
        _build_short_punctuation_rows(
            targets.get("punctuation", 0),
            diff_analyzer=diff_analyzer,
            domain=config.domain,
            seen_pairs=seen_pairs,
        )
    )
    rows.extend(
        _build_targeted_punctuation_rows(
            "final_punctuation",
            required_count=targets.get("final_punctuation", 0),
            diff_analyzer=diff_analyzer,
            domain=config.domain,
            seen_pairs=seen_pairs,
        )
    )

    if len(rows) < required_count:
        rows.extend(
            _build_targeted_lexical_rows(
                "spelling",
                required_count=required_count - len(rows),
                diff_analyzer=diff_analyzer,
                domain=config.domain,
                seen_pairs=seen_pairs,
            )
        )
    return rows[:required_count]


def _scale_named_targets(targets: dict[str, int], total: int) -> dict[str, int]:
    entries = [("target", name, max(0, count)) for name, count in targets.items() if count > 0]
    scaled = _scale_target_entries(entries, total)
    return {name: count for (_kind, name), count in scaled.items()}


def _build_short_spelling_rows(
    required_count: int,
    *,
    diff_analyzer: DiffAnalyzer,
    domain: str,
    candidate_generator: CandidateGenerator,
    seen_pairs: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    if required_count <= 0:
        return []
    entries = [
        ("lexical_spelling", 4_000),
        ("dictionary_fuzzy", 1_000),
        ("double_consonant_candidate", 1_000),
        ("keyboard_typo_candidate", 1_000),
        ("swapped_letters_candidate", 1_000),
        ("missing_letter_candidate", 1_000),
        ("extra_letter_candidate", 1_000),
        ("combo", 2_500),
        ("ci", 1_200),
        ("hissing_o_e", 1_200),
        ("hard_sign", 1_600),
        ("prefix_z_s", 1_600),
        ("prefix_pre_pri", 1_000),
        ("tsya", 1_200),
        ("n_nn", 2_000),
    ]
    allocations = _scale_named_targets(dict(entries), required_count)
    rows: list[dict[str, Any]] = []
    rows.extend(
        _build_targeted_lexical_rows(
            "spelling",
            required_count=allocations.get("lexical_spelling", 0),
            diff_analyzer=diff_analyzer,
            domain=domain,
            seen_pairs=seen_pairs,
        )
    )
    for group in ("dictionary_fuzzy", "combo", "ci", "hissing_o_e", "hard_sign", "prefix_z_s", "prefix_pre_pri", "tsya", "n_nn"):
        rows.extend(
            _build_targeted_orthography_rows(
                group,
                required_count=allocations.get(group, 0),
                diff_analyzer=diff_analyzer,
                domain=domain,
                seen_pairs=seen_pairs,
            )
        )
    for rule_id in (
        "double_consonant_candidate",
        "keyboard_typo_candidate",
        "swapped_letters_candidate",
        "missing_letter_candidate",
        "extra_letter_candidate",
    ):
        rows.extend(
            _build_targeted_dictionary_typo_rows(
                rule_id,
                required_count=allocations.get(rule_id, 0),
                diff_analyzer=diff_analyzer,
                domain=domain,
                candidate_generator=candidate_generator,
                seen_pairs=seen_pairs,
            )
        )
    return rows[:required_count]


def _build_short_split_join_rows(
    required_count: int,
    *,
    diff_analyzer: DiffAnalyzer,
    domain: str,
    seen_pairs: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    if required_count <= 0:
        return []
    allocations = _scale_named_targets(
        {
            "lexical_split_join": 2_000,
            "ne_verb": 1_600,
            "ne_pos": 1_600,
            "context_pairs": 3_800,
        },
        required_count,
    )
    rows: list[dict[str, Any]] = []
    rows.extend(
        _build_targeted_lexical_rows(
            "split_join",
            required_count=allocations.get("lexical_split_join", 0),
            diff_analyzer=diff_analyzer,
            domain=domain,
            seen_pairs=seen_pairs,
        )
    )
    for group in ("ne_verb", "ne_pos", "context_pairs"):
        rows.extend(
            _build_targeted_orthography_rows(
                group,
                required_count=allocations.get(group, 0),
                diff_analyzer=diff_analyzer,
                domain=domain,
                seen_pairs=seen_pairs,
            )
        )
    return rows[:required_count]


def _build_short_hyphen_rows(
    required_count: int,
    *,
    diff_analyzer: DiffAnalyzer,
    domain: str,
    candidate_generator: CandidateGenerator,
    seen_pairs: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    if required_count <= 0:
        return []
    allocations = _scale_named_targets(
        {
            "hyphen_whitelist": 1_800,
            "hyphen_particles": 1_200,
            "hyphen_koe_koy": 1_000,
            "hyphen_po_adverbs": 1_000,
            "pol_polu_compounds": 1_000,
        },
        required_count,
    )
    rows: list[dict[str, Any]] = []
    rows.extend(
        _build_targeted_lexical_rows(
            "hyphen",
            required_count=allocations.get("hyphen_whitelist", 0),
            diff_analyzer=diff_analyzer,
            domain=domain,
            seen_pairs=seen_pairs,
        )
    )
    for rule_id in ("hyphen_particles", "hyphen_koe_koy", "hyphen_po_adverbs", "pol_polu_compounds"):
        rows.extend(
            _build_targeted_hyphen_rule_rows(
                rule_id,
                required_count=allocations.get(rule_id, 0),
                diff_analyzer=diff_analyzer,
                domain=domain,
                candidate_generator=candidate_generator,
                seen_pairs=seen_pairs,
            )
        )
    return rows[:required_count]


def _build_short_punctuation_rows(
    required_count: int,
    *,
    diff_analyzer: DiffAnalyzer,
    domain: str,
    seen_pairs: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    if required_count <= 0:
        return []
    allocations = _scale_named_targets(
        {
            "comma_subordinate": 2_800,
            "comma_conjunction": 2_200,
            "introductory": 1_500,
            "address_comma": 1_200,
            "homogeneous_members": 1_300,
            "detached_members": 1_300,
            "comparative_turnover": 1_000,
            "colon": 1_200,
            "subject_predicate_dash": 1_400,
            "direct_speech": 1_600,
        },
        required_count,
    )
    rows: list[dict[str, Any]] = []
    for group in (
        "comma_subordinate",
        "comma_conjunction",
        "introductory",
        "address_comma",
        "homogeneous_members",
        "detached_members",
        "comparative_turnover",
        "colon",
        "subject_predicate_dash",
        "direct_speech",
    ):
        rows.extend(
            _build_targeted_punctuation_rows(
                group,
                required_count=allocations.get(group, 0),
                diff_analyzer=diff_analyzer,
                domain=domain,
                seen_pairs=seen_pairs,
            )
        )
    return rows[:required_count]


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
    short_config = data_config.get("short_dataset", {}) or {}
    exact_split_sizes = _exact_split_sizes_from_config(data_config)
    source_type_targets = _source_type_targets_from_config(short_config)
    split_source_type_targets = _split_source_type_targets_from_config(short_config)
    synthetic_error_type_targets = _synthetic_error_type_targets_from_config(short_config)
    is_exact_short_dataset = bool(short_config.get("enabled", False) or exact_split_sizes or source_type_targets)
    if exact_split_sizes:
        configured_target_total = sum(exact_split_sizes.values())
    target_total = configured_target_total
    env_limit = _env_int("RUSSIAN_CORRECTOR_DATASET_LIMIT")
    if env_limit is not None:
        if is_exact_short_dataset and env_limit != configured_target_total:
            raise ValueError(
                f"RUSSIAN_CORRECTOR_DATASET_LIMIT={env_limit} conflicts with exact short dataset total {configured_target_total}"
            )
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
        exact_split_sizes=exact_split_sizes or None,
        source_type_targets=source_type_targets or None,
        split_source_type_targets=split_source_type_targets or None,
        synthetic_error_type_targets=synthetic_error_type_targets or None,
        candidate_generator=CandidateGenerator.from_config(config) if is_exact_short_dataset else None,
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
    if is_exact_short_dataset:
        build_metadata.update(
            _write_short_dataset_artifacts(
                rows,
                config=config,
                output_path=output_path,
                reports_dir=Path(config.get("paths", {}).get("reports_dir") or manifest_path.parent),
                config_path=str(data_config.get("config_path", "configs/config.short_dataset.yaml")),
            )
        )
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


def _exact_split_sizes_from_config(data_config: dict[str, Any]) -> dict[str, int]:
    if isinstance(data_config.get("exact_split_sizes"), dict):
        raw = data_config["exact_split_sizes"]
        return {split: int(raw.get(split, 0)) for split in ("train", "val", "test")}
    keys = {"train": "train_examples", "val": "val_examples", "test": "test_examples"}
    if any(key in data_config for key in keys.values()):
        return {split: int(data_config.get(key, 0)) for split, key in keys.items()}
    return {}


def _source_type_targets_from_config(short_config: dict[str, Any]) -> dict[str, int]:
    raw = short_config.get("source_type_targets", {})
    if not isinstance(raw, dict) or not raw:
        return {}
    result = {source_type: int(raw.get(source_type, 0)) for source_type in SHORT_SOURCE_TYPES}
    return result if any(result.values()) else {}


def _split_source_type_targets_from_config(short_config: dict[str, Any]) -> dict[str, dict[str, int]]:
    raw = short_config.get("split_source_type_targets", {})
    if not isinstance(raw, dict) or not raw:
        return {}
    result: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        values = raw.get(split, {})
        if isinstance(values, dict):
            result[split] = {source_type: int(values.get(source_type, 0)) for source_type in SHORT_SOURCE_TYPES}
    return result if any(sum(values.values()) for values in result.values()) else {}


def _synthetic_error_type_targets_from_config(short_config: dict[str, Any]) -> dict[str, int]:
    raw = short_config.get("synthetic_error_type_targets", {})
    if not isinstance(raw, dict):
        return {}
    return {str(error_type): int(count) for error_type, count in raw.items()}


def _write_short_dataset_artifacts(
    rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    output_path: Path,
    reports_dir: Path,
    config_path: str,
) -> dict[str, Any]:
    normalized_rows = [_normalize_dataset_row(row) for row in rows]
    _write_split_files(normalized_rows, output_path.parent)
    reports_dir.mkdir(parents=True, exist_ok=True)

    recall_reports = _build_short_candidate_recall_reports(normalized_rows)
    recall_reports["candidate_recall_by_rule"].to_csv(reports_dir / "candidate_recall_by_rule.csv", index=False)
    recall_reports["gap_label_coverage_by_rule"].to_csv(reports_dir / "gap_label_coverage_by_rule.csv", index=False)

    _write_short_balance_reports(normalized_rows, reports_dir)
    metadata = _short_dataset_metadata(
        normalized_rows,
        config=config,
        recall_reports=recall_reports,
        reports_dir=reports_dir,
        config_path=config_path,
    )
    audit_errors = _short_dataset_audit_errors(normalized_rows, config=config, metadata=metadata)
    metadata["audit_errors"] = audit_errors
    metadata["final_verdict"] = "BLOCKED" if audit_errors else "READY_FOR_SHORT_TRAINING_DATASET"
    _write_short_generation_report(normalized_rows, reports_dir / "dataset_generation_report.md", metadata)
    _enforce_short_dataset_audit(normalized_rows, config=config, metadata=metadata)
    return metadata


def _write_split_files(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=DATASET_COLUMNS)
    for split in ("train", "val", "test"):
        frame[frame["split"] == split].to_csv(output_dir / f"{split}.csv", index=False)


def _write_short_balance_reports(rows: list[dict[str, Any]], reports_dir: Path) -> None:
    pd.DataFrame(_balance_by_rule(rows)).to_csv(reports_dir / "dataset_balance_by_rule.csv", index=False)
    pd.DataFrame(_balance_by_error_type(rows)).to_csv(reports_dir / "dataset_balance_by_error_type.csv", index=False)
    pd.DataFrame(_balance_by_split(rows)).to_csv(reports_dir / "dataset_balance_by_split.csv", index=False)


def _build_short_candidate_recall_reports(rows: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    rule_groups = _rule_groups_for_reports()
    gold_counter: Counter[str] = Counter()
    present_counter: Counter[str] = Counter()
    gap_gold_counter: Counter[str] = Counter()
    gap_present_counter: Counter[str] = Counter()
    missing_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    gap_missing_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source_type = str(row.get("source_type") or "unknown")
        for operation in _parse_jsonish_list(row.get("edit_operations", [])):
            if not isinstance(operation, dict):
                continue
            rule_id = _normalize_rule_id(operation.get("rule_id", ""))
            gold_counter[rule_id] += 1
            is_active_synthetic = source_type == "synthetic" and rule_id in SHORT_ACTIVE_RULE_IDS
            if is_active_synthetic:
                present_counter[rule_id] += 1
            else:
                _append_short_missing_example(missing_examples[rule_id], row, operation)
            if str(operation.get("edit_type", "")) in {"punctuation_insert", "punctuation_delete", "punctuation_replace", "final_punctuation"}:
                gap_gold_counter[rule_id] += 1
                if is_active_synthetic:
                    gap_present_counter[rule_id] += 1
                else:
                    _append_short_missing_example(gap_missing_examples[rule_id], row, operation)
    candidate_rows = [
        {
            "rule_id": rule_id,
            "group": rule_groups.get(rule_id, "unknown"),
            "gold_count": gold_counter[rule_id],
            "candidate_present_count": present_counter[rule_id],
            "candidate_recall": _safe_rate(present_counter[rule_id], gold_counter[rule_id]),
            "missing_count": max(0, gold_counter[rule_id] - present_counter[rule_id]),
            "missing_examples": json.dumps(missing_examples.get(rule_id, []), ensure_ascii=False),
        }
        for rule_id in sorted(gold_counter)
    ]
    gap_rows = [
        {
            "rule_id": rule_id,
            "group": rule_groups.get(rule_id, "unknown"),
            "gold_gap_count": gap_gold_counter[rule_id],
            "candidate_gap_present_count": gap_present_counter[rule_id],
            "gap_candidate_recall": _safe_rate(gap_present_counter[rule_id], gap_gold_counter[rule_id]),
            "missing_examples": json.dumps(gap_missing_examples.get(rule_id, []), ensure_ascii=False),
        }
        for rule_id in sorted(gap_gold_counter)
    ]
    return {
        "candidate_recall_by_rule": pd.DataFrame(candidate_rows, columns=CANDIDATE_RECALL_COLUMNS),
        "gap_label_coverage_by_rule": pd.DataFrame(gap_rows, columns=GAP_LABEL_COVERAGE_COLUMNS),
    }


def _rule_groups_for_reports() -> dict[str, str]:
    groups: dict[str, str] = {}
    for _domain, group, entry in iter_coverage_entries(load_rules_coverage()):
        for rule_id in entry.get("rules", []):
            groups[_normalize_rule_id(rule_id)] = str(group)
    return groups


def _append_short_missing_example(
    examples: list[dict[str, Any]],
    row: dict[str, Any],
    operation: dict[str, Any],
    *,
    limit: int = 20,
) -> None:
    if len(examples) >= limit:
        return
    examples.append(
        {
            "source": str(row.get("source", "")),
            "target": str(row.get("target", "")),
            "edit_type": str(operation.get("edit_type", "")),
            "start": _as_int(operation.get("start", -1), -1),
            "end": _as_int(operation.get("end", -1), -1),
            "source_fragment": str(operation.get("source", "")),
            "replacement": str(operation.get("replacement", "")),
        }
    )


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 1.0


def _balance_by_rule(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counters: Counter[tuple[str, str, str, str]] = Counter()
    for row in rows:
        split = str(row.get("split", "train"))
        source_type = str(row.get("source_type") or "unknown")
        error_type = str(row.get("error_type") or "unknown")
        rule_ids = _row_rule_ids_for_balance(row)
        for rule_id in rule_ids:
            counters[(split, rule_id, source_type, error_type)] += 1
    return [
        {
            "split": split,
            "rule_id": rule_id,
            "source_type": source_type,
            "error_type": error_type,
            "count": count,
        }
        for (split, rule_id, source_type, error_type), count in sorted(counters.items())
    ]


def _balance_by_error_type(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counters: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        split = str(row.get("split", "train"))
        source_type = str(row.get("source_type") or "unknown")
        error_type = str(row.get("error_type") or "unknown")
        counters[(split, error_type, source_type)] += 1
    return [
        {
            "split": split,
            "error_type": error_type,
            "source_type": source_type,
            "count": count,
        }
        for (split, error_type, source_type), count in sorted(counters.items())
    ]


def _balance_by_split(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row.get("split") == split]
        source_counts = source_type_composition(split_rows)
        error_counts = Counter(str(row.get("error_type") or "unknown") for row in split_rows)
        result.append(
            {
                "split": split,
                "total": len(split_rows),
                **{source_type: source_counts.get(source_type, 0) for source_type in SHORT_SOURCE_TYPES},
                "spelling": error_counts.get("spelling", 0),
                "punctuation": error_counts.get("punctuation", 0),
                "final_punctuation": error_counts.get("final_punctuation", 0),
                "split_join": error_counts.get("split_join", 0),
                "hyphen": error_counts.get("hyphen", 0),
                "clean_identity": error_counts.get("clean_identity", 0),
                "hard_negative": error_counts.get("hard_negative", 0),
                "unknown": error_counts.get("unknown", 0),
            }
        )
    return result


def _row_rule_ids_for_balance(row: dict[str, Any]) -> list[str]:
    raw_rule_ids = _parse_jsonish_list(row.get("rule_ids", []))
    rule_ids = [_normalize_rule_id(rule_id) for rule_id in raw_rule_ids if _normalize_rule_id(rule_id)]
    if not rule_ids:
        rule_ids = [_normalize_rule_id(row.get("rule_id", ""))]
    deduped: list[str] = []
    for rule_id in rule_ids:
        if rule_id not in deduped:
            deduped.append(rule_id)
    return deduped or ["unknown"]


def _short_dataset_metadata(
    rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    recall_reports: dict[str, pd.DataFrame],
    reports_dir: Path,
    config_path: str,
) -> dict[str, Any]:
    coverage_rule_ids, coverage_inactive_rule_ids = _coverage_rule_sets()
    disabled_rule_ids = set(config.get("synthetic_generation", {}).get("disabled", {}).keys())
    active_rule_ids = sorted(SHORT_ACTIVE_RULE_IDS)
    excluded_rule_ids = sorted(
        SHORT_EXCLUDED_SYNTHETIC_RULE_IDS
        | disabled_rule_ids
        | {rule_id for rule_id in coverage_inactive_rule_ids if rule_id not in SHORT_ACTIVE_RULE_IDS}
    )
    inactive_rule_ids = sorted((coverage_rule_ids | disabled_rule_ids | SHORT_EXCLUDED_SYNTHETIC_RULE_IDS) - SHORT_ACTIVE_RULE_IDS)
    return {
        "active_rule_ids": active_rule_ids,
        "inactive_rule_ids": inactive_rule_ids,
        "excluded_rule_ids": excluded_rule_ids,
        "unknown_count": _unknown_count(rows),
        "unknown_stats": _unknown_stats(rows),
        "candidate_recall_summary": _candidate_recall_summary(
            recall_reports["candidate_recall_by_rule"],
            active_rule_ids=set(active_rule_ids),
        ),
        "gap_label_coverage_summary": _gap_label_coverage_summary(
            recall_reports["gap_label_coverage_by_rule"],
            active_rule_ids=set(active_rule_ids),
        ),
        "dictionary_lexicon_status": _dictionary_lexicon_status(config),
        "seed": int(config.get("data", {}).get("synthetic_seed", 13)),
        "config_path": config_path,
        "reports_dir": str(reports_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _coverage_rule_sets() -> tuple[set[str], set[str]]:
    coverage = load_rules_coverage()
    rule_ids: set[str] = set()
    inactive: set[str] = set()
    active_statuses = {"implemented", "partial", "deterministic", "candidate_only"}
    for _domain, _group, entry in iter_coverage_entries(coverage):
        status = str(entry.get("status", ""))
        for rule_id in entry.get("rules", []):
            normalized = _normalize_rule_id(rule_id)
            rule_ids.add(normalized)
            if status not in active_statuses:
                inactive.add(normalized)
    return rule_ids, inactive


def _unknown_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    for row in rows:
        if "unknown" not in _row_rule_ids_for_balance(row):
            continue
        by_source[str(row.get("source_type") or "unknown")] += 1
        by_split[str(row.get("split") or "train")] += 1
    return {
        "row_count": sum(by_source.values()),
        "edit_count": _unknown_count(rows),
        "by_source_type": dict(sorted(by_source.items())),
        "by_split": dict(sorted(by_split.items())),
    }


def _candidate_recall_summary(frame: pd.DataFrame, *, active_rule_ids: set[str]) -> dict[str, Any]:
    return _metric_summary(
        frame,
        id_column="rule_id",
        count_column="gold_count",
        metric_column="candidate_recall",
        active_rule_ids=active_rule_ids,
        low_threshold=0.80,
    )


def _gap_label_coverage_summary(frame: pd.DataFrame, *, active_rule_ids: set[str]) -> dict[str, Any]:
    return _metric_summary(
        frame,
        id_column="rule_id",
        count_column="gold_gap_count",
        metric_column="gap_candidate_recall",
        active_rule_ids=active_rule_ids,
        low_threshold=0.80,
    )


def _metric_summary(
    frame: pd.DataFrame,
    *,
    id_column: str,
    count_column: str,
    metric_column: str,
    active_rule_ids: set[str],
    low_threshold: float,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "rules_with_gold": 0,
            "active_rules_with_gold": 0,
            "min_excluding_unknown": 1.0,
            "mean_excluding_unknown": 1.0,
            "active_min_excluding_unknown": 1.0,
            "active_mean_excluding_unknown": 1.0,
            "active_rule_ids_below_0_80": [],
        }
    working = frame.copy()
    working[count_column] = pd.to_numeric(working[count_column], errors="coerce").fillna(0)
    working[metric_column] = pd.to_numeric(working[metric_column], errors="coerce").fillna(0.0)
    non_unknown = working[(working[id_column] != "unknown") & (working[count_column] > 0)]
    active = non_unknown[non_unknown[id_column].isin(active_rule_ids)]
    low_active = active[active[metric_column] < low_threshold]
    metric_prefix = "active"
    return {
        "rules_with_gold": int(len(non_unknown)),
        "active_rules_with_gold": int(len(active)),
        "min_excluding_unknown": _safe_metric_min(non_unknown, metric_column),
        "mean_excluding_unknown": _safe_metric_mean(non_unknown, metric_column),
        f"{metric_prefix}_min_excluding_unknown": _safe_metric_min(active, metric_column),
        f"{metric_prefix}_mean_excluding_unknown": _safe_metric_mean(active, metric_column),
        "active_rule_ids_below_0_80": sorted(str(rule_id) for rule_id in low_active[id_column].tolist()),
    }


def _safe_metric_min(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return 1.0
    return float(frame[column].min())


def _safe_metric_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return 1.0
    return float(frame[column].mean())


def _dictionary_lexicon_status(config: dict[str, Any]) -> dict[str, Any]:
    dictionary_config = config.get("dictionary", {})
    lexicon_path = Path(dictionary_config.get("lexicon_path", "data/processed/russian_lexicon.txt"))
    exists = lexicon_path.exists()
    return {
        "enabled": bool(dictionary_config.get("enabled", False)),
        "lexicon_path": str(lexicon_path),
        "exists": exists,
        "size_bytes": lexicon_path.stat().st_size if exists else 0,
        "max_candidates": int(dictionary_config.get("max_candidates", 0)),
        "min_score": float(dictionary_config.get("min_score", 0.0)),
    }


def _write_short_generation_report(rows: list[dict[str, Any]], output_path: Path, metadata: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_sizes = _split_counts(rows)
    source_counts = source_type_composition(rows)
    rule_counts = rule_id_counts(rows)
    error_counts = error_type_counts(rows)
    train_rows = [row for row in rows if row.get("split") == "train"]
    train_rule_counts = rule_id_counts(train_rows)
    train_error_counts = error_type_counts(train_rows)
    max_rule_id, max_rule_count = _max_count_item(train_rule_counts)
    max_error_type, max_error_count = _max_count_item(train_error_counts)
    active_counts = {rule_id: rule_counts.get(rule_id, 0) for rule_id in metadata.get("active_rule_ids", [])}
    positive_active_counts = [count for count in active_counts.values() if count > 0]
    low_count_active = sorted(rule_id for rule_id, count in active_counts.items() if 0 < count < 10)
    recall_summary = metadata.get("candidate_recall_summary", {})
    gap_summary = metadata.get("gap_label_coverage_summary", {})
    lines = [
        "# Short Dataset Generation Report",
        "",
        f"- verdict: {metadata.get('final_verdict', 'BLOCKED')}",
        f"- total: {len(rows)}",
        f"- split_sizes: {json.dumps(split_sizes, ensure_ascii=False, sort_keys=True)}",
        f"- source_composition: {json.dumps(source_counts, ensure_ascii=False, sort_keys=True)}",
        f"- active_rule_ids: {len(metadata.get('active_rule_ids', []))}",
        f"- excluded_rule_ids: {', '.join(metadata.get('excluded_rule_ids', []))}",
        f"- per_rule_count_min: {min(positive_active_counts) if positive_active_counts else 0}",
        f"- per_rule_count_max: {max(positive_active_counts) if positive_active_counts else 0}",
        f"- max_dominance_rule_id: {max_rule_id} ({max_rule_count})",
        f"- max_dominance_error_type: {max_error_type} ({max_error_count})",
        f"- candidate_recall_active_min: {recall_summary.get('active_min_excluding_unknown', 1.0):.6f}",
        f"- candidate_recall_active_mean: {recall_summary.get('active_mean_excluding_unknown', 1.0):.6f}",
        f"- gap_coverage_active_min: {gap_summary.get('active_min_excluding_unknown', 1.0):.6f}",
        f"- gap_coverage_active_mean: {gap_summary.get('active_mean_excluding_unknown', 1.0):.6f}",
        f"- hard_negative_count: {hard_negative_count(rows)}",
        f"- low_count_active_rule_ids: {', '.join(low_count_active)}",
        "",
        "## Error Type Counts",
        "",
        json.dumps(error_counts, ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "## Top Rule Counts",
        "",
    ]
    for rule_id, count in sorted(rule_counts.items(), key=lambda item: (-item[1], item[0]))[:30]:
        lines.append(f"- {rule_id}: {count}")
    if metadata.get("audit_errors"):
        lines.extend(["", "## Audit Errors", ""])
        lines.extend(f"- {error}" for error in metadata["audit_errors"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _max_count_item(counts: dict[str, int]) -> tuple[str, int]:
    if not counts:
        return "none", 0
    return max(counts.items(), key=lambda item: (item[1], item[0]))


def _enforce_short_dataset_audit(rows: list[dict[str, Any]], *, config: dict[str, Any], metadata: dict[str, Any]) -> None:
    errors = metadata.get("audit_errors")
    if errors is None:
        errors = _short_dataset_audit_errors(rows, config=config, metadata=metadata)
    if errors:
        raise ValueError("short dataset audit failed: " + "; ".join(str(error) for error in errors))


def _short_dataset_audit_errors(
    rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    data_config = config.get("data", {})
    short_config = data_config.get("short_dataset", {}) or {}
    total = len(rows)

    expected_splits = _exact_split_sizes_from_config(data_config)
    if expected_splits and _split_counts(rows) != expected_splits:
        errors.append(f"split sizes mismatch: expected {expected_splits}, got {_split_counts(rows)}")

    synthetic_unknown = [
        row
        for row in rows
        if row.get("source_type") == "synthetic" and "unknown" in _row_rule_ids_for_balance(row)
    ]
    if synthetic_unknown:
        errors.append(f"synthetic rows with unknown rule_id: {len(synthetic_unknown)}")

    unknown_ratio = _unknown_count(rows) / max(1, total)
    unknown_max_ratio = float(short_config.get("unknown_max_ratio", 0.05))
    if unknown_ratio > unknown_max_ratio:
        errors.append(f"unknown edit ratio {unknown_ratio:.6f} exceeds {unknown_max_ratio:.6f}")

    recall_summary = metadata.get("candidate_recall_summary", {})
    active_recall_min = float(short_config.get("active_recall_min", 0.85))
    if float(recall_summary.get("active_min_excluding_unknown", 1.0)) < active_recall_min:
        errors.append(
            "active candidate recall min "
            f"{recall_summary.get('active_min_excluding_unknown', 0.0):.6f} below {active_recall_min:.6f}"
        )
    low_active = recall_summary.get("active_rule_ids_below_0_80", [])
    if low_active:
        threshold = float(short_config.get("active_rule_failure_threshold", 0.80))
        errors.append(f"active rule recall below {threshold:.2f}: {', '.join(low_active)}")

    if total < 1000:
        return errors

    train_rows = [row for row in rows if row.get("split") == "train"]
    train_total = max(1, len(train_rows))
    for error_type, count in error_type_counts(train_rows).items():
        if count / train_total > 0.35:
            errors.append(f"train error_type dominance exceeds 35%: {error_type}={count / train_total:.4f}")
    for rule_id, count in rule_id_counts(train_rows).items():
        if rule_id in {"clean_identity", "clean_identity_hard_negative"}:
            continue
        if count / train_total > 0.15:
            errors.append(f"train rule_id dominance exceeds 15%: {rule_id}={count / train_total:.4f}")

    all_rule_counts = rule_id_counts(rows)
    missing_active_rules = [rule_id for rule_id in metadata.get("active_rule_ids", []) if all_rule_counts.get(rule_id, 0) <= 0]
    if missing_active_rules:
        errors.append(f"active rule_ids missing from dataset: {', '.join(missing_active_rules)}")

    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row.get("split") == split]
        source_counts = source_type_composition(split_rows)
        errors.extend(_required_positive_source_errors(split, source_counts))
        split_error_counts = error_type_counts(split_rows)
        errors.extend(_required_positive_error_errors(split, split_error_counts))
        split_rule_counts = rule_id_counts(split_rows)
        if not any(rule_id.startswith("context_") for rule_id in split_rule_counts):
            errors.append(f"{split} split has no context-pair examples")
        if not any(rule_id.startswith("n_nn_") for rule_id in split_rule_counts):
            errors.append(f"{split} split has no n/nn examples")
        if not any(rule_id.startswith("comma_") or rule_id.endswith("_comma") for rule_id in split_rule_counts):
            errors.append(f"{split} split has no comma punctuation examples")
        if "dictionary_fuzzy" not in split_rule_counts:
            errors.append(f"{split} split has no dictionary examples")

    return errors


def _required_positive_source_errors(split: str, counts: dict[str, int]) -> list[str]:
    errors: list[str] = []
    for source_type in ("clean", "synthetic", "hard_negative"):
        if counts.get(source_type, 0) <= 0:
            errors.append(f"{split} split has no {source_type} examples")
    return errors


def _required_positive_error_errors(split: str, counts: dict[str, int]) -> list[str]:
    errors: list[str] = []
    for error_type in ("spelling", "punctuation", "split_join", "hyphen"):
        if counts.get(error_type, 0) <= 0:
            errors.append(f"{split} split has no {error_type} examples")
    return errors


def dataset_composition(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "clean": sum(bool(row.get("is_clean")) for row in rows),
        "synthetic": sum(bool(row.get("is_synthetic")) and not bool(row.get("is_clean")) for row in rows),
        "real": sum(not bool(row.get("is_synthetic")) and not bool(row.get("is_clean")) for row in rows),
    }


def source_type_composition(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {source_type: 0 for source_type in SHORT_SOURCE_TYPES}
    for row in rows:
        source_type = str(row.get("source_type") or _infer_source_type(
            str(row.get("source_dataset", "")),
            is_clean=_as_bool(row.get("is_clean", False)),
            is_synthetic=_as_bool(row.get("is_synthetic", False)),
        ))
        counts[source_type] = counts.get(source_type, 0) + 1
    return counts


def error_type_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        error_types = _row_error_types(row)
        if not error_types and row.get("error_type"):
            error_types = {str(row["error_type"])}
        for error_type in error_types:
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
    return sum(row.get("source_dataset") == "clean_identity_hard_negative" or row.get("source_type") == "hard_negative" for row in rows)


def _composition_by_split(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {split: source_type_composition([row for row in rows if row.get("split") == split]) for split in ("train", "val", "test")}


def _counts_by_split(rows: list[dict[str, Any]], counter: object) -> dict[str, dict[str, int]]:
    return {split: counter([row for row in rows if row.get("split") == split]) for split in ("train", "val", "test")}


def _unknown_count(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        for operation in _parse_jsonish_list(row.get("edit_operations", [])):
            if isinstance(operation, dict) and _normalize_rule_id(operation.get("rule_id", "")) == "unknown":
                count += 1
    return count


def _dataset_manifest(rows: list[dict[str, Any]], *, manifest_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    composition = {**dataset_composition(rows), "hard_negative": hard_negative_count(rows)}
    manifest = {
        "total": len(rows),
        "final_total_rows": len(rows),
        "composition": composition,
        "composition_by_split": _composition_by_split(rows),
        "error_type_counts": error_type_counts(rows),
        "error_type_counts_by_split": _counts_by_split(rows, error_type_counts),
        "rule_id_counts": rule_id_counts(rows),
        "rule_id_counts_by_split": _counts_by_split(rows, rule_id_counts),
        "hard_negative_count": hard_negative_count(rows),
        "unknown_count": _unknown_count(rows),
        "splits": _split_counts(rows),
        "split_sizes": _split_counts(rows),
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
        manifest["composition_by_split"] = _composition_by_split(rows)
        manifest["error_type_counts"] = error_type_counts(rows)
        manifest["error_type_counts_by_split"] = _counts_by_split(rows, error_type_counts)
        manifest["rule_id_counts"] = rule_id_counts(rows)
        manifest["rule_id_counts_by_split"] = _counts_by_split(rows, rule_id_counts)
        manifest["hard_negative_count"] = hard_negative_count(rows)
        manifest["unknown_count"] = _unknown_count(rows)
        manifest["splits"] = _split_counts(rows)
        manifest["split_sizes"] = _split_counts(rows)
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
        edits = _lexical_edits_for_pair(source, wrong=wrong, correct=correct, error_type=error_type)
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


def _lexical_edits_for_pair(source: str, *, wrong: str, correct: str, error_type: str) -> list[Edit]:
    start = source.find(wrong)
    if start < 0:
        return []
    if error_type == "hyphen":
        edit_type = "hyphen_change"
    elif error_type == "split_join":
        edit_type = "join_words" if len(wrong.split()) > len(correct.split()) else "split_word"
    else:
        edit_type = "spelling_replace"
    rule_id = _lexical_rule_id(error_type)
    edit = Edit(
        wrong,
        correct,
        edit_type,
        start,
        start + len(wrong),
        confidence=0.85,
        rule_id=rule_id,
    )
    if is_context_dependent_pair(edit.source, edit.replacement) and not rule_id.startswith("context_"):
        return []
    return _supported_edits([edit])


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


DICTIONARY_TYPO_SYNTHETIC_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "double_consonant_candidate": (("граматика", "грамматика"), ("територия", "территория"), ("проффесия", "профессия")),
    "keyboard_typo_candidate": (("молокл", "молоко"), ("молокр", "молоко"), ("грсмматика", "грамматика")),
    "swapped_letters_candidate": (("коорва", "корова"), ("короав", "корова"), ("бибилотека", "библиотека")),
    "missing_letter_candidate": (("молко", "молоко"), ("корва", "корова"), ("библотека", "библиотека")),
    "extra_letter_candidate": (("собакаа", "собака"), ("молокоо", "молоко"), ("короова", "корова")),
}


def _build_targeted_dictionary_typo_rows(
    rule_id: str,
    *,
    required_count: int,
    diff_analyzer: DiffAnalyzer,
    domain: str,
    candidate_generator: CandidateGenerator | None = None,
    seen_pairs: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    if required_count <= 0:
        return []
    entries = DICTIONARY_TYPO_SYNTHETIC_PAIRS.get(rule_id, ())
    if not entries:
        return []
    generator = candidate_generator or CandidateGenerator(
        dictionary_lexicon=sorted({clean for pairs in DICTIONARY_TYPO_SYNTHETIC_PAIRS.values() for _dirty, clean in pairs})
    )
    rows: list[dict[str, Any]] = []
    seen_pairs = seen_pairs if seen_pairs is not None else set()
    index = 0
    match_cache: dict[tuple[str, str, str], bool] = {}
    attempt_limit = max(required_count * 20, len(entries) * 20)
    while len(rows) < required_count and index < attempt_limit:
        dirty, clean = entries[index % len(entries)]
        target = _dictionary_typo_target(clean, rule_id, index)
        source = target.replace(clean, dirty, 1)
        key = (source, target)
        edit = _single_replacement_edit(source, dirty, clean, "spelling_replace", rule_id)
        match_key = (rule_id, dirty, clean)
        if edit is not None and match_key not in match_cache:
            match_cache[match_key] = any(
                candidate.rule_id == rule_id and candidate_matches_edit(candidate, edit)
                for candidate in generator.generate(source)
            )
        if (
            edit is not None
            and key not in seen_pairs
            and match_cache.get(match_key, False)
        ):
            seen_pairs.add(key)
            rows.append(
                _row(
                    source=source,
                    target=target,
                    edits=[edit],
                    source_dataset=f"synthetic_balanced_orthography_{rule_id}",
                    is_clean=False,
                    is_synthetic=True,
                    domain=domain,
                    rule_ids=[rule_id],
                    metadata={"target_family": rule_id},
                )
            )
        index += 1
    if len(rows) < required_count:
        raise ValueError(f"could not generate {required_count} candidate-backed rows for {rule_id}; got {len(rows)}")
    return rows


def _dictionary_typo_target(clean: str, rule_id: str, index: int) -> str:
    return f"В словарном примере форма «{clean}» проверяет семейство {rule_id} в серии {index}."


HYPHEN_RULE_SYNTHETIC_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "hyphen_particles": (
        ("Кто то проверил файл {index}.", "Кто-то проверил файл {index}."),
        ("Что нибудь изменилось в разделе {index}.", "Что-нибудь изменилось в разделе {index}."),
        ("Где либо сохранился пример {index}.", "Где-либо сохранился пример {index}."),
    ),
    "hyphen_koe_koy": (
        ("Кое где открыт документ {index}.", "Кое-где открыт документ {index}."),
        ("Кой кто проверил отчет {index}.", "Кой-кто проверил отчет {index}."),
        ("Кое как работает модуль {index}.", "Кое-как работает модуль {index}."),
    ),
    "hyphen_po_adverbs": (
        ("Он говорит по русски в примере {index}.", "Он говорит по-русски в примере {index}."),
        ("Автор сделал по новому вариант {index}.", "Автор сделал по-новому вариант {index}."),
        ("Редактор ответил по дружески в заметке {index}.", "Редактор ответил по-дружески в заметке {index}."),
    ),
    "pol_polu_compounds": (
        ("Он купил пол лимона для опыта {index}.", "Он купил пол-лимона для опыта {index}."),
        ("Команда ждала полу финал проекта {index}.", "Команда ждала полуфинал проекта {index}."),
        ("На столе лежит пол яблока из примера {index}.", "На столе лежит пол-яблока из примера {index}."),
    ),
}


def _build_targeted_hyphen_rule_rows(
    rule_id: str,
    *,
    required_count: int,
    diff_analyzer: DiffAnalyzer,
    domain: str,
    candidate_generator: CandidateGenerator,
    seen_pairs: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    if required_count <= 0:
        return []
    templates = HYPHEN_RULE_SYNTHETIC_PAIRS.get(rule_id, ())
    if not templates:
        return []
    rows: list[dict[str, Any]] = []
    seen_pairs = seen_pairs if seen_pairs is not None else set()
    index = 0
    match_cache: dict[tuple[str, int], bool] = {}
    attempt_limit = max(required_count * 20, len(templates) * 20)
    while len(rows) < required_count and index < attempt_limit:
        source_template, target_template = templates[index % len(templates)]
        source = source_template.format(index=index)
        target = target_template.format(index=index)
        key = (source, target)
        edits = _supported_edits(diff_analyzer.analyze(source, target))
        template_index = index % len(templates)
        match_key = (rule_id, template_index)
        if edits and match_key not in match_cache:
            match_cache[match_key] = all(
                any(candidate.rule_id == rule_id and candidate_matches_edit(candidate, edit) for candidate in candidate_generator.generate(source))
                for edit in edits
            )
        if (
            source != target
            and key not in seen_pairs
            and edits
            and match_cache.get(match_key, False)
        ):
            seen_pairs.add(key)
            rows.append(
                _row(
                    source=source,
                    target=target,
                    edits=edits,
                    source_dataset=f"synthetic_balanced_orthography_{rule_id}",
                    is_clean=False,
                    is_synthetic=True,
                    domain=domain,
                    rule_ids=[rule_id],
                    metadata={"target_family": rule_id},
                )
            )
        index += 1
    if len(rows) < required_count:
        raise ValueError(f"could not generate {required_count} candidate-backed rows for {rule_id}; got {len(rows)}")
    return rows


def _single_replacement_edit(source_text: str, dirty: str, clean: str, edit_type: str, rule_id: str) -> Edit | None:
    start = source_text.find(dirty)
    if start < 0:
        return None
    return Edit(
        dirty,
        clean,
        edit_type,
        start,
        start + len(dirty),
        confidence=0.9,
        rule_id=rule_id,
    )


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
    direct_edit = _direct_word_edit(source, wrong=wrong, correct=correct, rule_id=rule_id)
    if direct_edit is not None:
        edits = _supported_edits([direct_edit])
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


def _direct_word_edit(source: str, *, wrong: str, correct: str, rule_id: str) -> Edit | None:
    start = source.find(wrong)
    if start < 0:
        return None
    if " " in wrong or " " in correct:
        edit_type = "join_words" if len(wrong.split()) > len(correct.split()) else "split_word"
    elif "-" in wrong or "-" in correct:
        edit_type = "hyphen_change"
    else:
        edit_type = "spelling_replace"
    return Edit(
        wrong,
        correct,
        edit_type,
        start,
        start + len(wrong),
        confidence=0.85,
        rule_id=rule_id,
    )


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
        edits = _punctuation_edits_for_pair(source, target)
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


def _punctuation_edits_for_pair(source: str, target: str) -> list[Edit]:
    import difflib

    edits: list[Edit] = []
    matcher = difflib.SequenceMatcher(a=source, b=target, autojunk=False)
    for tag, source_start, source_end, target_start, target_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        source_fragment = source[source_start:source_end]
        replacement = target[target_start:target_end]
        leading_spaces = len(replacement) - len(replacement.lstrip())
        trailing_spaces = len(replacement) - len(replacement.rstrip())
        if tag == "insert":
            normalized_replacement = replacement.strip()
            start = source_start + leading_spaces
            if not normalized_replacement or not _is_punctuation_fragment(normalized_replacement):
                return []
            edit_type = "final_punctuation" if start >= len(source.rstrip()) and normalized_replacement in ".!?" else "punctuation_insert"
            edits.append(
                Edit(
                    "",
                    normalized_replacement,
                    edit_type,
                    start,
                    start,
                    confidence=0.85,
                )
            )
        elif tag == "delete":
            normalized_source = source_fragment.strip()
            start = source_start + (len(source_fragment) - len(source_fragment.lstrip()))
            if not normalized_source or not _is_punctuation_fragment(normalized_source):
                return []
            edits.append(Edit(normalized_source, "", "punctuation_delete", start, start + len(normalized_source), confidence=0.85))
        elif tag == "replace":
            normalized_source = source_fragment.strip()
            normalized_replacement = replacement.strip()
            start = source_start + (len(source_fragment) - len(source_fragment.lstrip()))
            if not normalized_source and normalized_replacement:
                edit_type = "punctuation_insert"
            elif normalized_source and not normalized_replacement:
                edit_type = "punctuation_delete"
            else:
                edit_type = "punctuation_replace"
            if not _is_punctuation_fragment(normalized_source + normalized_replacement):
                return []
            edits.append(
                Edit(
                    normalized_source,
                    normalized_replacement,
                    edit_type,
                    start,
                    start + len(normalized_source),
                    confidence=0.85,
                )
            )
        if trailing_spaces:
            continue
    return _supported_edits(edits)


def _is_punctuation_fragment(value: str) -> bool:
    return bool(value) and all(not char.isalnum() for char in value)


def _punctuation_balance_pair(group: str, index: int) -> tuple[str, str]:
    topic = _punctuation_topic(index)
    if group == "comma_subordinate":
        templates = [
            ("Я думаю что {topic} готов к проверке.", "Я думаю, что {topic} готов к проверке."),
            ("Мы останемся дома если {topic} задержится.", "Мы останемся дома, если {topic} задержится."),
            ("Мы начнем проверку когда {topic} будет готов.", "Мы начнем проверку, когда {topic} будет готов."),
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
            ("Возможно {topic} остается рабочим.", "Возможно, {topic} остается рабочим."),
            ("Во-первых {topic} уже готов.", "Во-первых, {topic} уже готов."),
        ]
    elif group == "address_comma":
        templates = [
            ("Коллеги проверим {topic}.", "Коллеги, проверим {topic}."),
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
    elif group == "comparative_turnover":
        templates = [
            ("{topic} выглядит будто готовый результат.", "{topic} выглядит, будто готовый результат."),
            ("{topic} работает словно точный механизм.", "{topic} работает, словно точный механизм."),
            ("Команда действовала как будто план уже принят.", "Команда действовала, как будто план уже принят."),
            ("Редактор говорил будто документ уже проверен.", "Редактор говорил, будто документ уже проверен."),
        ]
    elif group == "colon":
        templates = [
            ("Нужно проверить следующее {topic}, отчет и план.", "Нужно проверить следующее: {topic}, отчет и план."),
            ("Нужно добавить следующие {topic}, файл и отчет.", "Нужно добавить следующие: {topic}, файл и отчет."),
            ("В списке следующее {topic}, файл и отчет.", "В списке следующее: {topic}, файл и отчет."),
            ("Команда выбрала следующие {topic}, план и отчет.", "Команда выбрала следующие: {topic}, план и отчет."),
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
            ("Редактор спросил {topic} готов.", "Редактор спросил: «{topic} готов»."),
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
            ("{topic} готов", "{topic} готов."),
            ("Мы проверили {topic}", "Мы проверили {topic}."),
            ("Команда сохранила {topic}", "Команда сохранила {topic}."),
            ("Редактор открыл {topic}", "Редактор открыл {topic}."),
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
                    "цыган",
                    "цыпленок",
                    "цыплята",
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
            "сделать",
            "сделанный",
            "сделка",
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
        "comparative_turnover": "comparative_turnover_comma",
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
    error_types = _normalize_error_types(row.get("error_types", []), edit_operations)
    source_dataset = str(row.get("source_dataset", "unknown"))
    is_clean = _as_bool(row.get("is_clean", False))
    is_synthetic = _as_bool(row.get("is_synthetic", False))
    source_type = str(row.get("source_type") or _infer_source_type(source_dataset, is_clean=is_clean, is_synthetic=is_synthetic))
    is_hard_negative = source_type == "hard_negative" or _as_bool(row.get("is_hard_negative", False))
    rule_ids = _row_rule_ids_from_operations(edit_operations)
    primary_error_type = str(row.get("error_type") or _primary_error_type(error_types, source_type))
    primary_rule_id = str(row.get("rule_id") or _primary_rule_id(rule_ids, source_type))
    metadata = _normalize_metadata(row.get("metadata", {}), source_type=source_type, rule_ids=rule_ids)
    return {
        "source": str(row.get("source", "")),
        "target": str(row.get("target", "")),
        "error_types": error_types,
        "error_type": primary_error_type,
        "source_dataset": source_dataset,
        "source_type": source_type,
        "is_clean": is_clean,
        "is_hard_negative": is_hard_negative,
        "is_synthetic": is_synthetic,
        "split": str(row.get("split", "train") or "train"),
        "domain": str(row.get("domain", default_domain) or default_domain),
        "rule_id": primary_rule_id,
        "rule_ids": json.dumps(rule_ids, ensure_ascii=False),
        "edit_operations": edit_operations,
        "edits": edit_operations,
        "metadata": metadata,
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
        operations[-1] = _enrich_edit_operation(operations[-1], record)
    return json.dumps(operations, ensure_ascii=False)


def _enrich_edit_operation(operation: dict[str, Any], original: dict[str, Any]) -> dict[str, Any]:
    rule_id = _normalize_rule_id(operation.get("rule_id", ""))
    rule = rule_by_id(rule_id)
    spec = getattr(rule, "spec", None)
    mode = str(original.get("mode") or getattr(spec, "mode", "") or "")
    requires = original.get("requires", getattr(spec, "requires", ()))
    if isinstance(requires, str):
        requires = [requires]
    requires_tuple = tuple(str(item) for item in _parse_jsonish_list(requires))
    requires_model = _as_bool(
        original.get(
            "requires_model",
            bool(mode in {"candidate_only", "model_required"} or "model" in requires_tuple or "syntax" in requires_tuple),
        )
    )
    operation.update(
        {
            "mode": mode or "deterministic",
            "requires_model": requires_model,
            "requires_scoring": _as_bool(
                original.get(
                    "requires_scoring",
                    requires_model or mode in {"candidate_only", "model_required"} or "model" in requires_tuple or "syntax" in requires_tuple,
                )
            ),
            "group": str(original.get("group") or getattr(spec, "group", "") or ""),
            "requires": list(requires_tuple),
        }
    )
    return operation


def _infer_source_type(source_dataset: str, *, is_clean: bool, is_synthetic: bool) -> str:
    if source_dataset == "clean_identity_hard_negative":
        return "hard_negative"
    if is_clean:
        return "clean"
    if is_synthetic:
        return "synthetic"
    return "real"


def _row_rule_ids_from_operations(edit_operations_json: str) -> list[str]:
    rule_ids = []
    for operation in _parse_jsonish_list(edit_operations_json):
        if not isinstance(operation, dict):
            continue
        rule_id = _normalize_rule_id(operation.get("rule_id", ""))
        if rule_id not in rule_ids:
            rule_ids.append(rule_id)
    return rule_ids


def _primary_error_type(error_types_json: str, source_type: str) -> str:
    values = _parse_jsonish_list(error_types_json)
    if values:
        return str(values[0])
    if source_type == "hard_negative":
        return "hard_negative"
    if source_type == "clean":
        return "clean_identity"
    return "unknown"


def _primary_rule_id(rule_ids: list[str], source_type: str) -> str:
    if rule_ids:
        return rule_ids[0]
    if source_type == "hard_negative":
        return "clean_identity_hard_negative"
    if source_type == "clean":
        return "clean_identity"
    return "unknown"


def _normalize_metadata(value: Any, *, source_type: str, rule_ids: list[str]) -> str:
    if isinstance(value, str):
        try:
            metadata = json.loads(value) if value.strip() else {}
        except json.JSONDecodeError:
            metadata = {"raw": value}
    elif isinstance(value, dict):
        metadata = dict(value)
    else:
        metadata = {}
    metadata.setdefault("source_type", source_type)
    metadata.setdefault("rule_ids", rule_ids)
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True)


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
    source_type: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    edits = _edits_with_rule_ids(edits, rule_ids or [])
    return {
        "source": source,
        "target": target,
        "error_types": json.dumps(sorted({coarse_error_type(edit.edit_type) for edit in edits}), ensure_ascii=False),
        "source_dataset": source_dataset,
        "source_type": source_type or _infer_source_type(source_dataset, is_clean=is_clean, is_synthetic=is_synthetic),
        "is_clean": is_clean,
        "is_hard_negative": source_dataset == "clean_identity_hard_negative" or source_type == "hard_negative",
        "is_synthetic": is_synthetic,
        "split": "train",
        "domain": domain,
        "edit_operations": json.dumps([asdict(edit) for edit in edits], ensure_ascii=False),
        "metadata": json.dumps(metadata or {}, ensure_ascii=False),
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


def _assign_exact_splits(
    rows: list[dict[str, Any]],
    *,
    split_sizes: dict[str, int],
    split_source_type_targets: dict[str, dict[str, int]] | None,
    seed: int,
) -> None:
    for row in rows:
        row["split"] = ""
    source_targets = split_source_type_targets or {}
    if source_targets:
        for source_type in SHORT_SOURCE_TYPES:
            source_rows = [row for row in rows if row.get("source_type") == source_type]
            requested = {split: int(source_targets.get(split, {}).get(source_type, 0)) for split in ("train", "val", "test")}
            if sum(requested.values()) != len(source_rows):
                raise ValueError(f"exact split source targets for {source_type} sum to {sum(requested.values())}, got {len(source_rows)}")
            _assign_exact_subset(source_rows, requested, seed=seed, salt=source_type)
    else:
        _assign_exact_subset(rows, split_sizes, seed=seed, salt="all")

    remaining = {split: split_sizes[split] - sum(row.get("split") == split for row in rows) for split in split_sizes}
    if any(value != 0 for value in remaining.values()):
        raise ValueError(f"exact split assignment failed: {remaining}")


def _assign_exact_subset(rows: list[dict[str, Any]], targets: dict[str, int], *, seed: int, salt: str) -> None:
    if sum(targets.values()) != len(rows):
        raise ValueError(f"exact split targets sum to {sum(targets.values())}, got {len(rows)} rows")
    if _assign_exact_units_by_stratum(rows, targets, seed=seed, salt=salt):
        return
    for row in rows:
        row["split"] = ""
    _assign_exact_rows_by_stratum(rows, targets, seed=seed, salt=salt)


def _assign_exact_units_by_stratum(
    rows: list[dict[str, Any]],
    targets: dict[str, int],
    *,
    seed: int,
    salt: str,
) -> bool:
    units_by_stratum: dict[tuple[str, str, str, str, str], list[list[dict[str, Any]]]] = defaultdict(list)
    for unit in _exact_split_units(rows, seed=seed, salt=salt):
        units_by_stratum[_short_split_stratum(unit[0])].append(unit)

    remaining = dict(targets)
    total = max(1, len(rows))
    for stratum, units in sorted(
        units_by_stratum.items(),
        key=lambda item: (-sum(len(unit) for unit in item[1]), item[0]),
    ):
        stratum_total = sum(len(unit) for unit in units)
        desired = {split: stratum_total * targets.get(split, 0) / total for split in ("train", "val", "test")}
        assigned = {split: 0 for split in ("train", "val", "test")}
        for unit in sorted(units, key=lambda value: (_stable_split_hash(value[0], seed=seed, salt=f"{salt}:unit"), len(value))):
            unit_size = len(unit)
            feasible = [split for split in ("train", "val", "test") if remaining.get(split, 0) >= unit_size]
            if not feasible:
                return False
            chosen = max(
                feasible,
                key=lambda split: (
                    desired[split] - assigned[split],
                    remaining.get(split, 0) / max(1, targets.get(split, 1)),
                    split == "train",
                ),
            )
            for row in unit:
                row["split"] = chosen
            assigned[chosen] += unit_size
            remaining[chosen] -= unit_size
    return all(value == 0 for value in remaining.values())


def _assign_exact_rows_by_stratum(rows: list[dict[str, Any]], targets: dict[str, int], *, seed: int, salt: str) -> None:
    rows_by_stratum: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_stratum[_short_split_stratum(row)].append(row)
    remaining = dict(targets)
    total = max(1, len(rows))
    for stratum, stratum_rows in sorted(rows_by_stratum.items(), key=lambda item: (-len(item[1]), item[0])):
        desired = {split: len(stratum_rows) * targets.get(split, 0) / total for split in ("train", "val", "test")}
        assigned = {split: 0 for split in ("train", "val", "test")}
        for row in sorted(stratum_rows, key=lambda value: _stable_split_hash(value, seed=seed, salt=f"{salt}:row")):
            feasible = [split for split in ("train", "val", "test") if remaining.get(split, 0) > 0]
            if not feasible:
                raise ValueError(f"exact split row assignment exhausted targets for {salt}")
            chosen = max(
                feasible,
                key=lambda split: (
                    desired[split] - assigned[split],
                    remaining.get(split, 0) / max(1, targets.get(split, 1)),
                    split == "train",
                ),
            )
            row["split"] = chosen
            assigned[chosen] += 1
            remaining[chosen] -= 1
    if any(value != 0 for value in remaining.values()):
        raise ValueError(f"exact row split assignment failed for {salt}: {remaining}")


def _exact_split_units(rows: list[dict[str, Any]], *, seed: int, salt: str) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[tuple[str, str, str, str, str], str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(_short_split_stratum(row), _normalize_for_split(row.get("target", "")))].append(row)
    return sorted(
        grouped.values(),
        key=lambda unit: (
            _short_split_stratum(unit[0]),
            _stable_split_hash(unit[0], seed=seed, salt=salt),
            len(unit),
        ),
    )


def _short_split_stratum(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    source_type = str(row.get("source_type", "unknown"))
    error_type = str(row.get("error_type", "unknown"))
    rule_id = str(row.get("rule_id", "unknown"))
    domain = "punctuation" if error_type in {"punctuation", "final_punctuation"} else "orthography"
    risk = "model_required" if _row_requires_scoring(row) else "deterministic"
    return (source_type, error_type, rule_id, domain, risk)


def _stable_split_hash(row: dict[str, Any], *, seed: int, salt: str) -> int:
    import hashlib

    key = f"{seed}:{salt}:{row.get('target', '')}:{row.get('source', '')}:{row.get('rule_id', '')}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)


def _row_requires_scoring(row: dict[str, Any]) -> bool:
    for operation in _parse_jsonish_list(row.get("edit_operations", [])):
        if isinstance(operation, dict) and _as_bool(operation.get("requires_scoring", False)):
            return True
    return False


def _assert_exact_splits(rows: list[dict[str, Any]], split_sizes: dict[str, int]) -> None:
    counts = _split_counts(rows)
    expected = {split: split_sizes[split] for split in ("train", "val", "test")}
    if counts != expected:
        raise ValueError(f"exact split sizes failed: expected {expected}, got {counts}")


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
