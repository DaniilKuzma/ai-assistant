from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES_PATH = PROJECT_ROOT / "configs" / "rules.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "syntax_module"

INVENTORY_COLUMNS = [
    "matrix_key",
    "source_section",
    "orfogrammka_id",
    "title",
    "parent_path",
    "entry_type",
    "current_rule_ids",
    "current_status",
    "current_candidate_path_exists",
    "current_synthetic_support",
    "current_validator_support",
    "current_training_eligible_now",
    "current_production_ready_now",
    "syntax_family",
    "implementation_group",
    "implementation_difficulty",
    "can_be_candidate_backed_now",
    "needs_new_syntax_feature",
    "needs_model_scoring",
    "needs_validator_guard",
    "needs_decoder_constraint",
    "recommended_action",
    "reason",
]

FAMILY_SUMMARY_COLUMNS = [
    "syntax_family",
    "total_entries",
    "leaf_entries",
    "current_candidate_backed",
    "implement_now_count",
    "bounded_pattern_count",
    "blocked_count",
    "expected_new_candidate_rule_ids",
    "expected_training_eligible_gain",
    "priority",
]

SYNTAX_FAMILIES = (
    "subordinate_clause_comma",
    "coordinating_conjunction_comma",
    "introductory_words",
    "address_comma",
    "homogeneous_members",
    "detached_adverbial_phrases",
    "detached_participial_phrases",
    "detached_applications",
    "clarification_members",
    "comparative_turnovers",
    "subject_predicate_dash",
    "asyndetic_complex_sentence",
    "consequence_dash",
    "explanation_colon",
    "enumeration_colon_dash",
    "semicolon",
    "direct_speech_syntax",
    "quote_bracket_balance",
    "punctuation_combinations",
    "ne_with_parts_of_speech",
    "ni_stable_and_context",
    "n_nn_context",
    "tsya_tsya_context",
    "context_pairs",
    "grammatical_endings_context",
)

IMPLEMENT_NOW = "IMPLEMENT_NOW"
IMPLEMENT_WITH_BOUNDED_PATTERN = "IMPLEMENT_WITH_BOUNDED_PATTERN"
NEEDS_SEMANTIC_MODEL = "NEEDS_SEMANTIC_MODEL"
NEEDS_DICTIONARY = "NEEDS_DICTIONARY"
NEEDS_NER = "NEEDS_NER"
KEEP_METADATA_ONLY = "KEEP_METADATA_ONLY"
BLOCK_FOR_NOW = "BLOCK_FOR_NOW"

IMPLEMENTABLE_ACTIONS = {IMPLEMENT_NOW, IMPLEMENT_WITH_BOUNDED_PATTERN}

SUBORDINATE_CLAUSE_COMMA_KEYS = {
    "punctuation_7_2_2",
    "punctuation_7_2_3",
    "punctuation_7_2_4_1",
    "punctuation_7_2_4_2",
}
COORDINATING_CONJUNCTION_COMMA_KEYS = {"punctuation_4_3"}
INTRODUCTORY_WORDS_KEYS = {
    "punctuation_6_1_2",
    "punctuation_6_1_3",
    "punctuation_6_1_5",
    "punctuation_6_1_6",
    "punctuation_6_1_7",
    "punctuation_6_3",
    "punctuation_6_4",
    "punctuation_6_5",
}
HOMOGENEOUS_MEMBERS_KEYS = {
    "punctuation_4_1",
    "punctuation_4_4",
    "punctuation_4_5",
    "punctuation_4_10",
    "punctuation_4_11",
    "punctuation_4_14",
}
DETACHED_ADVERBIAL_KEYS = {
    "punctuation_5_4_2",
    "punctuation_5_4_3",
    "punctuation_5_4_5",
    "punctuation_5_4_6",
    "punctuation_5_4_7",
    "punctuation_5_4_8",
    "punctuation_5_4_9",
    "punctuation_5_4_10",
}
DETACHED_PARTICIPIAL_KEYS = {
    "punctuation_5_1_1_1",
    "punctuation_5_1_1_2",
    "punctuation_5_1_2",
    "punctuation_5_1_3",
}
DETACHED_APPLICATION_KEYS = {"punctuation_5_3"}
CLARIFICATION_KEYS = {"punctuation_5_5", "punctuation_5_6_2", "punctuation_5_6_3"}
COMPARATIVE_KEYS = {
    "punctuation_5_8_3",
    "punctuation_5_8_5",
    "punctuation_5_8_6",
    "punctuation_5_8_7",
    "punctuation_5_8_10",
    "punctuation_5_8_11",
    "punctuation_5_8_12",
}
SUBJECT_PREDICATE_DASH_KEYS = {
    "punctuation_1_4",
    "punctuation_2_1_1",
    "punctuation_2_1_3",
    "punctuation_2_1_4",
    "punctuation_2_1_5",
    "punctuation_2_1_6",
    "punctuation_2_1_7",
    "punctuation_2_1_8",
    "punctuation_2_1_9",
    "punctuation_2_1_10",
    "punctuation_2_1_11",
    "punctuation_2_2",
}
ASYNDETIC_COMPLEX_KEYS = {"punctuation_7_3_1"}
ENUMERATION_COLON_DASH_KEYS = {"punctuation_4_8"}
DIRECT_SPEECH_KEYS = {"punctuation_8_1_3", "punctuation_8_1_4", "punctuation_8_1_6", "punctuation_8_2"}

STABLE_PUNCTUATION_DICTIONARY_KEYS = {
    "punctuation_5_7_1",
    "punctuation_5_7_2",
    "punctuation_5_7_3",
    "punctuation_5_7_4",
    "punctuation_5_7_5",
    "punctuation_5_7_6",
    "punctuation_5_7_7",
    "punctuation_5_7_8",
    "punctuation_5_7_9",
    "punctuation_5_7_10",
}
PUNCTUATION_COMBINATION_KEYS = {
    "punctuation_1_3_5",
    "punctuation_1_5",
    "punctuation_9_1",
    "punctuation_9_2",
    "punctuation_9_3",
    "punctuation_9_4",
    "punctuation_9_7",
    "punctuation_9_8",
} | STABLE_PUNCTUATION_DICTIONARY_KEYS

GRAMMATICAL_ENDINGS_CONTEXT_KEYS = {
    "orthography_1_2_7_1",
    "orthography_1_2_7_2",
    "orthography_1_2_7_3",
    "orthography_1_2_7_4",
    "orthography_1_2_7_5",
    "orthography_1_2_7_6",
    "orthography_4_11_2",
}
NE_WITH_PARTS_OF_SPEECH_KEYS = {
    "orthography_3_5_4",
    "orthography_3_7_1_3",
    "orthography_3_7_1_4_2",
    "orthography_3_7_1_4_3",
    "orthography_3_7_1_4_4",
    "orthography_3_7_1_4_5",
    "orthography_3_7_1_5",
    "orthography_3_7_2_2",
    "orthography_3_7_2_3",
    "orthography_3_7_2_4",
    "orthography_3_7_2_5",
    "orthography_3_7_2_6",
    "orthography_3_7_2_8",
    "orthography_3_7_2_9",
    "orthography_3_7_2_10",
}
CONTEXT_PAIR_KEYS = {
    "orthography_3_2_3_1",
    "orthography_3_4_1",
    "orthography_3_4_3_2",
    "orthography_3_5_1_1",
    "orthography_3_5_1_2",
    "orthography_3_5_1_3",
    "orthography_3_5_1_4",
    "orthography_3_5_1_5",
    "orthography_3_5_1_6",
    "orthography_3_5_1_7",
    "orthography_3_5_2_1",
    "orthography_3_5_2_2",
    "orthography_3_5_2_3",
    "orthography_3_5_2_4",
    "orthography_3_5_2_5",
    "orthography_3_5_3_1",
    "orthography_3_6_1_2",
    "orthography_3_6_1_4",
    "orthography_3_6_2_3",
    "orthography_3_6_2_4",
    "orthography_3_6_4_1",
    "orthography_3_6_4_2",
}

NEEDS_DICTIONARY_KEYS = {
    "orthography_1_2_7_1",
    "orthography_1_2_7_2",
    "orthography_1_2_7_3",
    "orthography_1_2_7_4",
    "orthography_1_2_7_5",
    "orthography_1_2_7_6",
    "orthography_3_2_3_1",
    "orthography_3_5_1_1",
    "orthography_3_5_1_2",
    "orthography_3_5_1_3",
    "orthography_3_5_1_4",
    "orthography_3_5_1_5",
    "orthography_3_5_1_6",
    "orthography_3_5_1_7",
    "orthography_3_5_2_1",
    "orthography_3_5_2_2",
    "orthography_3_5_2_3",
    "orthography_3_5_2_4",
    "orthography_3_5_2_5",
    "orthography_3_5_3_1",
    "orthography_3_6_1_4",
    "orthography_3_6_2_3",
    "orthography_3_6_2_4",
    "orthography_3_7_1_3",
    "orthography_3_7_2_2",
    "orthography_3_7_2_3",
    "orthography_3_7_2_6",
} | STABLE_PUNCTUATION_DICTIONARY_KEYS
NEEDS_SEMANTIC_MODEL_KEYS = {
    "orthography_3_7_1_4_2",
    "orthography_3_7_1_4_3",
    "orthography_3_7_2_4",
    "orthography_3_7_2_5",
    "punctuation_5_8_7",
    "punctuation_5_8_10",
    "punctuation_7_3_1",
}
BLOCK_FOR_NOW_KEYS = {"orthography_4_11_2"}
KEEP_METADATA_ONLY_KEYS = {"punctuation_9_7"}
IMPLEMENT_NOW_KEYS = COORDINATING_CONJUNCTION_COMMA_KEYS

IMPLEMENTATION_GROUP_BY_FAMILY = {
    "subordinate_clause_comma": "syntax_clause_commas",
    "coordinating_conjunction_comma": "syntax_conjunction_commas",
    "introductory_words": "syntax_parenthetical_commas",
    "address_comma": "syntax_address_commas",
    "homogeneous_members": "syntax_homogeneous_members",
    "detached_adverbial_phrases": "syntax_detached_adverbials",
    "detached_participial_phrases": "syntax_detached_participials",
    "detached_applications": "syntax_detached_applications",
    "clarification_members": "syntax_clarification_members",
    "comparative_turnovers": "syntax_comparative_turnovers",
    "subject_predicate_dash": "syntax_predicative_dash",
    "asyndetic_complex_sentence": "syntax_asyndetic_complex",
    "consequence_dash": "syntax_consequence_dash",
    "explanation_colon": "syntax_explanation_colon",
    "enumeration_colon_dash": "syntax_enumeration_colon_dash",
    "semicolon": "syntax_semicolon",
    "direct_speech_syntax": "syntax_direct_speech",
    "quote_bracket_balance": "syntax_quote_bracket_balance",
    "punctuation_combinations": "syntax_punctuation_combinations",
    "ne_with_parts_of_speech": "syntax_ne_split_join",
    "ni_stable_and_context": "syntax_ni_context",
    "n_nn_context": "syntax_n_nn_context",
    "tsya_tsya_context": "syntax_tsya_context",
    "context_pairs": "syntax_context_pairs",
    "grammatical_endings_context": "syntax_grammatical_endings",
}

EXPECTED_RULE_IDS_BY_FAMILY = {
    "subordinate_clause_comma": ("comma_subordinate",),
    "coordinating_conjunction_comma": ("comma_conjunction",),
    "introductory_words": ("introductory_comma",),
    "address_comma": ("address_comma",),
    "homogeneous_members": ("homogeneous_comma", "enumeration_colon"),
    "detached_adverbial_phrases": ("detached_adverbial_comma",),
    "detached_participial_phrases": ("detached_participial_comma",),
    "detached_applications": ("detached_application_comma",),
    "clarification_members": ("clarification_comma",),
    "comparative_turnovers": ("comparative_turnover_comma",),
    "subject_predicate_dash": ("subject_predicate_dash",),
    "asyndetic_complex_sentence": ("asyndetic_dash", "semicolon"),
    "consequence_dash": ("consequence_dash",),
    "explanation_colon": ("explanation_colon",),
    "enumeration_colon_dash": ("enumeration_colon",),
    "semicolon": ("semicolon",),
    "direct_speech_syntax": ("direct_speech_colon", "direct_speech_dash", "direct_speech_quotes"),
    "quote_bracket_balance": ("quote_pair_balance", "bracket_pair_balance"),
    "punctuation_combinations": ("punctuation_delete_replace",),
    "ne_with_parts_of_speech": ("ne_adjective", "ne_adverb", "ne_participle", "ne_verb"),
    "ni_stable_and_context": ("ni_stable_expression",),
    "n_nn_context": ("n_nn_adjective", "n_nn_participle", "n_nn_deverbal_adjective", "n_nn_short_form"),
    "tsya_tsya_context": ("tsya_soft_delete", "tsya_soft_insert"),
    "context_pairs": (
        "context_chto_by",
        "context_nesmotrya",
        "context_tak_zhe",
        "context_to_zhe",
        "context_vsledstvie",
        "context_za_to",
    ),
    "grammatical_endings_context": (),
}

TOP_PRIORITY_FAMILIES = (
    "subject_predicate_dash",
    "detached_adverbial_phrases",
    "introductory_words",
    "homogeneous_members",
    "subordinate_clause_comma",
)


def build_syntax_required_inventory(rules_config_path: str | Path = DEFAULT_RULES_PATH) -> dict[str, Any]:
    config = _load_rules(rules_config_path)
    rows = [
        _inventory_row(section, matrix_key, entry)
        for section in ("orthography", "punctuation")
        for matrix_key, entry in config.get(section, {}).items()
        if (entry.get("implementation") or {}).get("status") == "syntax_required"
    ]
    summary_rows = _family_summary_rows(rows)
    summary = _summary(rows, summary_rows)
    return {"inventory_rows": rows, "family_summary_rows": summary_rows, "summary": summary}


def write_syntax_required_inventory_outputs(
    *,
    rules_config_path: str | Path = DEFAULT_RULES_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, str]:
    result = build_syntax_required_inventory(rules_config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    paths = {
        "syntax_required_inventory_csv": output / "syntax_required_inventory.csv",
        "syntax_required_inventory_md": output / "syntax_required_inventory.md",
        "syntax_required_family_summary_csv": output / "syntax_required_family_summary.csv",
        "syntax_module_implementation_plan": output / "syntax_module_implementation_plan.md",
    }

    _write_csv(paths["syntax_required_inventory_csv"], result["inventory_rows"], INVENTORY_COLUMNS)
    _write_csv(paths["syntax_required_family_summary_csv"], result["family_summary_rows"], FAMILY_SUMMARY_COLUMNS)
    paths["syntax_required_inventory_md"].write_text(
        _inventory_markdown(result["inventory_rows"], result["family_summary_rows"], result["summary"]),
        encoding="utf-8",
    )
    paths["syntax_module_implementation_plan"].write_text(
        _implementation_plan_markdown(result["family_summary_rows"], result["summary"]),
        encoding="utf-8",
    )
    return {key: str(path) for key, path in paths.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write syntax_required inventory reports without data or training side effects.")
    parser.add_argument("--rules", default=str(DEFAULT_RULES_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)

    outputs = write_syntax_required_inventory_outputs(
        rules_config_path=Path(args.rules),
        output_dir=Path(args.output_dir),
    )
    result = build_syntax_required_inventory(Path(args.rules))
    print(_summary_text(result["summary"], outputs))
    return 0


def _load_rules(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("rules.yaml must be a mapping")
    return data


def _inventory_row(section: str, matrix_key: str, entry: dict[str, Any]) -> dict[str, Any]:
    implementation = entry.get("implementation") or {}
    dataset = entry.get("dataset") or {}
    family = _syntax_family(section, matrix_key)
    action = _recommended_action(matrix_key)
    return {
        "matrix_key": matrix_key,
        "source_section": section,
        "orfogrammka_id": str(entry.get("orfogrammka_id") or ""),
        "title": str(entry.get("title") or ""),
        "parent_path": " > ".join(str(item) for item in entry.get("parent_path", []) or []),
        "entry_type": str(entry.get("entry_type") or ""),
        "current_rule_ids": json.dumps(implementation.get("rule_ids", []) or [], ensure_ascii=False),
        "current_status": str(implementation.get("status") or ""),
        "current_candidate_path_exists": _bool(dataset.get("current_candidate_path")),
        "current_synthetic_support": _bool(dataset.get("current_synthetic_support")),
        "current_validator_support": _bool(dataset.get("current_validator_support")),
        "current_training_eligible_now": _bool(dataset.get("training_eligible_now")),
        "current_production_ready_now": _bool(dataset.get("production_ready_now")),
        "syntax_family": family,
        "implementation_group": IMPLEMENTATION_GROUP_BY_FAMILY[family],
        "implementation_difficulty": _implementation_difficulty(matrix_key, family, action),
        "can_be_candidate_backed_now": _bool(action in IMPLEMENTABLE_ACTIONS),
        "needs_new_syntax_feature": _bool(action in IMPLEMENTABLE_ACTIONS or action == NEEDS_SEMANTIC_MODEL),
        "needs_model_scoring": _bool(action not in {KEEP_METADATA_ONLY, BLOCK_FOR_NOW}),
        "needs_validator_guard": _bool(action in IMPLEMENTABLE_ACTIONS),
        "needs_decoder_constraint": _bool(_needs_decoder_constraint(section, family, action)),
        "recommended_action": action,
        "reason": _reason(matrix_key, family, action),
    }


def _syntax_family(section: str, matrix_key: str) -> str:
    if section == "punctuation":
        if matrix_key in SUBORDINATE_CLAUSE_COMMA_KEYS:
            return "subordinate_clause_comma"
        if matrix_key in COORDINATING_CONJUNCTION_COMMA_KEYS:
            return "coordinating_conjunction_comma"
        if matrix_key in INTRODUCTORY_WORDS_KEYS:
            return "introductory_words"
        if matrix_key in HOMOGENEOUS_MEMBERS_KEYS:
            return "homogeneous_members"
        if matrix_key in DETACHED_ADVERBIAL_KEYS:
            return "detached_adverbial_phrases"
        if matrix_key in DETACHED_PARTICIPIAL_KEYS:
            return "detached_participial_phrases"
        if matrix_key in DETACHED_APPLICATION_KEYS:
            return "detached_applications"
        if matrix_key in CLARIFICATION_KEYS:
            return "clarification_members"
        if matrix_key in COMPARATIVE_KEYS:
            return "comparative_turnovers"
        if matrix_key in SUBJECT_PREDICATE_DASH_KEYS:
            return "subject_predicate_dash"
        if matrix_key in ASYNDETIC_COMPLEX_KEYS:
            return "asyndetic_complex_sentence"
        if matrix_key in ENUMERATION_COLON_DASH_KEYS:
            return "enumeration_colon_dash"
        if matrix_key in DIRECT_SPEECH_KEYS:
            return "direct_speech_syntax"
        if matrix_key in PUNCTUATION_COMBINATION_KEYS:
            return "punctuation_combinations"
        raise ValueError(f"Unclassified syntax_required punctuation entry: {matrix_key}")

    if matrix_key in GRAMMATICAL_ENDINGS_CONTEXT_KEYS:
        return "grammatical_endings_context"
    if matrix_key in NE_WITH_PARTS_OF_SPEECH_KEYS:
        return "ne_with_parts_of_speech"
    if matrix_key in CONTEXT_PAIR_KEYS:
        return "context_pairs"
    raise ValueError(f"Unclassified syntax_required orthography entry: {matrix_key}")


def _recommended_action(matrix_key: str) -> str:
    if matrix_key in IMPLEMENT_NOW_KEYS:
        return IMPLEMENT_NOW
    if matrix_key in NEEDS_DICTIONARY_KEYS:
        return NEEDS_DICTIONARY
    if matrix_key in NEEDS_SEMANTIC_MODEL_KEYS:
        return NEEDS_SEMANTIC_MODEL
    if matrix_key in KEEP_METADATA_ONLY_KEYS:
        return KEEP_METADATA_ONLY
    if matrix_key in BLOCK_FOR_NOW_KEYS:
        return BLOCK_FOR_NOW
    return IMPLEMENT_WITH_BOUNDED_PATTERN


def _implementation_difficulty(matrix_key: str, family: str, action: str) -> str:
    if action == IMPLEMENT_NOW:
        return "low"
    if action in {NEEDS_DICTIONARY, NEEDS_SEMANTIC_MODEL, BLOCK_FOR_NOW}:
        return "high"
    if family in {
        "subject_predicate_dash",
        "detached_participial_phrases",
        "direct_speech_syntax",
        "punctuation_combinations",
        "ne_with_parts_of_speech",
        "context_pairs",
    }:
        return "high" if matrix_key in {"punctuation_2_1_5", "punctuation_8_1_6"} else "medium"
    return "medium"


def _needs_decoder_constraint(section: str, family: str, action: str) -> bool:
    if section != "punctuation" or action not in IMPLEMENTABLE_ACTIONS:
        return False
    return family in {
        "detached_adverbial_phrases",
        "detached_participial_phrases",
        "detached_applications",
        "clarification_members",
        "direct_speech_syntax",
        "homogeneous_members",
        "punctuation_combinations",
        "subject_predicate_dash",
    }


def _reason(matrix_key: str, family: str, action: str) -> str:
    if action == IMPLEMENT_NOW:
        return "Existing adversative-comma candidate shape can be extended with bounded syntax guards and model scoring."
    if action == IMPLEMENT_WITH_BOUNDED_PATTERN:
        return (
            f"{family} can be represented as candidate-backed local edits with explicit syntax features, "
            "model scoring, validator guards, and no broad rewrite."
        )
    if action == NEEDS_DICTIONARY:
        return "Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first."
    if action == NEEDS_SEMANTIC_MODEL:
        return "Local syntax can propose probes, but production candidates need semantic disambiguation beyond bounded patterns."
    if action == KEEP_METADATA_ONLY:
        return "Footnote formatting is document-layout metadata and should remain outside the spelling/punctuation correction path."
    if action == BLOCK_FOR_NOW:
        return "This capitalization choice is stylistic/formality-sensitive and is not safe as strict spelling or punctuation correction."
    raise ValueError(f"Unhandled action for {matrix_key}: {action}")


def _family_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_family[str(row["syntax_family"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for family in SYNTAX_FAMILIES:
        family_rows = rows_by_family.get(family, [])
        action_counts = Counter(row["recommended_action"] for row in family_rows)
        implementable_count = action_counts[IMPLEMENT_NOW] + action_counts[IMPLEMENT_WITH_BOUNDED_PATTERN]
        blocked_count = len(family_rows) - implementable_count
        expected_rule_ids = EXPECTED_RULE_IDS_BY_FAMILY.get(family, ())
        if implementable_count == 0:
            expected_rule_ids = ()
        summary_rows.append(
            {
                "syntax_family": family,
                "total_entries": len(family_rows),
                "leaf_entries": sum(1 for row in family_rows if row["entry_type"] == "leaf_rule"),
                "current_candidate_backed": sum(1 for row in family_rows if _truthy_string(row["current_candidate_path_exists"])),
                "implement_now_count": action_counts[IMPLEMENT_NOW],
                "bounded_pattern_count": action_counts[IMPLEMENT_WITH_BOUNDED_PATTERN],
                "blocked_count": blocked_count,
                "expected_new_candidate_rule_ids": json.dumps(expected_rule_ids, ensure_ascii=False),
                "expected_training_eligible_gain": implementable_count,
                "priority": _priority(family, implementable_count),
            }
        )
    return summary_rows


def _priority(family: str, implementable_count: int) -> int:
    if implementable_count <= 0:
        return 4
    if family in TOP_PRIORITY_FAMILIES:
        return 1
    if family in {
        "coordinating_conjunction_comma",
        "detached_participial_phrases",
        "direct_speech_syntax",
        "clarification_members",
        "enumeration_colon_dash",
    }:
        return 2
    if family in {"context_pairs", "ne_with_parts_of_speech", "punctuation_combinations", "comparative_turnovers"}:
        return 3
    return 4


def _summary(rows: list[dict[str, Any]], family_rows: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts = Counter(row["recommended_action"] for row in rows)
    expected_gain = sum(int(row["expected_training_eligible_gain"]) for row in family_rows)
    nonzero_families = sum(1 for row in family_rows if int(row["total_entries"]) > 0)
    return {
        "total_syntax_required_entries": len(rows),
        "total_leaf_syntax_required_entries": sum(1 for row in rows if row["entry_type"] == "leaf_rule"),
        "families_total": len(SYNTAX_FAMILIES),
        "families_nonzero": nonzero_families,
        "implement_now_count": action_counts[IMPLEMENT_NOW],
        "bounded_pattern_count": action_counts[IMPLEMENT_WITH_BOUNDED_PATTERN],
        "blocked_count": len(rows) - action_counts[IMPLEMENT_NOW] - action_counts[IMPLEMENT_WITH_BOUNDED_PATTERN],
        "expected_training_eligible_gain": expected_gain,
        "top_priority_families": list(TOP_PRIORITY_FAMILIES),
    }


def _inventory_markdown(
    rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    lines = [
        "# Syntax Required Inventory",
        "",
        "This audit reads `configs/rules.yaml` only. It does not build a dataset, train, change thresholds, or change checkpoints.",
        "",
        "## Summary",
        "",
        f"- total syntax_required entries: {summary['total_syntax_required_entries']}",
        f"- total leaf syntax_required entries: {summary['total_leaf_syntax_required_entries']}",
        f"- syntax families: {summary['families_total']} total, {summary['families_nonzero']} nonzero",
        f"- IMPLEMENT_NOW: {summary['implement_now_count']}",
        f"- IMPLEMENT_WITH_BOUNDED_PATTERN: {summary['bounded_pattern_count']}",
        f"- blocked/deferred: {summary['blocked_count']}",
        f"- expected training_eligible gain: {summary['expected_training_eligible_gain']}",
        "",
        "## Family Summary",
        "",
        "| syntax_family | total | implement_now | bounded | blocked | priority |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in family_rows:
        lines.append(
            f"| {row['syntax_family']} | {row['total_entries']} | {row['implement_now_count']} | "
            f"{row['bounded_pattern_count']} | {row['blocked_count']} | {row['priority']} |"
        )
    lines.extend(
        [
            "",
            "## Inventory",
            "",
            "| matrix_key | title | syntax_family | recommended_action | reason |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        title = str(row["title"]).replace("|", "\\|")
        reason = str(row["reason"]).replace("|", "\\|")
        lines.append(
            f"| {row['matrix_key']} | {title} | {row['syntax_family']} | {row['recommended_action']} | {reason} |"
        )
    return "\n".join(lines) + "\n"


def _implementation_plan_markdown(family_rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    implementable = [row for row in family_rows if int(row["expected_training_eligible_gain"]) > 0]
    blocked = [row for row in family_rows if int(row["blocked_count"]) > 0]
    lines = [
        "# Syntax Module Implementation Plan",
        "",
        "This is an implementation handoff for future syntax-module work. It does not build data, run training, change thresholds, or change model checkpoints.",
        "",
        "## Baseline",
        "",
        f"- total syntax_required entries: {summary['total_syntax_required_entries']}",
        f"- total leaf syntax_required entries: {summary['total_leaf_syntax_required_entries']}",
        f"- families count: {summary['families_total']} total, {summary['families_nonzero']} nonzero",
        f"- implement_now count: {summary['implement_now_count']}",
        f"- bounded_pattern count: {summary['bounded_pattern_count']}",
        f"- blocked count: {summary['blocked_count']}",
        f"- expected training_eligible gain: {summary['expected_training_eligible_gain']}",
        "",
        "## Files To Create",
        "",
        "- `src/nlp/syntax_features.py`: dependency helpers over `SyntaxToken`, clause heads, finite verb detection, subtree spans, agreement, coordination chains, and protected gap checks.",
        "- `src/rules/syntax_punctuation.py`: bounded candidate builders grouped by syntax punctuation family.",
        "- `src/rules/syntax_orthography.py`: bounded syntax-aware split/join candidates for `не` and context-pair families.",
        "",
        "## Candidate And Validator Changes",
        "",
        "- Extend `Candidate` and `PunctuationGapCandidate` with optional `syntax_family`, `implementation_group`, `constraint_group`, and `bundle_id` fields.",
        "- Update `CandidateGenerator` to parse syntax once per text and pass syntax features into punctuation and span candidate builders.",
        "- Add `StrictValidator` guards per syntax family; keep every model-backed edit behind trusted candidates.",
        "- Add decoder constraints: one punctuation action per gap, paired direct-speech/quote/bracket bundles, comma-pair bundles, and no conflicting punctuation operations.",
        "",
        "## Implementation Families",
        "",
    ]
    for row in implementable:
        rule_ids = json.loads(str(row["expected_new_candidate_rule_ids"]))
        lines.append(
            f"- priority {row['priority']}: `{row['syntax_family']}` covers "
            f"{row['expected_training_eligible_gain']} entries; candidate rule ids: {', '.join(rule_ids) if rule_ids else 'none'}."
        )
    lines.extend(
        [
            "",
            "## Blocked Or Deferred",
            "",
        ]
    )
    for row in blocked:
        lines.append(f"- `{row['syntax_family']}`: {row['blocked_count']} blocked/deferred entries.")
    lines.extend(
        [
            "",
            "## Rules YAML Updates",
            "",
            "- Update `configs/rules.yaml` only after real candidate builders and tests exist.",
            "- Change implemented entries from `syntax_required` to `model_required`, set `executable: true`, add canonical `rule_ids`, and update dataset metadata.",
            "- Do not add fake rules, do not change thresholds, and do not change checkpoints.",
            "",
            "## Tests",
            "",
            "- Add syntax feature unit tests for clause heads, finite verbs, subtree spans, agreement, coordination, and protected gaps.",
            "- Add candidate tests for each implemented family with negative overcorrection examples.",
            "- Add validator tests for risky punctuation, paired bundles, context pairs, and `не` split/join.",
            "- Keep `python -m pytest -q tests/test_syntax_required_inventory.py` passing after every audit refresh.",
            "",
            "SYNTAX_REQUIRED_AUDIT_COMPLETE",
            "",
        ]
    )
    return "\n".join(lines)


def _summary_text(summary: dict[str, Any], outputs: dict[str, str]) -> str:
    report_paths = "\n".join(f"- {path}" for path in outputs.values())
    top = ", ".join(summary["top_priority_families"])
    return "\n".join(
        [
            f"total syntax_required entries: {summary['total_syntax_required_entries']}",
            f"total leaf syntax_required entries: {summary['total_leaf_syntax_required_entries']}",
            f"families count: {summary['families_total']} total, {summary['families_nonzero']} nonzero",
            f"implement_now count: {summary['implement_now_count']}",
            f"bounded_pattern count: {summary['bounded_pattern_count']}",
            f"blocked count: {summary['blocked_count']}",
            f"expected training_eligible gain: {summary['expected_training_eligible_gain']}",
            f"top priority families: {top}",
            "reports paths:",
            report_paths,
            "verdict: SYNTAX_REQUIRED_AUDIT_COMPLETE",
        ]
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def _truthy_string(value: Any) -> bool:
    return str(value).strip().lower() == "true"


if __name__ == "__main__":
    raise SystemExit(main())
