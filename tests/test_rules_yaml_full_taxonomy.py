from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.rules.registry import rule_by_id


RULES_PATH = Path("configs/rules.yaml")
SNAPSHOT_PATH = Path("reports/rules_taxonomy/orfogrammka_extraction_snapshot.json")
REPORT_PATH = Path("reports/rules_taxonomy/rules_taxonomy_update_report.md")

ALLOWED_STATUSES = {
    "implemented",
    "partial",
    "candidate_only",
    "model_required",
    "syntax_required",
    "dictionary_model_required",
    "ner_required",
    "planned",
    "metadata_only",
    "disabled",
}
ALLOWED_ENTRY_TYPES = {"group", "numbered_rule", "leaf_rule"}
REQUIRED_ENTRY_FIELDS = {
    "source_section",
    "orfogrammka_id",
    "title",
    "normalized_title",
    "entry_type",
    "parent_key",
    "parent_path",
    "depth",
    "order",
    "source_url",
    "implementation",
    "dataset",
}
REQUIRED_IMPLEMENTATION_FIELDS = {"status", "executable", "rule_ids", "aliases", "requires", "notes"}
REQUIRED_DATASET_FIELDS = {
    "eligible_now",
    "reason",
    "last_known_candidate_recall",
    "last_known_eval_count",
}
DICTIONARY_CANDIDATE_IDS = {
    "dictionary_fuzzy",
    "double_consonant_candidate",
    "keyboard_typo_candidate",
    "swapped_letters_candidate",
    "missing_letter_candidate",
    "extra_letter_candidate",
    "yo_e_candidate",
}


def _load_rules() -> dict:
    return yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))


def _entries(config: dict):
    for section in ("orthography", "punctuation"):
        for key, value in config[section].items():
            yield section, key, value


def _candidate_path_exists(rule_id: str) -> bool:
    if rule_id in DICTIONARY_CANDIDATE_IDS:
        return True
    rule = rule_by_id(rule_id)
    if rule is None:
        return False
    spec = getattr(rule, "spec", None)
    scope = getattr(spec, "scope", "")
    return (
        scope == "punctuation_gap"
        or hasattr(rule, "generate_candidates")
        or hasattr(rule, "generate")
        or hasattr(rule, "generate_span")
    )


def test_rules_yaml_exists_and_uses_v3_schema():
    assert RULES_PATH.exists()
    config = _load_rules()

    assert isinstance(config, dict)
    assert config["schema_version"] >= 3
    assert config["source"]["taxonomy_provider"] == "orfogrammka"
    assert isinstance(config["orthography"], dict)
    assert isinstance(config["punctuation"], dict)


def test_rules_yaml_contains_full_extracted_taxonomy_counts():
    config = _load_rules()

    assert len(config["orthography"]) > 100
    assert len(config["punctuation"]) > 40


def test_every_taxonomy_entry_has_required_v3_fields():
    config = _load_rules()

    for section, key, entry in _entries(config):
        assert REQUIRED_ENTRY_FIELDS <= set(entry), f"{section}.{key} is missing v3 fields"
        assert entry["source_section"] == section
        assert entry["entry_type"] in ALLOWED_ENTRY_TYPES
        assert isinstance(entry["parent_path"], list)
        assert isinstance(entry["depth"], int)
        assert isinstance(entry["order"], int)

        implementation = entry["implementation"]
        dataset = entry["dataset"]
        assert REQUIRED_IMPLEMENTATION_FIELDS <= set(implementation), f"{section}.{key} implementation is incomplete"
        assert implementation["status"] in ALLOWED_STATUSES
        assert isinstance(implementation["executable"], bool)
        assert isinstance(implementation["rule_ids"], list)
        assert isinstance(implementation["aliases"], list)
        assert isinstance(implementation["requires"], list)
        assert REQUIRED_DATASET_FIELDS <= set(dataset), f"{section}.{key} dataset metadata is incomplete"
        assert isinstance(dataset["eligible_now"], bool)
        assert isinstance(dataset["reason"], str) and dataset["reason"]


def test_no_duplicate_numbered_orfogrammka_ids_per_section():
    config = _load_rules()
    seen: set[tuple[str, str]] = set()

    for section, key, entry in _entries(config):
        orfogrammka_id = entry.get("orfogrammka_id")
        if not orfogrammka_id:
            continue
        identifier = (section, str(orfogrammka_id))
        assert identifier not in seen, f"duplicate numbered id {identifier} at {key}"
        seen.add(identifier)


def test_all_numbered_ids_from_extraction_snapshot_exist_in_rules_yaml():
    config = _load_rules()
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))

    for section in ("orthography", "punctuation"):
        ids_in_yaml = {
            str(entry["orfogrammka_id"])
            for entry in config[section].values()
            if entry.get("orfogrammka_id")
        }
        ids_in_snapshot = {
            str(entry["orfogrammka_id"])
            for entry in snapshot[section]["entries"]
            if entry.get("orfogrammka_id")
        }
        assert ids_in_snapshot <= ids_in_yaml


def test_executable_implemented_rule_ids_exist_in_registry():
    config = _load_rules()

    for section, key, entry in _entries(config):
        implementation = entry["implementation"]
        if implementation["status"] != "implemented":
            continue
        assert implementation["executable"], f"{section}.{key} implemented entries must be executable"
        assert implementation["rule_ids"], f"{section}.{key} implemented entries must list rule_ids"
        for rule_id in implementation["rule_ids"]:
            assert rule_by_id(rule_id) is not None, f"{section}.{key} references missing rule_id {rule_id}"


def test_planned_and_metadata_entries_do_not_claim_executability():
    config = _load_rules()

    for section, key, entry in _entries(config):
        implementation = entry["implementation"]
        if implementation["status"] in {"planned", "metadata_only", "disabled"}:
            assert implementation["executable"] is False, f"{section}.{key} must not be executable"


def test_dataset_eligible_entries_have_candidate_path():
    config = _load_rules()

    for section, key, entry in _entries(config):
        if not entry["dataset"]["eligible_now"]:
            continue
        rule_ids = entry["implementation"]["rule_ids"]
        assert rule_ids, f"{section}.{key} is dataset-eligible without rule_ids"
        assert all(_candidate_path_exists(rule_id) for rule_id in rule_ids), f"{section}.{key} lacks candidate path"


def test_rules_taxonomy_update_report_exists():
    assert REPORT_PATH.exists()
    assert "RULES_TAXONOMY" in REPORT_PATH.read_text(encoding="utf-8")
