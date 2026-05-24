from __future__ import annotations

from pathlib import Path

import yaml

from src.rules.capabilities import (
    active_rule_ids_for_training,
    capability_for_taxonomy_entry,
    capability_matrix_frame,
    expanded_activation_source_map,
    expanded_training_candidate_rule_ids,
    iter_taxonomy_rules,
    load_rule_capabilities,
)


def _entry(
    *,
    key: str = "fixture_rule",
    domain: str = "orthography",
    entry_type: str = "leaf_rule",
    status: str = "model_required",
    rule_ids: list[str] | None = None,
    requires: list[str] | None = None,
    executable: bool = True,
    existing_decision: str = "",
) -> dict[str, object]:
    return {
        "taxonomy_key": key,
        "domain": domain,
        "source_section": domain,
        "orfogrammka_id": "fixture.1",
        "title": "Fixture rule",
        "entry_type": entry_type,
        "implementation": {
            "status": status,
            "executable": executable,
            "rule_ids": rule_ids or [],
            "requires": requires or [],
        },
        "dataset": {
            "training_eligibility_decision": existing_decision,
            "training_eligibility_reason": "fixture existing decision" if existing_decision else "",
            "risk_level": "low",
        },
    }


def _modules(**overrides: object) -> dict[str, object]:
    modules: dict[str, object] = {
        "candidate_rule_ids": set(),
        "synthetic_rule_ids": set(),
        "hard_negative_rule_ids": set(),
        "validator_rule_ids": set(),
        "dictionary_supported": True,
        "syntax_supported": True,
        "morphology_supported": True,
        "ner_supported": False,
    }
    modules.update(overrides)
    return modules


def test_metadata_group_rule_gets_block_metadata_only():
    capability = capability_for_taxonomy_entry(
        _entry(entry_type="group", status="metadata_only", executable=False),
        available_modules=_modules(),
    )

    assert capability.training_decision == "BLOCK_METADATA_ONLY"
    assert capability.training_eligible is False


def test_dictionary_required_rule_without_dictionary_support_is_blocked():
    capability = capability_for_taxonomy_entry(
        _entry(rule_ids=["dictionary_rule"], requires=["dictionary"]),
        available_modules=_modules(
            candidate_rule_ids={"dictionary_rule"},
            synthetic_rule_ids={"dictionary_rule"},
            hard_negative_rule_ids={"dictionary_rule"},
            validator_rule_ids={"dictionary_rule"},
            dictionary_supported=False,
        ),
    )

    assert capability.training_decision == "BLOCK_NEEDS_DICTIONARY"
    assert capability.training_eligible is False


def test_syntax_required_rule_without_syntax_support_is_blocked():
    capability = capability_for_taxonomy_entry(
        _entry(rule_ids=["syntax_rule"], requires=["syntax"]),
        available_modules=_modules(
            candidate_rule_ids={"syntax_rule"},
            synthetic_rule_ids={"syntax_rule"},
            hard_negative_rule_ids={"syntax_rule"},
            validator_rule_ids={"syntax_rule"},
            syntax_supported=False,
        ),
    )

    assert capability.training_decision == "BLOCK_NEEDS_SYNTAX"
    assert capability.training_eligible is False


def test_executable_rule_with_complete_runtime_path_is_included_now():
    capability = capability_for_taxonomy_entry(
        _entry(rule_ids=["safe_rule"], requires=["model"]),
        available_modules=_modules(
            candidate_rule_ids={"safe_rule"},
            synthetic_rule_ids={"safe_rule"},
            hard_negative_rule_ids={"safe_rule"},
            validator_rule_ids={"safe_rule"},
        ),
    )

    assert capability.training_decision == "INCLUDE_NOW"
    assert capability.training_eligible is True


def test_disabled_rule_gets_block_disabled():
    capability = capability_for_taxonomy_entry(
        _entry(status="disabled", executable=False, rule_ids=["disabled_rule"]),
        available_modules=_modules(
            candidate_rule_ids={"disabled_rule"},
            synthetic_rule_ids={"disabled_rule"},
            hard_negative_rule_ids={"disabled_rule"},
            validator_rule_ids={"disabled_rule"},
        ),
    )

    assert capability.training_decision == "BLOCK_DISABLED"
    assert capability.training_eligible is False


def test_active_rule_ids_for_training_excludes_blocked_rules():
    active = capability_for_taxonomy_entry(
        _entry(key="active", rule_ids=["safe_rule"]),
        available_modules=_modules(
            candidate_rule_ids={"safe_rule"},
            synthetic_rule_ids={"safe_rule"},
            hard_negative_rule_ids={"safe_rule"},
            validator_rule_ids={"safe_rule"},
        ),
    )
    blocked = capability_for_taxonomy_entry(
        _entry(key="blocked", rule_ids=["blocked_rule"], requires=["dictionary"]),
        available_modules=_modules(dictionary_supported=False),
    )

    assert active_rule_ids_for_training([active, blocked]) == ["safe_rule"]


def test_capability_matrix_frame_contains_all_taxonomy_entries(tmp_path: Path):
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 3,
                "orthography": {
                    "metadata_group": _entry(key="metadata_group", entry_type="group", status="metadata_only", executable=False),
                    "safe_rule": _entry(key="safe_rule", rule_ids=["safe_rule"]),
                },
                "punctuation": {},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    entries = iter_taxonomy_rules(rules_path)
    capabilities = [
        capability_for_taxonomy_entry(
            entry,
            available_modules=_modules(
                candidate_rule_ids={"safe_rule"},
                synthetic_rule_ids={"safe_rule"},
                hard_negative_rule_ids={"safe_rule"},
                validator_rule_ids={"safe_rule"},
            ),
        )
        for entry in entries
    ]

    frame = capability_matrix_frame(capabilities)

    assert set(frame["taxonomy_key"]) == {"metadata_group", "safe_rule"}
    assert len(frame) == len(entries) == 2


def test_expanded_source_map_uses_live_runtime_sources_not_reports():
    capabilities = load_rule_capabilities("configs/rules.yaml")
    source_map = expanded_activation_source_map(capabilities)

    assert "configs_rules_yaml" in source_map["pattern_жы_жи"]
    assert "rule_registry" in source_map["pattern_жы_жи"]
    assert "corruption_operator_registry" in source_map["pattern_жы_жи"]
    assert "syntax_module" in source_map["comma_subordinate"]
    assert "reports/pre_dataset_capability" not in set().union(*source_map.values())


def test_expanded_training_candidates_keep_quote_open_close_blocked():
    quote_open = capability_for_taxonomy_entry(
        _entry(rule_ids=["quote_open"], existing_decision="INCLUDE_AFTER_THRESHOLD_CALIBRATION"),
        available_modules=_modules(
            candidate_rule_ids={"quote_open"},
            synthetic_rule_ids={"quote_open"},
            hard_negative_rule_ids={"quote_open"},
            validator_rule_ids={"quote_open"},
        ),
    )
    quote_pair = capability_for_taxonomy_entry(
        _entry(key="quote_pair", rule_ids=["quote_pair_balance"], existing_decision="INCLUDE_AFTER_THRESHOLD_CALIBRATION"),
        available_modules=_modules(
            candidate_rule_ids={"quote_pair_balance"},
            synthetic_rule_ids={"quote_pair_balance"},
            hard_negative_rule_ids={"quote_pair_balance"},
            validator_rule_ids={"quote_pair_balance"},
        ),
    )

    assert expanded_training_candidate_rule_ids([quote_open, quote_pair]) == ["quote_pair_balance"]
