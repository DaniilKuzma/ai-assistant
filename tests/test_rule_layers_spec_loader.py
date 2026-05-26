from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.grammar_gen.randomness import RandomSource
from src.rule_layers.spec_loader import load_layer_specs


def test_load_layer_specs_rejects_missing_required_fields(tmp_path: Path) -> None:
    _write_yaml(
        tmp_path,
        "bad",
        "missing.yaml",
        {
            "layer": "bad",
            "family": "dictionary_typo",
            "cases": [{"mode": "positive", "source": "\u041e\u043d \u043f\u0438\u0441\u0430\u043b.", "target": "\u041e\u043d \u043f\u0438\u0441\u0430\u043b."}],
        },
    )

    with pytest.raises(ValueError, match="missing 'rule_id'"):
        load_layer_specs("bad", root=tmp_path, rng=RandomSource(1))


def test_load_layer_specs_supports_grouped_cases_slots_and_templates(tmp_path: Path) -> None:
    _write_yaml(
        tmp_path,
        "compound",
        "rules.yaml",
        {
            "layer": "compound",
            "rule_id": "ne_verb",
            "family": "compound_spelling",
            "description": "Layered span examples.",
            "explanation": "ne_verb",
            "slots": {
                "persons": [
                    {"value": "\u0410\u043d\u043d\u0430", "weight": 0.0},
                    {"value": "\u0411\u043e\u0440\u0438\u0441", "weight": 1.0},
                ],
                "documents": ["\u043e\u0442\u0447\u0451\u0442"],
            },
            "cases": {
                "positive": [
                    {
                        "sub_rule_id": "merge_na_schet",
                        "source_template": "{persons} \u043f\u0440\u043e\u0432\u0435\u0440\u0438\u043b {documents}.",
                        "target_template": "{persons} \u043f\u0440\u043e\u0432\u0435\u0440\u0438\u043b {documents}.",
                        "token_operations": [
                            {
                                "kind": "token_span",
                                "label": "SPAN_REPLACE_BY_LEXICON",
                                "source_pattern": "{documents}",
                                "target_pattern": "{documents}",
                            }
                        ],
                    }
                ],
                "hard_negative": [
                    {
                        "sub_rule_id": "identity",
                        "source": "\u0411\u043e\u0440\u0438\u0441 \u043f\u0440\u043e\u0432\u0435\u0440\u0438\u043b \u043e\u0442\u0447\u0451\u0442.",
                        "target": "\u0411\u043e\u0440\u0438\u0441 \u043f\u0440\u043e\u0432\u0435\u0440\u0438\u043b \u043e\u0442\u0447\u0451\u0442.",
                    }
                ],
            },
        },
    )

    specs = load_layer_specs("compound", root=tmp_path, rng=RandomSource(7))

    assert len(specs) == 1
    assert specs[0].rule_id == "ne_verb"
    assert [case.mode for case in specs[0].cases] == ["positive", "hard_negative"]
    assert specs[0].cases[0].source_text == "\u0411\u043e\u0440\u0438\u0441 \u043f\u0440\u043e\u0432\u0435\u0440\u0438\u043b \u043e\u0442\u0447\u0451\u0442."
    assert specs[0].cases[0].token_operations[0].source_pattern == "\u043e\u0442\u0447\u0451\u0442"


def test_load_layer_specs_supports_flat_cases_and_reproducible_weighted_slots(tmp_path: Path) -> None:
    payload = {
        "layer": "typos",
        "rule_id": "ne_verb",
        "family": "dictionary_typo",
        "slots": {
            "nouns": [
                {"value": "\u043f\u043b\u0430\u043d", "weight": 0.0},
                {"value": "\u0442\u0435\u043a\u0441\u0442", "weight": 1.0},
            ]
        },
        "cases": [
            {
                "mode": "clean_identity",
                "sub_rule_id": "clean",
                "source_template": "\u041e\u043d \u0447\u0438\u0442\u0430\u043b {nouns}.",
                "target_template": "\u041e\u043d \u0447\u0438\u0442\u0430\u043b {nouns}.",
                "weight": 2.0,
            }
        ],
    }
    _write_yaml(tmp_path, "typos", "rules.yaml", payload)

    first = load_layer_specs("typos", root=tmp_path, rng=RandomSource(11))
    second = load_layer_specs("typos", root=tmp_path, rng=RandomSource(11))

    assert first == second
    assert first[0].cases[0].source_text == "\u041e\u043d \u0447\u0438\u0442\u0430\u043b \u0442\u0435\u043a\u0441\u0442."
    assert first[0].cases[0].weight == 2.0


def test_load_layer_specs_does_not_scan_orthography_yaml(tmp_path: Path) -> None:
    orthography_dir = tmp_path / "orthography"
    orthography_dir.mkdir()
    (orthography_dir / "rule.yaml").write_text("rule_id: should_not_load\n", encoding="utf-8")
    (tmp_path / "layers" / "empty").mkdir(parents=True)

    specs = load_layer_specs("empty", root=tmp_path / "layers", rng=RandomSource(1))

    assert specs == ()


def _write_yaml(root: Path, layer: str, name: str, payload: dict[str, object]) -> None:
    path = root / layer
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
