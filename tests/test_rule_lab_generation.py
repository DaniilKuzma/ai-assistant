from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.candidates.candidate_generator import Candidate
from src.data.dataset_contract import LAYER_ATOMIC_HARD_NEGATIVE, LAYER_ATOMIC_POSITIVE
from src.data.rule_lab_generation import generate_rule_lab_rows, load_rule_lab_recipes


class RuleLabCandidateGenerator:
    CONTEXT_PAIRS = {
        "context_chto_by": ("что бы", "чтобы"),
        "context_tak_zhe": ("так же", "также"),
    }

    def generate(self, text: str, **_kwargs):
        candidates: list[Candidate] = []
        lower = text.lower()
        for rule_id, (source, replacement) in self.CONTEXT_PAIRS.items():
            start = lower.find(source)
            if start >= 0:
                candidates.append(
                    Candidate(
                        source=text[start : start + len(source)],
                        replacement=replacement,
                        edit_type="split_join",
                        start=start,
                        end=start + len(source),
                        confidence=0.35,
                        requires_model=True,
                        rule_id=rule_id,
                        mode="model_required",
                        requires=("syntax", "model"),
                        group="context_split_join",
                    )
                )
        start = lower.find("длиный")
        if start >= 0:
            candidates.append(
                Candidate(
                    source=text[start : start + len("длиный")],
                    replacement="длинный",
                    edit_type="spelling",
                    start=start,
                    end=start + len("длиный"),
                    confidence=1.0,
                    requires_model=True,
                    rule_id="n_nn_adjective",
                    mode="model_required",
                    requires=("morphology", "syntax", "dictionary", "model"),
                    group="n_nn",
                )
            )
        start = lower.find("сказал проект")
        if start >= 0:
            insert_at = start + len("сказал")
            candidates.append(
                Candidate(
                    source="",
                    replacement=":",
                    edit_type="punctuation_insert",
                    start=insert_at,
                    end=insert_at,
                    confidence=1.0,
                    requires_model=True,
                    rule_id="direct_speech_colon",
                    mode="model_required",
                    action="INSERT",
                    label="COLON",
                    requires=("syntax", "model"),
                    group="punctuation",
                    syntax_family="direct_speech_syntax",
                )
            )
        if "«" in text and "»" not in text:
            insert_at = text.rfind(".")
            candidates.append(
                Candidate(
                    source="",
                    replacement="»",
                    edit_type="punctuation_insert",
                    start=insert_at,
                    end=insert_at,
                    confidence=1.0,
                    requires_model=True,
                    rule_id="quote_pair_balance",
                    mode="model_required",
                    action="INSERT",
                    label="QUOTE_CLOSE",
                    requires=("syntax", "model"),
                    group="punctuation",
                    syntax_family="quote_pair_balance",
                )
            )
        return candidates


class EmptyCandidateGenerator:
    def generate(self, text: str, **_kwargs):
        del text
        return []


def _recipe_path(tmp_path: Path, rules: dict) -> Path:
    path = tmp_path / "rule_lab_recipes.yaml"
    path.write_text(yaml.safe_dump({"rules": rules}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _config(path: Path, **rule_lab_overrides) -> dict:
    rule_lab = {
        "enabled": True,
        "fill_underfilled_rules": True,
        "max_generated_per_rule": 20,
        "max_hard_negatives_per_rule": 20,
        "fail_on_low_diversity": True,
        "max_per_template_share": 1.0,
        "max_per_slot_value_share": 1.0,
        "context_injection_max_share": 0.5,
    }
    rule_lab.update(rule_lab_overrides)
    return {
        "data": {
            "candidate_opportunity": {
                "paths": {"rule_lab_recipes_config": str(path)},
                "rule_quota": {
                    "min_atomic_positives_per_active_rule": 1,
                    "preferred_atomic_positives_per_active_rule": 4,
                    "min_hard_negatives_per_active_rule": 1,
                },
                "rule_lab": rule_lab,
            }
        }
    }


def _chto_by_rule() -> dict:
    return {
        "enabled": True,
        "family": "split_join",
        "mutation": {"target_fragment": "чтобы", "source_fragment": "что бы"},
        "positive_templates": [
            {
                "template_id": "purpose_basic",
                "target": "{subject} {verb_past}, чтобы {infinitive} {object} {time}.",
            },
            {
                "template_id": "purpose_review",
                "target": "{subject} {verb_past}, чтобы {infinitive} {object} после встречи.",
            },
        ],
        "hard_negative_templates": [
            {
                "template_id": "concessive_trap",
                "text": "Что бы {subject} ни {verb_past}, комиссия проверит {object} {time}.",
            }
        ],
        "slots": {
            "subject": {"values": ["редактор", "эксперт", "инспектор"]},
            "verb_past": {"values": ["пришёл", "вернулся"]},
            "infinitive": {"values": ["проверить", "уточнить"]},
            "object": {"values": ["документ", "отчёт"]},
            "time": {"values": ["утром", "вечером"]},
        },
        "diversity": {"max_per_template_share": 1.0, "max_per_slot_value_share": 1.0},
        "seed": 7,
    }


def test_rule_lab_does_not_repeat_identical_examples_to_reach_quota(tmp_path: Path):
    path = _recipe_path(
        tmp_path,
        {
            "context_chto_by": {
                **_chto_by_rule(),
                "positive_templates": [{"template_id": "single", "target": "Редактор пришёл, чтобы проверить документ утром."}],
                "slots": {},
            }
        },
    )

    result = generate_rule_lab_rows(
        ["context_chto_by"],
        RuleLabCandidateGenerator(),
        _config(path, fail_on_low_diversity=False),
        positive_counts_by_rule={"context_chto_by": 0},
        hard_counts_by_rule={"context_chto_by": 1},
        positive_target=3,
        hard_target=1,
    )

    assert len(result.atomic_positive_rows) == 1
    assert {row["source"] for row in result.atomic_positive_rows} == {
        "Редактор пришёл, что бы проверить документ утром."
    }
    assert len({row["source"] for row in result.atomic_positive_rows}) == len(result.atomic_positive_rows)
    assert result.generation_rows[0]["positive_requested"] == 3
    assert result.generation_rows[0]["positive_generated"] == 1


def test_slot_expansion_creates_multiple_unique_rows_from_same_template(tmp_path: Path):
    path = _recipe_path(tmp_path, {"context_chto_by": _chto_by_rule()})

    result = generate_rule_lab_rows(
        ["context_chto_by"],
        RuleLabCandidateGenerator(),
        _config(path),
        positive_counts_by_rule={"context_chto_by": 0},
        hard_counts_by_rule={"context_chto_by": 1},
        positive_target=4,
        hard_target=1,
    )

    assert len(result.atomic_positive_rows) == 4
    assert len({row["source"] for row in result.atomic_positive_rows}) == 4
    assert len({row["target"] for row in result.atomic_positive_rows}) == 4
    assert {row["dataset_layer"] for row in result.atomic_positive_rows} == {LAYER_ATOMIC_POSITIVE}


def test_positive_row_with_multi_edit_diff_is_rejected(tmp_path: Path):
    path = _recipe_path(
        tmp_path,
        {
            "context_chto_by": {
                **_chto_by_rule(),
                "positive_templates": [
                    {
                        "template_id": "multi_edit",
                        "source": "Редактор пришёл что бы проверить документ утром",
                        "target": "Редактор пришёл, чтобы проверить документ утром.",
                    }
                ],
                "slots": {},
            }
        },
    )

    result = generate_rule_lab_rows(
        ["context_chto_by"],
        RuleLabCandidateGenerator(),
        _config(path, fail_on_low_diversity=False),
        positive_counts_by_rule={"context_chto_by": 0},
        hard_counts_by_rule={"context_chto_by": 1},
        positive_target=1,
        hard_target=1,
    )

    assert result.atomic_positive_rows == []
    assert any(row["reason"] == "non_atomic_edit_count" for row in result.rejection_rows)


def test_positive_row_with_candidate_missing_is_rejected(tmp_path: Path):
    path = _recipe_path(tmp_path, {"context_chto_by": _chto_by_rule()})

    result = generate_rule_lab_rows(
        ["context_chto_by"],
        EmptyCandidateGenerator(),
        _config(path, fail_on_low_diversity=False),
        positive_counts_by_rule={"context_chto_by": 0},
        hard_counts_by_rule={"context_chto_by": 1},
        positive_target=1,
        hard_target=1,
    )

    assert result.atomic_positive_rows == []
    assert any(row["reason"] == "candidate_missing" for row in result.rejection_rows)


def test_hard_negative_with_unbalanced_quote_is_rejected(tmp_path: Path):
    path = _recipe_path(
        tmp_path,
        {
            "context_tak_zhe": {
                "enabled": True,
                "family": "split_join",
                "positive_templates": [],
                "hard_negative_templates": [
                    {"template_id": "bad_quote", "text": '"Так же, как раньше, редакция проверила отчет утром.'}
                ],
            }
        },
    )

    result = generate_rule_lab_rows(
        ["context_tak_zhe"],
        RuleLabCandidateGenerator(),
        _config(path, fail_on_low_diversity=False),
        positive_counts_by_rule={"context_tak_zhe": 1},
        hard_counts_by_rule={"context_tak_zhe": 0},
        positive_target=1,
        hard_target=1,
    )

    assert result.hard_negative_rows == []
    assert any(row["reason"] == "hard_negative_quality_failed:unbalanced_ascii_quotes" for row in result.rejection_rows)


def test_dominant_template_share_over_threshold_blocks_rule_lab_rows(tmp_path: Path):
    path = _recipe_path(tmp_path, {"context_chto_by": _chto_by_rule()})

    result = generate_rule_lab_rows(
        ["context_chto_by"],
        RuleLabCandidateGenerator(),
        _config(path, max_per_template_share=0.24, max_per_slot_value_share=1.0),
        positive_counts_by_rule={"context_chto_by": 0},
        hard_counts_by_rule={"context_chto_by": 1},
        positive_target=4,
        hard_target=1,
    )

    assert result.atomic_positive_rows == []
    diversity = result.diversity_rows[0]
    assert diversity.status == "blocked"
    assert "dominant_template_share" in diversity.reason


def test_dominant_slot_value_share_over_threshold_blocks_rule_lab_rows(tmp_path: Path):
    rule = _chto_by_rule()
    rule["slots"]["subject"] = {"values": ["редактор"]}
    path = _recipe_path(tmp_path, {"context_chto_by": rule})

    result = generate_rule_lab_rows(
        ["context_chto_by"],
        RuleLabCandidateGenerator(),
        _config(path, max_per_template_share=1.0, max_per_slot_value_share=0.35),
        positive_counts_by_rule={"context_chto_by": 0},
        hard_counts_by_rule={"context_chto_by": 1},
        positive_target=4,
        hard_target=1,
    )

    assert result.atomic_positive_rows == []
    diversity = result.diversity_rows[0]
    assert diversity.status == "blocked"
    assert "dominant_slot_value_share" in diversity.reason


def test_clean_corpus_context_injection_produces_valid_atomic_positive(tmp_path: Path):
    path = _recipe_path(
        tmp_path,
        {
            "context_chto_by": {
                "enabled": True,
                "family": "split_join",
                "mutation": {"target_fragment": "чтобы", "source_fragment": "что бы"},
                "context_injection": {"enabled": True, "mode": "append_clause", "max_share": 1.0},
                "positive_templates": [
                    {
                        "template_id": "context_append",
                        "target": "{context}, чтобы {infinitive} {object}.",
                    }
                ],
                "slots": {
                    "infinitive": {"values": ["подготовить", "уточнить"]},
                    "object": {"values": ["заключение", "отчёт"]},
                },
                "diversity": {"max_per_template_share": 1.0, "max_per_slot_value_share": 1.0},
            }
        },
    )

    result = generate_rule_lab_rows(
        ["context_chto_by"],
        RuleLabCandidateGenerator(),
        _config(path),
        clean_rows=[{"text": "Редактор внимательно изучил материалы дела."}],
        positive_counts_by_rule={"context_chto_by": 0},
        hard_counts_by_rule={"context_chto_by": 1},
        positive_target=1,
        hard_target=1,
    )

    assert len(result.atomic_positive_rows) == 1
    row = result.atomic_positive_rows[0]
    assert row["target"].startswith("Редактор внимательно изучил материалы дела, чтобы ")
    assert row["source"].startswith("Редактор внимательно изучил материалы дела, что бы ")
    assert row["source"].replace("что бы", "чтобы", 1) == row["target"]
    assert row["count_toward_rule_quota"] is True


def test_disabled_recipe_is_reported_and_not_used(tmp_path: Path):
    path = _recipe_path(
        tmp_path,
        {
            "tsya_soft_insert": {
                "enabled": False,
                "disabled_reason": "validator_missing",
                "family": "morphology",
                "positive_templates": [{"template_id": "unsafe", "target": "Они могут появиться завтра."}],
            }
        },
    )

    recipes = load_rule_lab_recipes(path)
    result = generate_rule_lab_rows(
        ["tsya_soft_insert"],
        RuleLabCandidateGenerator(),
        _config(path),
        positive_counts_by_rule={"tsya_soft_insert": 0},
        hard_counts_by_rule={"tsya_soft_insert": 0},
        positive_target=1,
        hard_target=1,
    )

    assert recipes["tsya_soft_insert"].enabled is False
    assert result.atomic_positive_rows == []
    assert result.hard_negative_rows == []
    assert result.recipe_status_rows[0]["status"] == "disabled"
    assert result.recipe_status_rows[0]["disabled_reason"] == "validator_missing"


def test_rule_lab_rows_count_only_atomic_positives_toward_quota(tmp_path: Path):
    path = _recipe_path(tmp_path, {"context_chto_by": _chto_by_rule()})

    result = generate_rule_lab_rows(
        ["context_chto_by"],
        RuleLabCandidateGenerator(),
        _config(path, fail_on_low_diversity=False),
        positive_counts_by_rule={"context_chto_by": 0},
        hard_counts_by_rule={"context_chto_by": 0},
        positive_target=1,
        hard_target=1,
    )

    assert result.atomic_positive_rows[0]["count_toward_rule_quota"] is True
    assert result.hard_negative_rows[0]["dataset_layer"] == LAYER_ATOMIC_HARD_NEGATIVE
    assert result.hard_negative_rows[0]["count_toward_rule_quota"] is False
    assert json.loads(result.hard_negative_rows[0]["edits"]) == []
