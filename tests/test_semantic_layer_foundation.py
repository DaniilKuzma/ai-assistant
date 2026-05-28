from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config.load_config import load_config
from src.runtime.edit_realizer import apply_token_edit_labels
from src.runtime.corrector import Corrector
from src.runtime.orthographic_lexicon import OrthographicCorrectionLexicon
from src.runtime.tokenization import tokenize_runtime_words
from src.schema import rule_id_to_label, rule_tag_to_id


ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_RULE_IDS = (
    "semantic_service_words",
    "semantic_derived_prepositions",
    "semantic_ne_ni",
    "semantic_introductory_context",
    "semantic_comparative_context",
)


def test_semantic_rule_ids_are_registered() -> None:
    for rule_id in SEMANTIC_RULE_IDS:
        assert rule_id_to_label(rule_tag_to_id(rule_id)) == rule_id


def test_load_semantic_specs_accepts_minimal_grouped_yaml(tmp_path: Path) -> None:
    from src.rule_layers.semantic import load_semantic_specs

    _write_yaml(
        tmp_path,
        "cases.yaml",
        {
            "rules": [
                {
                    "rule_id": "semantic_service_words",
                    "description": "Semantic service-word cases.",
                    "cases": {
                        "positive": [
                            {
                                "sub_rule_id": "chtoby_conjunction_merge",
                                "source": "Автор пришёл раньше, что бы подготовить доклад.",
                                "target": "Автор пришёл раньше, чтобы подготовить доклад.",
                                "token_operations": [
                                    {
                                        "kind": "token_span",
                                        "label": "SPAN_REPLACE_BY_LEXICON",
                                        "source_pattern": "что бы",
                                        "target_pattern": "чтобы",
                                    }
                                ],
                                "metadata": {
                                    "semantic_case_type": "service_word_merge",
                                    "ambiguity_pair": "что бы/чтобы",
                                    "semantic_signal": "purpose infinitive",
                                    "operation": "merge",
                                    "source": "что бы",
                                    "target": "чтобы",
                                },
                            }
                        ],
                        "hard_negative": [
                            {
                                "sub_rule_id": "chtoby_particle_identity",
                                "source": "Редактор спросил, что бы ещё проверить.",
                                "target": "Редактор спросил, что бы ещё проверить.",
                                "metadata": {
                                    "semantic_case_type": "service_word_guard",
                                    "ambiguity_pair": "что бы/чтобы",
                                    "semantic_signal": "interrogative pronoun with particle",
                                    "operation": "identity_guard",
                                    "source": "что бы",
                                    "target": "что бы",
                                },
                            }
                        ],
                    },
                }
            ]
        },
    )

    specs = load_semantic_specs(tmp_path)

    assert len(specs) == 1
    assert specs[0].layer == "semantic"
    assert specs[0].family == "semantic"
    assert specs[0].rule_id == "semantic_service_words"
    assert {case.mode for case in specs[0].cases} == {"positive", "hard_negative"}
    assert specs[0].cases[0].metadata["source_file"] == "cases.yaml"


def test_semantic_loader_rejects_legacy_rule_ids(tmp_path: Path) -> None:
    from src.rule_layers.semantic import load_semantic_specs

    _write_yaml(
        tmp_path,
        "legacy.yaml",
        {
            "rule_id": "takzhe_tak_zhe",
            "cases": {
                "positive": [
                    {
                        "sub_rule_id": "bad_duplicate",
                        "source": "Редактор так же проверил отчёт.",
                        "target": "Редактор также проверил отчёт.",
                        "token_operations": [
                            {
                                "label": "MERGE_TAK_ZHE_TO_TAKZHE",
                                "source_pattern": "так же",
                                "target_pattern": "также",
                            }
                        ],
                        "metadata": {
                            "semantic_case_type": "service_word_merge",
                            "ambiguity_pair": "так же/также",
                            "semantic_signal": "additive meaning",
                            "operation": "merge",
                            "source": "так же",
                            "target": "также",
                        },
                    }
                ]
            },
        },
    )

    with pytest.raises(ValueError, match="legacy.*rule_id"):
        load_semantic_specs(tmp_path)


def test_semantic_loader_allows_specialized_labels_for_legacy_overlap_subrules(tmp_path: Path) -> None:
    from src.rule_layers.semantic import load_semantic_specs

    _write_yaml(
        tmp_path,
        "overlap.yaml",
        {
            "rule_id": "semantic_service_words",
            "cases": {
                "positive": [
                    {
                        "sub_rule_id": "takzhe_additive_merge",
                        "source": "Редактор так же проверил отчёт.",
                        "target": "Редактор также проверил отчёт.",
                        "token_operations": [
                            {
                                "label": "MERGE_TAK_ZHE_TO_TAKZHE",
                                "source_pattern": "так же",
                                "target_pattern": "также",
                            }
                        ],
                        "metadata": {
                            "semantic_case_type": "service_word_merge",
                            "ambiguity_pair": "так же/также",
                            "semantic_signal": "additive meaning",
                            "operation": "merge",
                            "source": "так же",
                            "target": "также",
                        },
                    }
                ]
            },
        },
    )

    specs = load_semantic_specs(tmp_path)

    operation = specs[0].cases[0].token_operations[0]
    assert operation.label == "MERGE_TAK_ZHE_TO_TAKZHE"


def test_semantic_corrections_load_into_orthographic_lexicon() -> None:
    lexicon = OrthographicCorrectionLexicon.from_root(ROOT / "lexicon")

    entries = lexicon.lookup("что бы", rule_id="semantic_service_words")

    assert [(entry.target, entry.sub_rule_id, entry.operation) for entry in entries] == [
        ("чтобы", "chtoby_conjunction_merge", "merge")
    ]


def test_semantic_corrections_are_not_trusted_deterministic_lexicon_edits() -> None:
    config = load_config(ROOT / "configs" / "config.yaml")
    config["runtime"] = {
        **config.get("runtime", {}),
        "deterministic_lexicon": True,
        "deterministic_final_punctuation": False,
        "neural_token_edits": False,
        "neural_punctuation": False,
    }
    corrector = Corrector(neural_backend=None, config=config)

    result = corrector.correct("Совещание отменили вследствии болезни директора.")

    assert result.corrected_text == "Совещание отменили вследствии болезни директора."
    assert result.edits == []


def test_semantic_span_replace_can_use_dict_replace_entry_with_neural_label() -> None:
    text = "Совещание отменили вследствии болезни директора."
    tokens = tokenize_runtime_words(text)
    lexicon = OrthographicCorrectionLexicon.from_root(ROOT / "lexicon")

    corrected, edits = apply_token_edit_labels(
        text,
        tokens,
        ["KEEP", "KEEP", "SPAN_REPLACE_BY_LEXICON", "KEEP", "KEEP"],
        [1.0, 1.0, 0.99, 1.0, 1.0],
        threshold=0.70,
        rule_ids=["none", "none", "semantic_derived_prepositions", "none", "none"],
        orthographic_lexicon=lexicon,
    )

    assert corrected == "Совещание отменили вследствие болезни директора."
    assert [(edit.source, edit.replacement, edit.rule_id, edit.edit_type) for edit in edits] == [
        ("вследствии", "вследствие", "semantic_derived_prepositions", "spelling")
    ]


def _write_yaml(root: Path, name: str, payload: dict[str, object]) -> None:
    path = root / "semantic"
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
