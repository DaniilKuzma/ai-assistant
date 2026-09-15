from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.runtime.corrector import Corrector
from src.runtime.edit_realizer import apply_token_edit_labels
from src.runtime.neural_backend import DirectNeuralBackend
from src.runtime.orthographic_lexicon import OrthographicCorrectionLexicon
from src.runtime.tokenization import tokenize_runtime_words
from src.schema.labels import GAP_ID_TO_LABEL, RULE_ID_TO_LABEL, TOKEN_ID_TO_LABEL


KOZHENNY_SENTENCE = "\u0412 \u0441\u043b\u043e\u0432\u0430\u0440\u0435 \u0443\u043a\u0430\u0437\u0430\u043d\u043e \u0441\u043b\u043e\u0432\u043e \u00ab\u043a\u043e\u0436\u0435\u043d\u043d\u044b\u0439\u00bb."
KOZHANY_SENTENCE = "\u0412 \u0441\u043b\u043e\u0432\u0430\u0440\u0435 \u0443\u043a\u0430\u0437\u0430\u043d\u043e \u0441\u043b\u043e\u0432\u043e \u00ab\u043a\u043e\u0436\u0430\u043d\u044b\u0439\u00bb."


@dataclass(frozen=True)
class FakePrediction:
    token_labels: list[str]
    token_confidences: list[float]
    gap_labels: list[str]
    gap_confidences: list[float]
    rule_ids: list[str]
    token_margins: list[float]
    gap_margins: list[float]


class DictReplaceBackend:
    def predict(self, text: str) -> FakePrediction:
        tokens = tokenize_runtime_words(text)
        token_labels = ["KEEP"] * len(tokens)
        rule_ids = ["none"] * len(tokens)
        for index, token in enumerate(tokens):
            if token.text == "\u043a\u043e\u0436\u0435\u043d\u043d\u044b\u0439":
                token_labels[index] = "DICT_REPLACE"
                rule_ids[index] = "suffix_enn_yan"
        return FakePrediction(
            token_labels=token_labels,
            token_confidences=[0.99] * len(tokens),
            gap_labels=["NONE"] * len(tokens),
            gap_confidences=[1.0] * len(tokens),
            rule_ids=rule_ids,
            token_margins=[0.99] * len(tokens),
            gap_margins=[1.0] * len(tokens),
        )


def test_orthographic_lexicon_returns_suffix_enn_yan_replacement() -> None:
    entries = OrthographicCorrectionLexicon.default().lookup(
        "\u043a\u043e\u0436\u0435\u043d\u043d\u044b\u0439",
        rule_id="suffix_enn_yan",
    )

    assert [entry.target for entry in entries] == ["\u043a\u043e\u0436\u0430\u043d\u044b\u0439"]


def test_apply_token_edit_labels_applies_dict_replace_with_lexicon() -> None:
    tokens = tokenize_runtime_words(KOZHENNY_SENTENCE)

    corrected, edits = apply_token_edit_labels(
        KOZHENNY_SENTENCE,
        tokens,
        ["KEEP", "KEEP", "KEEP", "KEEP", "DICT_REPLACE"],
        [1.0, 1.0, 1.0, 1.0, 0.96],
        threshold=0.7,
        rule_ids=["none", "none", "none", "none", "suffix_enn_yan"],
        orthographic_lexicon=OrthographicCorrectionLexicon.default(),
    )

    assert corrected == KOZHANY_SENTENCE
    assert [(edit.source, edit.replacement, edit.rule_id) for edit in edits] == [
        ("\u043a\u043e\u0436\u0435\u043d\u043d\u044b\u0439", "\u043a\u043e\u0436\u0430\u043d\u044b\u0439", "suffix_enn_yan")
    ]


def test_apply_token_edit_labels_applies_dict_replace_when_contexts_share_target() -> None:
    text = "\u041d\u0430 \u0441\u0442\u043e\u043b\u0435 \u043b\u0435\u0436\u0430\u043b \u043c\u0430\u0441\u043b\u044f\u043d\u044b\u0439 \u043d\u043e\u0436."
    tokens = tokenize_runtime_words(text)

    corrected, edits = apply_token_edit_labels(
        text,
        tokens,
        ["KEEP", "KEEP", "KEEP", "DICT_REPLACE", "KEEP"],
        [1.0] * len(tokens),
        threshold=0.7,
        rule_ids=["none", "none", "none", "suffix_enn_yan", "none"],
        orthographic_lexicon=OrthographicCorrectionLexicon.default(),
    )

    assert corrected == "\u041d\u0430 \u0441\u0442\u043e\u043b\u0435 \u043b\u0435\u0436\u0430\u043b \u043c\u0430\u0441\u043b\u0435\u043d\u044b\u0439 \u043d\u043e\u0436."
    assert [(edit.source, edit.replacement, edit.rule_id) for edit in edits] == [
        ("\u043c\u0430\u0441\u043b\u044f\u043d\u044b\u0439", "\u043c\u0430\u0441\u043b\u0435\u043d\u044b\u0439", "suffix_enn_yan")
    ]


def test_corrector_applies_fake_neural_dict_replace_prediction() -> None:
    corrector = Corrector(
        neural_backend=DictReplaceBackend(),
        config={"runtime": {"deterministic_first": False}},
    )

    result = corrector.correct(KOZHENNY_SENTENCE)

    assert result.corrected_text == KOZHANY_SENTENCE
    assert [(edit.source, edit.replacement, edit.rule_id) for edit in result.edits] == [
        ("\u043a\u043e\u0436\u0435\u043d\u043d\u044b\u0439", "\u043a\u043e\u0436\u0430\u043d\u044b\u0439", "suffix_enn_yan")
    ]


def test_direct_neural_backend_rejects_stale_labels_before_torch_load(monkeypatch, tmp_path: Path) -> None:
    heads_dir = tmp_path / "heads"
    heads_dir.mkdir()
    (heads_dir / "heads.pt").write_bytes(b"not loaded")
    (heads_dir / "architecture.json").write_text(
        json.dumps({"architecture": "direct_edit_tagger_v1"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (heads_dir / "labels.json").write_text(
        json.dumps(
            {
                "token_id_to_label": [*list(TOKEN_ID_TO_LABEL), "STALE_TOKEN_LABEL"],
                "gap_id_to_label": list(GAP_ID_TO_LABEL),
                "rule_id_to_label": list(RULE_ID_TO_LABEL),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def fail_torch_load(*args, **kwargs):
        raise AssertionError("torch.load must not be called for stale labels")

    monkeypatch.setattr("torch.load", fail_torch_load)

    with pytest.raises(RuntimeError, match="label schema.*token_id_to_label"):
        DirectNeuralBackend.from_config(
            {
                "model": {"lora": {"enabled": False}, "local_files_only": True},
                "paths": {
                    "heads_output_dir": str(heads_dir),
                    "adapter_output_dir": str(tmp_path / "adapters"),
                },
            }
        )
