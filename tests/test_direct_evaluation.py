from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.evaluation.evaluate import evaluate_corrector
from src.schema import CorrectionResult, GeneratedExample, RuntimeEdit, WordToken
from src.schema.serialization import write_jsonl_examples


def test_direct_evaluation_writes_runtime_reports(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "val.jsonl"
    output_dir = tmp_path / "reports"
    config_path = tmp_path / "config.yaml"
    write_jsonl_examples(dataset_path, _frozen_examples(30))
    config_path.write_text(yaml.safe_dump({"runtime": {"neural_token_edits": False}}, allow_unicode=True), encoding="utf-8")

    class FakeCorrector:
        @classmethod
        def from_config(cls, config):
            return cls()

        def correct(self, text: str) -> CorrectionResult:
            if text == "\u041e\u043d \u043d\u0435\u0437\u043d\u0430\u043b":
                return CorrectionResult(
                    source_text=text,
                    corrected_text="\u041e\u043d \u043d\u0435 \u0437\u043d\u0430\u043b",
                    edits=[
                        RuntimeEdit(
                            start=3,
                            end=9,
                            source="\u043d\u0435\u0437\u043d\u0430\u043b",
                            replacement="\u043d\u0435 \u0437\u043d\u0430\u043b",
                            edit_type="split_join",
                            rule_id="ne_verb",
                            confidence=0.99,
                        )
                    ],
                    metadata={"rejected_by_threshold": 1},
                )
            if text == "\u0427\u0438\u0441\u0442\u044b\u0439 \u0442\u0435\u043a\u0441\u0442.":
                return CorrectionResult(
                    source_text=text,
                    corrected_text="\u0427\u0438\u0441\u0442\u044b\u0439 \u0442\u0435\u043a\u0441\u0442!",
                    edits=[
                        RuntimeEdit(
                            start=13,
                            end=14,
                            source=".",
                            replacement="!",
                            edit_type="punctuation",
                            rule_id="final_punctuation",
                            confidence=0.91,
                        )
                    ],
                    metadata={"rejected_by_scope_guard": 1},
                )
            return CorrectionResult(source_text=text, corrected_text=text, edits=[])

    monkeypatch.setattr("src.evaluation.evaluate.Corrector", FakeCorrector)

    summary = evaluate_corrector(config_path, dataset_path, output_dir)

    summary_path = output_dir / "evaluation_summary.json"
    per_rule_path = output_dir / "per_rule_metrics.csv"
    worst_path = output_dir / "worst_examples.jsonl"
    assert summary_path.exists()
    assert per_rule_path.exists()
    assert worst_path.exists()
    assert "exact_match" in summary
    assert "overcorrection_rate" in summary

    summary_from_disk = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "exact_match" in summary_from_disk
    assert "overcorrection_rate" in summary_from_disk


def _frozen_examples(count: int) -> list[GeneratedExample]:
    examples: list[GeneratedExample] = []
    for index in range(count):
        remainder = index % 3
        if remainder == 0:
            source = "\u041e\u043d \u043d\u0435\u0437\u043d\u0430\u043b"
            target = "\u041e\u043d \u043d\u0435 \u0437\u043d\u0430\u043b"
            tokens = _tokens(source)
            examples.append(
                GeneratedExample(
                    source_text=source,
                    target_text=target,
                    source_tokens=tokens,
                    token_edit_labels=["KEEP", "SPLIT_NE_VERB"],
                    gap_labels=["NONE", "NONE"],
                    rule_ids=["none", "ne_verb"],
                    primary_rule_id="ne_verb",
                    mode="positive",
                    explanation_ids=["ne_verb"],
                    metadata={"index": index},
                )
            )
        elif remainder == 1:
            text = "\u0427\u0438\u0441\u0442\u044b\u0439 \u0442\u0435\u043a\u0441\u0442."
            tokens = _tokens(text)
            examples.append(
                GeneratedExample(
                    source_text=text,
                    target_text=text,
                    source_tokens=tokens,
                    token_edit_labels=["KEEP", "KEEP"],
                    gap_labels=["NONE", "DOT"],
                    rule_ids=["none", "clean_identity"],
                    primary_rule_id="clean_identity",
                    mode="clean_identity",
                    explanation_ids=[],
                    metadata={"index": index},
                )
            )
        else:
            text = "\u041e\u043d \u043d\u0435\u043d\u0430\u0432\u0438\u0434\u0435\u043b \u0448\u0443\u043c."
            tokens = _tokens(text)
            examples.append(
                GeneratedExample(
                    source_text=text,
                    target_text=text,
                    source_tokens=tokens,
                    token_edit_labels=["KEEP", "KEEP", "KEEP"],
                    gap_labels=["NONE", "NONE", "DOT"],
                    rule_ids=["none", "ne_verb", "clean_identity"],
                    primary_rule_id="ne_verb",
                    mode="hard_negative",
                    explanation_ids=["ne_verb"],
                    metadata={"index": index},
                )
            )
    return examples


def _tokens(text: str) -> list[WordToken]:
    tokens: list[WordToken] = []
    start: int | None = None
    for index, char in enumerate(text):
        if char.isalpha() and start is None:
            start = index
        elif not char.isalpha() and start is not None:
            tokens.append(WordToken(text=text[start:index], start=start, end=index))
            start = None
    if start is not None:
        tokens.append(WordToken(text=text[start:], start=start, end=len(text)))
    return tokens
