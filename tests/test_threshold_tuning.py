from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.tune_thresholds import tune_thresholds
from src.schema import CorrectionResult, GeneratedExample, RuntimeEdit, WordToken
from src.schema.serialization import write_jsonl_examples


def test_tune_thresholds_writes_yaml_and_penalizes_overcorrection(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    dataset_path = tmp_path / "val.jsonl"
    output_path = tmp_path / "runtime_thresholds.yaml"
    report_path = tmp_path / "reports" / "threshold_tuning.json"

    config_path.write_text(
        yaml.safe_dump(
            {
                "runtime": {
                    "confidence_thresholds": {
                        "token_edit": 0.0,
                        "punctuation": 0.7,
                    }
                }
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    write_jsonl_examples(dataset_path, _examples_for_overcorrection_penalty())

    class FakeCorrector:
        def __init__(self, threshold: float) -> None:
            self.threshold = threshold

        @classmethod
        def from_config(cls, config):
            threshold = float(config["runtime"]["confidence_thresholds"]["token_edit"])
            return cls(threshold)

        def correct(self, text: str) -> CorrectionResult:
            if self.threshold <= 0.4:
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
                                confidence=0.4,
                            )
                        ],
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
                                confidence=0.4,
                            )
                        ],
                    )
            return CorrectionResult(source_text=text, corrected_text=text, edits=[])

    monkeypatch.setattr("scripts.tune_thresholds.Corrector", FakeCorrector)

    result = tune_thresholds(
        config_path=config_path,
        dataset_path=dataset_path,
        output_path=output_path,
        report_path=report_path,
        threshold_grid=[0.0, 0.4, 0.5],
    )

    assert output_path.exists()
    assert report_path.exists()
    selected = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert selected["runtime"]["confidence_thresholds"]["token_edit"] == 0.5
    assert result["selected_thresholds"]["token_edit"] == 0.5

    report = json.loads(report_path.read_text(encoding="utf-8"))
    low_trial = next(trial for trial in report["token_edit_trials"] if trial["threshold"] == 0.4)
    selected_trial = next(trial for trial in report["token_edit_trials"] if trial["threshold"] == 0.5)
    assert low_trial["exact_match"] > selected_trial["exact_match"]
    assert low_trial["combined_score"] < selected_trial["combined_score"]


def _examples_for_overcorrection_penalty() -> list[GeneratedExample]:
    examples = [_dirty_example(), _dirty_example(), _dirty_example()]
    examples.append(_clean_example())
    return examples


def _dirty_example() -> GeneratedExample:
    source = "\u041e\u043d \u043d\u0435\u0437\u043d\u0430\u043b"
    return GeneratedExample(
        source_text=source,
        target_text="\u041e\u043d \u043d\u0435 \u0437\u043d\u0430\u043b",
        source_tokens=_tokens(source),
        token_edit_labels=["KEEP", "SPLIT_NE_VERB"],
        gap_labels=["NONE", "NONE"],
        rule_ids=["none", "ne_verb"],
        primary_rule_id="ne_verb",
        mode="positive",
        explanation_ids=["ne_verb"],
        metadata={},
    )


def _clean_example() -> GeneratedExample:
    text = "\u0427\u0438\u0441\u0442\u044b\u0439 \u0442\u0435\u043a\u0441\u0442."
    return GeneratedExample(
        source_text=text,
        target_text=text,
        source_tokens=_tokens(text),
        token_edit_labels=["KEEP", "KEEP"],
        gap_labels=["NONE", "DOT"],
        rule_ids=["none", "clean_identity"],
        primary_rule_id="clean_identity",
        mode="clean_identity",
        explanation_ids=[],
        metadata={},
    )


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
