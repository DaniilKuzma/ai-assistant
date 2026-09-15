from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.evaluation.evaluate import evaluate_corrector, gold_runtime_edits
from src.runtime.orthographic_lexicon import OrthographicCorrectionLexicon
from src.schema import CorrectionResult, GeneratedExample, RuntimeEdit, WordToken
from src.schema.serialization import write_jsonl_examples


def test_direct_evaluation_writes_runtime_reports(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "val.jsonl"
    output_dir = tmp_path / "reports"
    config_path = tmp_path / "config.yaml"
    write_jsonl_examples(dataset_path, _frozen_examples(30))
    config_path.write_text(
        yaml.safe_dump({"runtime": {"neural_token_edits": False, "neural_punctuation": False}}, allow_unicode=True),
        encoding="utf-8",
    )

    class FakeCorrector:
        @classmethod
        def from_config(cls, config, *, strict_neural=False, allow_fallback=False):
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
    assert summary_from_disk["backend_kind"] == "deterministic_fallback"


def test_direct_evaluation_fails_on_silent_fallback_without_flag(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "val.jsonl"
    output_dir = tmp_path / "reports"
    config_path = tmp_path / "config.yaml"
    write_jsonl_examples(dataset_path, _frozen_examples(3))
    config_path.write_text(
        yaml.safe_dump({"runtime": {"neural_token_edits": True, "neural_punctuation": False}}, allow_unicode=True),
        encoding="utf-8",
    )

    seen: dict[str, object] = {}

    class FallbackCorrector:
        neural_backend = None

        @classmethod
        def from_config(cls, config, *, strict_neural=False, allow_fallback=False):
            seen["strict_neural"] = strict_neural
            seen["allow_fallback"] = allow_fallback
            return cls()

        def correct(self, text: str) -> CorrectionResult:
            return CorrectionResult(source_text=text, corrected_text=text, edits=[], metadata={})

    monkeypatch.setattr("src.evaluation.evaluate.Corrector", FallbackCorrector)

    with pytest.raises(RuntimeError, match="deterministic fallback"):
        evaluate_corrector(config_path, dataset_path, output_dir)

    assert seen == {"strict_neural": True, "allow_fallback": False}


def test_direct_evaluation_allows_fallback_only_when_explicit(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "val.jsonl"
    output_dir = tmp_path / "reports"
    config_path = tmp_path / "config.yaml"
    write_jsonl_examples(dataset_path, _frozen_examples(3))
    config_path.write_text(
        yaml.safe_dump({"runtime": {"neural_token_edits": True, "neural_punctuation": False}}, allow_unicode=True),
        encoding="utf-8",
    )

    class FallbackCorrector:
        neural_backend = None

        @classmethod
        def from_config(cls, config, *, strict_neural=False, allow_fallback=False):
            return cls()

        def correct(self, text: str) -> CorrectionResult:
            return CorrectionResult(
                source_text=text,
                corrected_text=text,
                edits=[],
                metadata={"backend_kind": "deterministic_fallback", "model_loaded": False},
            )

    monkeypatch.setattr("src.evaluation.evaluate.Corrector", FallbackCorrector)

    summary = evaluate_corrector(config_path, dataset_path, output_dir, allow_fallback=True)
    summary_from_disk = json.loads((output_dir / "evaluation_summary.json").read_text(encoding="utf-8"))

    assert summary["backend_kind"] == "deterministic_fallback"
    assert summary_from_disk["backend_kind"] == "deterministic_fallback"


def test_gold_runtime_edits_uses_orthographic_lexicon_for_dict_replace() -> None:
    source = "\u0412 \u0441\u043b\u043e\u0432\u0430\u0440\u0435 \u0443\u043a\u0430\u0437\u0430\u043d\u043e \u0441\u043b\u043e\u0432\u043e \u00ab\u043a\u043e\u0436\u0435\u043d\u043d\u044b\u0439\u00bb."
    target = "\u0412 \u0441\u043b\u043e\u0432\u0430\u0440\u0435 \u0443\u043a\u0430\u0437\u0430\u043d\u043e \u0441\u043b\u043e\u0432\u043e \u00ab\u043a\u043e\u0436\u0430\u043d\u044b\u0439\u00bb."
    example = GeneratedExample(
        source_text=source,
        target_text=target,
        source_tokens=_tokens(source),
        token_edit_labels=["KEEP", "KEEP", "KEEP", "KEEP", "DICT_REPLACE"],
        gap_labels=["NONE", "NONE", "NONE", "NONE", "NONE"],
        rule_ids=["none", "none", "none", "none", "suffix_enn_yan"],
        primary_rule_id="suffix_enn_yan",
        mode="positive",
        explanation_ids=["suffix_enn_yan"],
        metadata={},
    )

    edits = gold_runtime_edits(example, OrthographicCorrectionLexicon.default())

    assert [(edit.source, edit.replacement, edit.rule_id) for edit in edits] == [
        ("\u043a\u043e\u0436\u0435\u043d\u043d\u044b\u0439", "\u043a\u043e\u0436\u0430\u043d\u044b\u0439", "suffix_enn_yan")
    ]


def test_manual_pair_dataset_evaluator_writes_tag_metrics(monkeypatch, tmp_path: Path) -> None:
    from scripts import evaluate_pair_dataset

    dataset_path = tmp_path / "manual.jsonl"
    output_dir = tmp_path / "manual_report"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("runtime:\n  neural_token_edits: true\n  neural_punctuation: false\n", encoding="utf-8")
    _write_jsonl(
        dataset_path,
        [
            {"source": "Он незнал", "target": "Он не знал", "tags": ["ne_verb"], "scope": "in_scope"},
            {"source": "Чистый текст.", "target": "Чистый текст.", "tags": ["clean"], "scope": "in_scope"},
        ],
    )

    class FakeCorrector:
        neural_backend = object()

        @classmethod
        def from_config(cls, config, *, strict_neural=False, allow_fallback=False):
            assert strict_neural is True
            return cls()

        def correct(self, text: str) -> CorrectionResult:
            corrected = "Он не знал" if text == "Он незнал" else text
            return CorrectionResult(source_text=text, corrected_text=corrected, edits=[], metadata={"backend_kind": "direct_neural"})

    monkeypatch.setattr(evaluate_pair_dataset, "Corrector", FakeCorrector)

    summary = evaluate_pair_dataset.evaluate_pair_dataset(config_path, dataset_path, output_dir)
    summary_from_disk = json.loads((output_dir / "evaluation_summary.json").read_text(encoding="utf-8"))

    assert summary["exact_match"] == 1.0
    assert summary["per_tag"]["ne_verb"]["exact_match"] == 1.0
    assert summary_from_disk["unchanged_when_clean"] == 1.0
    assert (output_dir / "worst_examples.jsonl").exists()


def test_external_spellcheck_evaluator_reports_diagnostic_subsets(monkeypatch, tmp_path: Path) -> None:
    from scripts import evaluate_external_spellcheck

    dataset_path = tmp_path / "test.json"
    output_dir = tmp_path / "external_report"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("runtime:\n  neural_token_edits: true\n  neural_punctuation: false\n", encoding="utf-8")
    _write_jsonl(
        dataset_path,
        [
            {"source": "Он незнал", "correction": "Он не знал"},
            {"source": "Чистый текст.", "correction": "Чистый текст."},
        ],
    )

    class FakeCorrector:
        neural_backend = object()

        @classmethod
        def from_config(cls, config, *, strict_neural=False, allow_fallback=False):
            assert strict_neural is True
            return cls()

        def correct(self, text: str) -> CorrectionResult:
            corrected = "Он не знал" if text == "Он незнал" else text
            return CorrectionResult(source_text=text, corrected_text=corrected, edits=[], metadata={"backend_kind": "direct_neural"})

    monkeypatch.setattr(evaluate_external_spellcheck, "Corrector", FakeCorrector)

    summary = evaluate_external_spellcheck.evaluate_external_spellcheck(config_path, dataset_path, output_dir, limit=1000)

    assert {"all", "in_scope_by_simple_diff_filter", "clean_no_change"} <= set(summary)
    assert summary["all"]["example_count"] == 2
    assert summary["clean_no_change"]["unchanged_when_clean"] == 1.0
    assert (output_dir / "evaluation_summary.json").exists()


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


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
