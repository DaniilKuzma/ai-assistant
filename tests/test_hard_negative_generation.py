import json

import pandas as pd

from src.candidates.candidate_generator import Candidate
from src.data.dataset_contract import DATASET_CONTRACT, HARD_NEGATIVE_OPEN, LAYER_ATOMIC_HARD_NEGATIVE
from src.data.hard_negative_generation import generate_atomic_hard_negatives, write_hard_negative_reports
from src.training.tensorization import DebugTokenizer, build_training_feature


class StaticCandidateGenerator:
    def __init__(self, candidates_by_text):
        self.candidates_by_text = candidates_by_text

    def generate(self, text: str, **_kwargs):
        return list(self.candidates_by_text.get(text, []))


class EmptyCandidateGenerator:
    def generate(self, text: str, **_kwargs):
        del text
        return []


def _context_candidate(text: str, rule_id: str, source: str, replacement: str) -> Candidate:
    start = text.index(source)
    return Candidate(
        source=source,
        replacement=replacement,
        edit_type="split_join",
        start=start,
        end=start + len(source),
        requires_model=True,
        rule_id=rule_id,
        mode="model_required",
        requires=("syntax", "morphology", "model"),
        group="context_split_join",
    )


def _metadata(row: dict) -> dict:
    return json.loads(row["metadata"])


def test_hard_negative_row_keeps_identity_text_and_candidate_metadata():
    text = "Он сделал так же, как раньше."
    candidate = _context_candidate(text, "context_tak_zhe", "так же", "также")
    generator = StaticCandidateGenerator({text: [candidate]})

    result = generate_atomic_hard_negatives(
        clean_rows=[{"text": text, "source_name": "unit", "source_subcorpus": "context", "domain": "open_clean"}],
        rule_ids=["context_tak_zhe"],
        candidate_generator=generator,
        min_per_rule=1,
        preferred_per_rule=1,
        seed=13,
        max_scan_rows=10,
        fallback_templates_enabled=False,
    )

    assert len(result.rows) == 1
    row = result.rows[0]
    metadata = _metadata(row)

    assert row["source"] == text
    assert row["target"] == text
    assert row["source_type"] == HARD_NEGATIVE_OPEN
    assert row["rule_id"] == "clean_identity_hard_negative"
    assert json.loads(row["rule_ids"]) == ["clean_identity_hard_negative"]
    assert json.loads(row["edits"]) == []
    assert json.loads(row["edit_operations"]) == []
    assert row["dataset_contract"] == DATASET_CONTRACT
    assert row["dataset_layer"] == LAYER_ATOMIC_HARD_NEGATIVE
    assert row["is_atomic"] is True
    assert row["is_hard_negative"] is True
    assert row["target_rule_id"] == "context_tak_zhe"
    assert row["candidate_source"] == "так же"
    assert row["candidate_replacement"] == "также"
    assert row["candidate_start"] == candidate.start
    assert row["candidate_end"] == candidate.end
    assert row["count_toward_rule_quota"] is False
    assert row["loss_weight"] == 1.0
    assert row["gold_edit_count"] == 0
    assert metadata["dataset_contract"] == DATASET_CONTRACT
    assert metadata["dataset_layer"] == LAYER_ATOMIC_HARD_NEGATIVE
    assert metadata["target_rule_id"] == "context_tak_zhe"
    assert metadata["candidate_source"] == "так же"
    assert metadata["candidate_replacement"] == "также"
    assert metadata["count_toward_rule_quota"] is False
    assert metadata["loss_weight"] == 1.0


def test_hard_negative_candidate_labels_are_negative_in_training_features():
    text = "Он сделал так же, как раньше."
    candidate = _context_candidate(text, "context_tak_zhe", "так же", "также")
    generator = StaticCandidateGenerator({text: [candidate]})

    result = generate_atomic_hard_negatives(
        clean_rows=[{"text": text}],
        rule_ids=["context_tak_zhe"],
        candidate_generator=generator,
        min_per_rule=1,
        preferred_per_rule=1,
        seed=1,
        max_scan_rows=10,
        fallback_templates_enabled=False,
    )
    row = result.rows[0]

    feature = build_training_feature(
        row["source"],
        row["target"],
        tokenizer=DebugTokenizer(),
        punctuation_label_map={"NONE": 0},
        error_type_label_map={"keep": 0, "split_join": 1},
        max_length=16,
        max_candidates=4,
        candidate_generator=generator,
    )
    candidate_index = feature.candidate_rule_ids.index("context_tak_zhe")

    assert feature.candidate_labels[candidate_index] == 0.0


def test_fallback_template_is_rejected_when_candidate_is_not_generated(tmp_path):
    result = generate_atomic_hard_negatives(
        clean_rows=[],
        rule_ids=["context_chto_by"],
        candidate_generator=EmptyCandidateGenerator(),
        min_per_rule=1,
        preferred_per_rule=1,
        seed=1,
        max_scan_rows=10,
        fallback_templates_enabled=True,
    )

    assert result.rows == []
    assert result.counts_by_rule == {"context_chto_by": 0}
    assert any(
        row["rule_id"] == "context_chto_by" and row["reason"] == "candidate_not_generated_for_template"
        for row in result.rejection_rows
    )

    write_hard_negative_reports(result, tmp_path)
    coverage = pd.read_csv(tmp_path / "hard_negative_coverage_report.csv")
    rejection = pd.read_csv(tmp_path / "hard_negative_rejection_report.csv")

    assert coverage.loc[0, "rule_id"] == "context_chto_by"
    assert coverage.loc[0, "generated_count"] == 0
    assert coverage.loc[0, "status"] == "under_min"
    assert rejection.loc[0, "reason"] == "candidate_not_generated_for_template"


def test_counts_by_rule_counts_accepted_rows_per_target_rule():
    tak_text = "Он сделал так же, как раньше."
    za_text = "Он отвечает за то решение."
    generator = StaticCandidateGenerator(
        {
            tak_text: [_context_candidate(tak_text, "context_tak_zhe", "так же", "также")],
            za_text: [_context_candidate(za_text, "context_za_to", "за то", "зато")],
        }
    )

    result = generate_atomic_hard_negatives(
        clean_rows=[{"text": tak_text}, {"text": za_text}],
        rule_ids=["context_tak_zhe", "context_za_to"],
        candidate_generator=generator,
        min_per_rule=1,
        preferred_per_rule=1,
        seed=1,
        max_scan_rows=10,
        fallback_templates_enabled=False,
    )

    assert result.counts_by_rule == {
        "context_tak_zhe": 1,
        "context_za_to": 1,
    }
    assert {row["target_rule_id"] for row in result.rows} == {"context_tak_zhe", "context_za_to"}
