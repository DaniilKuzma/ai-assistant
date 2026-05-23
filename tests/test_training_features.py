import json
from pathlib import Path

import pandas as pd
import torch

from src.candidates.candidate_generator import CandidateGenerator
from src.training.diagnostics import (
    training_label_distribution_by_rule_frame,
    training_positive_counts,
    write_training_label_distribution_by_rule,
)
from src.training.tensorization import DebugTokenizer, EditBatchCollator, build_features_from_rows, build_training_feature


def test_training_feature_marks_gold_candidate_and_punctuation_labels():
    tokenizer = DebugTokenizer()

    feature = build_training_feature(
        "Я незнаю что делать",
        "Я не знаю, что делать.",
        tokenizer=tokenizer,
        punctuation_label_map={"NONE": 0, "COMMA": 1, "DOT": 2},
        error_type_label_map={"keep": 0, "split_join": 1, "punctuation": 2, "final_punctuation": 3},
        max_length=16,
        max_candidates=8,
    )

    assert feature.sample_weight == 1.0
    gold_index = feature.candidate_replacements.index("не знаю")
    assert feature.candidate_labels[gold_index] == 1.0
    assert 1 in feature.punctuation_labels
    assert 2 in feature.punctuation_labels
    assert sum(feature.punctuation_gap_mask) == 4
    assert sum(feature.punctuation_mask) == 4
    assert feature.punctuation_gap_indices[:4] == [0, 1, 2, 3]
    assert feature.punctuation_confidence_labels[:4] == [0.0, 1.0, 0.0, 1.0]
    assert feature.punctuation_error_type_labels[:4] == [0, 2, 0, 3]


def test_training_feature_candidate_budget_keeps_late_edit_candidate():
    tokenizer = DebugTokenizer()
    source = " ".join([f"слово{i}" for i in range(30)]) + " недумаю"
    target = " ".join([f"слово{i}" for i in range(30)]) + " не думаю"

    feature = build_training_feature(
        source,
        target,
        tokenizer=tokenizer,
        punctuation_label_map={"NONE": 0},
        error_type_label_map={"keep": 0, "split_join": 1},
        max_length=80,
        max_candidates=16,
    )

    assert "не думаю" in [replacement.lower() for replacement in feature.candidate_replacements]


def test_training_feature_uses_word_gap_mask_instead_of_token_mask_for_punctuation():
    tokenizer = DebugTokenizer()

    feature = build_training_feature(
        "Я думаю, что это важно.",
        "Я думаю что это важно.",
        tokenizer=tokenizer,
        punctuation_label_map={"NONE": 0, "COMMA": 1, "DOT": 2},
        error_type_label_map={"keep": 0, "punctuation": 1, "final_punctuation": 2},
        max_length=16,
        max_candidates=8,
    )

    assert sum(feature.attention_mask) == 7
    assert sum(feature.punctuation_gap_mask) == 5
    assert feature.punctuation_labels[1] == 0
    assert feature.punctuation_confidence_labels[1] == 1.0
    assert feature.punctuation_error_type_labels[1] == 1
    assert feature.punctuation_labels[-1] == 0


def test_training_feature_marks_punctuation_action_labels():
    tokenizer = DebugTokenizer()
    punctuation_labels = {"NONE": 0, "COMMA": 1, "DOT": 2, "COLON": 3}
    action_labels = {"KEEP_NONE": 0, "KEEP_EXISTING": 1, "INSERT": 2, "DELETE": 3, "REPLACE": 4}
    error_labels = {"keep": 0, "punctuation": 1, "final_punctuation": 2}

    insert_feature = build_training_feature(
        "Я думаю что это важно.",
        "Я думаю, что это важно.",
        tokenizer=tokenizer,
        punctuation_label_map=punctuation_labels,
        punctuation_action_label_map=action_labels,
        error_type_label_map=error_labels,
        max_length=16,
        max_candidates=8,
    )
    delete_feature = build_training_feature(
        "Я думаю, что это важно.",
        "Я думаю что это важно.",
        tokenizer=tokenizer,
        punctuation_label_map=punctuation_labels,
        punctuation_action_label_map=action_labels,
        error_type_label_map=error_labels,
        max_length=16,
        max_candidates=8,
    )
    replace_feature = build_training_feature(
        "Он сказал, привет.",
        "Он сказал: привет.",
        tokenizer=tokenizer,
        punctuation_label_map=punctuation_labels,
        punctuation_action_label_map=action_labels,
        error_type_label_map=error_labels,
        max_length=16,
        max_candidates=8,
    )
    keep_feature = build_training_feature(
        "Я думаю, что это важно.",
        "Я думаю, что это важно.",
        tokenizer=tokenizer,
        punctuation_label_map=punctuation_labels,
        punctuation_action_label_map=action_labels,
        error_type_label_map=error_labels,
        max_length=16,
        max_candidates=8,
    )

    assert insert_feature.punctuation_action_labels[1] == action_labels["INSERT"]
    assert delete_feature.punctuation_action_labels[1] == action_labels["DELETE"]
    assert replace_feature.punctuation_action_labels[1] == action_labels["REPLACE"]
    assert keep_feature.punctuation_action_labels[0] == action_labels["KEEP_NONE"]
    assert keep_feature.punctuation_action_labels[1] == action_labels["KEEP_EXISTING"]


def test_training_feature_uses_aligned_edit_span_for_repeated_candidates():
    tokenizer = DebugTokenizer()

    feature = build_training_feature(
        "Я незнаю и незнаю что делать",
        "Я не знаю и незнаю, что делать.",
        tokenizer=tokenizer,
        punctuation_label_map={"NONE": 0, "COMMA": 1, "DOT": 2},
        error_type_label_map={"keep": 0, "split_join": 1, "punctuation": 2, "final_punctuation": 3},
        max_length=20,
        max_candidates=12,
    )
    split_candidate_indexes = [
        index
        for index, replacement in enumerate(feature.candidate_replacements)
        if replacement == "не знаю" and feature.candidate_mask[index]
    ]

    assert len(split_candidate_indexes) == 2
    assert [feature.candidate_labels[index] for index in split_candidate_indexes] == [1.0, 0.0]


def test_training_feature_encodes_candidate_replacements():
    tokenizer = DebugTokenizer()

    feature = build_training_feature(
        "Я незнаю что делать",
        "Я не знаю, что делать.",
        tokenizer=tokenizer,
        punctuation_label_map={"NONE": 0, "COMMA": 1, "DOT": 2},
        error_type_label_map={"keep": 0, "split_join": 1, "punctuation": 2, "final_punctuation": 3},
        max_length=16,
        max_candidates=8,
    )
    replacement_index = feature.candidate_replacements.index("не знаю")

    assert feature.candidate_replacement_mask[replacement_index]
    assert any(token_id > 0 for token_id in feature.candidate_replacement_ids[replacement_index])


def test_training_feature_preserves_ranked_candidate_metadata():
    tokenizer = DebugTokenizer()

    feature = build_training_feature(
        "Он учится каждый день.",
        "Он учится каждый день.",
        tokenizer=tokenizer,
        punctuation_label_map={"NONE": 0, "DOT": 1},
        error_type_label_map={"keep": 0, "spelling": 1},
        max_length=16,
        max_candidates=8,
    )
    candidate_index = next(
        index
        for index, replacement in enumerate(feature.candidate_replacements)
        if replacement.lower() == "учиться" and feature.candidate_mask[index]
    )

    assert feature.candidate_rule_ids[candidate_index] == "tsya_soft_insert"
    assert feature.candidate_modes[candidate_index] == "model_required"
    assert feature.candidate_requires_model[candidate_index] is True
    assert feature.candidate_requires_scoring[candidate_index] is True


def test_training_feature_pads_candidate_metadata_with_inactive_mask():
    tokenizer = DebugTokenizer()

    feature = build_training_feature(
        "Жызнь.",
        "Жизнь.",
        tokenizer=tokenizer,
        punctuation_label_map={"NONE": 0, "DOT": 1},
        error_type_label_map={"keep": 0, "spelling": 1},
        max_length=8,
        max_candidates=6,
    )

    assert len(feature.candidate_rule_ids) == 6
    assert len(feature.candidate_modes) == 6
    assert len(feature.candidate_requires_model) == 6
    assert len(feature.candidate_requires_scoring) == 6
    for index, active in enumerate(feature.candidate_mask):
        if active:
            continue
        assert feature.candidate_rule_ids[index] == ""
        assert feature.candidate_modes[index] == "deterministic"
        assert feature.candidate_requires_model[index] is False
        assert feature.candidate_requires_scoring[index] is False


def test_clean_training_feature_marks_keep_candidates_positive():
    tokenizer = DebugTokenizer()

    feature = build_training_feature(
        "Чистый текст.",
        "Чистый текст.",
        tokenizer=tokenizer,
        punctuation_label_map={"NONE": 0, "DOT": 2},
        error_type_label_map={"keep": 0},
        max_length=8,
        max_candidates=4,
    )

    active_labels = [label for label, mask in zip(feature.candidate_labels, feature.candidate_mask, strict=False) if mask]
    assert active_labels
    assert all(label == 1.0 for label in active_labels)


def test_batch_collator_returns_tensors_with_candidate_masks():
    tokenizer = DebugTokenizer()
    feature = build_training_feature(
        "Я незнаю что делать",
        "Я не знаю, что делать.",
        tokenizer=tokenizer,
        punctuation_label_map={"NONE": 0, "COMMA": 1, "DOT": 2},
        error_type_label_map={"keep": 0, "split_join": 1, "punctuation": 2, "final_punctuation": 3},
        max_length=12,
        max_candidates=6,
    )

    batch = EditBatchCollator().collate([feature])

    assert batch["input_ids"].shape == torch.Size([1, 12])
    assert batch["candidate_spans"].shape == torch.Size([1, 6, 2])
    assert batch["candidate_replacement_ids"].shape[0:2] == torch.Size([1, 6])
    assert batch["punctuation_gap_indices"].shape == torch.Size([1, 12])
    assert batch["punctuation_right_gap_indices"].shape == torch.Size([1, 12])
    assert batch["labels"]["candidate_labels"].shape == torch.Size([1, 6])
    assert batch["labels"]["sample_weight"].shape == torch.Size([1])
    assert batch["labels"]["sample_weight"].item() == 1.0
    assert batch["labels"]["punctuation_mask"].sum().item() == 4
    assert batch["labels"]["punctuation_action_labels"].shape == torch.Size([1, 12])
    assert batch["labels"]["punctuation_confidence_labels"].shape == torch.Size([1, 12])
    assert batch["labels"]["punctuation_error_type_labels"].shape == torch.Size([1, 12])
    assert batch["labels"]["candidate_mask"].sum().item() >= 2
    assert batch["candidate_rule_ids"] == [feature.candidate_rule_ids]
    assert batch["candidate_modes"] == [feature.candidate_modes]
    assert batch["candidate_requires_model"] == [feature.candidate_requires_model]
    assert batch["candidate_requires_scoring"] == [feature.candidate_requires_scoring]


def test_build_features_supports_progress_option():
    tokenizer = DebugTokenizer()

    features = build_features_from_rows(
        [{"source": "Я незнаю что делать", "target": "Я не знаю, что делать."}],
        tokenizer=tokenizer,
        punctuation_label_map={"NONE": 0, "COMMA": 1, "DOT": 2},
        error_type_label_map={"keep": 0, "split_join": 1, "punctuation": 2, "final_punctuation": 3},
        max_length=12,
        max_candidates=6,
        show_progress=True,
    )

    assert len(features) == 1
    assert features[0].source == "Я незнаю что делать"


def test_build_features_reads_sample_weight_from_row_loss_weight():
    tokenizer = DebugTokenizer()

    features = build_features_from_rows(
        [{"source": "Я незнаю что делать", "target": "Я не знаю, что делать.", "loss_weight": 0.4}],
        tokenizer=tokenizer,
        punctuation_label_map={"NONE": 0, "COMMA": 1, "DOT": 2},
        error_type_label_map={"keep": 0, "split_join": 1, "punctuation": 2, "final_punctuation": 3},
        max_length=12,
        max_candidates=6,
    )

    assert features[0].sample_weight == 0.4


def test_build_features_reads_sample_weight_from_metadata_fallback():
    tokenizer = DebugTokenizer()

    features = build_features_from_rows(
        [
            {
                "source": "Я незнаю что делать",
                "target": "Я не знаю, что делать.",
                "metadata": json.dumps({"loss_weight": 0.6}, ensure_ascii=False),
            },
            {
                "source": "Кто то пришел",
                "target": "Кто-то пришел.",
                "metadata": {"loss_weight": 0.7},
            },
        ],
        tokenizer=tokenizer,
        punctuation_label_map={"NONE": 0, "COMMA": 1, "DOT": 2},
        error_type_label_map={"keep": 0, "split_join": 1, "punctuation": 2, "final_punctuation": 3, "hyphen": 4},
        max_length=12,
        max_candidates=6,
    )

    assert features[0].sample_weight == 0.6
    assert features[1].sample_weight == 0.7


def test_build_features_clamps_invalid_sample_weight_and_profiles_warning():
    class Profiler:
        def __init__(self):
            self.records = []

        def should_profile(self, _row_index):
            return True

        def record(self, row):
            self.records.append(row)

    profiler = Profiler()

    features = build_features_from_rows(
        [{"source": "Я незнаю что делать", "target": "Я не знаю, что делать.", "loss_weight": -0.1}],
        tokenizer=DebugTokenizer(),
        punctuation_label_map={"NONE": 0, "COMMA": 1, "DOT": 2},
        error_type_label_map={"keep": 0, "split_join": 1, "punctuation": 2, "final_punctuation": 3},
        max_length=12,
        max_candidates=6,
        profiler=profiler,
    )

    assert features[0].sample_weight == 1.0
    assert profiler.records[0]["invalid_sample_weight"] is True
    assert profiler.records[0]["raw_sample_weight"] == -0.1


def test_build_features_clamps_non_finite_sample_weight_to_default():
    features = build_features_from_rows(
        [{"source": "Я незнаю что делать", "target": "Я не знаю, что делать.", "loss_weight": float("nan")}],
        tokenizer=DebugTokenizer(),
        punctuation_label_map={"NONE": 0, "COMMA": 1, "DOT": 2},
        error_type_label_map={"keep": 0, "split_join": 1, "punctuation": 2, "final_punctuation": 3},
        max_length=12,
        max_candidates=6,
    )

    assert features[0].sample_weight == 1.0


def test_stress_row_loss_weight_passes_through_collator():
    features = build_features_from_rows(
        [
            {
                "source": "Я незнаю что делать",
                "target": "Я не знаю, что делать.",
                "dataset_layer": "stress_multi_error",
                "is_stress": True,
                "loss_weight": 0.4,
            }
        ],
        tokenizer=DebugTokenizer(),
        punctuation_label_map={"NONE": 0, "COMMA": 1, "DOT": 2},
        error_type_label_map={"keep": 0, "split_join": 1, "punctuation": 2, "final_punctuation": 3},
        max_length=12,
        max_candidates=6,
    )

    batch = EditBatchCollator().collate(features)

    assert features[0].sample_weight == 0.4
    assert batch["labels"]["sample_weight"].shape == torch.Size([1])
    assert torch.isclose(batch["labels"]["sample_weight"], torch.tensor([0.4])).all()


def test_word_candidate_positive_labels_exist():
    tokenizer = DebugTokenizer()
    labels = {
        "punctuation": {"NONE": 0, "COMMA": 1, "DOT": 2},
        "punctuation_actions": {"KEEP_NONE": 0, "KEEP_EXISTING": 1, "INSERT": 2},
        "error_types": {"keep": 0, "spelling": 1, "punctuation": 2, "split_join": 3, "hyphen": 4, "final_punctuation": 6},
    }
    dictionary_generator = CandidateGenerator(
        dictionary_lexicon=["молоко", "корова", "библиотека"],
        dictionary_min_score=85,
        syntax_provider=lambda _text: (),
    )

    features = [
        build_training_feature(
            "Редактор проверил молокл.",
            "Редактор проверил молоко.",
            tokenizer=tokenizer,
            punctuation_label_map=labels["punctuation"],
            punctuation_action_label_map=labels["punctuation_actions"],
            error_type_label_map=labels["error_types"],
            max_length=24,
            max_candidates=16,
            candidate_generator=dictionary_generator,
        ),
        build_training_feature(
            "Кто то пришел",
            "Кто-то пришел.",
            tokenizer=tokenizer,
            punctuation_label_map=labels["punctuation"],
            punctuation_action_label_map=labels["punctuation_actions"],
            error_type_label_map=labels["error_types"],
            max_length=24,
            max_candidates=16,
        ),
        build_training_feature(
            "Я незнаю что делать",
            "Я не знаю, что делать.",
            tokenizer=tokenizer,
            punctuation_label_map=labels["punctuation"],
            punctuation_action_label_map=labels["punctuation_actions"],
            error_type_label_map=labels["error_types"],
            max_length=24,
            max_candidates=16,
        ),
    ]

    counts = training_positive_counts(features)

    assert counts["spelling_positive_count"] > 0
    assert counts["hyphen_positive_count"] > 0
    assert counts["split_join_positive_count"] > 0
    assert counts["punctuation_positive_count"] > 0


def test_training_label_distribution_contains_spelling_hyphen_split_join_positives(tmp_path: Path):
    tokenizer = DebugTokenizer()
    labels = {
        "punctuation": {"NONE": 0, "COMMA": 1, "DOT": 2},
        "punctuation_actions": {"KEEP_NONE": 0, "KEEP_EXISTING": 1, "INSERT": 2},
        "error_types": {"keep": 0, "spelling": 1, "punctuation": 2, "split_join": 3, "hyphen": 4, "final_punctuation": 6},
    }
    generator = CandidateGenerator(
        dictionary_lexicon=["молоко", "корова", "библиотека"],
        dictionary_min_score=85,
        syntax_provider=lambda _text: (),
    )
    features = build_features_from_rows(
        [
            {"source": "Редактор проверил молокл.", "target": "Редактор проверил молоко.", "split": "train"},
            {"source": "Кто то пришел", "target": "Кто-то пришел.", "split": "train"},
            {"source": "Я незнаю что делать", "target": "Я не знаю, что делать.", "split": "train"},
        ],
        tokenizer=tokenizer,
        punctuation_label_map=labels["punctuation"],
        punctuation_action_label_map=labels["punctuation_actions"],
        error_type_label_map=labels["error_types"],
        max_length=24,
        max_candidates=16,
        candidate_generator=generator,
    )

    output_path = tmp_path / "training_label_distribution_by_rule.csv"
    write_training_label_distribution_by_rule(features, output_path)
    frame = training_label_distribution_by_rule_frame(features)
    written = pd.read_csv(output_path)

    assert list(written.columns) == [
        "rule_id",
        "edit_type",
        "split",
        "candidate_count",
        "positive_label_count",
        "negative_label_count",
        "positive_rate",
        "avg_candidate_rank",
        "examples",
    ]
    by_type = frame.groupby("edit_type")["positive_label_count"].sum()
    assert by_type["spelling"] > 0
    assert by_type["hyphen"] > 0
    assert by_type["split_join"] > 0
