import torch

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

    gold_index = feature.candidate_replacements.index("не знаю")
    assert feature.candidate_labels[gold_index] == 1.0
    assert 1 in feature.punctuation_labels
    assert 2 in feature.punctuation_labels


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
    assert batch["labels"]["candidate_labels"].shape == torch.Size([1, 6])
    assert batch["labels"]["candidate_mask"].sum().item() >= 2


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
