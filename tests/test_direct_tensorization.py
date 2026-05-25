from __future__ import annotations

from dataclasses import fields

from src.schema import GeneratedExample, WordToken
from src.schema.labels import gap_label_to_id, rule_tag_to_id, token_label_to_id
from src.training.tensorization import (
    DebugTokenizer,
    DirectTrainingFeature,
    build_direct_training_feature,
)


def test_generated_example_builds_direct_feature_shapes() -> None:
    feature = build_direct_training_feature(_example(), DebugTokenizer(), max_length=8)

    assert isinstance(feature, DirectTrainingFeature)
    assert len(feature.input_ids) == 8
    assert len(feature.attention_mask) == 8
    assert len(feature.offset_mapping) == 8
    assert len(feature.word_token_indices) == 8
    assert len(feature.word_token_mask) == 8
    assert len(feature.gap_left_indices) == 8
    assert len(feature.gap_right_indices) == 8
    assert len(feature.gap_mask) == 8
    assert len(feature.token_edit_label_ids) == 8
    assert len(feature.gap_label_ids) == 8
    assert len(feature.rule_tag_ids) == 8
    assert feature.word_token_mask[:3] == [True, True, True]
    assert feature.word_token_mask[3:] == [False] * 5
    assert feature.gap_mask[:3] == [True, True, True]
    assert feature.gap_mask[3:] == [False] * 5


def test_direct_feature_maps_schema_labels_to_ids() -> None:
    feature = build_direct_training_feature(_example(), DebugTokenizer(), max_length=8)

    assert feature.token_edit_label_ids[:3] == [
        token_label_to_id("KEEP"),
        token_label_to_id("SPLIT_NE_VERB"),
        token_label_to_id("KEEP"),
    ]
    assert feature.gap_label_ids[:3] == [
        gap_label_to_id("NONE"),
        gap_label_to_id("NONE"),
        gap_label_to_id("DOT"),
    ]
    assert feature.rule_tag_ids[:3] == [
        rule_tag_to_id("clean_identity"),
        rule_tag_to_id("ne_verb"),
        rule_tag_to_id("final_punctuation"),
    ]
    assert feature.token_edit_label_ids[3:] == [-100] * 5
    assert feature.gap_label_ids[3:] == [-100] * 5
    assert feature.rule_tag_ids[3:] == [-100] * 5
    assert feature.gap_left_indices[:3] == feature.word_token_indices[:3]
    assert feature.gap_right_indices[0] == feature.word_token_indices[1]
    assert feature.gap_right_indices[1] == feature.word_token_indices[2]
    assert feature.gap_right_indices[2] == -1


def test_direct_feature_has_no_candidate_fields() -> None:
    field_names = {field.name for field in fields(DirectTrainingFeature)}
    feature = build_direct_training_feature(_example(), DebugTokenizer(), max_length=8)

    assert not any("candidate" in name for name in field_names)
    assert not hasattr(feature, "candidate_spans")
    assert not hasattr(feature, "candidate_replacements")


def _example() -> GeneratedExample:
    source_text = "РћРЅ РЅРµР·РЅР°Р» РѕС‚РІРµС‚Р°"
    return GeneratedExample(
        source_text=source_text,
        target_text="РћРЅ РЅРµ Р·РЅР°Р» РѕС‚РІРµС‚Р°.",
        source_tokens=[
            WordToken(text="РћРЅ", start=0, end=2, lemma="РѕРЅ", pos="PRON"),
            WordToken(text="РЅРµР·РЅР°Р»", start=3, end=9, lemma="Р·РЅР°С‚СЊ", pos="VERB"),
            WordToken(text="РѕС‚РІРµС‚Р°", start=10, end=16, lemma="РѕС‚РІРµС‚", pos="NOUN"),
        ],
        token_edit_labels=["KEEP", "SPLIT_NE_VERB", "KEEP"],
        gap_labels=["NONE", "NONE", "DOT"],
        rule_ids=["clean_identity", "ne_verb", "final_punctuation"],
        primary_rule_id="ne_verb",
        mode="positive",
        explanation_ids=["ne_verb"],
        metadata={"unit": True},
    )

