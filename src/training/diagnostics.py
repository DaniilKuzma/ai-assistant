from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.training.tensorization import TrainingFeature


PUNCTUATION_CANDIDATE_TYPES = frozenset(
    {
        "punctuation_insert",
        "punctuation_delete",
        "punctuation_replace",
        "final_punctuation",
    }
)

TRAINING_LABEL_DISTRIBUTION_COLUMNS = [
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


def training_positive_counts(features: list[TrainingFeature]) -> dict[str, int]:
    counts = {
        "word_candidate_positive_count": 0,
        "spelling_positive_count": 0,
        "hyphen_positive_count": 0,
        "split_join_positive_count": 0,
        "punctuation_positive_count": 0,
    }
    for feature in features:
        for index, active in enumerate(feature.candidate_mask):
            if not active or feature.candidate_labels[index] <= 0:
                continue
            edit_type = feature.candidate_edit_types[index]
            if edit_type == "spelling":
                counts["spelling_positive_count"] += 1
            elif edit_type == "hyphen":
                counts["hyphen_positive_count"] += 1
            elif edit_type == "split_join":
                counts["split_join_positive_count"] += 1
            if edit_type and edit_type != "keep" and edit_type not in PUNCTUATION_CANDIDATE_TYPES:
                counts["word_candidate_positive_count"] += 1

        for label, active in zip(feature.punctuation_confidence_labels, feature.punctuation_gap_mask, strict=False):
            if active and label > 0:
                counts["punctuation_positive_count"] += 1
    return counts


def training_label_distribution_by_rule_frame(features: list[TrainingFeature]) -> pd.DataFrame:
    buckets: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(_label_bucket)
    for feature in features:
        split = str(feature.split or "train")
        for index, active in enumerate(feature.candidate_mask):
            if not active:
                continue
            rule_id = str(feature.candidate_rule_ids[index] or "")
            edit_type = str(feature.candidate_edit_types[index] or "")
            if not rule_id or edit_type == "keep":
                continue
            bucket = buckets[(rule_id, edit_type, split)]
            bucket["candidate_count"] += 1
            bucket["rank_sum"] += index + 1
            label = float(feature.candidate_labels[index])
            if label > 0:
                bucket["positive_label_count"] += 1
                bucket["positive_rank_sum"] += index + 1
                _append_example(bucket, feature, index)
            else:
                bucket["negative_label_count"] += 1
                if not bucket["examples"]:
                    _append_example(bucket, feature, index)

    rows: list[dict[str, Any]] = []
    for (rule_id, edit_type, split), bucket in sorted(buckets.items()):
        candidate_count = int(bucket["candidate_count"])
        positive_count = int(bucket["positive_label_count"])
        rank_denominator = positive_count or candidate_count
        rank_sum = float(bucket["positive_rank_sum"] or bucket["rank_sum"])
        rows.append(
            {
                "rule_id": rule_id,
                "edit_type": edit_type,
                "split": split,
                "candidate_count": candidate_count,
                "positive_label_count": positive_count,
                "negative_label_count": int(bucket["negative_label_count"]),
                "positive_rate": round(positive_count / candidate_count, 6) if candidate_count else 0.0,
                "avg_candidate_rank": round(rank_sum / rank_denominator, 3) if rank_denominator else 0.0,
                "examples": " | ".join(bucket["examples"]),
            }
        )
    return pd.DataFrame(rows, columns=TRAINING_LABEL_DISTRIBUTION_COLUMNS)


def write_training_label_distribution_by_rule(features: list[TrainingFeature], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    training_label_distribution_by_rule_frame(features).to_csv(output, index=False)


def _label_bucket() -> dict[str, Any]:
    return {
        "candidate_count": 0,
        "positive_label_count": 0,
        "negative_label_count": 0,
        "rank_sum": 0,
        "positive_rank_sum": 0,
        "examples": [],
    }


def _append_example(bucket: dict[str, Any], feature: TrainingFeature, index: int) -> None:
    examples = bucket["examples"]
    if len(examples) >= 3:
        return
    source = feature.candidate_sources[index]
    replacement = feature.candidate_replacements[index]
    examples.append(f"{source}->{replacement}")
