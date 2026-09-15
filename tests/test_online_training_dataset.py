from __future__ import annotations

from itertools import islice
from pathlib import Path

from src.config.load_config import load_config
from src.training.online_dataset import OnlineGrammarDataset
from src.training.tensorization import DebugTokenizer, DirectBatchCollator, DirectTrainingFeature


ROOT = Path(__file__).resolve().parents[1]


def test_online_dataset_yields_direct_features_without_train_csv() -> None:
    train_csv = ROOT / "data" / "processed" / "train.csv"
    assert not train_csv.exists()

    dataset = OnlineGrammarDataset(_config(), DebugTokenizer(), split="train", seed=13)
    features = list(islice(iter(dataset), 5))

    assert len(features) == 5
    assert all(isinstance(feature, DirectTrainingFeature) for feature in features)
    assert not train_csv.exists()


def test_direct_collator_returns_tensors_for_online_features() -> None:
    features = list(islice(iter(OnlineGrammarDataset(_config(), DebugTokenizer(), seed=13)), 2))

    batch = DirectBatchCollator().collate(features)

    assert batch["input_ids"].shape == (2, 24)
    assert batch["attention_mask"].shape == (2, 24)
    assert batch["word_token_indices"].shape == (2, 24)
    assert batch["word_token_mask"].shape == (2, 24)
    assert batch["gap_left_indices"].shape == (2, 24)
    assert batch["gap_right_indices"].shape == (2, 24)
    assert batch["gap_mask"].shape == (2, 24)
    assert batch["labels"]["token_edit_label_ids"].shape == (2, 24)
    assert batch["labels"]["gap_label_ids"].shape == (2, 24)
    assert batch["labels"]["rule_tag_ids"].shape == (2, 24)
    assert batch["labels"]["sample_weight"].shape == (2,)


def _config() -> dict:
    config = load_config(ROOT / "configs" / "config.yaml")
    config["model"] = {**config.get("model", {}), "max_sequence_length": 24}
    config["generation"] = {**config.get("generation", {}), "samples_per_epoch": 5}
    config["training"] = {**config.get("training", {}), "num_workers": 0}
    return config

