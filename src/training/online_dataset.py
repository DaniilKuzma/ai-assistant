from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset, IterableDataset, get_worker_info

from src.grammar_gen.factory import online_generator_from_config
from src.schema.serialization import read_jsonl_examples
from src.training.tensorization import DirectTrainingFeature, build_direct_training_feature


class OnlineGrammarDataset(IterableDataset[DirectTrainingFeature]):
    def __init__(
        self,
        config: Mapping[str, Any],
        tokenizer: Any,
        split: str = "train",
        seed: int = 13,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        self.seed = int(seed) + _split_seed_offset(split)
        self.max_length = _max_sequence_length(config)
        self.samples_per_epoch = _samples_per_epoch(config)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = max(0, int(epoch))

    def __iter__(self) -> Iterator[DirectTrainingFeature]:
        worker = get_worker_info()
        worker_id = int(worker.id) if worker is not None else 0
        worker_count = int(worker.num_workers) if worker is not None else 1
        generator = online_generator_from_config(self.config, seed=self.seed)

        if self.samples_per_epoch is None:
            index = worker_id
            while True:
                yield build_direct_training_feature(
                    generator.sample_by_index(index),
                    self.tokenizer,
                    self.max_length,
                )
                index += worker_count

        epoch_start = self._epoch * self.samples_per_epoch
        for local_index in range(worker_id, self.samples_per_epoch, worker_count):
            yield build_direct_training_feature(
                generator.sample_by_index(epoch_start + local_index),
                self.tokenizer,
                self.max_length,
            )
        self._epoch += 1


class FrozenJsonlDataset(Dataset[DirectTrainingFeature]):
    def __init__(
        self,
        path_or_config: str | Path | Mapping[str, Any],
        tokenizer: Any,
        split: str = "val",
    ) -> None:
        self.path = _resolve_jsonl_path(path_or_config, split)
        self.tokenizer = tokenizer
        self.split = split
        self.max_length = _max_sequence_length(path_or_config) if isinstance(path_or_config, Mapping) else 128
        self.examples = read_jsonl_examples(self.path) if self.path.exists() else []

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> DirectTrainingFeature:
        return build_direct_training_feature(
            self.examples[index],
            self.tokenizer,
            self.max_length,
        )


def _max_sequence_length(config: Mapping[str, Any]) -> int:
    model = config.get("model", {}) if isinstance(config, Mapping) else {}
    raw_value = model.get("max_sequence_length", 128) if isinstance(model, Mapping) else 128
    return max(1, int(raw_value))


def _samples_per_epoch(config: Mapping[str, Any]) -> int | None:
    generation = config.get("generation", {}) if isinstance(config, Mapping) else {}
    raw_value = generation.get("samples_per_epoch") if isinstance(generation, Mapping) else None
    if raw_value is None:
        return None
    value = int(raw_value)
    return value if value > 0 else None


def _resolve_jsonl_path(path_or_config: str | Path | Mapping[str, Any], split: str) -> Path:
    if isinstance(path_or_config, Mapping):
        paths = path_or_config.get("paths", {})
        generation = path_or_config.get("generation", {})
        frozen_eval = generation.get("frozen_eval", {}) if isinstance(generation, Mapping) else {}
        raw_dir = None
        if isinstance(paths, Mapping):
            raw_dir = paths.get("generated_eval_dir")
        if raw_dir is None and isinstance(frozen_eval, Mapping):
            raw_dir = frozen_eval.get("output_dir")
        return Path(str(raw_dir or "data/generated_eval")) / f"{split}.jsonl"

    path = Path(path_or_config)
    return path / f"{split}.jsonl" if path.suffix != ".jsonl" else path


def _split_seed_offset(split: str) -> int:
    return {
        "train": 0,
        "val": 1_000_000,
        "test": 2_000_000,
        "regression": 3_000_000,
    }.get(split, 0)


__all__ = ["FrozenJsonlDataset", "OnlineGrammarDataset"]
