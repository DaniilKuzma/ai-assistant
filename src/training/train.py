from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


DISABLE_MODEL_TRAINING_ENV = "RUSSIAN_CORRECTOR_DISABLE_MODEL_TRAINING"
LEGACY_TRAINING_ENTRYPOINT_MESSAGE = (
    "Legacy training entrypoint has been removed. Direct online training will be implemented "
    "in src/training in the next migration step."
)


class TrainedModelCorrector:
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TrainedModelCorrector":
        raise RuntimeError(LEGACY_TRAINING_ENTRYPOINT_MESSAGE)


def train(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    raise RuntimeError(LEGACY_TRAINING_ENTRYPOINT_MESSAGE)


def evaluate_trained_model(
    config_path: str | Path = "configs/config.yaml",
    *,
    split: str = "test",
    corrector: Any | None = None,
) -> dict[str, Any]:
    raise RuntimeError(LEGACY_TRAINING_ENTRYPOINT_MESSAGE)


def _run_model_training(config: dict[str, Any], features: Any, **kwargs: Any) -> dict[str, Any]:
    raise RuntimeError(LEGACY_TRAINING_ENTRYPOINT_MESSAGE)


def _build_features(config: dict[str, Any], rows: list[dict[str, Any]]) -> list[Any]:
    raise RuntimeError(LEGACY_TRAINING_ENTRYPOINT_MESSAGE)


def _load_evaluation_rows(config: dict[str, Any], fallback_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raise RuntimeError(LEGACY_TRAINING_ENTRYPOINT_MESSAGE)


def _select_evaluation_corrector(
    config: dict[str, Any],
    *,
    model_training_ran: bool,
    model_training_disabled_source: str = "",
) -> tuple[Any, dict[str, Any]]:
    raise RuntimeError(LEGACY_TRAINING_ENTRYPOINT_MESSAGE)


def _build_evaluation_corrector(config: dict[str, Any], model_training_ran: bool) -> Any:
    raise RuntimeError(LEGACY_TRAINING_ENTRYPOINT_MESSAGE)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Removed legacy training entrypoint.")
    parser.add_argument("config", nargs="?", default="configs/config.yaml")
    args = parser.parse_args(argv)
    train(args.config)


if __name__ == "__main__":
    main()
