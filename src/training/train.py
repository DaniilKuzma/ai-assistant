from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.training.trainer import train_model


def train(
    config_path: str | Path = "configs/config.yaml",
    *,
    overrides: dict[str, Any] | None = None,
    debug_model: bool = False,
) -> dict[str, Any]:
    return train_model(config_path, overrides=overrides, debug_model=debug_model)


def evaluate_trained_model(*args: Any, **kwargs: Any) -> dict[str, Any]:
    raise RuntimeError("Legacy candidate-aware evaluation entrypoint has been removed from direct online training.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train direct online Russian edit corrector.")
    parser.add_argument("config", nargs="?", default="configs/config.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--debug-model", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--samples-per-epoch", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--no-mixed-precision", action="store_true")
    args = parser.parse_args(argv)

    overrides: dict[str, Any] = {}
    for key in (
        "steps",
        "epochs",
        "batch_size",
        "samples_per_epoch",
        "learning_rate",
        "gradient_accumulation_steps",
    ):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value
    if args.smoke:
        overrides["smoke"] = True
    if args.no_mixed_precision:
        overrides["mixed_precision"] = False

    result = train_model(args.config, overrides=overrides, debug_model=args.debug_model)
    print(json.dumps({"summary_path": result["summary_path"], "heads_path": result["heads_path"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
