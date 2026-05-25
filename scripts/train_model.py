from __future__ import annotations

import argparse

from src.training.train import main as training_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the direct online correction model.")
    parser.add_argument("config", nargs="?")
    parser.add_argument("--config", dest="config_option")
    args, remaining = parser.parse_known_args()
    config = args.config or args.config_option or "configs/config.yaml"
    return training_main([config, *remaining])


if __name__ == "__main__":
    raise SystemExit(main())
