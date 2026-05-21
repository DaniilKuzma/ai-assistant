from __future__ import annotations

import argparse

from src.training.train import train


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the canonical correction model.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
