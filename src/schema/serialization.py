from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.schema.examples import GeneratedExample


def read_jsonl_examples(path: str | Path) -> list[GeneratedExample]:
    examples: list[GeneratedExample] = []
    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                examples.append(GeneratedExample.from_json(stripped))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid GeneratedExample JSONL at {path_obj}:{line_number}: {exc}") from exc
    return examples


def write_jsonl_examples(path: str | Path, examples: Iterable[GeneratedExample]) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(example.to_json())
            handle.write("\n")


def validate_jsonl_examples(path: str | Path) -> int:
    return len(read_jsonl_examples(path))


__all__ = ["read_jsonl_examples", "validate_jsonl_examples", "write_jsonl_examples"]
