from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.candidates.morphology import morph_analyzer


RUSSIAN_LEXICON_RE = re.compile(r"^[а-яё]+(?:-[а-яё]+)*$")
DEFAULT_OUTPUT_PATH = Path("data/processed/russian_lexicon.txt")


def iter_lexicon_entries(*, max_entries: int | None = None) -> list[str]:
    entries: set[str] = set()
    for item in morph_analyzer().dictionary.iter_known_words():
        word = _word_from_dictionary_item(item)
        normalized = word.strip().lower().replace("ѐ", "ё")
        if not RUSSIAN_LEXICON_RE.fullmatch(normalized):
            continue
        entries.add(normalized)
        if max_entries is not None and len(entries) >= max_entries:
            break
    return sorted(entries)


def build_lexicon(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    max_entries: int | None = None,
) -> dict[str, Any]:
    entries = iter_lexicon_entries(max_entries=max_entries)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(entries) + ("\n" if entries else ""), encoding="utf-8")
    return {"path": str(output), "entries": len(entries), "written": len(entries)}


def _word_from_dictionary_item(item: object) -> str:
    if isinstance(item, tuple) and item:
        return str(item[0])
    return str(item)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normalized Russian lexicon from pymorphy OpenCorpora forms.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--max-entries", type=int, default=None)
    args = parser.parse_args()
    result = build_lexicon(args.output, max_entries=args.max_entries)
    print(f"wrote {result['entries']} entries to {result['path']}")


if __name__ == "__main__":
    main()
