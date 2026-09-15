from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence


@lru_cache(maxsize=16)
def load_dictionary_lexicon(path: str | Path, max_entries: int | None = None) -> tuple[str, ...]:
    """Load and cache a normalized UTF-8 dictionary lexicon."""

    lexicon_path = Path(path)
    if not lexicon_path.exists():
        return ()

    limit = None if max_entries is None else max(0, int(max_entries))
    entries: list[str] = []
    seen: set[str] = set()
    with lexicon_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            word = line.strip().lower()
            if not word or word in seen:
                continue
            seen.add(word)
            entries.append(word)
            if limit is not None and len(entries) >= limit:
                break
    return tuple(entries)


@dataclass(frozen=True)
class FileDictionaryProvider:
    path: str | Path
    max_entries: int | None = None

    def get_lexicon(self) -> Sequence[str]:
        return load_dictionary_lexicon(self.path, self.max_entries)


def dictionary_provider_from_config(config: dict[str, Any]) -> FileDictionaryProvider | None:
    dictionary_config = config.get("dictionary", {})
    if not dictionary_config or not bool(dictionary_config.get("enabled", False)):
        return None

    path = dictionary_config.get("lexicon_path")
    if not path:
        return None

    max_entries = dictionary_config.get("max_entries")
    return FileDictionaryProvider(path=path, max_entries=None if max_entries is None else int(max_entries))
