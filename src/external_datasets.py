"""External real-pair datasets for V10 training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import pandas as pd

from edit_labels import ACTION_KEEP, build_training_example
from quality_guard import levenshtein_distance
from text_utils import normalize_word, word_tokens


AI_FOREVER_REPO = "ai-forever/spellcheck_punctuation_benchmark"
RUSPELLGOLD_REPO = "RussianNLP/RuSpellGold"
HF_DATASET_BASE = "https://huggingface.co/datasets"
REAL_PAIR_COLUMNS = [
    "error_text",
    "correct_text",
    "difficulty",
    "error_types",
    "source_kind",
    "source_dataset",
    "source_domain",
]

AI_FOREVER_TRAIN_FILES: tuple[tuple[str, str, str], ...] = (
    ("RUSpellRU", "train", "data/RUSpellRU/train.json"),
    ("MultidomainGold", "train", "data/MultidomainGold/train.json"),
)

AI_FOREVER_TEST_FILES: tuple[tuple[str, str, str], ...] = (
    ("RUSpellRU", "test", "data/RUSpellRU/test.json"),
    ("MultidomainGold", "test", "data/MultidomainGold/test.json"),
    ("MedSpellchecker", "test", "data/MedSpellchecker/test.json"),
    ("GitHubTypoCorpusRu", "test", "data/GitHubTypoCorpusRu/test.json"),
)

RUSPELLGOLD_TEST_FILES: tuple[tuple[str, str, str], ...] = (
    ("RuSpellGold", "test", "data/complete_test/test.json"),
)


@dataclass(frozen=True)
class ExternalPair:
    error_text: str
    correct_text: str
    source_dataset: str
    source_domain: str = ""


def normalize_yo(text: str) -> str:
    """Normalize yo to avoid forcing optional ё restoration as a model target."""
    return str(text).replace("Ё", "Е").replace("ё", "е")


def cache_hf_file(repo: str, remote_path: str, cache_dir: str | Path = "data/external") -> Path:
    """Download one HF dataset file into a local cache and return its path."""
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    filename = f"{repo.replace('/', '__')}__{remote_path.replace('/', '__')}"
    local_path = cache_root / filename
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    quoted_path = urllib.parse.quote(remote_path)
    url = f"{HF_DATASET_BASE}/{repo}/resolve/main/{quoted_path}"
    with urllib.request.urlopen(url, timeout=120) as response:
        local_path.write_bytes(response.read())
    return local_path


def read_json_records(path: str | Path) -> Iterator[Mapping[str, object]]:
    """Read either JSONL or a JSON list/object file."""
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return
    if stripped[0] in "[{":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            for row in parsed:
                if isinstance(row, Mapping):
                    yield row
            return
        if isinstance(parsed, Mapping):
            yield parsed
            return

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if isinstance(row, Mapping):
            yield row


def extract_pair(row: Mapping[str, object], source_dataset: str) -> ExternalPair | None:
    source = row.get("source", row.get("sources", row.get("incorrect")))
    correction = row.get("correction", row.get("corrections", row.get("correct")))
    if source is None or correction is None:
        return None
    domain = row.get("domain", row.get("source_domain", ""))
    return ExternalPair(
        error_text=_normalize_external_text(str(source)),
        correct_text=_normalize_external_text(str(correction)),
        source_dataset=source_dataset,
        source_domain=str(domain or ""),
    )


def load_external_pairs(
    repo: str,
    files: Sequence[tuple[str, str, str]],
    *,
    cache_dir: str | Path = "data/external",
) -> list[ExternalPair]:
    pairs: list[ExternalPair] = []
    for config_name, split_name, remote_path in files:
        path = cache_hf_file(repo, remote_path, cache_dir=cache_dir)
        source_dataset = f"{repo}:{config_name}:{split_name}"
        for row in read_json_records(path):
            pair = extract_pair(row, source_dataset)
            if pair is not None:
                pairs.append(pair)
    return pairs


def filter_real_pairs(
    pairs: Iterable[ExternalPair],
    *,
    max_chars: int = 240,
    max_tokens: int = 96,
    require_edit_compatible: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, object]] = []
    stats: dict[str, int] = {"input": 0, "kept": 0}

    for pair in pairs:
        stats["input"] += 1
        reason = real_pair_filter_reason(
            pair.error_text,
            pair.correct_text,
            max_chars=max_chars,
            max_tokens=max_tokens,
            require_edit_compatible=require_edit_compatible,
        )
        if reason:
            stats[reason] = stats.get(reason, 0) + 1
            continue

        labels = classify_real_error_types(pair.error_text, pair.correct_text, max_tokens=max_tokens)
        rows.append(
            {
                "error_text": pair.error_text,
                "correct_text": pair.correct_text,
                "difficulty": "real_spellcheck_punctuation",
                "error_types": "|".join(labels),
                "source_kind": "real",
                "source_dataset": pair.source_dataset,
                "source_domain": pair.source_domain,
            }
        )
        stats["kept"] += 1

    return pd.DataFrame(rows, columns=REAL_PAIR_COLUMNS), stats


def real_pair_filter_reason(
    error_text: str,
    correct_text: str,
    *,
    max_chars: int = 240,
    max_tokens: int = 96,
    require_edit_compatible: bool = True,
) -> str:
    error = str(error_text or "").strip()
    correct = str(correct_text or "").strip()
    if not error or not correct:
        return "empty"
    if error == correct:
        return "identical"
    if _has_spacing_difference(error, correct):
        return "spacing_only"
    if len(error) > max_chars or len(correct) > max_chars:
        return "too_long"

    source_words = word_tokens(error, include_numbers=True)
    target_words = word_tokens(correct, include_numbers=True)
    if not source_words or not target_words:
        return "no_words"
    if len(source_words) > max_tokens or len(target_words) > max_tokens:
        return "too_many_tokens"

    token_delta = abs(len(source_words) - len(target_words))
    if token_delta > 3 and token_delta / max(len(target_words), 1) > 0.20:
        return "large_token_count_shift"

    similarity = 1.0 - (levenshtein_distance(error, correct) / max(len(correct), 1))
    if similarity < 0.72:
        return "heavy_rewrite"

    changed_ratio = _changed_word_ratio(source_words, target_words)
    if changed_ratio > 0.55:
        return "heavy_word_rewrite"

    if require_edit_compatible and build_training_example(error, correct, max_tokens=max_tokens) is None:
        return "not_edit_compatible"

    return ""


def classify_real_error_types(error_text: str, correct_text: str, *, max_tokens: int = 128) -> list[str]:
    labels: list[str] = []
    example = build_training_example(error_text, correct_text, max_tokens=max_tokens)
    if example is not None:
        if any(action != ACTION_KEEP for action in example.action_labels):
            labels.append("real_spelling")
        if any(src != tgt for src, tgt in zip(example.source_punct_labels, example.target_punct_labels)):
            labels.append("real_punctuation")
    else:
        source_norm = [normalize_word(word) for word in word_tokens(error_text, include_numbers=True)]
        target_norm = [normalize_word(word) for word in word_tokens(correct_text, include_numbers=True)]
        if source_norm != target_norm:
            labels.append("real_spelling")

    return labels or ["real_other"]


def build_ai_forever_train_val(
    *,
    cache_dir: str | Path = "data/external",
    val_ratio: float = 0.10,
    repeat_train: int = 6,
    seed: int = 42,
    max_chars: int = 240,
    max_tokens: int = 96,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    pairs = load_external_pairs(AI_FOREVER_REPO, AI_FOREVER_TRAIN_FILES, cache_dir=cache_dir)
    df, stats = filter_real_pairs(pairs, max_chars=max_chars, max_tokens=max_tokens)
    if df.empty:
        df["split"] = []
        return df.copy(), df.copy(), stats

    rng = random.Random(seed)
    indices = list(df.index)
    rng.shuffle(indices)
    val_count = max(1, int(round(len(indices) * float(val_ratio))))
    val_indices = set(indices[:val_count])

    val_df = df.loc[df.index.isin(val_indices)].copy().reset_index(drop=True)
    train_df = df.loc[~df.index.isin(val_indices)].copy().reset_index(drop=True)

    if repeat_train > 1 and not train_df.empty:
        train_df = pd.concat([train_df.copy() for _ in range(int(repeat_train))], ignore_index=True)

    train_df["split"] = "train"
    val_df["split"] = "val"
    stats["train_rows_before_repeat"] = int(len(df) - len(val_df))
    stats["train_rows_after_repeat"] = int(len(train_df))
    stats["val_rows"] = int(len(val_df))
    return train_df, val_df, stats


def build_ai_forever_external_eval(
    *,
    cache_dir: str | Path = "data/external",
    max_chars: int = 280,
    max_tokens: int = 128,
) -> tuple[pd.DataFrame, dict[str, int]]:
    pairs = load_external_pairs(AI_FOREVER_REPO, AI_FOREVER_TEST_FILES, cache_dir=cache_dir)
    df, stats = filter_real_pairs(
        pairs,
        max_chars=max_chars,
        max_tokens=max_tokens,
        require_edit_compatible=False,
    )
    df["split"] = "test"
    return df, stats


def build_ruspellgold_external_eval(
    *,
    cache_dir: str | Path = "data/external",
    max_chars: int = 280,
    max_tokens: int = 128,
) -> tuple[pd.DataFrame, dict[str, int]]:
    pairs = load_external_pairs(RUSPELLGOLD_REPO, RUSPELLGOLD_TEST_FILES, cache_dir=cache_dir)
    df, stats = filter_real_pairs(
        pairs,
        max_chars=max_chars,
        max_tokens=max_tokens,
        require_edit_compatible=False,
    )
    df["split"] = "test"
    return df, stats


def _normalize_external_text(text: str) -> str:
    return normalize_yo(str(text)).replace("\u00a0", " ").strip()


def _changed_word_ratio(source_words: Sequence[str], target_words: Sequence[str]) -> float:
    source_norm = [normalize_word(word) for word in source_words]
    target_norm = [normalize_word(word) for word in target_words]
    if not source_norm and not target_norm:
        return 0.0
    distance = levenshtein_distance(" ".join(source_norm), " ".join(target_norm))
    return distance / max(len(" ".join(target_norm)), 1)


def _has_spacing_difference(error_text: str, correct_text: str) -> bool:
    return "".join(str(error_text).split()) == "".join(str(correct_text).split())
