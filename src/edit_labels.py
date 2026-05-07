"""Convert correction pairs into token-level edit labels."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional

import pandas as pd

from text_utils import extract_word_slots, normalize_word, punctuation_labels_for_slots


ACTION_KEEP = "KEEP"
ACTION_REPLACE = "REPLACE"
ACTION_DELETE = "DELETE"
ACTION_REPLACE_PREFIX = "REPLACE_"


def replace_action(rank: int) -> str:
    return f"{ACTION_REPLACE_PREFIX}{int(rank)}"


def action_replace_rank(action: str) -> int | None:
    action = str(action)
    if action == ACTION_REPLACE:
        return 0
    if not action.startswith(ACTION_REPLACE_PREFIX):
        return None
    try:
        return int(action[len(ACTION_REPLACE_PREFIX) :])
    except ValueError:
        return None


def candidate_list(value: object) -> List[str]:
    """Normalize a legacy single candidate or a top-k candidate list."""
    if isinstance(value, (list, tuple)):
        values = [str(item) for item in value if str(item)]
        return values or [""]
    text = str(value)
    return [text] if text else [""]


@dataclass
class HybridTrainingExample:
    source_words: List[str]
    candidate_words: List[List[str]]
    action_labels: List[str]
    source_punct_labels: List[str]
    target_punct_labels: List[str]
    target_words: List[str]
    is_clean: bool = False
    target_candidate_ranks: List[int] = field(default_factory=list)
    oracle_injected_flags: List[bool] = field(default_factory=list)

    @property
    def punct_labels(self) -> List[str]:
        """Backward-compatible alias for target punctuation labels."""
        return self.target_punct_labels


def build_training_example(
    error_text: str,
    correct_text: str,
    max_tokens: int = 128,
) -> Optional[HybridTrainingExample]:
    """Build one aligned training example.

    The model stays edit-based: every source token receives KEEP, REPLACE or
    DELETE, and a punctuation label for the output after this token.
    """
    src_slots = extract_word_slots(error_text)
    tgt_slots = extract_word_slots(correct_text)
    if not src_slots or not tgt_slots:
        return None

    src_words = [slot.word for slot in src_slots]
    tgt_words = [slot.word for slot in tgt_slots]
    src_norm = [normalize_word(w) for w in src_words]
    tgt_norm = [normalize_word(w) for w in tgt_words]

    source_out: List[str] = []
    candidates_out: List[List[str]] = []
    actions_out: List[str] = []
    source_punct_out: List[str] = []
    target_punct_out: List[str] = []
    target_out: List[str] = []

    src_punct_all = punctuation_labels_for_slots(str(error_text), src_slots)
    tgt_punct_all = punctuation_labels_for_slots(str(correct_text), tgt_slots)

    matcher = SequenceMatcher(None, src_norm, tgt_norm)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        src_segment = src_words[i1:i2]
        tgt_segment = tgt_words[j1:j2]
        src_punct = src_punct_all[i1:i2]
        tgt_punct = tgt_punct_all[j1:j2]

        if tag == "equal":
            for src_word, tgt_word, src_p, tgt_p in zip(src_segment, tgt_segment, src_punct, tgt_punct):
                source_out.append(src_word)
                candidates_out.append([tgt_word if src_word != tgt_word else src_word])
                actions_out.append(ACTION_KEEP if src_word == tgt_word else replace_action(0))
                source_punct_out.append(src_p)
                target_punct_out.append(tgt_p)
                target_out.append(tgt_word)
            continue

        if tag == "replace":
            if len(src_segment) == len(tgt_segment):
                for src_word, tgt_word, src_p, tgt_p in zip(src_segment, tgt_segment, src_punct, tgt_punct):
                    source_out.append(src_word)
                    candidates_out.append([tgt_word])
                    actions_out.append(ACTION_KEEP if src_word == tgt_word else replace_action(0))
                    source_punct_out.append(src_p)
                    target_punct_out.append(tgt_p)
                    target_out.append(tgt_word)
                continue

            if len(src_segment) == 1 and 1 <= len(tgt_segment) <= 3:
                source_out.append(src_segment[0])
                candidates_out.append([" ".join(tgt_segment)])
                actions_out.append(replace_action(0))
                source_punct_out.append(src_punct[0] if src_punct else "")
                target_punct_out.append(tgt_punct[-1] if tgt_punct else "")
                target_out.append(" ".join(tgt_segment))
                continue

            if len(tgt_segment) == 1 and len(src_segment) >= 1:
                source_out.append(src_segment[0])
                candidates_out.append([tgt_segment[0]])
                actions_out.append(replace_action(0))
                source_punct_out.append(src_punct[0] if src_punct else "")
                target_punct_out.append(tgt_punct[0] if tgt_punct else "")
                target_out.append(tgt_segment[0])
                for src_word, src_p in zip(src_segment[1:], src_punct[1:]):
                    source_out.append(src_word)
                    candidates_out.append([src_word])
                    actions_out.append(ACTION_DELETE)
                    source_punct_out.append(src_p)
                    target_punct_out.append("")
                    target_out.append("")
                continue

            return None

        if tag == "delete":
            for src_word, src_p in zip(src_segment, src_punct):
                source_out.append(src_word)
                candidates_out.append([src_word])
                actions_out.append(ACTION_DELETE)
                source_punct_out.append(src_p)
                target_punct_out.append("")
                target_out.append("")
            continue

        if tag == "insert":
            # Pure inserted words have no source token to attach to. They are
            # skipped instead of forcing free-form text generation.
            return None

    if not source_out or len(source_out) > max_tokens:
        return None

    return HybridTrainingExample(
        source_words=source_out,
        candidate_words=candidates_out,
        action_labels=actions_out,
        source_punct_labels=source_punct_out,
        target_punct_labels=target_punct_out,
        target_words=target_out,
        is_clean=str(error_text).strip() == str(correct_text).strip(),
    )


def build_examples_from_dataframe(
    df: pd.DataFrame,
    max_tokens: int = 128,
) -> tuple[List[HybridTrainingExample], Dict[str, int]]:
    examples: List[HybridTrainingExample] = []
    stats = {"rows": 0, "kept": 0, "skipped": 0, "clean": 0}

    for _, row in df.iterrows():
        stats["rows"] += 1
        example = build_training_example(
            str(row["error_text"]),
            str(row["correct_text"]),
            max_tokens=max_tokens,
        )
        if example is None:
            stats["skipped"] += 1
            continue
        examples.append(example)
        stats["kept"] += 1
        if example.is_clean:
            stats["clean"] += 1

    return examples, stats


def iter_all_words(examples: Iterable[HybridTrainingExample]) -> Iterable[str]:
    for example in examples:
        yield from example.source_words
        for candidates in example.candidate_words:
            yield from candidate_list(candidates)
