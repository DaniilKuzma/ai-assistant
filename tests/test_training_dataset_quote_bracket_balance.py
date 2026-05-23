from pathlib import Path

import pandas as pd

from src.data.training_quality_audit import (
    quote_bracket_balance_audit_frame,
    quote_bracket_balance_counts,
)


DATASET_PATH = Path("data/processed/correction_dataset.csv.gz")
POSITIVE_SOURCE_TYPES = {"synthetic_augmented_from_open_clean", "real_error_pair"}
FOCUS_RULE_IDS = {
    "quote_pair_balance",
    "direct_speech_quotes",
    "direct_speech_colon",
    "bracket_pair_balance",
    "direct_speech_dash",
    "punctuation_delete_replace",
}


def _row(source: str, target: str, rule_id: str = "quote_pair_balance") -> dict[str, object]:
    return {
        "source": source,
        "target": target,
        "split": "train",
        "source_type": "synthetic_augmented_from_open_clean",
        "rule_id": rule_id,
        "rule_ids": f'["{rule_id}"]',
        "metadata": '{"candidate_present":true}',
    }


def test_quote_bracket_balance_counts_unbalanced_positive_targets():
    frame = pd.DataFrame(
        [
            _row("Он сказал текст.", "Он сказал: «текст.", "quote_pair_balance"),
            _row("Он сказал текст.", 'Он сказал: "текст.', "direct_speech_colon"),
            _row("Проверь текст.", "Проверь текст (черновик.", "bracket_pair_balance"),
            _row("Проверь текст.", "Проверь текст [черновик.", "bracket_pair_balance"),
            _row("Проверь текст.", "Проверь текст {черновик.", "bracket_pair_balance"),
            {
                **_row("Проверь текст.", "Проверь текст (черновик.", "bracket_pair_balance"),
                "source_type": "clean_identity_from_open_clean",
            },
        ]
    )

    assert quote_bracket_balance_counts(frame) == {
        "unbalanced_target_guillemets": 1,
        "unbalanced_target_ascii_quotes": 1,
        "unbalanced_target_parentheses": 1,
        "unbalanced_target_square_brackets": 1,
        "unbalanced_target_curly_brackets": 1,
    }

    audit = quote_bracket_balance_audit_frame(frame)
    assert len(audit) == 5
    assert set(audit["action"]) == {"kept"}
    assert set(audit["reason"]) <= {
        "unbalanced_target_guillemets",
        "unbalanced_target_ascii_quotes",
        "unbalanced_target_parentheses",
        "unbalanced_target_square_brackets",
        "unbalanced_target_curly_brackets",
    }


def test_known_quote_bracket_regressions_are_absent_from_canonical_dataset():
    df = pd.read_csv(DATASET_PATH)
    bad_targets = {
        'Давайте радоваться жизни, радоваться игре и давайте пить пиво", - сказал футбольный чиновник.',
        "«Так что я не исключаю этого.",
        '"Может быть, он будет лучше - вполне допускаю, - заявил он.',
        '"Весной и летом Депардье должен будет несколько раз съездить из Парижа в Боде (Bodoe) и обратно.',
        'Тимошенко в четверг в суде сказала: "Мы отказываемся от какого-либо участия.',
    }

    present = set(df["target"].astype(str)) & bad_targets

    assert present == set()


def test_canonical_positive_targets_are_quote_bracket_balanced():
    df = pd.read_csv(DATASET_PATH)

    counts = quote_bracket_balance_counts(df)
    audit = quote_bracket_balance_audit_frame(df)

    assert counts == {
        "unbalanced_target_guillemets": 0,
        "unbalanced_target_ascii_quotes": 0,
        "unbalanced_target_parentheses": 0,
        "unbalanced_target_square_brackets": 0,
        "unbalanced_target_curly_brackets": 0,
    }
    assert audit.empty


def test_focus_rule_targets_are_quote_bracket_balanced():
    df = pd.read_csv(DATASET_PATH)
    focus = df[
        df["source_type"].isin(POSITIVE_SOURCE_TYPES)
        & df["rule_id"].astype(str).isin(FOCUS_RULE_IDS)
    ]

    assert not focus.empty
    assert quote_bracket_balance_audit_frame(focus).empty
