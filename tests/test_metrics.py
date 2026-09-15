import pytest

from src.evaluation.metrics import compute_metrics, compute_metrics_from_edits, combined_score
from src.validation.diff_analyzer import DiffAnalyzer as RealDiffAnalyzer


def test_metrics_count_exact_match_and_clean_overcorrection():
    rows = [
        {"source": "Я незнаю что делать", "target": "Я не знаю, что делать.", "prediction": "Я не знаю, что делать.", "is_clean": False},
        {"source": "Чистый текст.", "target": "Чистый текст.", "prediction": "Чистый текст,", "is_clean": True},
    ]

    metrics = compute_metrics(rows)

    assert metrics["exact_match"] == 0.5
    assert metrics["dirty_improved_rate"] == 1.0
    assert metrics["clean_overcorrection_rate"] == 1.0
    assert metrics["edit_f1"] > 0


def test_dirty_rates_treat_partial_gold_edits_as_improvement_not_worse():
    rows = [
        {
            "source": "Я незнаю что делать",
            "target": "Я не знаю, что делать.",
            "prediction": "Я не знаю что делать",
            "is_clean": False,
        }
    ]

    metrics = compute_metrics(rows)

    assert metrics["exact_match"] == 0.0
    assert metrics["dirty_improved_rate"] == 1.0
    assert metrics["dirty_worse_rate"] == 0.0
    assert metrics["edit_precision"] == 1.0
    assert 0.0 < metrics["edit_recall"] < 1.0


def test_edit_metrics_keep_repeated_edits_distinct_by_position():
    rows = [
        {
            "source": "Он пришел она ушла он вернулся она осталась",
            "target": "Он пришел, она ушла, он вернулся она осталась",
            "prediction": "Он пришел она ушла, он вернулся она осталась",
            "is_clean": False,
        }
    ]

    metrics = compute_metrics(rows)

    assert metrics["punctuation_precision"] == 1.0
    assert metrics["punctuation_recall"] == 0.5
    assert metrics["punctuation_f1"] == pytest.approx(2 / 3)


def test_combined_score_penalizes_clean_overcorrection():
    score = combined_score(
        {
            "exact_match": 1.0,
            "edit_f1": 1.0,
            "spelling_f1": 1.0,
            "punctuation_f1": 1.0,
            "dirty_improved_rate": 1.0,
            "dirty_worse_rate": 0.0,
            "clean_overcorrection_rate": 0.5,
        }
    )

    assert score == 3.5


def test_compute_metrics_includes_combined_score_and_dataset_slices():
    rows = [
        {
            "source": "Я незнаю что делать",
            "target": "Я не знаю, что делать.",
            "prediction": "Я не знаю, что делать.",
            "is_clean": False,
            "is_synthetic": False,
        },
        {
            "source": "Во первых это важно",
            "target": "Во-первых, это важно.",
            "prediction": "Во-первых, это важно.",
            "is_clean": False,
            "is_synthetic": True,
        },
        {
            "source": "Чистый текст.",
            "target": "Чистый текст.",
            "prediction": "Чистый текст.",
            "is_clean": True,
            "is_synthetic": False,
        },
    ]

    metrics = compute_metrics(rows)

    assert "combined_score" in metrics
    assert "real_exact_match" in metrics
    assert "synthetic_exact_match" in metrics
    assert "clean_exact_match" in metrics
    assert "real_combined_score" in metrics
    assert "synthetic_combined_score" in metrics
    assert "clean_combined_score" in metrics
    assert metrics["real_exact_match"] == 1.0
    assert metrics["synthetic_exact_match"] == 1.0
    assert metrics["clean_clean_overcorrection_rate"] == 0.0


def test_compute_metrics_analyzes_each_row_pair_once(monkeypatch):
    calls = {"count": 0}

    class CountingDiffAnalyzer:
        def __init__(self):
            self.inner = RealDiffAnalyzer()

        def analyze(self, *args, **kwargs):
            calls["count"] += 1
            return self.inner.analyze(*args, **kwargs)

    rows = [
        {
            "source": "Я незнаю что делать",
            "target": "Я не знаю, что делать.",
            "prediction": "Я не знаю что делать",
            "is_clean": False,
            "is_synthetic": False,
        },
        {
            "source": "Чистый текст.",
            "target": "Чистый текст.",
            "prediction": "Чистый текст.",
            "is_clean": True,
            "is_synthetic": False,
        },
    ]
    monkeypatch.setattr("src.evaluation.metrics.DiffAnalyzer", CountingDiffAnalyzer)

    compute_metrics(rows)

    assert calls["count"] == len(rows) * 2


def test_compute_metrics_from_edits_matches_string_diff_metrics():
    analyzer = RealDiffAnalyzer()
    rows = [
        {
            "source": "Я незнаю что делать",
            "target": "Я не знаю, что делать.",
            "prediction": "Я не знаю что делать",
            "is_clean": False,
            "is_synthetic": False,
        },
        {
            "source": "Чистый текст.",
            "target": "Чистый текст.",
            "prediction": "Чистый текст.",
            "is_clean": True,
            "is_synthetic": False,
        },
    ]
    predicted_edits = [analyzer.analyze(row["source"], row["prediction"]) for row in rows]
    gold_edits = [analyzer.analyze(row["source"], row["target"]) for row in rows]

    assert compute_metrics_from_edits(rows, predicted_edits, gold_edits) == compute_metrics(rows)
