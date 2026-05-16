from src.evaluation.metrics import compute_metrics, combined_score


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
