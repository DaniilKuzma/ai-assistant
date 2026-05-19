from src.data.short_dataset_v2 import assign_template_disjoint_splits, normalized_pair_hash, template_id_for_pair


def test_template_id_normalizes_numbers_quotes_and_case():
    first = template_id_for_pair("«Денвер» выиграл матч 2024 года.", "«Денвер» выиграл матч 2024 года.")
    second = template_id_for_pair('"денвер" выиграл матч 2025 года.', '"Денвер" выиграл матч 2025 года.')

    assert first == second
    assert normalized_pair_hash("Текст 12.", "Текст 12.") == normalized_pair_hash("текст 99.", "текст 99.")


def test_template_disjoint_split_keeps_same_template_in_one_split():
    rows = []
    templates = [
        ("Эксперты сообщили, что отчет {n} готов.", "Эксперты сообщили, что отчет {n} готов.", "clean_identity"),
        ("Жызнь в городе стала спокойнее {n}.", "Жизнь в городе стала спокойнее {n}.", "synthetic_augmented"),
        ("Редакция убрала запятую перед словом что {n}.", "Редакция убрала запятую перед словом, что {n}.", "synthetic_augmented"),
        ("Компания открыла новый подъезд к станции {n}.", "Компания открыла новый подъезд к станции {n}.", "hard_negative"),
        ("Автор сказал проект готов {n}.", "Автор сказал: «Проект готов» {n}.", "synthetic_augmented"),
        ("Команда проверила библиотеку утром {n}.", "Команда проверила библиотеку утром {n}.", "clean_identity"),
    ]
    for template_index, (source_template, target_template, source_type) in enumerate(templates):
        for index in range(10):
            rows.append(
                {
                    "source": source_template.format(n=index),
                    "target": target_template.format(n=index),
                    "source_type": source_type,
                    "error_type": "spelling" if source_type == "synthetic_augmented" else source_type,
                    "rule_id": "frequent_error_exact" if source_type == "synthetic_augmented" else source_type,
                    "template_bucket": template_index,
                }
            )

    assign_template_disjoint_splits(rows, {"train": 40, "val": 10, "test": 10}, seed=7)

    by_template = {}
    for row in rows:
        by_template.setdefault(row["template_id"], set()).add(row["split"])

    assert {row["split"] for row in rows} == {"train", "val", "test"}
    assert all(len(splits) == 1 for splits in by_template.values())
