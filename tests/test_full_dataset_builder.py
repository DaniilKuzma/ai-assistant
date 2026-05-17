import json
from pathlib import Path
import re

import pandas as pd

import src.data.full_dataset_builder as full_dataset_builder
from src.data.full_dataset_builder import (
    DatasetBuildConfig,
    _lexical_balance_target,
    build_dataset_rows,
    dataset_composition,
    write_dataset,
)
from src.validation.diff_analyzer import DiffAnalyzer
from src.validation.edit_classifier import is_allowed_edit_type


def test_full_dataset_builder_creates_required_schema_and_supported_rows():
    rows = build_dataset_rows(
        DatasetBuildConfig(
            target_total_examples=120,
            clean_identity_ratio=0.1,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=3,
        )
    )

    assert len(rows) == 120
    required = {"source", "target", "error_types", "source_dataset", "is_clean", "is_synthetic", "split", "domain", "edit_operations"}
    assert required.issubset(rows[0])
    assert any(row["is_clean"] for row in rows)
    assert any(row["is_synthetic"] and not row["is_clean"] for row in rows)
    assert {row["split"] for row in rows} == {"train", "val", "test"}

    for row in rows[:40]:
        for edit in json.loads(row["edit_operations"]):
            assert is_allowed_edit_type(edit["edit_type"])


def test_dataset_composition_and_write_dataset(tmp_path: Path):
    rows = build_dataset_rows(DatasetBuildConfig(target_total_examples=50, clean_identity_ratio=0.2, seed=5))
    output_path = tmp_path / "dataset.csv.gz"
    manifest_path = tmp_path / "manifest.json"

    write_dataset(rows, output_path, manifest_path)
    frame = pd.read_csv(output_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(frame) == 50
    assert manifest["total"] == 50
    assert manifest["composition"]["clean"] == 10
    assert manifest["error_type_counts"]["spelling"] > 0
    assert manifest["error_type_counts"]["split_join"] > 0
    assert manifest["error_type_counts"]["hyphen"] > 0
    assert dataset_composition(rows)["synthetic"] == 40


def test_full_dataset_builder_keeps_target_total_with_external_rows():
    external_rows = [
        {
            "source": "Я незнаю что делать",
            "target": "Я не знаю, что делать.",
            "error_types": '["split_join"]',
            "source_dataset": "unit_external",
            "is_clean": False,
            "is_synthetic": False,
            "split": "train",
            "domain": "unit",
            "edit_operations": "[]",
        }
        for _ in range(7)
    ]

    rows = build_dataset_rows(
        DatasetBuildConfig(target_total_examples=100, clean_identity_ratio=0.1, external_rows=tuple(external_rows))
    )

    assert len(rows) == 100
    assert dataset_composition(rows) == {"clean": 10, "synthetic": 83, "real": 7}


def test_full_dataset_builder_keeps_normalized_pairs_in_one_split():
    rows = build_dataset_rows(DatasetBuildConfig(target_total_examples=600, clean_identity_ratio=0.1, seed=11))
    split_by_key: dict[str, set[str]] = {}

    for row in rows:
        key = _normalize_for_leakage_check(row["target"])
        split_by_key.setdefault(key, set()).add(row["split"])

    leaked = {key: splits for key, splits in split_by_key.items() if len(splits) > 1}
    assert leaked == {}


def test_full_dataset_builder_keeps_punctuation_variants_in_one_split():
    external_rows = [
        {
            "source": "«Денвер» выиграл матч.",
            "target": "«Денвер» выиграл матч.",
            "error_types": "[]",
            "source_dataset": "unit_a",
            "is_clean": False,
            "is_synthetic": False,
            "split": "train",
            "domain": "unit",
            "edit_operations": "[]",
        },
        {
            "source": '"Денвер" выиграл матч.',
            "target": '"Денвер" выиграл матч.',
            "error_types": "[]",
            "source_dataset": "unit_b",
            "is_clean": False,
            "is_synthetic": False,
            "split": "train",
            "domain": "unit",
            "edit_operations": "[]",
        },
    ]

    rows = build_dataset_rows(
        DatasetBuildConfig(
            target_total_examples=80,
            clean_identity_ratio=0.1,
            external_rows=tuple(external_rows),
            seed=19,
        )
    )
    denver_splits = {row["split"] for row in rows if "Денвер" in row["target"]}

    assert len(denver_splits) == 1


def test_full_dataset_builder_uses_clean_corpus_texts_for_synthetic_targets():
    clean_texts = tuple(
        f"Периодически аналитики публикуют отчет о состоянии рынка и результатах исследования {index}."
        for index in range(40)
    )

    rows = build_dataset_rows(
        DatasetBuildConfig(target_total_examples=30, clean_identity_ratio=0.1, seed=7, clean_texts=clean_texts)
    )

    assert any("аналитики публикуют отчет" in row["target"] for row in rows)
    assert not any("документа 151223" in row["target"] for row in rows)


def test_full_dataset_builder_balances_required_lexical_error_types():
    rows = build_dataset_rows(
        DatasetBuildConfig(
            target_total_examples=180,
            clean_identity_ratio=0.1,
            seed=23,
            min_spelling_examples=45,
            min_split_join_examples=20,
            min_hyphen_examples=20,
        )
    )

    assert _count_rows_with_error_type(rows, "spelling") >= 45
    assert _count_rows_with_error_type(rows, "split_join") >= 20
    assert _count_rows_with_error_type(rows, "hyphen") >= 20
    assert dataset_composition(rows)["clean"] == 18


def test_supported_edits_excludes_context_dependent_pairs_from_training_labels():
    analyzer = DiffAnalyzer()
    edits = analyzer.analyze("Он пришел чтобы помочь.", "Он пришел что бы помочь.")

    supported_edits = full_dataset_builder._supported_edits(edits)

    assert supported_edits == []


def test_lexical_balance_targets_are_unique_beyond_base_template_capacity():
    targets = {_lexical_balance_target("не знаю", index) for index in range(20_500)}

    assert len(targets) == 20_500


def test_external_rows_respect_disable_env(monkeypatch):
    monkeypatch.setenv("RUSSIAN_CORRECTOR_DISABLE_EXTERNAL_SOURCES", "1")

    def fail_on_network_call(*_args, **_kwargs):
        raise AssertionError("HF loader should not be called in disabled external-source mode")

    monkeypatch.setattr(full_dataset_builder, "load_hf_jsonl_pairs", fail_on_network_call)

    assert full_dataset_builder._load_external_rows({"use_external_sources": True}) == []


def test_external_rows_pass_local_files_only_in_hf_offline_mode(monkeypatch):
    captured = {}

    def fake_loader(*, limit, local_files_only):
        captured["limit"] = limit
        captured["local_files_only"] = local_files_only
        return []

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(full_dataset_builder, "load_hf_jsonl_pairs", fake_loader)

    assert full_dataset_builder._load_external_rows({"use_external_sources": True, "max_external_examples": 17}) == []
    assert captured == {"limit": 17, "local_files_only": True}


def _normalize_for_leakage_check(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"\d+", "<NUM>", value)
    value = re.sub(r"[^\w\s<>]+", " ", value, flags=re.U)
    return re.sub(r"\s+", " ", value)


def _count_rows_with_error_type(rows: list[dict], error_type: str) -> int:
    return sum(error_type in json.loads(row["error_types"]) for row in rows)
