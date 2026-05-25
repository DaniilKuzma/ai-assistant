from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

REMOVED_DATA_MODULES = (
    "operator_dataset_builder.py",
    "full_dataset_builder.py",
    "_training_dataset_builder.py",
    "rule_data_compiler.py",
    "rule_lab_generation.py",
    "corruption_operators.py",
    "hard_negative_generation.py",
    "synthetic_generator.py",
    "stress_generation.py",
    "clean_sentence_pool.py",
    "clean_corpus_sources.py",
    "open_corpora_sources.py",
    "real_error_sources.py",
    "sage_sources.py",
    "source_downloads.py",
    "load_external.py",
    "dataset_builder.py",
    "dataset_quality.py",
    "dataset_stats.py",
    "dataset_verifiers.py",
    "training_dataset.py",
    "training_quality_audit.py",
    "strict_filter.py",
    "atomic_verifier.py",
    "matrix_eval_dataset.py",
)


def test_legacy_data_prep_modules_are_removed() -> None:
    src_data = ROOT / "src" / "data"

    for module_name in REMOVED_DATA_MODULES:
        assert not (src_data / module_name).exists(), module_name


def test_config_uses_online_ast_generation_without_candidate_dataset() -> None:
    config_path = ROOT / "configs" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["generation"]["mode"] == "online_ast"
    assert "candidate_opportunity" not in config.get("data", {})


def test_rule_lab_config_is_removed() -> None:
    assert not (ROOT / "configs" / "rule_lab_recipes.yaml").exists()


def test_materialized_training_artifacts_are_removed() -> None:
    processed = ROOT / "data" / "processed"
    forbidden_names = {
        "train.csv",
        "val.csv",
        "test.csv",
        "correction_dataset.csv.gz",
    }

    assert not forbidden_names.intersection({path.name for path in processed.iterdir()})


def test_ast_first_architecture_decision_exists() -> None:
    assert (ROOT / "docs" / "architecture_decision_ast_first.md").exists()
