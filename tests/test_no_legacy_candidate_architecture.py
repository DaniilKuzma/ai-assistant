from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("src", "scripts", "tests", "configs")
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".json"}


def _join(*parts: str) -> str:
    return "".join(parts)


BANNED_TERMS = (
    _join("Candidate", "Generator"),
    _join("Candidate", "AwareEditModel"),
    _join("src", ".", "candidates"),
    _join("candidate", "_opportunity"),
    _join("candidate", "_decision"),
    _join("clean", "_sentence", "_pool"),
    _join("correction", "_dataset"),
    _join("operator", "_dataset", "_builder"),
    _join("rule", "_data", "_compiler"),
    _join("rule", "_lab"),
    _join("Strict", "Validator"),
    _join("strict", "_validator"),
    _join("matrix", "_eval"),
    _join("threshold", "_sweep"),
    _join("candidate", "_recall"),
)

ACTIVE_LAYER_SCAN_PATHS = (
    Path("src") / "rule_layers",
    Path("lexicon") / "layers" / "quotation_dialogue",
    Path("lexicon") / "layers" / "casing",
    Path("lexicon") / "layers" / "semantic",
    Path("lexicon") / "layers" / "compound_spelling",
    Path("lexicon") / "layers" / "dictionary_typo",
    Path("lexicon") / "layers" / "syntax_punctuation",
    Path("configs") / "config.yaml",
)

ACTIVE_LAYER_BANNED_TERMS = (
    _join("Candidate", "Generator"),
    _join("candidate", "-", "aware"),
    _join("candidate", "_aware"),
    _join("dataset", "_builder"),
    _join("train", ".", "csv"),
    _join("val", ".", "csv"),
    _join("test", ".", "csv"),
    _join("seq", "2", "seq"),
    _join("Seq", "2", "Seq"),
    _join("Auto", "Model", "For", "Seq", "2", "Seq", "LM"),
    _join("encoder", "_decoder"),
    _join("clean", "_sentence", "_pool"),
    _join("clean", " ", "sentence", " ", "pool"),
    _join("rule", "_lab"),
)

DELETED_PATHS = (
    Path("src") / "candidates",
    Path("src") / "training" / "feature_cache.py",
    Path("src") / "training" / "feature_profile.py",
    Path("src") / "training" / "diagnostics.py",
    Path("src") / "validation" / _join("strict", "_validator.py"),
    Path("src") / "validation" / "edit_classifier.py",
    Path("src") / "alignment" / "edit_label_builder.py",
    Path("src") / "evaluation" / _join("candidate", "_recall.py"),
    Path("src") / "evaluation" / _join("matrix", "_eval.py"),
    Path("src") / "evaluation" / _join("matrix", "_eval", "_reports.py"),
    Path("src") / "evaluation" / _join("matrix", "_inventory.py"),
    Path("src") / "evaluation" / "pre_dataset_capability.py",
    Path("src") / "evaluation" / "activation_plan.py",
    Path("src") / "evaluation" / "final_syntax_capability.py",
    Path("src") / "evaluation" / "syntax_required_inventory.py",
    Path("src") / "evaluation" / "score_distribution.py",
    Path("src") / "evaluation" / _join("threshold", "_sweep.py"),
    Path("src") / "evaluation" / "fast_eval.py",
    Path("src") / "config" / _join("candidate", "_dataset_config.py"),
    Path("src") / "config" / "thresholds.py",
    Path("scripts") / "calibrate_thresholds.py",
    Path("scripts") / "threshold_safety.py",
)


def test_legacy_terms_are_absent_from_active_code_and_configs() -> None:
    offenders: list[str] = []
    for path in _text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for term in BANNED_TERMS:
            if term in text:
                offenders.append(f"{path.relative_to(ROOT)}: {term}")

    assert offenders == []


def test_legacy_files_and_folders_are_absent() -> None:
    existing = [str(path) for path in DELETED_PATHS if (ROOT / path).exists()]

    assert existing == []


def test_runtime_training_model_inference_do_not_import_deleted_package() -> None:
    import_term = _join("src", ".", "candidates")
    package_roots = [ROOT / "src" / name for name in ("runtime", "training", "model", "inference")]
    offenders: list[str] = []
    for package_root in package_roots:
        if not package_root.exists():
            continue
        for path in package_root.rglob("*.py"):
            if import_term in path.read_text(encoding="utf-8", errors="ignore"):
                offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []


def test_src_training_has_no_old_candidate_feature_fields() -> None:
    banned_fields = (
        _join("candidate", "_labels"),
        _join("candidate", "_mask"),
        _join("candidate", "_scores"),
        _join("candidate", "_rule", "_ids"),
        _join("candidate", "_edit", "_types"),
        _join("candidate", "_sources"),
        _join("candidate", "_replacements"),
        _join("candidate", "_spans"),
    )
    offenders: list[str] = []
    training_root = ROOT / "src" / "training"
    for path in training_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for field in banned_fields:
            if field in text:
                offenders.append(f"{path.relative_to(ROOT)}: {field}")

    assert offenders == []


def test_active_rule_layers_do_not_use_legacy_candidate_dataset_or_seq2seq_paths() -> None:
    offenders: list[str] = []
    for path in _active_layer_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for term in ACTIVE_LAYER_BANNED_TERMS:
            if term.lower() in text.lower():
                offenders.append(f"{path.relative_to(ROOT)}: {term}")

    assert offenders == []


def test_materialized_train_val_test_csv_pipeline_is_absent() -> None:
    generated_csvs = [
        path.relative_to(ROOT)
        for directory in (ROOT / "data" / "processed", ROOT / "data" / "generated_eval")
        if directory.exists()
        for path in directory.rglob("*.csv")
    ]

    assert generated_csvs == []


def _text_files() -> list[Path]:
    files: list[Path] = []
    for dirname in SCAN_DIRS:
        root = ROOT / dirname
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in TEXT_SUFFIXES:
                files.append(path)
    return files


def _active_layer_text_files() -> list[Path]:
    files: list[Path] = []
    for relative_path in ACTIVE_LAYER_SCAN_PATHS:
        path = ROOT / relative_path
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            files.append(path)
        elif path.is_dir():
            files.extend(item for item in path.rglob("*") if item.is_file() and item.suffix in TEXT_SUFFIXES)
    return files
