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
