from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INTEGRATED_YAML_LAYERS = (
    "compound_spelling",
    "dictionary_typo",
    "syntax_punctuation",
    "quotation_dialogue",
    "casing",
    "semantic",
)


def test_architecture_file_map_and_rule_authoring_docs_mention_integrated_yaml_layers() -> None:
    docs = {
        "ai_docs/ARCHITECTURE.md": (ROOT / "ai_docs" / "ARCHITECTURE.md").read_text(encoding="utf-8"),
        "ai_docs/FILE_MAP.md": (ROOT / "ai_docs" / "FILE_MAP.md").read_text(encoding="utf-8"),
        "docs/rule_authoring_guide.md": (ROOT / "docs" / "rule_authoring_guide.md").read_text(encoding="utf-8"),
    }

    missing = [
        f"{path}: {layer}"
        for path, text in docs.items()
        for layer in INTEGRATED_YAML_LAYERS
        if layer not in text
    ]

    assert missing == []
