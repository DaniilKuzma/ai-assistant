from pathlib import Path
import inspect
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.app import streamlit_app
from src.runtime.corrector import Corrector
from src.runtime.neural_backend import DirectNeuralBackend
from src.schema.edits import CorrectionResult, RuntimeEdit


ROOT = Path(__file__).resolve().parents[1]


class FakeRuntimeCorrector:
    received_config = None
    from_config_calls = 0
    next_neural_backend = None

    def __init__(self, neural_backend=None):
        self.neural_backend = neural_backend

    @classmethod
    def from_config(cls, config):
        cls.received_config = config
        cls.from_config_calls += 1
        return cls(cls.next_neural_backend)


def _reset_fakes():
    FakeRuntimeCorrector.received_config = None
    FakeRuntimeCorrector.from_config_calls = 0
    FakeRuntimeCorrector.next_neural_backend = None


def test_streamlit_corrector_returns_deterministic_fallback_without_model_artifacts(tmp_path: Path, monkeypatch):
    _reset_fakes()
    config = {"paths": {"adapter_output_dir": str(tmp_path / "missing_adapter"), "heads_output_dir": str(tmp_path / "missing_heads")}}

    monkeypatch.setattr(streamlit_app, "load_config", lambda _path: config, raising=False)
    monkeypatch.setattr(streamlit_app, "Corrector", FakeRuntimeCorrector)

    result = streamlit_app.build_streamlit_corrector(tmp_path / "config.yaml")

    assert isinstance(result.corrector, FakeRuntimeCorrector)
    assert result.kind == "deterministic_fallback"
    assert result.error is None
    assert FakeRuntimeCorrector.from_config_calls == 1
    assert FakeRuntimeCorrector.received_config == config


def test_streamlit_corrector_returns_direct_neural_when_runtime_backend_loaded(tmp_path: Path, monkeypatch):
    _reset_fakes()
    backend = object()
    FakeRuntimeCorrector.next_neural_backend = backend
    config = {"runtime": {"neural_token_edits": True, "neural_punctuation": True}}

    monkeypatch.setattr(streamlit_app, "load_config", lambda _path: config, raising=False)
    monkeypatch.setattr(streamlit_app, "Corrector", FakeRuntimeCorrector)

    result = streamlit_app.build_streamlit_corrector(tmp_path / "config.yaml")

    assert isinstance(result.corrector, FakeRuntimeCorrector)
    assert result.corrector.neural_backend is backend
    assert result.kind == "direct_neural"
    assert result.error is None
    assert FakeRuntimeCorrector.from_config_calls == 1
    assert FakeRuntimeCorrector.received_config == config


def test_streamlit_app_does_not_expose_legacy_trained_model_corrector():
    assert not hasattr(streamlit_app, "TrainedModelCorrector")


def test_streamlit_app_import_does_not_require_streamlit_package():
    script = """
import builtins
import importlib

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "streamlit":
        raise ModuleNotFoundError("streamlit intentionally blocked")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
module = importlib.import_module("src.app.streamlit_app")
assert hasattr(module, "build_streamlit_corrector")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_corrector_from_config_accepts_config_path(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "runtime:\n  neural_token_edits: false\n  neural_punctuation: false\n",
        encoding="utf-8",
    )

    corrector = Corrector.from_config(config_path)

    assert isinstance(corrector, Corrector)
    assert corrector.neural_backend is None


def test_direct_neural_backend_refuses_missing_architecture_marker(tmp_path: Path):
    heads_dir = tmp_path / "heads"
    heads_dir.mkdir()
    (heads_dir / "heads.pt").write_bytes(b"not a torch artifact")
    config = {
        "model": {"lora": {"enabled": False}},
        "paths": {
            "heads_output_dir": str(heads_dir),
            "adapter_output_dir": str(tmp_path / "adapters"),
        },
    }

    with pytest.raises(RuntimeError, match="architecture marker"):
        DirectNeuralBackend.from_config(config)


def test_direct_neural_backend_refuses_debug_marker_by_default(tmp_path: Path):
    heads_dir = tmp_path / "heads"
    heads_dir.mkdir()
    (heads_dir / "heads.pt").write_bytes(b"not a torch artifact")
    (heads_dir / "architecture.json").write_text(
        json.dumps({"architecture": "direct_edit_tagger_v1", "debug_model": True}),
        encoding="utf-8",
    )
    config = {
        "model": {"lora": {"enabled": False}},
        "paths": {
            "heads_output_dir": str(heads_dir),
            "adapter_output_dir": str(tmp_path / "adapters"),
        },
    }

    with pytest.raises(RuntimeError, match="debug_model"):
        DirectNeuralBackend.from_config(config)


def test_streamlit_api_exposes_no_memory_or_user_response_helpers():
    corrector_params = inspect.signature(streamlit_app.build_streamlit_corrector).parameters
    cached_params = inspect.signature(streamlit_app._cached_streamlit_corrector).parameters
    main_source = inspect.getsource(streamlit_app.main)

    assert "memory_enabled" not in corrector_params
    assert "doc_id" not in corrector_params
    assert "memory_enabled" not in cached_params
    assert "ID документа" not in main_source
    assert "doc_id" not in main_source
    obsolete_helper = "build_" + "feed" + "back_rows"
    assert not hasattr(streamlit_app, obsolete_helper)


def test_build_edit_rows_returns_compact_edit_rows():
    result = CorrectionResult(
        source_text="Она пришла сдесь утром.",
        corrected_text="Она пришла здесь утром.",
        edits=[
            RuntimeEdit(
                start=11,
                end=17,
                source="сдесь",
                replacement="здесь",
                edit_type="spelling_replace",
                rule_id="spelling_zdes",
                confidence=0.92,
                explanation="Исправлено написание слова.",
            )
        ],
    )

    assert streamlit_app.build_edit_rows(result.edits) == [
        {
            "source": "сдесь",
            "replacement": "здесь",
            "edit_type": "spelling_replace",
            "rule_id": "spelling_zdes",
            "confidence": 0.92,
            "explanation": "Исправлено написание слова.",
        }
    ]


def test_text_incremental_cache_key_is_internal_and_stable():
    assert streamlit_app.text_incremental_cache_key() == "default_text"


def test_build_incremental_summary_returns_segment_counts():
    result = SimpleNamespace(checked_segments=3, reused_segments=2, changed_segments=1)

    assert streamlit_app.build_incremental_summary(result) == {
        "checked": 3,
        "reused": 2,
        "changed": 1,
    }


def test_docx_cache_key_uses_upload_name_without_exposing_document_id():
    assert streamlit_app.docx_cache_key("file-a.docx") == "file-a.docx"
    assert streamlit_app.docx_cache_key("file-a.docx") != streamlit_app.docx_cache_key("file-b.docx")
    assert streamlit_app.docx_cache_key("  ") == "uploaded_docx"


def test_docx_correction_requires_explicit_button():
    main_source = inspect.getsource(streamlit_app.main)

    assert "if docx_correction_requested(uploaded.name):" in main_source


def test_docx_correction_requested_uses_primary_button(monkeypatch):
    calls = []

    def fake_button(label, **kwargs):
        calls.append((label, kwargs))
        return False

    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(button=fake_button))

    assert streamlit_app.docx_correction_requested("draft.docx") is False
    assert calls == [
        (
            "Исправить DOCX",
            {"type": "primary", "key": "correct_docx_draft.docx"},
        )
    ]


def test_build_docx_incremental_summary_returns_paragraph_counts_and_total_edits():
    result = SimpleNamespace(checked_paragraphs=4, reused_paragraphs=2, edits=[object(), object(), object()])

    assert streamlit_app.build_docx_incremental_summary(result) == {
        "checked": 4,
        "reused": 2,
        "total_edits": 3,
    }


def test_render_highlighted_diff_marks_insertions_and_deletions():
    html = streamlit_app.render_highlighted_diff("Я незнаю что делать", "Я не знаю, что делать.")

    assert "diff-insert" in html
    assert "diff-delete" in html
    assert "не знаю" in html
    assert "," in html


def test_render_highlighted_diff_escapes_user_text():
    html = streamlit_app.render_highlighted_diff("<script>незнаю</script>", "<script>не знаю</script>.")

    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "diff-insert" in html
