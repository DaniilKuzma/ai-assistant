from pathlib import Path
from types import SimpleNamespace

from src.app import streamlit_app
from src.inference.corrector import CorrectionResult
from src.validation.diff_analyzer import Edit


class FakeRuleCorrector:
    received_config = None
    from_config_calls = 0

    @classmethod
    def from_config(cls, config):
        cls.received_config = config
        cls.from_config_calls += 1
        return cls()


class FakeTrainedCorrector:
    received_config = None
    should_raise = False

    @classmethod
    def from_config(cls, config):
        cls.received_config = config
        if cls.should_raise:
            raise RuntimeError("trained load failed")
        return cls()


def _reset_fakes():
    FakeRuleCorrector.received_config = None
    FakeRuleCorrector.from_config_calls = 0
    FakeTrainedCorrector.received_config = None
    FakeTrainedCorrector.should_raise = False


def test_streamlit_corrector_prefers_trained_model_when_artifacts_exist(tmp_path: Path, monkeypatch):
    _reset_fakes()
    adapter_dir = tmp_path / "models" / "adapters" / "current"
    heads_dir = tmp_path / "models" / "heads" / "current"
    adapter_dir.mkdir(parents=True)
    heads_dir.mkdir(parents=True)
    (heads_dir / "heads.pt").write_bytes(b"unit")
    config = {"paths": {"adapter_output_dir": str(adapter_dir), "heads_output_dir": str(heads_dir)}}

    monkeypatch.setattr(streamlit_app, "load_config", lambda _path: config, raising=False)
    monkeypatch.setattr(streamlit_app, "TrainedModelCorrector", FakeTrainedCorrector, raising=False)
    monkeypatch.setattr(streamlit_app, "Corrector", FakeRuleCorrector)

    result = streamlit_app.build_streamlit_corrector(tmp_path / "config.yaml")

    assert isinstance(result.corrector, FakeTrainedCorrector)
    assert result.kind == "trained_model"
    assert result.error is None
    assert FakeTrainedCorrector.received_config == config
    assert FakeRuleCorrector.from_config_calls == 0


def test_streamlit_corrector_uses_configured_rule_fallback_when_artifacts_are_missing(tmp_path: Path, monkeypatch):
    _reset_fakes()
    config = {"paths": {"adapter_output_dir": str(tmp_path / "missing_adapter"), "heads_output_dir": str(tmp_path / "missing_heads")}}

    monkeypatch.setattr(streamlit_app, "load_config", lambda _path: config, raising=False)
    monkeypatch.setattr(streamlit_app, "TrainedModelCorrector", FakeTrainedCorrector, raising=False)
    monkeypatch.setattr(streamlit_app, "Corrector", FakeRuleCorrector)

    result = streamlit_app.build_streamlit_corrector(tmp_path / "config.yaml")

    assert isinstance(result.corrector, FakeRuleCorrector)
    assert result.kind == "rule_fallback"
    assert result.error is None
    assert FakeRuleCorrector.from_config_calls == 1
    assert FakeRuleCorrector.received_config == config


def test_streamlit_corrector_uses_configured_rule_fallback_when_trained_model_load_fails(
    tmp_path: Path,
    monkeypatch,
):
    _reset_fakes()
    adapter_dir = tmp_path / "models" / "adapters" / "current"
    heads_dir = tmp_path / "models" / "heads" / "current"
    adapter_dir.mkdir(parents=True)
    heads_dir.mkdir(parents=True)
    (heads_dir / "heads.pt").write_bytes(b"unit")
    config = {"paths": {"adapter_output_dir": str(adapter_dir), "heads_output_dir": str(heads_dir)}}
    FakeTrainedCorrector.should_raise = True

    monkeypatch.setattr(streamlit_app, "load_config", lambda _path: config, raising=False)
    monkeypatch.setattr(streamlit_app, "TrainedModelCorrector", FakeTrainedCorrector, raising=False)
    monkeypatch.setattr(streamlit_app, "Corrector", FakeRuleCorrector)

    result = streamlit_app.build_streamlit_corrector(tmp_path / "config.yaml")

    assert isinstance(result.corrector, FakeRuleCorrector)
    assert result.kind == "rule_fallback"
    assert result.error == "trained load failed"
    assert FakeRuleCorrector.from_config_calls == 1
    assert FakeRuleCorrector.received_config == config


def test_streamlit_corrector_applies_memory_overrides_to_config_copy(tmp_path: Path, monkeypatch):
    _reset_fakes()
    config = {
        "paths": {
            "adapter_output_dir": str(tmp_path / "missing_adapter"),
            "heads_output_dir": str(tmp_path / "missing_heads"),
        },
        "correction_memory": {
            "enabled": False,
            "storage_path": str(tmp_path / "memory.jsonl"),
            "context_window_chars": 24,
        },
    }

    monkeypatch.setattr(streamlit_app, "load_config", lambda _path: config, raising=False)
    monkeypatch.setattr(streamlit_app, "Corrector", FakeRuleCorrector)

    result = streamlit_app.build_streamlit_corrector(tmp_path / "config.yaml", memory_enabled=True, doc_id="doc-7")

    assert isinstance(result.corrector, FakeRuleCorrector)
    assert FakeRuleCorrector.received_config["correction_memory"]["enabled"] is True
    assert FakeRuleCorrector.received_config["correction_memory"]["doc_id"] == "doc-7"
    assert config["correction_memory"].get("doc_id") is None
    assert config["correction_memory"]["enabled"] is False


def test_build_feedback_rows_returns_compact_edit_rows():
    result = CorrectionResult(
        source_text="Она пришла сдесь утром.",
        corrected_text="Она пришла здесь утром.",
        edits=[
            Edit(
                source="сдесь",
                replacement="здесь",
                edit_type="spelling_replace",
                status="accepted",
                reason="allowed strict-scope edit",
            )
        ],
    )

    assert streamlit_app.build_feedback_rows(result) == [
        {
            "source": "сдесь",
            "replacement": "здесь",
            "type": "spelling_replace",
            "status": "accepted",
            "reason": "allowed strict-scope edit",
        }
    ]


def test_get_incremental_cache_key_depends_on_normalized_doc_id():
    assert streamlit_app.get_incremental_cache_key("doc-a") == "doc-a"
    assert streamlit_app.get_incremental_cache_key("doc-a") != streamlit_app.get_incremental_cache_key("doc-b")
    assert streamlit_app.get_incremental_cache_key("  ") == "default"


def test_build_incremental_summary_returns_segment_counts():
    result = SimpleNamespace(checked_segments=3, reused_segments=2, changed_segments=1)

    assert streamlit_app.build_incremental_summary(result) == {
        "checked": 3,
        "reused": 2,
        "changed": 1,
    }


def test_docx_cache_key_depends_on_normalized_doc_id():
    assert streamlit_app.docx_cache_key("doc-a") == "doc-a"
    assert streamlit_app.docx_cache_key("doc-a") != streamlit_app.docx_cache_key("doc-b")
    assert streamlit_app.docx_cache_key("  ") == "default"


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
