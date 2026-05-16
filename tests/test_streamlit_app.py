from pathlib import Path

from src.app import streamlit_app


class FakeRuleCorrector:
    pass


class FakeTrainedCorrector:
    received_config = None

    @classmethod
    def from_config(cls, config):
        cls.received_config = config
        return cls()


def test_streamlit_corrector_prefers_trained_model_when_artifacts_exist(tmp_path: Path, monkeypatch):
    adapter_dir = tmp_path / "models" / "adapters" / "latest"
    heads_dir = tmp_path / "models" / "heads" / "latest"
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


def test_streamlit_corrector_uses_rule_fallback_when_artifacts_are_missing(tmp_path: Path, monkeypatch):
    config = {"paths": {"adapter_output_dir": str(tmp_path / "missing_adapter"), "heads_output_dir": str(tmp_path / "missing_heads")}}

    monkeypatch.setattr(streamlit_app, "load_config", lambda _path: config, raising=False)
    monkeypatch.setattr(streamlit_app, "TrainedModelCorrector", FakeTrainedCorrector, raising=False)
    monkeypatch.setattr(streamlit_app, "Corrector", FakeRuleCorrector)

    result = streamlit_app.build_streamlit_corrector(tmp_path / "config.yaml")

    assert isinstance(result.corrector, FakeRuleCorrector)
    assert result.kind == "rule_fallback"
    assert result.error is None
