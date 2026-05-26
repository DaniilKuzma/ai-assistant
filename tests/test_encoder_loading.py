import os
import sys
import types
from pathlib import Path

import pytest

from src.model.encoder import EncoderLoadConfig, PRIMARY_ENCODER, load_encoder, load_tokenizer


def test_ensure_pytorch_transformers_backend_disables_tf_and_flax(monkeypatch):
    from src.model.encoder import ensure_pytorch_transformers_backend

    for key in ("USE_TF", "TRANSFORMERS_NO_TF", "USE_FLAX"):
        monkeypatch.delenv(key, raising=False)

    ensure_pytorch_transformers_backend()

    assert os.environ["USE_TF"] == "0"
    assert os.environ["TRANSFORMERS_NO_TF"] == "1"
    assert os.environ["USE_FLAX"] == "0"


def test_load_encoder_disables_unused_pooler(monkeypatch):
    calls = []

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            calls.append((model_name, kwargs))
            return object()

    import transformers

    monkeypatch.setattr(transformers, "AutoModel", FakeAutoModel)

    load_encoder(EncoderLoadConfig(model_name="fake-roberta", fallback_model_name="fake-bert"))

    assert calls[0][1]["add_pooling_layer"] is False


def test_direct_neural_backend_sets_pytorch_env_before_importing_peft(monkeypatch, tmp_path):
    from src.runtime import neural_backend

    for key in ("USE_TF", "TRANSFORMERS_NO_TF", "USE_FLAX"):
        monkeypatch.delenv(key, raising=False)

    observed: dict[str, str | None] = {}

    class FakePeftModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return "loaded-adapter"

    class FakePeftModule(types.ModuleType):
        def __getattribute__(self, name):
            if name == "PeftModel":
                observed["USE_TF"] = os.environ.get("USE_TF")
                observed["TRANSFORMERS_NO_TF"] = os.environ.get("TRANSFORMERS_NO_TF")
                observed["USE_FLAX"] = os.environ.get("USE_FLAX")
                return FakePeftModel
            return super().__getattribute__(name)

    monkeypatch.setitem(sys.modules, "peft", FakePeftModule("peft"))

    loaded = neural_backend._load_peft_adapter(object(), tmp_path)

    assert observed == {
        "USE_TF": "0",
        "TRANSFORMERS_NO_TF": "1",
        "USE_FLAX": "0",
    }
    assert loaded == "loaded-adapter"


def test_direct_neural_backend_broken_adapter_path_raises_in_strict_mode(monkeypatch, tmp_path):
    from src.runtime import neural_backend

    class FakePeftModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise ValueError("broken adapter")

    fake_module = types.ModuleType("peft")
    fake_module.PeftModel = FakePeftModel
    monkeypatch.setitem(sys.modules, "peft", fake_module)

    with pytest.raises(RuntimeError, match="broken adapter"):
        neural_backend._load_peft_adapter(object(), tmp_path)


def test_direct_neural_backend_can_soft_fallback_only_when_enabled(monkeypatch, tmp_path):
    from src.runtime import neural_backend

    encoder = object()

    class FakePeftModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise ValueError("broken adapter")

    fake_module = types.ModuleType("peft")
    fake_module.PeftModel = FakePeftModel
    monkeypatch.setitem(sys.modules, "peft", fake_module)

    assert neural_backend._load_peft_adapter(encoder, tmp_path, allow_fallback=True) is encoder


def test_direct_neural_backend_loads_saved_adapter_onto_plain_encoder(monkeypatch, tmp_path):
    from src.runtime import neural_backend

    heads_dir = tmp_path / "heads"
    adapter_dir = tmp_path / "adapter"
    heads_dir.mkdir()
    adapter_dir.mkdir()
    (heads_dir / "heads.pt").write_bytes(b"fake-heads")
    (heads_dir / "architecture.json").write_text(
        '{"architecture": "direct_edit_tagger_v1", "debug_model": false}\n',
        encoding="utf-8",
    )

    seen_configs = []

    class FakeHeads:
        def load_state_dict(self, state):
            self.state = state

    class FakeModule:
        def __init__(self) -> None:
            self.encoder = "plain-encoder"
            self.heads = FakeHeads()
            self.eval_called = False

        def eval(self) -> None:
            self.eval_called = True

    class FakeDirectEditTaggerModel:
        def __init__(self, config) -> None:
            seen_configs.append(config)
            self.module = FakeModule()

    def fake_load_adapter(encoder, path: Path, *, allow_fallback: bool = False):
        assert encoder == "plain-encoder"
        assert path == adapter_dir
        assert allow_fallback is False
        return "loaded-adapter"

    import torch

    monkeypatch.setattr(neural_backend, "load_tokenizer", lambda config: "tokenizer")
    monkeypatch.setattr(neural_backend, "DirectEditTaggerModel", FakeDirectEditTaggerModel)
    monkeypatch.setattr(neural_backend, "_load_peft_adapter", fake_load_adapter)
    monkeypatch.setattr(torch, "load", lambda path, map_location=None: {"heads": "state"})

    backend = neural_backend.DirectNeuralBackend.from_config(
        {
            "model": {"encoder": "fake", "lora": {"enabled": True}},
            "paths": {
                "adapter_output_dir": str(adapter_dir),
                "heads_output_dir": str(heads_dir),
            },
        }
    )

    assert seen_configs
    assert seen_configs[0].lora_enabled is False
    assert backend.model.encoder == "loaded-adapter"


def test_direct_neural_backend_moves_runtime_model_to_cuda_when_available(monkeypatch, tmp_path):
    from src.runtime import neural_backend

    heads_dir = tmp_path / "heads"
    adapter_dir = tmp_path / "adapter"
    heads_dir.mkdir()
    adapter_dir.mkdir()
    (heads_dir / "heads.pt").write_bytes(b"fake-heads")
    (heads_dir / "architecture.json").write_text(
        '{"architecture": "direct_edit_tagger_v1", "debug_model": false}\n',
        encoding="utf-8",
    )

    class FakeHeads:
        def load_state_dict(self, state):
            self.state = state

    class FakeModule:
        def __init__(self) -> None:
            self.encoder = "plain-encoder"
            self.heads = FakeHeads()
            self.to_device = None

        def to(self, device):
            self.to_device = str(device)
            return self

        def eval(self) -> None:
            pass

    class FakeDirectEditTaggerModel:
        def __init__(self, config) -> None:
            self.module = FakeModule()

    import torch

    monkeypatch.setattr(neural_backend, "load_tokenizer", lambda config: "tokenizer")
    monkeypatch.setattr(neural_backend, "DirectEditTaggerModel", FakeDirectEditTaggerModel)
    monkeypatch.setattr(neural_backend, "_load_peft_adapter", lambda encoder, path, **kwargs: encoder)
    monkeypatch.setattr(torch, "load", lambda path, map_location=None: {"heads": "state"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    backend = neural_backend.DirectNeuralBackend.from_config(
        {
            "model": {"encoder": "fake", "lora": {"enabled": True}},
            "paths": {
                "adapter_output_dir": str(adapter_dir),
                "heads_output_dir": str(heads_dir),
            },
        }
    )

    assert str(backend.device) == "cuda"
    assert backend.model.to_device == "cuda"


def test_ruroberta_large_tokenizer_and_encoder_forward_pass_from_local_cache():
    torch = pytest.importorskip("torch")
    config = EncoderLoadConfig(
        model_name=PRIMARY_ENCODER,
        fallback_model_name=PRIMARY_ENCODER,
        local_files_only=True,
    )

    try:
        tokenizer = load_tokenizer(config)
        encoder = load_encoder(config)
    except OSError as exc:
        pytest.skip(f"{PRIMARY_ENCODER} is not available in the local Hugging Face cache: {exc}")

    encoded = tokenizer("Это короткий русский текст.", return_tensors="pt")
    encoder.eval()
    with torch.no_grad():
        output = encoder(**encoded)

    assert output.last_hidden_state.shape[0] == 1
    assert output.last_hidden_state.shape[1] == encoded["input_ids"].shape[1]
    assert output.last_hidden_state.shape[-1] == encoder.config.hidden_size
