import os
import sys
import types
from pathlib import Path

import pytest

from src.model.encoder import EncoderLoadConfig, PRIMARY_ENCODER, load_encoder, load_tokenizer
from src.schema.labels import (
    BOUNDARY_AFTER_ID_TO_LABEL,
    BOUNDARY_BEFORE_ID_TO_LABEL,
    GAP_ID_TO_LABEL,
    RULE_ID_TO_LABEL,
    TOKEN_ID_TO_LABEL,
)


def _write_label_maps(path: Path) -> None:
    path.write_text(
        __import__("json").dumps(
            {
                "token_id_to_label": list(TOKEN_ID_TO_LABEL),
                "gap_id_to_label": list(GAP_ID_TO_LABEL),
                "boundary_before_id_to_label": list(BOUNDARY_BEFORE_ID_TO_LABEL),
                "boundary_after_id_to_label": list(BOUNDARY_AFTER_ID_TO_LABEL),
                "rule_id_to_label": list(RULE_ID_TO_LABEL),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _write_legacy_rule_prefix_label_maps(path: Path) -> None:
    path.write_text(
        __import__("json").dumps(
            {
                "token_id_to_label": list(TOKEN_ID_TO_LABEL),
                "gap_id_to_label": list(GAP_ID_TO_LABEL),
                "boundary_before_id_to_label": list(BOUNDARY_BEFORE_ID_TO_LABEL),
                "boundary_after_id_to_label": list(BOUNDARY_AFTER_ID_TO_LABEL),
                "rule_id_to_label": list(RULE_ID_TO_LABEL[:-3]),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _write_legacy_token_gap_prefix_label_maps(path: Path) -> None:
    path.write_text(
        __import__("json").dumps(
            {
                "token_id_to_label": list(TOKEN_ID_TO_LABEL[:-1]),
                "gap_id_to_label": list(GAP_ID_TO_LABEL[:-1]),
                "rule_id_to_label": list(RULE_ID_TO_LABEL),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


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
    _write_label_maps(heads_dir / "labels.json")
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


def test_direct_neural_backend_loads_selected_checkpoint_inside_heads_latest(monkeypatch, tmp_path):
    from src.runtime import neural_backend

    heads_dir = tmp_path / "heads" / "latest"
    checkpoint_dir = heads_dir / "checkpoint-epoch-2"
    adapter_dir = tmp_path / "adapter"
    checkpoint_dir.mkdir(parents=True)
    adapter_dir.mkdir()
    (checkpoint_dir / "heads.pt").write_bytes(b"fake-heads")
    _write_label_maps(checkpoint_dir / "labels.json")
    (heads_dir / "architecture.json").write_text(
        '{"architecture": "direct_edit_tagger_v1", "debug_model": false, "selected_epoch": 2}\n',
        encoding="utf-8",
    )
    (heads_dir / "training_summary.json").write_text('{"selected_epoch": 2}\n', encoding="utf-8")

    class FakeHeads:
        def load_state_dict(self, state):
            self.state = state

    class FakeModule:
        def __init__(self) -> None:
            self.encoder = "plain-encoder"
            self.heads = FakeHeads()

        def eval(self) -> None:
            pass

    class FakeDirectEditTaggerModel:
        def __init__(self, config) -> None:
            self.module = FakeModule()

    loaded_paths = []

    import torch

    monkeypatch.setattr(neural_backend, "load_tokenizer", lambda config: "tokenizer")
    monkeypatch.setattr(neural_backend, "DirectEditTaggerModel", FakeDirectEditTaggerModel)
    monkeypatch.setattr(neural_backend, "_load_peft_adapter", lambda encoder, path, **kwargs: encoder)
    monkeypatch.setattr(torch, "load", lambda path, map_location=None: loaded_paths.append(path) or {"heads": "state"})

    backend = neural_backend.DirectNeuralBackend.from_config(
        {
            "model": {"encoder": "fake", "lora": {"enabled": True}},
            "paths": {
                "adapter_output_dir": str(adapter_dir),
                "heads_output_dir": str(heads_dir),
            },
        }
    )

    assert loaded_paths == [checkpoint_dir / "heads.pt"]
    assert backend.heads_path == str(checkpoint_dir / "heads.pt")
    assert backend.selected_epoch == 2


def test_direct_neural_backend_skips_legacy_rule_head_when_saved_rule_ids_were_extended(monkeypatch, tmp_path):
    from src.runtime import neural_backend

    heads_dir = tmp_path / "heads"
    adapter_dir = tmp_path / "adapter"
    heads_dir.mkdir()
    adapter_dir.mkdir()
    (heads_dir / "heads.pt").write_bytes(b"fake-heads")
    _write_legacy_rule_prefix_label_maps(heads_dir / "labels.json")
    (heads_dir / "architecture.json").write_text(
        '{"architecture": "direct_edit_tagger_v1", "debug_model": false}\n',
        encoding="utf-8",
    )

    loaded_state = {}

    class FakeHeads:
        def load_state_dict(self, state, strict=True):
            loaded_state["state"] = state
            loaded_state["strict"] = strict

    class FakeModule:
        def __init__(self) -> None:
            self.encoder = "plain-encoder"
            self.heads = FakeHeads()

        def eval(self) -> None:
            pass

    class FakeDirectEditTaggerModel:
        def __init__(self, config) -> None:
            self.module = FakeModule()

    import torch

    monkeypatch.setattr(neural_backend, "load_tokenizer", lambda config: "tokenizer")
    monkeypatch.setattr(neural_backend, "DirectEditTaggerModel", FakeDirectEditTaggerModel)
    monkeypatch.setattr(neural_backend, "_load_peft_adapter", lambda encoder, path, **kwargs: encoder)
    monkeypatch.setattr(
        torch,
        "load",
        lambda path, map_location=None: {
            "token_edit.weight": object(),
            "rule.weight": object(),
            "rule.bias": object(),
            "gap_punctuation.weight": object(),
        },
    )

    backend = neural_backend.DirectNeuralBackend.from_config(
        {
            "model": {"encoder": "fake", "lora": {"enabled": True}},
            "paths": {
                "adapter_output_dir": str(adapter_dir),
                "heads_output_dir": str(heads_dir),
            },
        }
    )

    assert backend.rule_head_loaded is False
    assert loaded_state["strict"] is False
    assert "rule.weight" not in loaded_state["state"]
    assert "rule.bias" not in loaded_state["state"]


def test_direct_neural_backend_partially_loads_legacy_token_and_gap_heads(monkeypatch, tmp_path):
    from src.runtime import neural_backend

    heads_dir = tmp_path / "heads"
    adapter_dir = tmp_path / "adapter"
    heads_dir.mkdir()
    adapter_dir.mkdir()
    (heads_dir / "heads.pt").write_bytes(b"fake-heads")
    _write_legacy_token_gap_prefix_label_maps(heads_dir / "labels.json")
    (heads_dir / "architecture.json").write_text(
        '{"architecture": "direct_edit_tagger_v1", "debug_model": false}\n',
        encoding="utf-8",
    )

    import torch

    token_count = len(TOKEN_ID_TO_LABEL)
    gap_count = len(GAP_ID_TO_LABEL)
    rule_count = len(RULE_ID_TO_LABEL)
    hidden_size = 3
    gap_hidden_size = 5
    loaded_state = {}

    class FakeHeads:
        def state_dict(self):
            return {
                "token_edit.weight": torch.zeros(token_count, hidden_size),
                "token_edit.bias": torch.zeros(token_count),
                "gap_punctuation.weight": torch.zeros(gap_count, gap_hidden_size),
                "gap_punctuation.bias": torch.zeros(gap_count),
                "rule.weight": torch.zeros(rule_count, hidden_size),
                "rule.bias": torch.zeros(rule_count),
                "token_confidence.weight": torch.zeros(1, hidden_size),
                "token_confidence.bias": torch.zeros(1),
                "gap_confidence.weight": torch.zeros(1, gap_hidden_size),
                "gap_confidence.bias": torch.zeros(1),
            }

        def load_state_dict(self, state, strict=True):
            loaded_state["state"] = state
            loaded_state["strict"] = strict

    class FakeModule:
        def __init__(self) -> None:
            self.encoder = "plain-encoder"
            self.heads = FakeHeads()

        def eval(self) -> None:
            pass

    class FakeDirectEditTaggerModel:
        def __init__(self, config) -> None:
            self.module = FakeModule()

    legacy_state = {
        "token_edit.weight": torch.ones(token_count - 1, hidden_size),
        "token_edit.bias": torch.ones(token_count - 1),
        "gap_punctuation.weight": torch.full((gap_count - 1, gap_hidden_size), 2.0),
        "gap_punctuation.bias": torch.full((gap_count - 1,), 2.0),
        "rule.weight": torch.full((rule_count, hidden_size), 3.0),
        "rule.bias": torch.full((rule_count,), 3.0),
        "token_confidence.weight": torch.full((1, hidden_size), 4.0),
        "token_confidence.bias": torch.full((1,), 4.0),
        "gap_confidence.weight": torch.full((1, gap_hidden_size), 5.0),
        "gap_confidence.bias": torch.full((1,), 5.0),
    }

    monkeypatch.setattr(neural_backend, "load_tokenizer", lambda config: "tokenizer")
    monkeypatch.setattr(neural_backend, "DirectEditTaggerModel", FakeDirectEditTaggerModel)
    monkeypatch.setattr(neural_backend, "_load_peft_adapter", lambda encoder, path, **kwargs: encoder)
    monkeypatch.setattr(torch, "load", lambda path, map_location=None: legacy_state)

    backend = neural_backend.DirectNeuralBackend.from_config(
        {
            "model": {"encoder": "fake", "lora": {"enabled": True}},
            "paths": {
                "adapter_output_dir": str(adapter_dir),
                "heads_output_dir": str(heads_dir),
            },
        }
    )

    state = loaded_state["state"]
    assert backend.rule_head_loaded is True
    assert backend.boundary_heads_loaded is False
    assert loaded_state["strict"] is False
    assert torch.equal(state["token_edit.weight"][:-1], legacy_state["token_edit.weight"])
    assert torch.equal(state["gap_punctuation.weight"][:-1], legacy_state["gap_punctuation.weight"])
    assert torch.equal(state["rule.weight"], legacy_state["rule.weight"])
    assert "boundary_before.weight" not in legacy_state
    assert "boundary_after.weight" not in legacy_state
    assert state["token_edit.bias"][-1].item() <= -1000.0
    assert state["gap_punctuation.bias"][-1].item() <= -1000.0


def test_direct_neural_backend_moves_runtime_model_to_cuda_when_available(monkeypatch, tmp_path):
    from src.runtime import neural_backend

    heads_dir = tmp_path / "heads"
    adapter_dir = tmp_path / "adapter"
    heads_dir.mkdir()
    adapter_dir.mkdir()
    (heads_dir / "heads.pt").write_bytes(b"fake-heads")
    _write_label_maps(heads_dir / "labels.json")
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
