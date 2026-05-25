import os
import sys
import types

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
