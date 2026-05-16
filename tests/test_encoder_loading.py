from src.model.encoder import EncoderLoadConfig, load_encoder


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
