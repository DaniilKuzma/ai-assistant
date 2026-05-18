from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


PRIMARY_ENCODER = "ai-forever/ruRoberta-large"
FALLBACK_ENCODER = "ai-forever/ruRoberta-large"


@dataclass(frozen=True)
class EncoderLoadConfig:
    model_name: str = PRIMARY_ENCODER
    fallback_model_name: str = FALLBACK_ENCODER
    local_files_only: bool = False
    trust_remote_code: bool = False
    add_pooling_layer: bool = False


def load_tokenizer(config: EncoderLoadConfig) -> Any:
    _disable_unused_transformers_backends()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        config.model_name,
        local_files_only=config.local_files_only,
        trust_remote_code=config.trust_remote_code,
    )


def load_encoder(config: EncoderLoadConfig) -> Any:
    """Load an encoder-only base model. Seq2Seq classes are intentionally absent."""

    _disable_unused_transformers_backends()
    from transformers import AutoModel

    try:
        return _load_auto_model(AutoModel, config.model_name, config)
    except Exception:
        if config.model_name == config.fallback_model_name:
            raise
        return _load_auto_model(AutoModel, config.fallback_model_name, config)


def _load_auto_model(auto_model: Any, model_name: str, config: EncoderLoadConfig) -> Any:
    return auto_model.from_pretrained(
        model_name,
        local_files_only=config.local_files_only,
        trust_remote_code=config.trust_remote_code,
        add_pooling_layer=config.add_pooling_layer,
    )


def _disable_unused_transformers_backends() -> None:
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_FLAX", "0")
