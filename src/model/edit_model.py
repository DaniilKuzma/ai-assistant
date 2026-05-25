from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.model.encoder import EncoderLoadConfig, ensure_pytorch_transformers_backend, load_encoder
from src.model.heads import build_direct_edit_heads
from src.schema.labels import GAP_PUNCTUATION_LABELS, RULE_LABELS, TOKEN_EDIT_LABELS


@dataclass(frozen=True)
class DirectEditModelConfig:
    model_name: str = "ai-forever/ruRoberta-large"
    fallback_model_name: str = "ai-forever/ruRoberta-large"
    token_label_count: int = len(TOKEN_EDIT_LABELS)
    gap_label_count: int = len(GAP_PUNCTUATION_LABELS)
    rule_tag_count: int = len(RULE_LABELS)
    local_files_only: bool = False
    lora_enabled: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05


class DirectEditTaggerModel:
    def __init__(self, config: DirectEditModelConfig) -> None:
        self.config = config
        encoder_config = EncoderLoadConfig(
            model_name=config.model_name,
            fallback_model_name=config.fallback_model_name,
            local_files_only=config.local_files_only,
        )
        encoder = load_encoder(encoder_config)
        if config.lora_enabled:
            encoder = self._attach_lora(encoder, config)
        self.module = self._build_module(encoder, config)

    @classmethod
    def from_encoder(cls, encoder: Any, config: DirectEditModelConfig) -> "DirectEditTaggerModel":
        instance = cls.__new__(cls)
        instance.config = config
        instance.module = cls._build_module(encoder, config)
        return instance

    @staticmethod
    def _build_module(encoder: Any, config: DirectEditModelConfig) -> Any:
        import torch
        import torch.nn as nn

        class _Module(nn.Module):
            def __init__(self, base: Any, heads: dict[str, nn.Module]) -> None:
                super().__init__()
                self.encoder = base
                self.heads = nn.ModuleDict(heads)

            def forward(  # type: ignore[no-untyped-def]
                self,
                input_ids,
                attention_mask=None,
                word_token_indices=None,
                word_token_mask=None,
                gap_left_indices=None,
                gap_right_indices=None,
                gap_mask=None,
                **kwargs,
            ):
                output = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
                hidden = output.last_hidden_state
                batch_size = hidden.shape[0]

                if word_token_indices is None:
                    word_token_indices = torch.zeros((batch_size, 0), dtype=torch.long, device=hidden.device)
                word_representations = _gather_hidden(hidden, word_token_indices)
                token_edit_logits = self.heads["token_edit"](word_representations)
                token_confidence_logits = self.heads["token_confidence"](word_representations).squeeze(-1)
                rule_logits = self.heads["rule"](word_representations)

                word_token_mask = _mask_or_ones(word_token_mask, word_token_indices, hidden.device)
                token_edit_logits = _zero_masked(token_edit_logits, word_token_mask)
                token_confidence_logits = token_confidence_logits.masked_fill(~word_token_mask, 0.0)
                rule_logits = _zero_masked(rule_logits, word_token_mask)

                if gap_left_indices is None:
                    gap_left_indices = torch.zeros((batch_size, 0), dtype=torch.long, device=hidden.device)
                gap_representations = _gap_representations(hidden, gap_left_indices, gap_right_indices)
                gap_punctuation_logits = self.heads["gap_punctuation"](gap_representations)
                gap_confidence_logits = self.heads["gap_confidence"](gap_representations).squeeze(-1)

                gap_mask = _mask_or_ones(gap_mask, gap_left_indices, hidden.device)
                gap_punctuation_logits = _zero_masked(gap_punctuation_logits, gap_mask)
                gap_confidence_logits = gap_confidence_logits.masked_fill(~gap_mask, 0.0)

                return {
                    "hidden_states": hidden,
                    "token_edit_logits": token_edit_logits,
                    "token_confidence_logits": token_confidence_logits,
                    "gap_punctuation_logits": gap_punctuation_logits,
                    "gap_confidence_logits": gap_confidence_logits,
                    "rule_logits": rule_logits,
                }

        hidden_size = _encoder_hidden_size(encoder)
        return _Module(
            encoder,
            build_direct_edit_heads(
                hidden_size,
                config.token_label_count,
                config.gap_label_count,
                config.rule_tag_count,
            ),
        )

    def _attach_lora(self, encoder: Any, config: DirectEditModelConfig) -> Any:
        ensure_pytorch_transformers_backend()
        from peft import LoraConfig, TaskType, get_peft_model

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["query", "key", "value", "dense"],
        )
        return get_peft_model(encoder, lora_config)


def _gather_hidden(hidden, indices):  # type: ignore[no-untyped-def]
    import torch

    indices = indices.to(device=hidden.device, dtype=torch.long)
    sequence_length = hidden.shape[1]
    clamped = indices.clamp(min=0, max=max(0, sequence_length - 1))
    expanded = clamped.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
    return hidden.gather(dim=1, index=expanded)


def _gap_representations(hidden, left_indices, right_indices):  # type: ignore[no-untyped-def]
    import torch

    left_hidden = _gather_hidden(hidden, left_indices)
    cls_hidden = hidden[:, :1, :].expand_as(left_hidden)
    if right_indices is None:
        right_hidden = cls_hidden
    else:
        right_indices = right_indices.to(device=hidden.device, dtype=torch.long)
        right_missing = right_indices < 0
        right_hidden = _gather_hidden(hidden, right_indices)
        right_hidden = torch.where(right_missing.unsqueeze(-1), cls_hidden, right_hidden)
    return torch.cat([left_hidden, right_hidden, right_hidden - left_hidden], dim=-1)


def _mask_or_ones(mask, reference_indices, device):  # type: ignore[no-untyped-def]
    import torch

    if mask is None:
        return torch.ones(reference_indices.shape, dtype=torch.bool, device=device)
    return mask.to(device=device, dtype=torch.bool)


def _zero_masked(logits, mask):  # type: ignore[no-untyped-def]
    return logits.masked_fill(~mask.unsqueeze(-1), 0.0)


def _encoder_hidden_size(encoder: Any) -> int:
    config = getattr(encoder, "config", None)
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)

    base_model = getattr(encoder, "base_model", None)
    for candidate in (
        getattr(base_model, "config", None),
        getattr(getattr(base_model, "model", None), "config", None),
    ):
        hidden_size = getattr(candidate, "hidden_size", None)
        if hidden_size is not None:
            return int(hidden_size)

    raise AttributeError("Cannot resolve encoder hidden_size from encoder config")


__all__ = ["DirectEditModelConfig", "DirectEditTaggerModel"]

