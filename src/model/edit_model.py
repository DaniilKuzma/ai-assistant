from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.model.encoder import EncoderLoadConfig, ensure_pytorch_transformers_backend, load_encoder
from src.model.heads import build_linear_heads


@dataclass(frozen=True)
class EditModelConfig:
    model_name: str = "ai-forever/ruRoberta-large"
    fallback_model_name: str = "ai-forever/ruRoberta-large"
    punctuation_label_count: int = 13
    punctuation_action_count: int = 5
    error_type_count: int = 7
    local_files_only: bool = False
    lora_enabled: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05


class CandidateAwareEditModel:
    """Encoder-only multitask model for candidate-aware edit correction."""

    def __init__(self, config: EditModelConfig) -> None:
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
    def from_encoder(cls, encoder: Any, config: EditModelConfig) -> "CandidateAwareEditModel":
        instance = cls.__new__(cls)
        instance.config = config
        instance.module = cls._build_module(encoder, config)
        return instance

    @staticmethod
    def _build_module(encoder: Any, config: EditModelConfig) -> Any:
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
                candidate_spans=None,
                candidate_mask=None,
                candidate_replacement_ids=None,
                candidate_replacement_mask=None,
                punctuation_gap_indices=None,
                punctuation_right_gap_indices=None,
                punctuation_gap_mask=None,
                **kwargs,
            ):
                output = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
                hidden = output.last_hidden_state
                if candidate_spans is None:
                    batch_size = hidden.shape[0]
                    candidate_spans = torch.zeros((batch_size, 1, 2), dtype=torch.long, device=hidden.device)
                    candidate_spans[:, :, 1] = 1
                candidate_representations = _pool_candidate_spans(hidden, candidate_spans)
                replacement_representations = _pool_candidate_replacements(
                    self.encoder,
                    candidate_replacement_ids,
                    candidate_replacement_mask,
                    candidate_representations,
                )
                pair_representations = torch.cat(
                    [
                        candidate_representations,
                        replacement_representations,
                        replacement_representations - candidate_representations,
                    ],
                    dim=-1,
                )
                projected = self.heads["candidate_projection"](pair_representations)
                candidate_scores = self.heads["candidate_score"](projected).squeeze(-1)
                confidence_logits = self.heads["confidence"](projected).squeeze(-1)
                error_type_logits = self.heads["error_type"](projected)
                if candidate_mask is not None:
                    candidate_scores = candidate_scores.masked_fill(~candidate_mask, -1e4)
                    confidence_logits = confidence_logits.masked_fill(~candidate_mask, -1e4)
                punctuation_representations = hidden
                if punctuation_gap_indices is not None:
                    left_representations = _gather_gap_representations(hidden, punctuation_gap_indices)
                    if punctuation_right_gap_indices is None:
                        punctuation_right_gap_indices = punctuation_gap_indices
                    right_representations = _gather_gap_representations(hidden, punctuation_right_gap_indices)
                    cls_representations = hidden[:, :1, :].expand_as(left_representations)
                    punctuation_representations = torch.cat(
                        [
                            left_representations,
                            right_representations,
                            right_representations - left_representations,
                            cls_representations,
                        ],
                        dim=-1,
                    )
                    punctuation_representations = self.heads["punctuation_projection"](punctuation_representations)
                punctuation_confidence_logits = self.heads["punctuation_confidence"](punctuation_representations).squeeze(-1)
                return {
                    "hidden_states": hidden,
                    "candidate_scores": candidate_scores,
                    "punctuation_logits": self.heads["punctuation_gap"](punctuation_representations),
                    "punctuation_action_logits": self.heads["punctuation_action"](punctuation_representations),
                    "punctuation_confidence_logits": punctuation_confidence_logits,
                    "punctuation_error_type_logits": self.heads["punctuation_error_type"](punctuation_representations),
                    "confidence_logits": confidence_logits,
                    "error_type_logits": error_type_logits,
                }

        hidden_size = _encoder_hidden_size(encoder)
        return _Module(
            encoder,
            build_linear_heads(
                hidden_size,
                config.punctuation_label_count,
                config.error_type_count,
                config.punctuation_action_count,
            ),
        )

    def _attach_lora(self, encoder: Any, config: EditModelConfig) -> Any:
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


def _pool_candidate_spans(hidden, candidate_spans):  # type: ignore[no-untyped-def]
    import torch

    batch_size, candidate_count, _ = candidate_spans.shape
    hidden_size = hidden.shape[-1]
    pooled = torch.zeros((batch_size, candidate_count, hidden_size), dtype=hidden.dtype, device=hidden.device)
    sequence_length = hidden.shape[1]

    for batch_index in range(batch_size):
        for candidate_index in range(candidate_count):
            start = int(candidate_spans[batch_index, candidate_index, 0].item())
            end = int(candidate_spans[batch_index, candidate_index, 1].item())
            start = max(0, min(start, sequence_length - 1))
            end = max(start + 1, min(end, sequence_length))
            pooled[batch_index, candidate_index] = hidden[batch_index, start:end].mean(dim=0)

    return pooled


def _pool_candidate_replacements(encoder, replacement_ids, replacement_mask, fallback_representations):  # type: ignore[no-untyped-def]
    import torch

    if replacement_ids is None:
        return torch.zeros_like(fallback_representations)

    replacement_ids = replacement_ids.to(fallback_representations.device)
    if replacement_mask is None:
        replacement_mask = replacement_ids.ne(0)
    replacement_mask = replacement_mask.to(fallback_representations.device).float()

    embeddings = _input_embeddings(encoder)(replacement_ids)
    masked = embeddings * replacement_mask.unsqueeze(-1)
    lengths = replacement_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return masked.sum(dim=-2) / lengths


def _input_embeddings(encoder):  # type: ignore[no-untyped-def]
    if hasattr(encoder, "get_input_embeddings"):
        embeddings = encoder.get_input_embeddings()
        if embeddings is not None:
            return embeddings
    if hasattr(encoder, "embeddings"):
        return encoder.embeddings
    base_model = getattr(encoder, "base_model", None)
    if base_model is not None and hasattr(base_model, "get_input_embeddings"):
        embeddings = base_model.get_input_embeddings()
        if embeddings is not None:
            return embeddings
    raise AttributeError("Cannot resolve encoder input embeddings for candidate replacement pooling")


def _gather_gap_representations(hidden, gap_indices):  # type: ignore[no-untyped-def]
    gap_indices = gap_indices.to(hidden.device)
    sequence_length = hidden.shape[1]
    clamped = gap_indices.clamp(min=0, max=max(0, sequence_length - 1))
    expanded = clamped.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
    return hidden.gather(dim=1, index=expanded)


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
