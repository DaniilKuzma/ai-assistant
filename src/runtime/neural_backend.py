from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from src.model.edit_model import DirectEditModelConfig, DirectEditTaggerModel
from src.model.encoder import EncoderLoadConfig, ensure_pytorch_transformers_backend, load_tokenizer
from src.schema.labels import GAP_ID_TO_LABEL, RULE_ID_TO_LABEL, TOKEN_ID_TO_LABEL
from src.runtime.tokenization import tokenize_runtime_words


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DirectPrediction:
    token_labels: list[str]
    token_confidences: list[float]
    gap_labels: list[str]
    gap_confidences: list[float]
    rule_ids: list[str]
    token_margins: list[float]
    gap_margins: list[float]


class DirectNeuralBackend:
    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        max_length: int = 128,
        device: Any | None = None,
        *,
        adapter_path: str | Path | None = None,
        heads_path: str | Path | None = None,
        selected_epoch: int | None = None,
    ) -> None:
        import torch

        self.tokenizer = tokenizer
        self.model = model
        self.max_length = max_length
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adapter_path = str(adapter_path or "")
        self.heads_path = str(heads_path or "")
        self.selected_epoch = selected_epoch

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "DirectNeuralBackend":
        paths = config.get("paths", {}) if isinstance(config, Mapping) else {}
        adapter_dir = _resolve_path(paths.get("adapter_output_dir", "models/adapters/latest"))
        heads_dir = _resolve_path(paths.get("heads_output_dir", "models/heads/latest"))
        heads_path = heads_dir / "heads.pt"
        if not heads_path.exists():
            raise FileNotFoundError(f"Direct edit heads artifact is missing: {heads_path}")
        _validate_architecture_marker(heads_dir, config)

        model_config = config.get("model", {}) if isinstance(config, Mapping) else {}
        lora = model_config.get("lora", {}) if isinstance(model_config, Mapping) else {}
        lora_enabled = bool(lora.get("enabled", True)) if isinstance(lora, Mapping) else True
        runtime = config.get("runtime", {}) if isinstance(config, Mapping) else {}
        allow_adapter_fallback = bool(runtime.get("allow_neural_adapter_fallback", False)) if isinstance(runtime, Mapping) else False
        if lora_enabled and not adapter_dir.exists():
            if allow_adapter_fallback:
                lora_enabled = False
            else:
                raise FileNotFoundError(f"Direct edit adapter artifact is missing: {adapter_dir}")

        tokenizer = load_tokenizer(_encoder_config(model_config))
        direct = DirectEditTaggerModel(_direct_model_config(model_config, lora_enabled=False)).module
        if lora_enabled:
            direct.encoder = _load_peft_adapter(
                direct.encoder,
                adapter_dir,
                allow_fallback=allow_adapter_fallback,
            )

        import torch

        state = torch.load(heads_path, map_location="cpu")
        direct.heads.load_state_dict(state)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(direct, "to"):
            direct.to(device)
        direct.eval()
        return cls(
            tokenizer,
            direct,
            int(model_config.get("max_sequence_length", 128)),
            device=device,
            adapter_path=adapter_dir if lora_enabled else "",
            heads_path=heads_path,
            selected_epoch=_selected_epoch(heads_dir),
        )

    def predict(self, text: str) -> DirectPrediction:
        import torch
        import torch.nn.functional as functional

        words = tokenize_runtime_words(text)
        encoded = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        offsets = _offsets(encoded.pop("offset_mapping"))
        word_indices, word_mask = _word_indices(words, offsets, self.max_length)
        gap_left, gap_right, gap_mask = _gap_indices(words, offsets, word_indices, word_mask, self.max_length)

        batch = {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
            "word_token_indices": torch.tensor([word_indices], dtype=torch.long, device=self.device),
            "word_token_mask": torch.tensor([word_mask], dtype=torch.bool, device=self.device),
            "gap_left_indices": torch.tensor([gap_left], dtype=torch.long, device=self.device),
            "gap_right_indices": torch.tensor([gap_right], dtype=torch.long, device=self.device),
            "gap_mask": torch.tensor([gap_mask], dtype=torch.bool, device=self.device),
        }

        with torch.no_grad():
            outputs = self.model(**batch)

        token_probs = functional.softmax(outputs["token_edit_logits"][0], dim=-1)
        gap_probs = functional.softmax(outputs["gap_punctuation_logits"][0], dim=-1)
        rule_ids = outputs["rule_logits"][0].argmax(dim=-1).tolist()
        token_top = token_probs.topk(k=min(2, token_probs.shape[-1]), dim=-1)
        gap_top = gap_probs.topk(k=min(2, gap_probs.shape[-1]), dim=-1)

        token_count = len(words)
        return DirectPrediction(
            token_labels=[TOKEN_ID_TO_LABEL[int(index)] for index in token_probs.argmax(dim=-1).tolist()[:token_count]],
            token_confidences=[float(value) for value in token_top.values[:, 0].tolist()[:token_count]],
            gap_labels=[GAP_ID_TO_LABEL[int(index)] for index in gap_probs.argmax(dim=-1).tolist()[:token_count]],
            gap_confidences=[float(value) for value in gap_top.values[:, 0].tolist()[:token_count]],
            rule_ids=[RULE_ID_TO_LABEL[int(index)] for index in rule_ids[:token_count]],
            token_margins=_margins(token_top.values.tolist(), token_count),
            gap_margins=_margins(gap_top.values.tolist(), token_count),
        )


def _encoder_config(model_config: Mapping[str, Any]) -> EncoderLoadConfig:
    model_name = str(model_config.get("encoder") or model_config.get("primary_encoder") or "ai-forever/ruRoberta-large")
    return EncoderLoadConfig(
        model_name=model_name,
        fallback_model_name=str(model_config.get("fallback_encoder") or model_name),
        local_files_only=bool(model_config.get("local_files_only", False)),
    )


def _direct_model_config(model_config: Mapping[str, Any], *, lora_enabled: bool | None = None) -> DirectEditModelConfig:
    lora = model_config.get("lora", {}) if isinstance(model_config, Mapping) else {}
    encoder_config = _encoder_config(model_config)
    return DirectEditModelConfig(
        model_name=encoder_config.model_name,
        fallback_model_name=encoder_config.fallback_model_name,
        local_files_only=encoder_config.local_files_only,
        lora_enabled=bool(lora.get("enabled", True)) if lora_enabled is None and isinstance(lora, Mapping) else bool(lora_enabled),
        lora_r=int(lora.get("r", 8)) if isinstance(lora, Mapping) else 8,
        lora_alpha=int(lora.get("alpha", 16)) if isinstance(lora, Mapping) else 16,
        lora_dropout=float(lora.get("dropout", 0.05)) if isinstance(lora, Mapping) else 0.05,
    )


def _load_peft_adapter(encoder: Any, adapter_dir: Path, *, allow_fallback: bool = False) -> Any:
    ensure_pytorch_transformers_backend()
    try:
        from peft import PeftModel

        return PeftModel.from_pretrained(encoder, adapter_dir)
    except Exception as exc:
        if allow_fallback:
            return encoder
        raise RuntimeError(f"Failed to load PEFT adapter from {adapter_dir}: {exc}") from exc


def _resolve_path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _validate_architecture_marker(heads_dir: Path, config: Mapping[str, Any]) -> None:
    marker_path = heads_dir / "architecture.json"
    if not marker_path.exists():
        raise RuntimeError(f"Direct edit architecture marker is missing: {marker_path}")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Direct edit architecture marker is invalid: {marker_path}") from exc

    architecture = str(marker.get("architecture") or "")
    if architecture != "direct_edit_tagger_v1":
        raise RuntimeError(f"Unsupported direct edit architecture: {architecture!r}")

    runtime = config.get("runtime", {}) if isinstance(config, Mapping) else {}
    allow_debug_model = bool(runtime.get("allow_debug_model", False)) if isinstance(runtime, Mapping) else False
    if bool(marker.get("debug_model", False)) and not allow_debug_model:
        raise RuntimeError("Refusing to load debug_model direct edit heads without allow_debug_model=true.")


def _selected_epoch(heads_dir: Path) -> int | None:
    for name in ("checkpoint_meta.json", "selected_checkpoint_meta.json", "architecture.json"):
        path = heads_dir / name
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        value = data.get("selected_epoch") or data.get("epoch")
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _offsets(value: Any) -> list[tuple[int, int]]:
    raw = value[0].tolist() if hasattr(value, "tolist") else value[0]
    return [(int(start), int(end)) for start, end in raw]


def _word_indices(words: list[Any], offsets: list[tuple[int, int]], max_length: int) -> tuple[list[int], list[bool]]:
    indices = [0] * max_length
    mask = [False] * max_length
    for index, word in enumerate(words[:max_length]):
        token_index = _first_overlap(offsets, word.start, word.end)
        if token_index is None:
            continue
        indices[index] = token_index
        mask[index] = True
    return indices, mask


def _gap_indices(
    words: list[Any],
    offsets: list[tuple[int, int]],
    word_indices: list[int],
    word_mask: list[bool],
    max_length: int,
) -> tuple[list[int], list[int], list[bool]]:
    left = [0] * max_length
    right = [-1] * max_length
    mask = [False] * max_length
    for index, word in enumerate(words[:max_length]):
        if not word_mask[index]:
            continue
        left[index] = word_indices[index]
        if index + 1 < len(words):
            next_index = _first_overlap(offsets, words[index + 1].start, words[index + 1].end)
            right[index] = -1 if next_index is None else next_index
        mask[index] = True
    return left, right, mask


def _first_overlap(offsets: list[tuple[int, int]], start: int, end: int) -> int | None:
    for index, (token_start, token_end) in enumerate(offsets):
        if token_end > token_start and token_start < end and start < token_end:
            return index
    return None


def _margins(values: list[list[float]], count: int) -> list[float]:
    margins: list[float] = []
    for row in values[:count]:
        if len(row) < 2:
            margins.append(float(row[0]) if row else 1.0)
        else:
            margins.append(float(row[0]) - float(row[1]))
    return margins


__all__ = ["DirectNeuralBackend", "DirectPrediction"]
