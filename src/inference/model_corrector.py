from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.candidates.candidate_generator import Candidate, CandidateGenerator
from src.inference.corrector import CorrectionResult
from src.inference.edit_realizer import apply_candidate, ensure_final_punctuation
from src.inference.iterative_decoder import run_until_stable
from src.inference.postprocess import normalize_spacing
from src.model.edit_model import CandidateAwareEditModel, EditModelConfig
from src.model.encoder import EncoderLoadConfig, load_encoder, load_tokenizer
from src.preprocessing.tokenizer import tokenize_words
from src.validation.strict_validator import StrictValidator


MAX_REPLACEMENT_TOKENS = 8


@dataclass(frozen=True)
class ModelCandidatePrediction:
    candidate: Candidate
    score: float
    confidence: float


@dataclass(frozen=True)
class ModelPunctuationPrediction:
    word_index: int
    label: str
    confidence: float


class CandidateModelBackend(Protocol):
    def score_candidates(self, text: str, candidates: list[Candidate]) -> list[ModelCandidatePrediction]:
        ...

    def predict_punctuation(self, text: str) -> list[ModelPunctuationPrediction]:
        ...


class TrainedModelCorrector:
    """Inference facade backed by fine-tuned LoRA adapters and custom heads."""

    def __init__(
        self,
        backend: CandidateModelBackend,
        *,
        thresholds: dict[str, float] | None = None,
        max_passes: int = 3,
    ) -> None:
        self.backend = backend
        self.thresholds = thresholds or {}
        self.max_passes = max_passes
        self.candidates = CandidateGenerator()
        self.validator = StrictValidator()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TrainedModelCorrector":
        threshold_config = config.get("thresholds", {})
        mode = threshold_config.get("mode", "balanced")
        thresholds = threshold_config.get(mode, {})
        backend = TorchCandidateModelBackend.from_config(config)
        return cls(backend, thresholds=thresholds, max_passes=int(config.get("decoder", {}).get("max_passes", 3)))

    def correct(self, text: str) -> CorrectionResult:
        proposed = run_until_stable(text, self._single_pass, self.max_passes)
        validation = self.validator.validate(text, proposed)
        corrected = validation.apply_accepted()
        return CorrectionResult(text, corrected, validation.edits)

    def _single_pass(self, text: str) -> str:
        candidates = self.candidates.generate(text)
        selected = _select_candidates(self.backend.score_candidates(text, candidates), self.thresholds)
        proposed = _apply_candidates(text, selected)
        proposed = _apply_punctuation_predictions(proposed, self.backend.predict_punctuation(proposed), self.thresholds)
        proposed = ensure_final_punctuation(proposed, ".")
        return normalize_spacing(proposed)


class TorchCandidateModelBackend:
    def __init__(
        self,
        *,
        tokenizer: Any,
        module: Any,
        device: Any,
        punctuation_labels: dict[str, int],
        max_length: int,
        max_candidates: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.module = module
        self.device = device
        self.punctuation_labels = punctuation_labels
        self.punctuation_by_id = {value: key for key, value in punctuation_labels.items()}
        self.max_length = max_length
        self.max_candidates = max_candidates

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TorchCandidateModelBackend":
        import torch
        from peft import PeftModel

        model_config = config.get("model", {})
        paths = config.get("paths", {})
        adapter_dir = Path(paths.get("adapter_output_dir", "models/adapters/latest"))
        heads_path = Path(paths.get("heads_output_dir", "models/heads/latest")) / "heads.pt"
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Trained adapter directory not found: {adapter_dir}")
        if not heads_path.exists():
            raise FileNotFoundError(f"Trained heads checkpoint not found: {heads_path}")

        encoder_load_config = EncoderLoadConfig(
            model_name=model_config.get("primary_encoder", "ai-forever/ruRoberta-large"),
            fallback_model_name=model_config.get("fallback_encoder", "ai-forever/ruBert-base"),
            local_files_only=bool(model_config.get("local_files_only", False)),
        )
        tokenizer = load_tokenizer(encoder_load_config)
        base_encoder = load_encoder(encoder_load_config)
        encoder = PeftModel.from_pretrained(
            base_encoder,
            adapter_dir,
            local_files_only=bool(model_config.get("local_files_only", False)),
        )
        edit_model = CandidateAwareEditModel.from_encoder(
            encoder,
            EditModelConfig(
                model_name=model_config.get("primary_encoder", "ai-forever/ruRoberta-large"),
                fallback_model_name=model_config.get("fallback_encoder", "ai-forever/ruBert-base"),
                punctuation_label_count=len(config.get("labels", {}).get("punctuation", {})),
                error_type_count=len(config.get("labels", {}).get("error_types", {})),
                local_files_only=bool(model_config.get("local_files_only", False)),
                lora_enabled=False,
            ),
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        module = edit_model.module.to(device)
        module.heads.load_state_dict(torch.load(heads_path, map_location=device))
        module.eval()
        return cls(
            tokenizer=tokenizer,
            module=module,
            device=device,
            punctuation_labels=config.get("labels", {}).get("punctuation", {}),
            max_length=int(model_config.get("max_sequence_length", 128)),
            max_candidates=int(model_config.get("max_candidates", 16)),
        )

    def score_candidates(self, text: str, candidates: list[Candidate]) -> list[ModelCandidatePrediction]:
        import torch

        active_candidates = candidates[: self.max_candidates]
        encoded = _encode(self.tokenizer, text, self.max_length)
        spans = [_candidate_to_token_span(candidate, encoded["offset_mapping"]) for candidate in active_candidates]
        replacements = [_encode_replacement(self.tokenizer, candidate.replacement) for candidate in active_candidates]
        replacement_ids = [replacement["input_ids"] for replacement in replacements]
        replacement_masks = [[bool(value) for value in replacement["attention_mask"]] for replacement in replacements]
        candidate_mask = [True] * len(active_candidates)
        pad_count = self.max_candidates - len(active_candidates)
        spans.extend([(0, 0)] * pad_count)
        replacement_ids.extend([[0] * MAX_REPLACEMENT_TOKENS for _ in range(pad_count)])
        replacement_masks.extend([[False] * MAX_REPLACEMENT_TOKENS for _ in range(pad_count)])
        candidate_mask.extend([False] * pad_count)

        with torch.no_grad():
            outputs = self.module(
                input_ids=torch.tensor([encoded["input_ids"]], dtype=torch.long, device=self.device),
                attention_mask=torch.tensor([encoded["attention_mask"]], dtype=torch.long, device=self.device),
                candidate_spans=torch.tensor([spans], dtype=torch.long, device=self.device),
                candidate_mask=torch.tensor([candidate_mask], dtype=torch.bool, device=self.device),
                candidate_replacement_ids=torch.tensor([replacement_ids], dtype=torch.long, device=self.device),
                candidate_replacement_mask=torch.tensor([replacement_masks], dtype=torch.bool, device=self.device),
            )
            scores = torch.sigmoid(outputs["candidate_scores"][0]).detach().cpu().tolist()
            confidences = torch.sigmoid(outputs["confidence_logits"][0]).detach().cpu().tolist()

        return [
            ModelCandidatePrediction(candidate=candidate, score=float(scores[index]), confidence=float(confidences[index]))
            for index, candidate in enumerate(active_candidates)
        ]

    def predict_punctuation(self, text: str) -> list[ModelPunctuationPrediction]:
        import torch

        if not self.punctuation_labels:
            return []
        encoded = _encode(self.tokenizer, text, self.max_length)
        with torch.no_grad():
            outputs = self.module(
                input_ids=torch.tensor([encoded["input_ids"]], dtype=torch.long, device=self.device),
                attention_mask=torch.tensor([encoded["attention_mask"]], dtype=torch.long, device=self.device),
            )
            probabilities = torch.softmax(outputs["punctuation_logits"][0], dim=-1).detach().cpu()

        predictions: list[ModelPunctuationPrediction] = []
        for word_index, _word in enumerate(tokenize_words(text)):
            if word_index >= probabilities.shape[0]:
                break
            confidence, label_id = probabilities[word_index].max(dim=-1)
            label = self.punctuation_by_id.get(int(label_id.item()), "NONE")
            predictions.append(ModelPunctuationPrediction(word_index, label, float(confidence.item())))
        return predictions


def _select_candidates(
    predictions: list[ModelCandidatePrediction],
    thresholds: dict[str, float],
) -> list[Candidate]:
    selected: list[Candidate] = []
    occupied: list[tuple[int, int]] = []
    for prediction in sorted(predictions, key=lambda item: (item.candidate.start, -item.score)):
        candidate = prediction.candidate
        if candidate.edit_type == "keep":
            continue
        if prediction.score < _threshold_for_candidate(candidate, thresholds):
            continue
        if any(candidate.start < end and start < candidate.end for start, end in occupied):
            continue
        selected.append(
            Candidate(
                source=candidate.source,
                replacement=candidate.replacement,
                edit_type=candidate.edit_type,
                start=candidate.start,
                end=candidate.end,
                confidence=prediction.confidence,
            )
        )
        occupied.append((candidate.start, candidate.end))
    return selected


def _threshold_for_candidate(candidate: Candidate, thresholds: dict[str, float]) -> float:
    if candidate.edit_type == "split_join":
        return float(thresholds.get("split_join_threshold", 0.88))
    if candidate.edit_type == "hyphen":
        return float(thresholds.get("hyphen_threshold", 0.88))
    if candidate.edit_type == "case":
        return float(thresholds.get("case_threshold", 0.85))
    return float(thresholds.get("spelling_threshold", 0.85))


def _apply_candidates(text: str, candidates: list[Candidate]) -> str:
    proposed = text
    offset = 0
    for candidate in sorted(candidates, key=lambda item: item.start):
        shifted = Candidate(
            source=candidate.source,
            replacement=candidate.replacement,
            edit_type=candidate.edit_type,
            start=candidate.start + offset,
            end=candidate.end + offset,
            confidence=candidate.confidence,
        )
        before = proposed
        proposed = apply_candidate(proposed, shifted)
        offset += len(proposed) - len(before)
    return proposed


def _apply_punctuation_predictions(
    text: str,
    predictions: list[ModelPunctuationPrediction],
    thresholds: dict[str, float],
) -> str:
    punctuation_threshold = float(thresholds.get("punctuation_threshold", 0.82))
    proposed = text
    for prediction in predictions:
        if prediction.confidence < punctuation_threshold:
            continue
        if prediction.label == "NONE":
            proposed = _delete_punctuation_after_word(proposed, prediction.word_index)
            continue
        if prediction.label in {"QUOTE_OPEN", "BRACKET_OPEN"}:
            proposed = _insert_punctuation_before_word(proposed, prediction.word_index, _punctuation_mark(prediction.label))
        elif prediction.label in {"DOT", "QUESTION", "EXCLAMATION", "ELLIPSIS"} and _is_last_word_index(proposed, prediction.word_index):
            proposed = _replace_final_punctuation(proposed, _punctuation_mark(prediction.label))
        elif prediction.label in {
            "COMMA",
            "DOT",
            "QUESTION",
            "EXCLAMATION",
            "COLON",
            "DASH",
            "SEMICOLON",
            "ELLIPSIS",
            "QUOTE_CLOSE",
            "BRACKET_CLOSE",
        }:
            proposed = _insert_punctuation_after_word(proposed, prediction.word_index, _punctuation_mark(prediction.label))
    return proposed


def _encode(tokenizer: Any, text: str, max_length: int) -> dict[str, Any]:
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    input_ids = _to_list(encoded["input_ids"])
    attention_mask = _to_list(encoded["attention_mask"])
    offsets = [tuple(item) for item in _to_list(encoded["offset_mapping"])]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "offset_mapping": offsets}


def _encode_replacement(tokenizer: Any, text: str) -> dict[str, list[Any]]:
    kwargs = {
        "truncation": True,
        "padding": "max_length",
        "max_length": MAX_REPLACEMENT_TOKENS,
        "add_special_tokens": False,
    }
    try:
        encoded = tokenizer(text, return_offsets_mapping=False, **kwargs)
    except TypeError:
        encoded = tokenizer(
            text,
            return_offsets_mapping=False,
            truncation=True,
            padding="max_length",
            max_length=MAX_REPLACEMENT_TOKENS,
        )
    return {
        "input_ids": _to_list(encoded["input_ids"]),
        "attention_mask": _to_list(encoded["attention_mask"]),
    }


def _candidate_to_token_span(candidate: Candidate, offsets: list[tuple[int, int]]) -> tuple[int, int]:
    token_indexes = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > start and start < candidate.end and candidate.start < end
    ]
    if not token_indexes:
        return (0, 1)
    return (min(token_indexes), max(token_indexes) + 1)


def _to_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _punctuation_mark(label: str) -> str:
    return {
        "COMMA": ",",
        "DOT": ".",
        "QUESTION": "?",
        "EXCLAMATION": "!",
        "COLON": ":",
        "DASH": "—",
        "SEMICOLON": ";",
        "ELLIPSIS": "…",
        "QUOTE_OPEN": "«",
        "QUOTE_CLOSE": "»",
        "BRACKET_OPEN": "(",
        "BRACKET_CLOSE": ")",
    }.get(label, "")


def _is_last_word_index(text: str, word_index: int) -> bool:
    words = tokenize_words(text)
    return bool(words) and word_index == len(words) - 1


def _replace_final_punctuation(text: str, mark: str) -> str:
    if not mark:
        return text
    stripped = text.rstrip()
    if stripped and stripped[-1] in ".!?…":
        stripped = stripped[:-1]
    return stripped + mark


def _insert_punctuation_after_word(text: str, word_index: int, mark: str) -> str:
    if not mark:
        return text
    words = tokenize_words(text)
    if word_index < 0 or word_index >= len(words):
        return text
    position = words[word_index].end
    punct_position = _punctuation_position_after(text, position)
    if mark == "—":
        if punct_position is not None and text[punct_position] == mark:
            return text
        if punct_position is not None and text[punct_position] in ",.!?:;—…":
            return text[:position].rstrip() + " — " + text[punct_position + 1 :].lstrip()
        return text[:position].rstrip() + " — " + text[position:].lstrip()
    if punct_position is not None and text[punct_position] in ",.!?:;—…":
        if text[punct_position] == mark:
            return text
        return text[:punct_position] + mark + text[punct_position + 1 :]
    if punct_position is not None and text[punct_position] == mark:
        return text
    return text[:position] + mark + text[position:]


def _insert_punctuation_before_word(text: str, word_index: int, mark: str) -> str:
    if not mark:
        return text
    words = tokenize_words(text)
    if word_index < 0 or word_index >= len(words):
        return text
    position = words[word_index].start
    if position < len(text) and text[position] == mark:
        return text
    if position > 0 and text[position - 1] == mark:
        return text
    return text[:position] + mark + text[position:]


def _delete_punctuation_after_word(text: str, word_index: int) -> str:
    words = tokenize_words(text)
    if word_index < 0 or word_index >= len(words):
        return text
    position = words[word_index].end
    punct_position = _punctuation_position_after(text, position)
    if punct_position is None:
        return text
    mark = text[punct_position]
    if word_index == len(words) - 1 and mark in ".!?…":
        return text
    if mark == "—":
        return text[:position].rstrip() + " " + text[punct_position + 1 :].lstrip()
    return text[:punct_position] + text[punct_position + 1 :]


def _punctuation_position_after(text: str, position: int) -> int | None:
    index = position
    while index < len(text) and text[index].isspace():
        index += 1
    if index < len(text) and text[index] in ",.!?:;—…«»()[]":
        return index
    return None
