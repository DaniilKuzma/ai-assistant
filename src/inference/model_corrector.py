from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.candidates.candidate_generator import Candidate, CandidateGenerator
from src.candidates.candidate_ranking import rank_candidates_for_budget
from src.config.thresholds import threshold_for_candidate, threshold_for_punctuation_prediction
from src.inference.corrector import CorrectionResult
from src.inference.edit_realizer import apply_candidate
from src.inference.postprocess import normalize_spacing
from src.model.edit_model import CandidateAwareEditModel, EditModelConfig
from src.model.encoder import EncoderLoadConfig, ensure_pytorch_transformers_backend, load_encoder, load_tokenizer
from src.preprocessing.punctuation_gaps import word_gap_context_token_indices
from src.preprocessing.tokenizer import tokenize_words
from src.validation.diff_analyzer import DiffAnalyzer
from src.validation.strict_validator import StrictValidator


MAX_REPLACEMENT_TOKENS = 8


@dataclass(frozen=True)
class ModelCandidatePrediction:
    candidate: Candidate
    score: float
    confidence: float
    rule_id: str = ""


@dataclass(frozen=True)
class ModelPunctuationPrediction:
    gap_index: int
    label: str
    confidence: float
    action: str = "INSERT"
    rule_id: str = ""

    @property
    def word_index(self) -> int:
        return self.gap_index


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
        candidate_generator: CandidateGenerator | None = None,
    ) -> None:
        self.backend = backend
        self.thresholds = thresholds or {}
        self.max_passes = max_passes
        self.candidates = candidate_generator or CandidateGenerator()
        self.validator = StrictValidator(
            context_pair_threshold=float(self.thresholds.get("context_pair_threshold", 0.98)),
            tsya_threshold=float(
                self.thresholds.get(
                    "tsya_threshold",
                    self.thresholds.get("context_pair_threshold", 0.98),
                )
            ),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TrainedModelCorrector":
        threshold_config = config.get("thresholds", {})
        mode = threshold_config.get("mode", "balanced")
        thresholds = threshold_config.get(mode, {})
        backend = TorchCandidateModelBackend.from_config(config)
        return cls(
            backend,
            thresholds=thresholds,
            max_passes=int(config.get("decoder", {}).get("max_passes", 3)),
            candidate_generator=CandidateGenerator.from_config(config),
        )

    def correct(self, text: str) -> CorrectionResult:
        proposed, trusted_edits = self._decode_with_trusted_edits(text)
        validation = self.validator.validate(text, proposed, trusted_edits=trusted_edits)
        corrected = validation.apply_accepted()
        return CorrectionResult(text, corrected, validation.edits)

    def _single_pass(self, text: str) -> str:
        return self._single_pass_with_candidates(text)[0]

    def _decode_with_trusted_edits(self, text: str) -> tuple[str, list[Candidate]]:
        current = text
        trusted_edits: list[Candidate] = []
        for pass_index in range(self.max_passes):
            updated, selected = self._single_pass_with_candidates(current)
            if pass_index == 0:
                trusted_edits.extend(selected)
            if updated == current:
                break
            current = updated
        return current, trusted_edits

    def _single_pass_with_candidates(self, text: str) -> tuple[str, list[Candidate]]:
        candidates = self.candidates.generate(text)
        selected = _select_candidates(self.backend.score_candidates(text, candidates), self.thresholds)
        proposed = _apply_candidates(text, selected)
        proposed, punctuation_edits = _apply_punctuation_predictions(
            proposed,
            self.backend.predict_punctuation(proposed),
            self.thresholds,
        )
        return normalize_spacing(proposed), [*selected, *punctuation_edits]


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
        punctuation_action_labels: dict[str, int] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.module = module
        self.device = device
        self.punctuation_labels = punctuation_labels
        self.punctuation_by_id = {value: key for key, value in punctuation_labels.items()}
        self.punctuation_action_labels = punctuation_action_labels or _default_punctuation_action_labels()
        self.punctuation_action_by_id = {value: key for key, value in self.punctuation_action_labels.items()}
        self.max_length = max_length
        self.max_candidates = max_candidates

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TorchCandidateModelBackend":
        model_config = config.get("model", {})
        paths = config.get("paths", {})
        adapter_dir = Path(paths.get("adapter_output_dir", "models/adapters/latest"))
        heads_path = Path(paths.get("heads_output_dir", "models/heads/latest")) / "heads.pt"
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Trained adapter directory not found: {adapter_dir}")
        if not heads_path.exists():
            raise FileNotFoundError(f"Trained heads checkpoint not found: {heads_path}")

        ensure_pytorch_transformers_backend()
        import torch
        from peft import PeftModel

        encoder_load_config = EncoderLoadConfig(
            model_name=model_config.get("primary_encoder", "ai-forever/ruRoberta-large"),
            fallback_model_name=model_config.get("fallback_encoder", "ai-forever/ruRoberta-large"),
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
                fallback_model_name=model_config.get("fallback_encoder", "ai-forever/ruRoberta-large"),
                punctuation_label_count=len(config.get("labels", {}).get("punctuation", {})),
                punctuation_action_count=len(config.get("labels", {}).get("punctuation_actions", {})) or 5,
                error_type_count=len(config.get("labels", {}).get("error_types", {})),
                local_files_only=bool(model_config.get("local_files_only", False)),
                lora_enabled=False,
            ),
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        module = edit_model.module.to(device)
        heads_state = torch.load(heads_path, map_location=device)
        module.heads.load_state_dict(_prepare_heads_state_dict_for_module(heads_state, module.heads))
        module.eval()
        return cls(
            tokenizer=tokenizer,
            module=module,
            device=device,
            punctuation_labels=config.get("labels", {}).get("punctuation", {}),
            punctuation_action_labels=config.get("labels", {}).get("punctuation_actions", {}),
            max_length=int(model_config.get("max_sequence_length", 128)),
            max_candidates=int(model_config.get("max_candidates", 16)),
        )

    def score_candidates(self, text: str, candidates: list[Candidate]) -> list[ModelCandidatePrediction]:
        import torch

        active_candidates = rank_candidates_for_budget(candidates, self.max_candidates)
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
            ModelCandidatePrediction(
                candidate=candidate,
                score=float(scores[index]),
                confidence=float(confidences[index]),
                rule_id=candidate.rule_id,
            )
            for index, candidate in enumerate(active_candidates)
        ]

    def predict_punctuation(self, text: str) -> list[ModelPunctuationPrediction]:
        import torch

        if not self.punctuation_labels:
            return []
        encoded = _encode(self.tokenizer, text, self.max_length)
        gap_indices, right_gap_indices, gap_mask = word_gap_context_token_indices(text, encoded["offset_mapping"], self.max_length)
        if not any(gap_mask):
            return []
        with torch.no_grad():
            outputs = self.module(
                input_ids=torch.tensor([encoded["input_ids"]], dtype=torch.long, device=self.device),
                attention_mask=torch.tensor([encoded["attention_mask"]], dtype=torch.long, device=self.device),
                punctuation_gap_indices=torch.tensor([gap_indices], dtype=torch.long, device=self.device),
                punctuation_right_gap_indices=torch.tensor([right_gap_indices], dtype=torch.long, device=self.device),
                punctuation_gap_mask=torch.tensor([gap_mask], dtype=torch.bool, device=self.device),
            )
            probabilities = torch.softmax(outputs["punctuation_logits"][0], dim=-1).detach().cpu()
            action_probabilities = torch.softmax(outputs["punctuation_action_logits"][0], dim=-1).detach().cpu()
            punctuation_confidences = None
            if "punctuation_confidence_logits" in outputs:
                punctuation_confidences = torch.sigmoid(outputs["punctuation_confidence_logits"][0]).detach().cpu()

        predictions: list[ModelPunctuationPrediction] = []
        for gap_index, active in enumerate(gap_mask):
            if not active:
                continue
            if gap_index >= probabilities.shape[0]:
                break
            confidence, label_id = probabilities[gap_index].max(dim=-1)
            _action_confidence, action_id = action_probabilities[gap_index].max(dim=-1)
            if punctuation_confidences is not None and gap_index < punctuation_confidences.shape[0]:
                confidence = punctuation_confidences[gap_index]
            label = self.punctuation_by_id.get(int(label_id.item()), "NONE")
            action = self.punctuation_action_by_id.get(int(action_id.item()), "KEEP_NONE")
            predictions.append(
                ModelPunctuationPrediction(
                    gap_index,
                    label,
                    float(confidence.item()),
                    action=action,
                )
            )
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
        rule_id = prediction.rule_id or candidate.rule_id
        if prediction.score < threshold_for_candidate(candidate, thresholds, rule_id=rule_id):
            continue
        if any(_candidates_conflict(candidate.start, candidate.end, start, end) for start, end in occupied):
            continue
        selected.append(
            Candidate(
                source=candidate.source,
                replacement=candidate.replacement,
                edit_type=candidate.edit_type,
                start=candidate.start,
                end=candidate.end,
                confidence=prediction.confidence,
                requires_model=candidate.requires_model,
                rule_id=prediction.rule_id or candidate.rule_id,
                mode=candidate.mode,
                action=candidate.action,
                label=candidate.label,
                gap_index=candidate.gap_index,
                requires=candidate.requires,
                group=candidate.group,
            )
        )
        occupied.append((candidate.start, candidate.end))
    return selected


def _candidates_conflict(candidate_start: int, candidate_end: int, occupied_start: int, occupied_end: int) -> bool:
    if candidate_start == candidate_end or occupied_start == occupied_end:
        return candidate_start == occupied_start
    return candidate_start < occupied_end and occupied_start < candidate_end


def _default_punctuation_action_labels() -> dict[str, int]:
    return {"KEEP_NONE": 0, "KEEP_EXISTING": 1, "INSERT": 2, "DELETE": 3, "REPLACE": 4}


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
            requires_model=candidate.requires_model,
            rule_id=candidate.rule_id,
            mode=candidate.mode,
            action=candidate.action,
            label=candidate.label,
            gap_index=candidate.gap_index,
            requires=candidate.requires,
            group=candidate.group,
        )
        before = proposed
        proposed = apply_candidate(proposed, shifted)
        offset += len(proposed) - len(before)
    return proposed


def _apply_punctuation_predictions(
    text: str,
    predictions: list[ModelPunctuationPrediction],
    thresholds: dict[str, float],
) -> tuple[str, list[Candidate]]:
    proposed = text
    trusted_edits: list[Candidate] = []
    for prediction in predictions:
        if prediction.confidence < threshold_for_punctuation_prediction(prediction, thresholds):
            continue
        if prediction.action in {"KEEP_NONE", "KEEP_EXISTING"}:
            continue
        before = proposed
        if prediction.action == "DELETE":
            proposed = _delete_punctuation_after_word(proposed, prediction.gap_index)
            trusted_edits.extend(_trusted_punctuation_candidates(before, proposed, prediction))
            continue
        if prediction.label == "NONE":
            continue
        if prediction.action not in {"INSERT", "REPLACE"}:
            continue
        if prediction.label in {"QUOTE_OPEN", "BRACKET_OPEN"}:
            proposed = _insert_punctuation_before_word(proposed, prediction.gap_index + 1, _punctuation_mark(prediction.label))
        elif prediction.label in {"DOT", "QUESTION", "EXCLAMATION", "ELLIPSIS"}:
            if _is_last_word_index(proposed, prediction.gap_index):
                proposed = _replace_final_punctuation(proposed, _punctuation_mark(prediction.label))
        elif prediction.label in {
            "COMMA",
            "COLON",
            "DASH",
            "SEMICOLON",
            "QUOTE_CLOSE",
            "BRACKET_CLOSE",
        }:
            proposed = _insert_punctuation_after_word(proposed, prediction.gap_index, _punctuation_mark(prediction.label))
        trusted_edits.extend(_trusted_punctuation_candidates(before, proposed, prediction))
    return proposed, trusted_edits


def _trusted_punctuation_candidates(
    source: str,
    target: str,
    prediction: ModelPunctuationPrediction,
) -> list[Candidate]:
    if source == target:
        return []
    return [
        Candidate(
            source=edit.source,
            replacement=edit.replacement,
            edit_type=edit.edit_type,
            start=edit.start,
            end=edit.end,
            confidence=prediction.confidence,
            requires_model=True,
            rule_id=prediction.rule_id,
            mode="model_required",
            action=prediction.action,
            label=prediction.label,
            gap_index=prediction.gap_index,
            requires=("model",),
            group="punctuation",
        )
        for edit in DiffAnalyzer().analyze(source, target)
        if edit.edit_type
        in {
            "punctuation_insert",
            "punctuation_delete",
            "punctuation_replace",
            "final_punctuation",
        }
    ]


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


def _prepare_heads_state_dict_for_module(state_dict: dict[str, Any], heads: Any) -> dict[str, Any]:
    prepared = dict(state_dict)
    key = "candidate_projection.weight"
    if key not in prepared:
        return prepared

    target = heads.state_dict().get(key)
    source = prepared[key]
    if target is None or not hasattr(source, "shape") or tuple(source.shape) == tuple(target.shape):
        return prepared

    if _is_legacy_candidate_projection_shape(source, target):
        expanded = target.new_zeros(target.shape)
        expanded[:, : source.shape[1]] = source.to(device=target.device, dtype=target.dtype)
        prepared[key] = expanded
    return prepared


def _is_legacy_candidate_projection_shape(source: Any, target: Any) -> bool:
    return (
        len(source.shape) == 2
        and len(target.shape) == 2
        and source.shape[0] == target.shape[0]
        and source.shape[1] > 0
        and target.shape[1] == source.shape[1] * 3
    )


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
        if mark in {"»", ")"} and text[punct_position] in ".!?…":
            return text[:punct_position] + mark + text[punct_position:]
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
    char = text[punct_position]
    if char not in ",:;—«»()[]":
        return text
    if char == "—":
        left = text[:position].rstrip()
        right = text[punct_position + 1 :].lstrip()
        return f"{left} {right}" if right else left
    return text[:punct_position] + text[punct_position + 1 :]


def _punctuation_position_after(text: str, position: int) -> int | None:
    index = position
    while index < len(text) and text[index].isspace():
        index += 1
    if index < len(text) and text[index] in ",.!?:;—…«»()[]":
        return index
    return None
