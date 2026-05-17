from src.candidates.candidate_generator import Candidate
import pytest
import torch

from src.inference.model_corrector import (
    ModelCandidatePrediction,
    ModelPunctuationPrediction,
    TorchCandidateModelBackend,
    TrainedModelCorrector,
    _prepare_heads_state_dict_for_module,
)
from src.model.heads import build_linear_heads


class FakeBackend:
    def __init__(self, scores: dict[str, float], punctuation=None):
        self.scores = scores
        self.punctuation = punctuation or []

    def score_candidates(self, text: str, candidates: list[Candidate]) -> list[ModelCandidatePrediction]:
        return [
            ModelCandidatePrediction(
                candidate=candidate,
                score=self.scores.get(candidate.replacement, 0.0),
                confidence=self.scores.get(candidate.replacement, 0.0),
            )
            for candidate in candidates
        ]

    def predict_punctuation(self, text: str):
        return self.punctuation


def test_trained_model_corrector_does_not_apply_whitelist_candidate_when_model_score_is_low():
    corrector = TrainedModelCorrector(FakeBackend({}), thresholds={"split_join_threshold": 0.9})

    result = corrector.correct("Я незнаю что делать")

    assert result.corrected_text == "Я незнаю что делать."


def test_trained_model_corrector_applies_candidate_when_model_score_passes_threshold():
    corrector = TrainedModelCorrector(FakeBackend({"не знаю": 0.99}), thresholds={"split_join_threshold": 0.9})

    result = corrector.correct("Я незнаю что делать")

    assert result.corrected_text == "Я не знаю что делать."
    assert any(edit.edit_type == "split_word" and edit.status == "accepted" for edit in result.edits)


def test_trained_model_corrector_rejects_context_dependent_pair_without_strict_context():
    corrector = TrainedModelCorrector(FakeBackend({"так же": 0.99}), thresholds={"split_join_threshold": 0.9})

    result = corrector.correct("Он также пришел.")

    assert result.corrected_text == "Он также пришел."
    assert any(edit.source.lower() == "также" and edit.status == "rejected" for edit in result.edits)


def test_trained_model_corrector_can_apply_context_dependent_pair_with_high_confidence_and_strict_context():
    corrector = TrainedModelCorrector(
        FakeBackend({"так же": 0.99}),
        thresholds={"split_join_threshold": 0.9, "context_pair_threshold": 0.98},
    )

    result = corrector.correct("Он сделал также как я.")

    assert result.corrected_text == "Он сделал так же как я."
    assert any(edit.source.lower() == "также" and edit.status == "accepted" for edit in result.edits)


def test_trained_model_corrector_keeps_existing_punctuation_when_model_predicts_none_for_gap():
    corrector = TrainedModelCorrector(
        FakeBackend({}, punctuation=[ModelPunctuationPrediction(1, "NONE", 0.99)]),
        thresholds={"punctuation_threshold": 0.9},
    )

    result = corrector.correct("Я думаю, это важно.")

    assert result.corrected_text == "Я думаю, это важно."
    assert not any(edit.edit_type == "punctuation_delete" and edit.status == "accepted" for edit in result.edits)


def test_trained_model_corrector_deletes_existing_punctuation_only_with_delete_action_and_threshold():
    low_confidence = TrainedModelCorrector(
        FakeBackend({}, punctuation=[ModelPunctuationPrediction(1, "NONE", 0.91, action="DELETE")]),
        thresholds={"punctuation_threshold": 0.5, "punctuation_delete_threshold": 0.95},
    )
    high_confidence = TrainedModelCorrector(
        FakeBackend({}, punctuation=[ModelPunctuationPrediction(1, "NONE", 0.97, action="DELETE")]),
        thresholds={"punctuation_threshold": 0.5, "punctuation_delete_threshold": 0.95},
    )

    assert low_confidence.correct("Я думаю, это важно.").corrected_text == "Я думаю, это важно."
    result = high_confidence.correct("Я думаю, это важно.")

    assert result.corrected_text == "Я думаю это важно."
    assert any(edit.edit_type == "punctuation_delete" and edit.status == "accepted" for edit in result.edits)


def test_trained_model_corrector_replaces_existing_punctuation_with_colon():
    corrector = TrainedModelCorrector(
        FakeBackend({}, punctuation=[ModelPunctuationPrediction(1, "COLON", 0.99, action="REPLACE")]),
        thresholds={"punctuation_threshold": 0.9},
    )

    result = corrector.correct("Он сказал, привет.")

    assert result.corrected_text == "Он сказал: привет."
    assert any(edit.edit_type == "punctuation_replace" and edit.status == "accepted" for edit in result.edits)


def test_trained_model_corrector_does_not_insert_sentence_final_mark_inside_sentence():
    corrector = TrainedModelCorrector(
        FakeBackend({}, punctuation=[ModelPunctuationPrediction(2, "DOT", 0.99, action="INSERT")]),
        thresholds={"punctuation_threshold": 0.9},
    )

    result = corrector.correct("Он предназначен для роботов поисковых систем.")

    assert result.corrected_text == "Он предназначен для роботов поисковых систем."


def test_trained_model_corrector_inserts_dash_with_spacing():
    corrector = TrainedModelCorrector(
        FakeBackend({}, punctuation=[ModelPunctuationPrediction(0, "DASH", 0.99, action="INSERT")]),
        thresholds={"punctuation_threshold": 0.9},
    )

    result = corrector.correct("Москва это столица.")

    assert result.corrected_text == "Москва — это столица."


def test_trained_model_corrector_supports_simple_quotes_and_brackets():
    corrector = TrainedModelCorrector(
        FakeBackend(
            {},
            punctuation=[
                ModelPunctuationPrediction(1, "QUOTE_OPEN", 0.99),
                ModelPunctuationPrediction(2, "QUOTE_CLOSE", 0.99),
                ModelPunctuationPrediction(3, "BRACKET_OPEN", 0.99),
                ModelPunctuationPrediction(4, "BRACKET_CLOSE", 0.99),
            ],
        ),
        thresholds={"punctuation_threshold": 0.9},
    )

    result = corrector.correct("Он сказал привет это важно.")

    assert result.corrected_text == "Он сказал «привет» это (важно)."


def test_trained_model_corrector_uses_label_specific_punctuation_thresholds():
    corrector = TrainedModelCorrector(
        FakeBackend(
            {},
            punctuation=[
                ModelPunctuationPrediction(1, "COMMA", 0.86, action="INSERT"),
                ModelPunctuationPrediction(2, "COLON", 0.86, action="INSERT"),
            ],
        ),
        thresholds={"punctuation_threshold": 0.5, "comma_threshold": 0.9, "colon_threshold": 0.8},
    )

    result = corrector.correct("Он сказал привет дальше.")

    assert result.corrected_text == "Он сказал привет: дальше."


def test_torch_backend_passes_candidate_replacement_tokens_to_model():
    tokenizer = FakeTokenizer()
    module = CapturingModule(max_candidates=3)
    backend = TorchCandidateModelBackend(
        tokenizer=tokenizer,
        module=module,
        device=torch.device("cpu"),
        punctuation_labels={},
        max_length=8,
        max_candidates=3,
    )
    candidates = [
        Candidate("незнаю", "незнаю", "keep", 2, 8),
        Candidate("незнаю", "не знаю", "split_join", 2, 8),
    ]

    backend.score_candidates("Я незнаю", candidates)

    replacement_ids = module.last_kwargs["candidate_replacement_ids"]
    replacement_mask = module.last_kwargs["candidate_replacement_mask"]
    assert replacement_ids.shape[0:2] == torch.Size([1, 3])
    assert replacement_mask[0, 1].any()
    assert replacement_ids[0, 1].sum().item() > 0


def test_torch_backend_passes_word_gap_indices_to_model_for_punctuation():
    tokenizer = FakeTokenizer()
    module = CapturingModule(max_candidates=1, punctuation_label_count=3)
    backend = TorchCandidateModelBackend(
        tokenizer=tokenizer,
        module=module,
        device=torch.device("cpu"),
        punctuation_labels={"NONE": 0, "COMMA": 1, "DOT": 2},
        punctuation_action_labels={"KEEP_NONE": 0, "KEEP_EXISTING": 1, "INSERT": 2, "DELETE": 3, "REPLACE": 4},
        max_length=8,
        max_candidates=1,
    )

    backend.predict_punctuation("Я думаю, что")

    gap_indices = module.last_kwargs["punctuation_gap_indices"]
    right_gap_indices = module.last_kwargs["punctuation_right_gap_indices"]
    assert gap_indices.shape == torch.Size([1, 8])
    assert right_gap_indices.shape == torch.Size([1, 8])
    assert gap_indices.tolist()[0][:3] == [0, 5, 7]
    assert right_gap_indices.tolist()[0][:3] == [1, 7, 7]


def test_torch_backend_uses_punctuation_confidence_head_for_prediction_confidence():
    tokenizer = FakeTokenizer()
    module = CapturingModule(max_candidates=1, punctuation_label_count=3, punctuation_confidence_logit=2.0)
    backend = TorchCandidateModelBackend(
        tokenizer=tokenizer,
        module=module,
        device=torch.device("cpu"),
        punctuation_labels={"NONE": 0, "COMMA": 1, "DOT": 2},
        punctuation_action_labels={"KEEP_NONE": 0, "KEEP_EXISTING": 1, "INSERT": 2, "DELETE": 3, "REPLACE": 4},
        max_length=8,
        max_candidates=1,
    )

    predictions = backend.predict_punctuation("Я думаю что")

    assert predictions
    assert predictions[0].confidence == pytest.approx(torch.sigmoid(torch.tensor(2.0)).item())
    assert predictions[0].action in {"KEEP_NONE", "KEEP_EXISTING", "INSERT", "DELETE", "REPLACE"}


def test_legacy_candidate_projection_heads_are_expanded_for_current_model_shape():
    heads = torch.nn.ModuleDict(build_linear_heads(hidden_size=4, punctuation_labels=3, error_types=2))
    legacy_state = heads.state_dict()
    legacy_weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    legacy_state["candidate_projection.weight"] = legacy_weight

    prepared = _prepare_heads_state_dict_for_module(legacy_state, heads)

    assert prepared["candidate_projection.weight"].shape == torch.Size([4, 12])
    assert torch.equal(prepared["candidate_projection.weight"][:, :4], legacy_weight)
    assert torch.equal(prepared["candidate_projection.weight"][:, 4:], torch.zeros(4, 8))
    heads.load_state_dict(prepared)


class FakeTokenizer:
    def __call__(
        self,
        text,
        *,
        return_offsets_mapping=True,
        truncation=True,
        padding="max_length",
        max_length=8,
        add_special_tokens=True,
    ):
        tokens = [(char, index, index + 1) for index, char in enumerate(text) if not char.isspace()]
        tokens = tokens[:max_length]
        input_ids = [ord(char) % 97 + 2 for char, _start, _end in tokens]
        attention_mask = [1] * len(input_ids)
        offsets = [(start, end) for _char, start, end in tokens]
        pad = max_length - len(input_ids)
        input_ids += [0] * pad
        attention_mask += [0] * pad
        offsets += [(0, 0)] * pad
        result = {"input_ids": input_ids, "attention_mask": attention_mask}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


class CapturingModule:
    def __init__(
        self,
        max_candidates: int,
        punctuation_label_count: int = 0,
        punctuation_action_count: int = 5,
        punctuation_confidence_logit: float = 0.0,
    ):
        self.max_candidates = max_candidates
        self.punctuation_label_count = punctuation_label_count
        self.punctuation_action_count = punctuation_action_count
        self.punctuation_confidence_logit = punctuation_confidence_logit
        self.last_kwargs = None

    def __call__(self, **kwargs):
        self.last_kwargs = kwargs
        gap_count = kwargs.get("punctuation_gap_indices", torch.zeros(1, 0)).shape[1]
        return {
            "candidate_scores": torch.zeros(1, self.max_candidates),
            "confidence_logits": torch.zeros(1, self.max_candidates),
            "punctuation_logits": torch.zeros(1, gap_count, self.punctuation_label_count),
            "punctuation_action_logits": torch.zeros(1, gap_count, self.punctuation_action_count),
            "punctuation_confidence_logits": torch.full((1, gap_count), self.punctuation_confidence_logit),
        }
