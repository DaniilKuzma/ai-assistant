from src.candidates.candidate_generator import Candidate
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


def test_trained_model_corrector_can_apply_context_dependent_pair_when_score_is_high():
    corrector = TrainedModelCorrector(FakeBackend({"так же": 0.99}), thresholds={"split_join_threshold": 0.9})

    result = corrector.correct("Он также пришел.")

    assert result.corrected_text == "Он так же пришел."
    assert any(edit.edit_type == "split_word" and edit.status == "accepted" for edit in result.edits)


def test_trained_model_corrector_keeps_existing_punctuation_when_model_predicts_none():
    corrector = TrainedModelCorrector(
        FakeBackend({}, punctuation=[ModelPunctuationPrediction(1, "NONE", 0.99)]),
        thresholds={"punctuation_threshold": 0.9},
    )

    result = corrector.correct("Я думаю, это важно.")

    assert result.corrected_text == "Я думаю, это важно."
    assert not any(edit.edit_type == "punctuation_delete" and edit.status == "accepted" for edit in result.edits)


def test_trained_model_corrector_replaces_existing_punctuation_with_colon():
    corrector = TrainedModelCorrector(
        FakeBackend({}, punctuation=[ModelPunctuationPrediction(1, "COLON", 0.99)]),
        thresholds={"punctuation_threshold": 0.9},
    )

    result = corrector.correct("Он сказал, привет.")

    assert result.corrected_text == "Он сказал: привет."
    assert any(edit.edit_type == "punctuation_replace" and edit.status == "accepted" for edit in result.edits)


def test_trained_model_corrector_inserts_dash_with_spacing():
    corrector = TrainedModelCorrector(
        FakeBackend({}, punctuation=[ModelPunctuationPrediction(0, "DASH", 0.99)]),
        thresholds={"punctuation_threshold": 0.9},
    )

    result = corrector.correct("Москва это столица.")

    assert result.corrected_text == "Москва — это столица."


def test_trained_model_corrector_supports_simple_quotes_and_brackets():
    corrector = TrainedModelCorrector(
        FakeBackend(
            {},
            punctuation=[
                ModelPunctuationPrediction(2, "QUOTE_OPEN", 0.99),
                ModelPunctuationPrediction(2, "QUOTE_CLOSE", 0.99),
                ModelPunctuationPrediction(4, "BRACKET_OPEN", 0.99),
                ModelPunctuationPrediction(4, "BRACKET_CLOSE", 0.99),
            ],
        ),
        thresholds={"punctuation_threshold": 0.9},
    )

    result = corrector.correct("Он сказал привет это важно.")

    assert result.corrected_text == "Он сказал «привет» это (важно)."


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
    def __init__(self, max_candidates: int):
        self.max_candidates = max_candidates
        self.last_kwargs = None

    def __call__(self, **kwargs):
        self.last_kwargs = kwargs
        return {
            "candidate_scores": torch.zeros(1, self.max_candidates),
            "confidence_logits": torch.zeros(1, self.max_candidates),
        }
