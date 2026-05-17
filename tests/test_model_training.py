from types import SimpleNamespace

import torch

from src.model.edit_model import CandidateAwareEditModel, EditModelConfig
from src.model.losses import multitask_loss


class FakeEncoder(torch.nn.Module):
    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embeddings = torch.nn.Embedding(32, hidden_size)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return SimpleNamespace(last_hidden_state=self.embeddings(input_ids))


class WrappedFakeEncoder(torch.nn.Module):
    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.config = SimpleNamespace()
        self.base_model = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(hidden_size=hidden_size)))
        self.embeddings = torch.nn.Embedding(32, hidden_size)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return SimpleNamespace(last_hidden_state=self.embeddings(input_ids))


def test_candidate_aware_model_scores_each_candidate_span():
    model = CandidateAwareEditModel.from_encoder(
        FakeEncoder(),
        EditModelConfig(punctuation_label_count=4, error_type_count=5, lora_enabled=False),
    ).module

    outputs = model(
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        attention_mask=torch.tensor([[1, 1, 1, 1]]),
        candidate_spans=torch.tensor([[[1, 2], [2, 3], [0, 0]]]),
        candidate_mask=torch.tensor([[1, 1, 0]], dtype=torch.bool),
        punctuation_gap_indices=torch.tensor([[0, 2, 3]]),
    )

    assert outputs["candidate_scores"].shape == torch.Size([1, 3])
    assert outputs["confidence_logits"].shape == torch.Size([1, 3])
    assert outputs["error_type_logits"].shape == torch.Size([1, 3, 5])
    assert outputs["punctuation_logits"].shape == torch.Size([1, 3, 4])
    assert outputs["punctuation_action_logits"].shape == torch.Size([1, 3, 5])
    assert outputs["punctuation_confidence_logits"].shape == torch.Size([1, 3])
    assert outputs["punctuation_error_type_logits"].shape == torch.Size([1, 3, 5])


def test_candidate_aware_model_predicts_punctuation_on_gap_representations():
    torch.manual_seed(11)
    model = CandidateAwareEditModel.from_encoder(
        FakeEncoder(),
        EditModelConfig(punctuation_label_count=4, error_type_count=5, lora_enabled=False),
    ).module

    outputs = model(
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        attention_mask=torch.tensor([[1, 1, 1, 1]]),
        candidate_spans=torch.tensor([[[1, 2]]]),
        candidate_mask=torch.tensor([[1]], dtype=torch.bool),
        punctuation_gap_indices=torch.tensor([[1, 3]]),
        punctuation_right_gap_indices=torch.tensor([[2, 3]]),
    )

    assert outputs["punctuation_logits"].shape == torch.Size([1, 2, 4])
    assert outputs["punctuation_action_logits"].shape == torch.Size([1, 2, 5])
    assert outputs["punctuation_confidence_logits"].shape == torch.Size([1, 2])
    assert outputs["punctuation_error_type_logits"].shape == torch.Size([1, 2, 5])
    assert not torch.equal(outputs["punctuation_logits"][0, 0], outputs["punctuation_logits"][0, 1])


def test_candidate_aware_model_uses_right_context_for_punctuation_gaps():
    torch.manual_seed(13)
    model = CandidateAwareEditModel.from_encoder(
        FakeEncoder(),
        EditModelConfig(punctuation_label_count=4, punctuation_action_count=5, error_type_count=5, lora_enabled=False),
    ).module

    left_only = model(
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        attention_mask=torch.tensor([[1, 1, 1, 1]]),
        candidate_spans=torch.tensor([[[1, 2]]]),
        candidate_mask=torch.tensor([[1]], dtype=torch.bool),
        punctuation_gap_indices=torch.tensor([[1]]),
        punctuation_right_gap_indices=torch.tensor([[2]]),
    )
    changed_right = model(
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        attention_mask=torch.tensor([[1, 1, 1, 1]]),
        candidate_spans=torch.tensor([[[1, 2]]]),
        candidate_mask=torch.tensor([[1]], dtype=torch.bool),
        punctuation_gap_indices=torch.tensor([[1]]),
        punctuation_right_gap_indices=torch.tensor([[3]]),
    )

    assert not torch.equal(left_only["punctuation_logits"], changed_right["punctuation_logits"])


def test_candidate_aware_model_uses_replacement_tokens_to_score_candidates():
    torch.manual_seed(7)
    model = CandidateAwareEditModel.from_encoder(
        FakeEncoder(),
        EditModelConfig(punctuation_label_count=4, error_type_count=5, lora_enabled=False),
    ).module

    outputs = model(
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        attention_mask=torch.tensor([[1, 1, 1, 1]]),
        candidate_spans=torch.tensor([[[1, 2], [1, 2]]]),
        candidate_mask=torch.tensor([[1, 1]], dtype=torch.bool),
        candidate_replacement_ids=torch.tensor([[[5, 0, 0], [6, 7, 0]]]),
        candidate_replacement_mask=torch.tensor([[[1, 0, 0], [1, 1, 0]]], dtype=torch.bool),
    )

    assert outputs["candidate_scores"][0, 0] != outputs["candidate_scores"][0, 1]


def test_candidate_aware_model_reads_hidden_size_from_wrapped_peft_encoder():
    model = CandidateAwareEditModel.from_encoder(
        WrappedFakeEncoder(),
        EditModelConfig(punctuation_label_count=4, error_type_count=5, lora_enabled=False),
    ).module

    outputs = model(
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        attention_mask=torch.tensor([[1, 1, 1, 1]]),
        candidate_spans=torch.tensor([[[1, 2], [2, 3]]]),
        candidate_mask=torch.tensor([[1, 1]], dtype=torch.bool),
    )

    assert outputs["candidate_scores"].shape == torch.Size([1, 2])


def test_multitask_loss_ignores_padded_candidates():
    outputs = {
        "candidate_scores": torch.tensor([[2.0, -1.0, 20.0]]),
        "punctuation_logits": torch.randn(1, 4, 3),
        "punctuation_action_logits": torch.randn(1, 4, 5),
        "confidence_logits": torch.tensor([[2.0, -1.0, 20.0]]),
        "error_type_logits": torch.randn(1, 3, 4),
        "punctuation_confidence_logits": torch.tensor([[2.0, -1.0, 0.0, 20.0]]),
        "punctuation_error_type_logits": torch.randn(1, 4, 4),
    }
    labels = {
        "candidate_labels": torch.tensor([[1.0, 0.0, 0.0]]),
        "candidate_mask": torch.tensor([[1, 1, 0]], dtype=torch.bool),
        "punctuation_labels": torch.tensor([[0, 1, 0, 2]]),
        "punctuation_action_labels": torch.tensor([[0, 2, 0, 3]]),
        "punctuation_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.bool),
        "confidence_labels": torch.tensor([[1.0, 0.0, 0.0]]),
        "error_type_labels": torch.tensor([[1, 0, -100]]),
        "punctuation_confidence_labels": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "punctuation_error_type_labels": torch.tensor([[2, 0, 0, -100]]),
    }

    loss = multitask_loss(
        outputs,
        labels,
        {
            "punctuation_label_weights": [0.25, 2.0, 1.5],
            "punctuation_action_weights": [0.25, 1.0, 2.0, 2.5, 2.5],
            "punctuation_focal_gamma": 1.5,
            "punctuation_action_loss_weight": 0.7,
        },
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
