import torch

from src.model.losses import multitask_loss_with_components


def test_sample_weight_scales_candidate_and_punctuation_losses():
    outputs = {
        "candidate_scores": torch.tensor([[0.3, -0.7]]),
        "punctuation_logits": torch.tensor([[[1.0, -0.5], [0.1, 0.6]]]),
        "confidence_logits": torch.tensor([[0.5, -0.4]]),
        "error_type_logits": torch.tensor([[[0.1, 0.7], [0.8, -0.3]]]),
    }
    labels = {
        "candidate_labels": torch.tensor([[1.0, 0.0]]),
        "candidate_mask": torch.tensor([[1, 1]], dtype=torch.bool),
        "punctuation_labels": torch.tensor([[0, 1]]),
        "punctuation_mask": torch.tensor([[1, 1]], dtype=torch.bool),
        "confidence_labels": torch.tensor([[1.0, 0.0]]),
        "error_type_labels": torch.tensor([[1, 0]]),
    }

    unweighted_total, unweighted_components = multitask_loss_with_components(outputs, labels, {})
    weighted_total, weighted_components = multitask_loss_with_components(
        outputs,
        {**labels, "sample_weight": torch.tensor([0.4])},
        {},
    )

    assert torch.isclose(weighted_total, unweighted_total * 0.4)
    for key, value in unweighted_components.items():
        assert torch.isclose(weighted_components[key], value * 0.4)
