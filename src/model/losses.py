from __future__ import annotations


def multitask_loss(outputs: dict, labels: dict, weights: dict[str, float]):
    import torch
    import torch.nn.functional as functional

    loss = torch.tensor(0.0, device=outputs["candidate_scores"].device)
    candidate_mask = labels.get("candidate_mask")
    if "candidate_labels" in labels:
        word_loss = functional.binary_cross_entropy_with_logits(
            outputs["candidate_scores"], labels["candidate_labels"].float(), reduction="none"
        )
        loss = loss + weights.get("word_loss_weight", 1.0) * _masked_mean(word_loss, candidate_mask)
    if "punctuation_labels" in labels:
        punctuation_loss = functional.cross_entropy(
            outputs["punctuation_logits"].transpose(1, 2), labels["punctuation_labels"], reduction="none"
        )
        loss = loss + weights.get("punctuation_loss_weight", 1.0) * _masked_mean(
            punctuation_loss, labels.get("punctuation_mask")
        )
    if "confidence_labels" in labels:
        confidence_loss = functional.binary_cross_entropy_with_logits(
            outputs["confidence_logits"], labels["confidence_labels"].float(), reduction="none"
        )
        loss = loss + weights.get("confidence_loss_weight", 1.0) * _masked_mean(confidence_loss, candidate_mask)
    if "error_type_labels" in labels:
        loss = loss + weights.get("error_type_loss_weight", 1.0) * functional.cross_entropy(
            outputs["error_type_logits"].reshape(-1, outputs["error_type_logits"].shape[-1]),
            labels["error_type_labels"].reshape(-1),
            ignore_index=-100,
        )
    return loss


def _masked_mean(values, mask):  # type: ignore[no-untyped-def]
    if mask is None:
        return values.mean()
    mask = mask.to(values.device).float()
    return (values * mask).sum() / mask.sum().clamp_min(1.0)
