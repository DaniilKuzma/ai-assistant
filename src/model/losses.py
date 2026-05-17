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
    if "punctuation_confidence_labels" in labels and "punctuation_confidence_logits" in outputs:
        punctuation_confidence_loss = functional.binary_cross_entropy_with_logits(
            outputs["punctuation_confidence_logits"],
            labels["punctuation_confidence_labels"].float(),
            reduction="none",
        )
        loss = loss + weights.get("confidence_loss_weight", 1.0) * _masked_mean(
            punctuation_confidence_loss,
            labels.get("punctuation_mask"),
        )
    if "error_type_labels" in labels:
        loss = loss + weights.get("error_type_loss_weight", 1.0) * _cross_entropy_ignore_empty(
            outputs["error_type_logits"],
            labels["error_type_labels"],
        )
    if "punctuation_error_type_labels" in labels and "punctuation_error_type_logits" in outputs:
        loss = loss + weights.get("error_type_loss_weight", 1.0) * _cross_entropy_ignore_empty(
            outputs["punctuation_error_type_logits"],
            labels["punctuation_error_type_labels"],
        )
    return loss


def _masked_mean(values, mask):  # type: ignore[no-untyped-def]
    if mask is None:
        return values.mean()
    mask = mask.to(values.device).float()
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _cross_entropy_ignore_empty(logits, labels):  # type: ignore[no-untyped-def]
    import torch.nn.functional as functional

    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1).to(logits.device)
    valid_mask = flat_labels.ne(-100)
    if not valid_mask.any():
        return logits.new_tensor(0.0)
    return functional.cross_entropy(flat_logits[valid_mask], flat_labels[valid_mask])
