from __future__ import annotations


def multitask_loss(outputs: dict, labels: dict, weights: dict[str, float]):
    total, _components = multitask_loss_with_components(outputs, labels, weights)
    return total


def multitask_loss_components(outputs: dict, labels: dict, weights: dict[str, float]):
    _total, components = multitask_loss_with_components(outputs, labels, weights)
    return components


def multitask_loss_with_components(outputs: dict, labels: dict, weights: dict[str, float]):
    import torch
    import torch.nn.functional as functional

    loss = torch.tensor(0.0, device=outputs["candidate_scores"].device)
    components: dict[str, torch.Tensor] = {}
    candidate_mask = labels.get("candidate_mask")
    if "candidate_labels" in labels:
        word_loss = functional.binary_cross_entropy_with_logits(
            outputs["candidate_scores"], labels["candidate_labels"].float(), reduction="none"
        )
        components["word_loss"] = _masked_mean(word_loss, candidate_mask)
        loss = loss + weights.get("word_loss_weight", 1.0) * components["word_loss"]
    if "punctuation_labels" in labels:
        punctuation_loss = _weighted_token_cross_entropy(
            outputs["punctuation_logits"],
            labels["punctuation_labels"],
            weights.get("punctuation_label_weights"),
            float(weights.get("punctuation_focal_gamma", 0.0)),
        )
        components["punctuation_loss"] = _masked_mean(punctuation_loss, labels.get("punctuation_mask"))
        loss = loss + weights.get("punctuation_loss_weight", 1.0) * components["punctuation_loss"]
    if "punctuation_action_labels" in labels and "punctuation_action_logits" in outputs:
        punctuation_action_loss = _weighted_token_cross_entropy(
            outputs["punctuation_action_logits"],
            labels["punctuation_action_labels"],
            weights.get("punctuation_action_weights"),
            float(weights.get("punctuation_action_focal_gamma", weights.get("punctuation_focal_gamma", 0.0))),
        )
        components["punctuation_action_loss"] = _masked_mean(punctuation_action_loss, labels.get("punctuation_mask"))
        loss = loss + weights.get("punctuation_action_loss_weight", 1.0) * components["punctuation_action_loss"]
    if "confidence_labels" in labels:
        confidence_loss = functional.binary_cross_entropy_with_logits(
            outputs["confidence_logits"], labels["confidence_labels"].float(), reduction="none"
        )
        components["confidence_loss"] = _masked_mean(confidence_loss, candidate_mask)
        loss = loss + weights.get("confidence_loss_weight", 1.0) * components["confidence_loss"]
    if "punctuation_confidence_labels" in labels and "punctuation_confidence_logits" in outputs:
        punctuation_confidence_loss = functional.binary_cross_entropy_with_logits(
            outputs["punctuation_confidence_logits"],
            labels["punctuation_confidence_labels"].float(),
            reduction="none",
        )
        components["punctuation_confidence_loss"] = _masked_mean(
            punctuation_confidence_loss,
            labels.get("punctuation_mask"),
        )
        components["confidence_loss"] = components.get("confidence_loss", loss.new_tensor(0.0)) + components[
            "punctuation_confidence_loss"
        ]
        loss = loss + weights.get("confidence_loss_weight", 1.0) * components["punctuation_confidence_loss"]
    if "error_type_labels" in labels:
        components["error_type_loss"] = _cross_entropy_ignore_empty(
            outputs["error_type_logits"],
            labels["error_type_labels"],
        )
        loss = loss + weights.get("error_type_loss_weight", 1.0) * components["error_type_loss"]
    if "punctuation_error_type_labels" in labels and "punctuation_error_type_logits" in outputs:
        components["punctuation_error_type_loss"] = _cross_entropy_ignore_empty(
            outputs["punctuation_error_type_logits"],
            labels["punctuation_error_type_labels"],
        )
        components["error_type_loss"] = components.get("error_type_loss", loss.new_tensor(0.0)) + components[
            "punctuation_error_type_loss"
        ]
        loss = loss + weights.get("error_type_loss_weight", 1.0) * components["punctuation_error_type_loss"]
    return loss, components


def _weighted_token_cross_entropy(logits, labels, class_weights, focal_gamma: float = 0.0):  # type: ignore[no-untyped-def]
    import torch
    import torch.nn.functional as functional

    weight = None
    if class_weights:
        weight = torch.tensor(list(class_weights), dtype=logits.dtype, device=logits.device)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1).to(logits.device)
    loss = functional.cross_entropy(flat_logits, flat_labels, weight=weight, reduction="none")
    if focal_gamma > 0:
        probability = torch.exp(-loss)
        loss = ((1 - probability) ** focal_gamma) * loss
    return loss.reshape(labels.shape)


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
