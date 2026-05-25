from __future__ import annotations

from types import SimpleNamespace

import torch

import src.model as model_exports
from src.model.edit_model import DirectEditModelConfig, DirectEditTaggerModel
from src.schema.labels import GAP_PUNCTUATION_LABELS, RULE_LABELS, TOKEN_EDIT_LABELS


class FakeEncoder(torch.nn.Module):
    def __init__(self, hidden_size: int = 16, vocab_size: int = 64) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embeddings = torch.nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids, attention_mask=None, **kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(last_hidden_state=self.embeddings(input_ids))


def test_direct_edit_model_outputs_expected_shapes_and_masks_do_not_crash() -> None:
    batch_size = 2
    sequence_length = 12
    word_count = 5
    gap_count = 5
    hidden_size = 16
    token_label_count = 7
    gap_label_count = 4
    rule_tag_count = 3

    model = DirectEditTaggerModel.from_encoder(
        FakeEncoder(hidden_size=hidden_size),
        DirectEditModelConfig(
            token_label_count=token_label_count,
            gap_label_count=gap_label_count,
            rule_tag_count=rule_tag_count,
            lora_enabled=False,
        ),
    ).module

    outputs = model(
        input_ids=torch.arange(batch_size * sequence_length).view(batch_size, sequence_length),
        attention_mask=torch.ones(batch_size, sequence_length, dtype=torch.long),
        word_token_indices=torch.tensor(
            [
                [0, 1, 2, sequence_length + 4, -3],
                [3, 4, 5, 6, 7],
            ],
            dtype=torch.long,
        ),
        word_token_mask=torch.tensor(
            [
                [True, True, True, True, False],
                [True, False, True, True, True],
            ],
            dtype=torch.bool,
        ),
        gap_left_indices=torch.tensor(
            [
                [0, 1, sequence_length + 9, 3, 4],
                [2, 3, 4, 5, 6],
            ],
            dtype=torch.long,
        ),
        gap_right_indices=torch.tensor(
            [
                [1, -1, 3, sequence_length + 7, 5],
                [-1, 4, 5, 6, 7],
            ],
            dtype=torch.long,
        ),
        gap_mask=torch.tensor(
            [
                [True, True, True, False, True],
                [True, False, True, True, True],
            ],
            dtype=torch.bool,
        ),
    )

    assert outputs["hidden_states"].shape == torch.Size([batch_size, sequence_length, hidden_size])
    assert outputs["token_edit_logits"].shape == torch.Size([batch_size, word_count, token_label_count])
    assert outputs["token_confidence_logits"].shape == torch.Size([batch_size, word_count])
    assert outputs["gap_punctuation_logits"].shape == torch.Size([batch_size, gap_count, gap_label_count])
    assert outputs["gap_confidence_logits"].shape == torch.Size([batch_size, gap_count])
    assert outputs["rule_logits"].shape == torch.Size([batch_size, word_count, rule_tag_count])

    assert torch.equal(outputs["token_edit_logits"][0, 4], torch.zeros(token_label_count))
    assert outputs["token_confidence_logits"][0, 4].item() == 0.0
    assert torch.equal(outputs["rule_logits"][1, 1], torch.zeros(rule_tag_count))
    assert torch.equal(outputs["gap_punctuation_logits"][0, 3], torch.zeros(gap_label_count))
    assert outputs["gap_confidence_logits"][1, 1].item() == 0.0


def test_direct_model_config_defaults_use_schema_label_counts() -> None:
    config = DirectEditModelConfig(lora_enabled=False)

    assert config.token_label_count == len(TOKEN_EDIT_LABELS)
    assert config.gap_label_count == len(GAP_PUNCTUATION_LABELS)
    assert config.rule_tag_count == len(RULE_LABELS)


def test_model_package_exports_direct_model_only() -> None:
    assert model_exports.DirectEditModelConfig is DirectEditModelConfig
    assert model_exports.DirectEditTaggerModel is DirectEditTaggerModel
    legacy_model_name = "Candidate" + "Aware" + "EditModel"
    assert not hasattr(model_exports, legacy_model_name)

