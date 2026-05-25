from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Mapping

from src.schema.labels import gap_label_to_id, rule_tag_to_id, token_label_to_id


@dataclass
class WordToken:
    text: str
    start: int
    end: int
    lemma: str | None = None
    pos: str | None = None
    feats: dict[str, str] = field(default_factory=dict)


@dataclass
class GeneratedExample:
    source_text: str
    target_text: str
    source_tokens: list[WordToken]
    token_edit_labels: list[str]
    gap_labels: list[str]
    rule_ids: list[str]
    primary_rule_id: str
    mode: str
    explanation_ids: list[str]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.source_text.strip():
            raise ValueError("source_text must not be empty.")
        if not self.target_text.strip():
            raise ValueError("target_text must not be empty.")
        if not self.primary_rule_id.strip():
            raise ValueError("primary_rule_id must not be empty.")

        token_count = len(self.source_tokens)
        _require_length("token_edit_labels", self.token_edit_labels, token_count)
        _require_length("gap_labels", self.gap_labels, token_count)
        _require_length("rule_ids", self.rule_ids, token_count)

        for label in self.token_edit_labels:
            token_label_to_id(label)
        for label in self.gap_labels:
            gap_label_to_id(label)
        for rule_id in self.rule_ids:
            rule_tag_to_id(rule_id)
        rule_tag_to_id(self.primary_rule_id)

        for index, token in enumerate(self.source_tokens):
            _validate_token_span(self.source_text, token, index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_text": self.source_text,
            "target_text": self.target_text,
            "source_tokens": [asdict(token) for token in self.source_tokens],
            "token_edit_labels": list(self.token_edit_labels),
            "gap_labels": list(self.gap_labels),
            "rule_ids": list(self.rule_ids),
            "primary_rule_id": self.primary_rule_id,
            "mode": self.mode,
            "explanation_ids": list(self.explanation_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GeneratedExample:
        if not isinstance(data, Mapping):
            raise ValueError("GeneratedExample data must be a mapping.")
        tokens = [_word_token_from_mapping(token) for token in data["source_tokens"]]
        return cls(
            source_text=str(data["source_text"]),
            target_text=str(data["target_text"]),
            source_tokens=tokens,
            token_edit_labels=list(data["token_edit_labels"]),
            gap_labels=list(data["gap_labels"]),
            rule_ids=list(data["rule_ids"]),
            primary_rule_id=str(data["primary_rule_id"]),
            mode=str(data["mode"]),
            explanation_ids=list(data["explanation_ids"]),
            metadata=dict(data["metadata"]),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> GeneratedExample:
        return cls.from_dict(json.loads(raw))

    def stable_id(self) -> str:
        payload = self.to_json().encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _require_length(name: str, values: list[str], expected: int) -> None:
    if len(values) != expected:
        raise ValueError(f"{name} length must match source_tokens length: {len(values)} != {expected}.")


def _validate_token_span(source_text: str, token: WordToken, index: int) -> None:
    if not isinstance(token, WordToken):
        raise ValueError(f"source_tokens[{index}] must be a WordToken.")
    if not token.text:
        raise ValueError(f"source_tokens[{index}].text must not be empty.")
    if token.start < 0:
        raise ValueError(f"source_tokens[{index}].start must be non-negative.")
    if token.end <= token.start:
        raise ValueError(f"source_tokens[{index}].end must be greater than start.")
    if token.end > len(source_text):
        raise ValueError(f"source_tokens[{index}].end is outside source_text.")

    substring = source_text[token.start : token.end]
    if substring != token.text:
        raise ValueError(
            f"source_tokens[{index}] does not match source_text span: "
            f"{substring!r} != {token.text!r}."
        )


def _word_token_from_mapping(data: Mapping[str, Any] | WordToken) -> WordToken:
    if isinstance(data, WordToken):
        return data
    if not isinstance(data, Mapping):
        raise ValueError("WordToken data must be a mapping.")
    return WordToken(
        text=str(data["text"]),
        start=int(data["start"]),
        end=int(data["end"]),
        lemma=data.get("lemma"),
        pos=data.get("pos"),
        feats=dict(data.get("feats", {})),
    )


__all__ = ["GeneratedExample", "WordToken"]

