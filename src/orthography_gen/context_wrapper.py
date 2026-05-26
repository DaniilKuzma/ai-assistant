from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.runtime.tokenization import tokenize_runtime_words
from src.schema import WordToken


@dataclass(frozen=True)
class WrappedContext:
    source_text: str
    target_text: str
    source_tokens: list[WordToken]
    edited_token_index: int
    context: dict[str, Any]
    construction_id: str


class ContextWrapper:
    def wrap(
        self,
        *,
        source_word: str,
        target_word: str,
        pos: str,
        context: Mapping[str, Any] | None = None,
    ) -> WrappedContext:
        payload = dict(context or {})
        template = str(payload.get("template") or "").strip()
        construction_id = str(payload.get("construction_id") or "").strip()
        if not template:
            template, construction_id = self._default_template(pos, payload)
        if not construction_id:
            construction_id = _construction_for_template(template, pos)

        format_args = {"word": source_word, "variant": payload.get("variant", "")}
        source_text = template.format(**format_args)
        format_args["word"] = target_word
        target_text = template.format(**format_args)
        if payload.get("vary_shell", True) is not False and payload.get("variant"):
            source_text = _add_variant_shell(source_text, payload["variant"])
            target_text = _add_variant_shell(target_text, payload["variant"])
        tokens = tokenize_runtime_words(source_text)
        edited_index = _find_token(tokens, source_word)
        return WrappedContext(
            source_text=source_text,
            target_text=target_text,
            source_tokens=tokens,
            edited_token_index=edited_index,
            context=payload,
            construction_id=construction_id,
        )

    def _default_template(self, pos: str, context: Mapping[str, Any]) -> tuple[str, str]:
        if bool(context.get("quote", False)):
            return "В словаре указано слово «{word}».", "orthography_word_quote"
        if pos == "NOUN":
            return "Редактор проверил слово «{word}».", "orthography_word_quote"
        if pos in {"ADJ", "ADJF", "PRTF"}:
            noun = str(context.get("noun") or "пример")
            subject = str(context.get("subject") or "Редактор")
            return f"{subject} заметил {{word}} {noun}.", "orthography_adjective_safe_context"
        return "В словаре указано слово «{word}».", "orthography_word_quote"


def _find_token(tokens: list[WordToken], source_word: str) -> int:
    lowered = source_word.lower()
    for index, token in enumerate(tokens):
        if token.text.lower() == lowered:
            return index
    raise ValueError(f"Wrapped source word {source_word!r} was not found as a token.")


def _construction_for_template(template: str, pos: str) -> str:
    if "«{word}»" in template:
        return "orthography_word_quote"
    if pos == "NOUN":
        return "orthography_noun_safe_context"
    return "orthography_adjective_safe_context"


def _add_variant_shell(text: str, variant: Any) -> str:
    return f"В примере {variant} сказано: {text}"


__all__ = ["ContextWrapper", "WrappedContext"]
