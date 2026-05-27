from __future__ import annotations

from src.grammar_gen.randomness import RandomSource
from src.grammar_gen.rules.common import gap_labels_from_text
from src.runtime.tokenization import tokenize_runtime_words
from src.rule_layers.base import LayerDirectCase
from src.schema import GeneratedExample, WordToken


VARIED_FAMILIES = frozenset({"compound_spelling", "dictionary_typo", "syntax_punctuation"})

DOCUMENTS: tuple[str, ...] = (
    "черновике",
    "отчёте",
    "протоколе",
    "инструкции",
    "заметке",
    "регламенте",
    "журнале правок",
    "учебном задании",
    "деловом письме",
    "текстовой сводке",
    "редакторской карточке",
    "рабочей памятке",
)
DOCUMENTS_GEN: tuple[str, ...] = (
    "черновика",
    "отчёта",
    "протокола",
    "инструкции",
    "заметки",
    "регламента",
    "журнала правок",
    "учебного задания",
    "делового письма",
    "текстовой сводки",
    "редакторской карточки",
    "рабочей памятки",
)
ACTORS: tuple[str, ...] = (
    "корректор",
    "редактор",
    "методист",
    "лингвист",
    "эксперт",
    "автор",
    "проверяющий",
    "куратор",
)
VERBS: tuple[str, ...] = (
    "оставил",
    "сверил",
    "проверил",
    "выделил",
    "подчеркнул",
    "обсудил",
    "заметил",
    "вынес в список",
)
SHELLS: tuple[str, ...] = (
    "В {doc} {actor} оставил строку: {text}",
    "{actor_cap} сверил фразу в {doc}: {text}",
    "На полях {doc_gen} осталась запись: {text}",
    "После проверки {doc_gen} сохранилась строка: {text}",
    "В рабочей версии {doc_gen} указали фразу: {text}",
    "{actor_cap} {verb} такой фрагмент: {text}",
    "В карточке задания записали: {text}",
    "Для редакторской проверки оставили: {text}",
)


def contextualize_case(case: LayerDirectCase, rng: RandomSource) -> LayerDirectCase:
    if not _should_contextualize(case):
        return case

    doc_index = rng.randint(0, len(DOCUMENTS) - 1)
    actor = rng.choice(ACTORS)
    shell = rng.choice(SHELLS)
    verb = rng.choice(VERBS)
    source_text = _render_shell(
        shell,
        text=case.source_text,
        doc=DOCUMENTS[doc_index],
        doc_gen=DOCUMENTS_GEN[doc_index],
        actor=actor,
        verb=verb,
    )
    target_text = _render_shell(
        shell,
        text=case.target_text,
        doc=DOCUMENTS[doc_index],
        doc_gen=DOCUMENTS_GEN[doc_index],
        actor=actor,
        verb=verb,
    )
    metadata = dict(case.metadata)
    metadata["context_variation"] = True
    metadata["context_shell"] = shell.replace("{text}", "").strip()
    return LayerDirectCase(
        rule_id=case.rule_id,
        family=case.family,
        sub_rule_id=case.sub_rule_id,
        mode=case.mode,
        source_text=source_text,
        target_text=target_text,
        token_operations=case.token_operations,
        gap_operations=case.gap_operations,
        expected_token_edit_count=case.expected_token_edit_count,
        expected_gap_edit_count=case.expected_gap_edit_count,
        metadata=metadata,
        weight=case.weight,
        direct_token_labels=case.direct_token_labels,
        direct_gap_labels=case.direct_gap_labels,
    )


def contextualize_example(example: GeneratedExample, rng: RandomSource) -> GeneratedExample:
    if _truthy(example.metadata.get("disable_context_variation")):
        return example

    for _ in range(8):
        candidate = _contextualize_example_once(example, rng)
        if candidate == example:
            return example
        if not _validation_reasons(candidate):
            return candidate
    return example


def _contextualize_example_once(example: GeneratedExample, rng: RandomSource) -> GeneratedExample:
    prefix, shell_id = _sample_prefix(rng)
    prefix_tokens = tokenize_runtime_words(prefix)
    if not prefix_tokens:
        return example

    shifted_tokens = [
        WordToken(
            text=token.text,
            start=token.start + len(prefix),
            end=token.end + len(prefix),
            lemma=token.lemma,
            pos=token.pos,
            feats=dict(token.feats),
        )
        for token in example.source_tokens
    ]
    metadata = dict(example.metadata)
    metadata["context_variation"] = True
    metadata["context_shell"] = shell_id
    return GeneratedExample(
        source_text=f"{prefix}{example.source_text}",
        target_text=f"{prefix}{example.target_text}",
        source_tokens=[*prefix_tokens, *shifted_tokens],
        token_edit_labels=["KEEP"] * len(prefix_tokens) + list(example.token_edit_labels),
        gap_labels=gap_labels_from_text(prefix, prefix_tokens) + list(example.gap_labels),
        rule_ids=["none"] * len(prefix_tokens) + list(example.rule_ids),
        primary_rule_id=example.primary_rule_id,
        mode=example.mode,
        explanation_ids=list(example.explanation_ids),
        metadata=metadata,
    )


def _validation_reasons(example: GeneratedExample) -> list[str]:
    from src.grammar_gen.safety import validate_generated_pair

    return validate_generated_pair(example)


def _should_contextualize(case: LayerDirectCase) -> bool:
    if case.family not in VARIED_FAMILIES:
        return False
    if _truthy(case.metadata.get("disable_context_variation")):
        return False
    return not case.direct_token_labels and not case.direct_gap_labels


def _render_shell(
    shell: str,
    *,
    text: str,
    doc: str,
    doc_gen: str,
    actor: str,
    verb: str,
) -> str:
    return shell.format(
        text=text,
        doc=doc,
        doc_gen=doc_gen,
        actor=actor,
        actor_cap=actor.capitalize(),
        verb=verb,
    )


def _sample_prefix(rng: RandomSource) -> tuple[str, str]:
    doc_index = rng.randint(0, len(DOCUMENTS) - 1)
    actor = rng.choice(ACTORS)
    shell = rng.choice(SHELLS)
    verb = rng.choice(VERBS)
    prefix = _render_shell(
        shell,
        text="",
        doc=DOCUMENTS[doc_index],
        doc_gen=DOCUMENTS_GEN[doc_index],
        actor=actor,
        verb=verb,
    )
    return prefix, shell.replace("{text}", "").strip()


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


__all__ = ["contextualize_case", "contextualize_example"]
