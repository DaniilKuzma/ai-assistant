from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
import re
from typing import Any

from src.grammar_gen.rules.base import GenerationMode
from src.grammar_gen.safety import allowed_source_surface_failures, validate_generated_pair, validate_surface
from src.schema import GeneratedExample
from src.schema.labels import gap_label_to_id, rule_tag_to_id, token_label_to_id


LATIN_RE = re.compile(r"[A-Za-z]")
FORBIDDEN_PHRASES = (
    ("\u0434\u0435\u0432\u043e\u0447\u043a\u0430 \u043f\u043e\u0448\u0451\u043b", "forbidden_devochka_poshyol"),
    ("\u0434\u0435\u0432\u043e\u0447\u043a\u0430 \u043f\u043e\u0448\u0435\u043b", "forbidden_devochka_poshel"),
    ("\u043a\u043e\u043c\u0438\u0441\u0441\u0438\u044f \u0440\u0435\u0448\u0438\u043b", "forbidden_komissiya_reshil"),
    ("\u0432\u043e \u043e\u0433\u043e\u0440\u043e\u0434", "forbidden_vo_ogorod"),
    ("\u043a \u043e\u0433\u043e\u0440\u043e\u0434", "forbidden_k_ogorod"),
    ("\u043f\u043e \u0440\u0443\u0441\u043a\u0438\u0439", "forbidden_po_ruskiy"),
    ("письменная соседка", "forbidden_written_neighbor"),
    ("письменный студент", "forbidden_written_student"),
    ("внимательный банк", "forbidden_attentive_bank"),
    ("краткое министерство", "forbidden_brief_ministry"),
    ("личная редакция", "forbidden_personal_editorial_office"),
    ("городская цитата — сообщение", "forbidden_city_quote_message_dash"),
    ("заявка — заключение", "forbidden_request_conclusion_dash"),
    ("план — главная сводка", "forbidden_plan_summary_dash"),
    ("заключение — личный принцип", "forbidden_conclusion_principle_dash"),
    ("подписал абзац", "forbidden_signed_paragraph"),
    ("подписала абзац", "forbidden_signed_paragraph"),
    ("исправил инцидент", "forbidden_corrected_incident"),
    ("исправила инцидент", "forbidden_corrected_incident"),
    ("открыл справку", "forbidden_opened_certificate"),
)
TAKZHE_KAK_RE = re.compile(
    r"\b\u0442\u0430\u043a\u0436\u0435\s*,?\s+\u043a\u0430\u043a\b",
    re.IGNORECASE,
)
NUMBERED_EXAMPLE_SHELL_RE = re.compile(r"^\u0412 \u043f\u0440\u0438\u043c\u0435\u0440\u0435 \d+ \u0441\u043a\u0430\u0437\u0430\u043d\u043e:")


def audit_example(example: GeneratedExample) -> list[str]:
    reasons: list[str] = []

    if not isinstance(example, GeneratedExample):
        return [f"not_generated_example:{type(example).__name__}"]

    try:
        validated = GeneratedExample.from_dict(example.to_dict())
    except Exception as exc:
        return [f"generated_example_invariant:{type(exc).__name__}:{exc}"]

    if validated != example:
        reasons.append("generated_example_roundtrip_mismatch")

    reasons.extend(_surface_reasons(example))
    reasons.extend(f"pair_validation:{reason}" for reason in validate_generated_pair(example))
    reasons.extend(_label_reasons(example))
    reasons.extend(_mode_reasons(example))
    reasons.extend(_forbidden_phrase_reasons(example))

    return _dedupe(reasons)


def audit_batch(examples: Iterable[GeneratedExample]) -> dict[str, Any]:
    example_list = list(examples)
    failure_reasons: Counter[str] = Counter()
    first_failed_examples: list[dict[str, Any]] = []
    failed_examples_count = 0

    for index, example in enumerate(example_list):
        reasons = audit_example(example)
        if not reasons:
            continue
        failed_examples_count += 1
        failure_reasons.update(reasons)
        if len(first_failed_examples) < 10:
            first_failed_examples.append(
                {
                    "index": index,
                    "stable_id": _stable_id_or_empty(example),
                    "primary_rule_id": getattr(example, "primary_rule_id", ""),
                    "mode": getattr(example, "mode", ""),
                    "reasons": reasons,
                    "source_text": getattr(example, "source_text", ""),
                    "target_text": getattr(example, "target_text", ""),
                }
            )

    return {
        "count": len(example_list),
        "failed_examples_count": failed_examples_count,
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "rule_distribution": dict(sorted(Counter(example.primary_rule_id for example in example_list).items())),
        "mode_distribution": dict(sorted(Counter(example.mode for example in example_list).items())),
        "first_failed_examples": first_failed_examples,
    }


def _surface_reasons(example: GeneratedExample) -> list[str]:
    reasons: list[str] = []
    allowed_source = allowed_source_surface_failures(example)
    for reason in validate_surface(example.source_text):
        if reason not in allowed_source:
            reasons.append(f"source_surface:{reason}")
    for reason in validate_surface(example.target_text):
        reasons.append(f"target_surface:{reason}")

    for field_name, text in (("source", example.source_text), ("target", example.target_text)):
        if "{" in text or "}" in text:
            reasons.append(f"{field_name}_braces")
        if LATIN_RE.search(text):
            reasons.append(f"{field_name}_latin_letters")
    return reasons


def _label_reasons(example: GeneratedExample) -> list[str]:
    reasons: list[str] = []
    for label in example.token_edit_labels:
        try:
            token_label_to_id(label)
        except ValueError:
            reasons.append(f"unknown_token_label:{label}")
    for label in example.gap_labels:
        try:
            gap_label_to_id(label)
        except ValueError:
            reasons.append(f"unknown_gap_label:{label}")
    for rule_id in example.rule_ids + [example.primary_rule_id]:
        try:
            rule_tag_to_id(rule_id)
        except ValueError:
            reasons.append(f"unknown_rule_tag:{rule_id}")
    return reasons


def _mode_reasons(example: GeneratedExample) -> list[str]:
    reasons: list[str] = []
    if example.mode == GenerationMode.POSITIVE.value:
        if example.source_text == example.target_text:
            reasons.append("positive_source_equals_target")
        if all(label == "KEEP" for label in example.token_edit_labels) and all(
            label == "NONE" for label in example.gap_labels
        ):
            reasons.append("positive_has_no_active_labels")
    elif example.mode in {GenerationMode.HARD_NEGATIVE.value, GenerationMode.CLEAN_IDENTITY.value}:
        if example.source_text != example.target_text:
            reasons.append(f"{example.mode}_source_differs_from_target")
    else:
        reasons.append(f"unknown_generation_mode:{example.mode}")
    return reasons


def _forbidden_phrase_reasons(example: GeneratedExample) -> list[str]:
    reasons: list[str] = []
    texts = (example.source_text.lower(), example.target_text.lower())
    for phrase, reason in FORBIDDEN_PHRASES:
        if any(_contains_forbidden_phrase(text, phrase) for text in texts):
            reasons.append(reason)

    if example.mode == GenerationMode.HARD_NEGATIVE.value and any(TAKZHE_KAK_RE.search(text) for text in texts):
        reasons.append("forbidden_hard_negative_takzhe_kak")
    if any(NUMBERED_EXAMPLE_SHELL_RE.search(text) for text in (example.source_text, example.target_text)):
        reasons.append("forbidden_numbered_example_shell")
    return reasons


def _stable_id_or_empty(example: GeneratedExample) -> str:
    try:
        return example.stable_id()
    except Exception:
        return ""


def _contains_forbidden_phrase(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    if not (_is_cyrillic_word_char(phrase[0]) or _is_cyrillic_word_char(phrase[-1])):
        return phrase in text
    left = r"(?<![А-Яа-яЁё-])" if _is_cyrillic_word_char(phrase[0]) else ""
    right = r"(?![А-Яа-яЁё-])" if _is_cyrillic_word_char(phrase[-1]) else ""
    return re.search(f"{left}{re.escape(phrase)}{right}", text, re.IGNORECASE) is not None


def _is_cyrillic_word_char(value: str) -> bool:
    return bool(re.fullmatch(r"[А-Яа-яЁё-]", value))


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


__all__ = ["audit_batch", "audit_example"]

