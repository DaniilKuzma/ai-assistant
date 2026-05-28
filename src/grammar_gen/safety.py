from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
import re
from typing import Any

from src.grammar_gen.ast import (
    Clause,
    ComplexSentence,
    DashSubjectPredicateSentence,
    HomogeneousSentence,
    IntroductorySentence,
    NounPhrase,
    SimpleSentence,
    VerbPhrase,
)
from src.grammar_gen.lexicon import Lexicon, NounEntry, object_lemma_allowed_for_frame
from src.grammar_gen.morphology import MorphologyEngine, is_valid_prepositional_phrase
from src.grammar_gen.semantic_safety import reject_semantic_nonsense, validate_clause_semantics
from src.grammar_gen.semantics import PrepSlot, VerbFrame
from src.schema import GeneratedExample


LATIN_RE = re.compile(r"[A-Za-z]")
CONTENT_WORD_RE = re.compile(r"[А-Яа-яЁё-]+")
REPEATED_PUNCTUATION_RE = re.compile(r"([,!?;:])\1+|\.{2}(?!\.)|\.{4,}")
BROKEN_PUNCTUATION_SPACING_RE = re.compile(r"\s+[,.!?;:]|[,;:](?=\S)|[.!?](?=[А-Яа-яЁё])")
FINAL_PUNCTUATION_RE = re.compile(r"(\.\.\.|[.!?\u2026])$")
FINAL_PUNCTUATION_LABELS = {
    ".": "DOT",
    "?": "QUESTION",
    "!": "EXCLAMATION",
    "...": "ELLIPSIS",
    "\u2026": "ELLIPSIS",
}
CONTENT_WORD_STOPLIST = frozenset(
    {
        "без",
        "был",
        "была",
        "были",
        "было",
        "весь",
        "для",
        "его",
        "если",
        "или",
        "как",
        "кое",
        "кто",
        "над",
        "нас",
        "наш",
        "него",
        "нее",
        "них",
        "она",
        "они",
        "оно",
        "при",
        "про",
        "сам",
        "так",
        "там",
        "тот",
        "уже",
        "что",
        "это",
    }
)
VO_RE = re.compile(r"(^|\s)во\s+", re.IGNORECASE)
BAD_PAIR_REASONS = (
    ("bad_pair_devochka_poshel", re.compile(r"\bдевочка\s+пош[её]л\b", re.IGNORECASE)),
    ("bad_pair_devochka_proveril", re.compile(r"\bдевочка\s+проверил\b", re.IGNORECASE)),
    ("bad_pair_komissiya_reshil", re.compile(r"\bкомиссия\s+решил\b", re.IGNORECASE)),
    ("bad_pair_student_poshla", re.compile(r"\bстудент\s+пошла\b", re.IGNORECASE)),
    ("bad_pair_student_podgotovila", re.compile(r"\bстудент\s+подготовила\b", re.IGNORECASE)),
    ("bad_pair_redaktor_skazala", re.compile(r"\bредактор\s+сказала\b", re.IGNORECASE)),
    ("bad_pair_vo_ogorod", re.compile(r"\bво\s+огород\b", re.IGNORECASE)),
    ("bad_pair_dom_kupil_produkty", re.compile(r"\bдом\s+купил\s+продукты\b", re.IGNORECASE)),
    ("bad_pair_okno_obyasnilo_reshenie", re.compile(r"\bокно\s+объяснило\s+решение\b", re.IGNORECASE)),
    ("bad_pair_dokument_sel_yabloko", re.compile(r"\bдокумент\s+съел\s+яблоко\b", re.IGNORECASE)),
    ("bad_pair_otchet_skazal_fakt", re.compile(r"\bотч[её]т\s+сказал\s+факт\b", re.IGNORECASE)),
    ("bad_pair_stol_reshil_vopros", re.compile(r"\bстол\s+решил\s+вопрос\b", re.IGNORECASE)),
    ("bad_pair_produkty_prochitali_dokument", re.compile(r"\bпродукты\s+прочитали\s+документ\b", re.IGNORECASE)),
)
BAD_PAIR_REASONS = (
    *BAD_PAIR_REASONS,
    ("bad_present_plural:dannye_pokazyvaet", re.compile(r"\bданные\s+показывает\b", re.IGNORECASE)),
    ("bad_present_plural:dannye_soderzhit", re.compile(r"\bданные\s+содержит\b", re.IGNORECASE)),
    ("bad_present_plural:dannye_vklyuchaet", re.compile(r"\bданные\s+включает\b", re.IGNORECASE)),
    ("bad_answer_direct_object:request", re.compile(r"\bответил[аи]?\s+заявк[ауи]\b", re.IGNORECASE)),
    ("bad_answer_direct_object:question", re.compile(r"\bответил[аи]?\s+вопрос\b", re.IGNORECASE)),
    ("bad_subordinate_main:hold_meeting_chto", re.compile(r"\bпров[её]л[аи]?\s+собрани[ея],\s+что\b", re.IGNORECASE)),
    ("bad_subordinate_main:send_notification_chto", re.compile(r"\bотправил[аи]?\s+уведомлени[ея],\s+что\b", re.IGNORECASE)),
    ("bad_subordinate_main:compare_document_chto", re.compile(r"\bсравнил[аи]?\s+документ,\s+что\b", re.IGNORECASE)),
    ("bad_dash_pair:city_quote", re.compile(r"\bгородская\s+цитата\b", re.IGNORECASE)),
    (
        "bad_context_pair:writer_personal_contract",
        re.compile(r"\bписатель\s+создал\s+личн(?:ый|ого)\s+договор\b", re.IGNORECASE),
    ),
    (
        "bad_question_context:girl_ate_fish",
        re.compile(r"\bвчера\s+девочка\s+съела\s+рыб[уы]\?", re.IGNORECASE),
    ),
    (
        "bad_object_pair:bolnitsa_ispravila_tsitatu",
        re.compile(r"\bбольница\s+исправила\s+(?:[а-яё-]+\s+){0,2}цитат[ауые]\b", re.IGNORECASE),
    ),
)

BAD_TARGET_MORPHOLOGY_REASONS = (
    ("target_infinitive_as_finite:провести", re.compile(r"\bпровести\b", re.IGNORECASE)),
    ("target_infinitive_as_finite:открыть", re.compile(r"\bоткрыть\b", re.IGNORECASE)),
    ("target_infinitive_as_finite:найти", re.compile(r"\bнайти\b", re.IGNORECASE)),
    ("target_infinitive_as_finite:закрыть", re.compile(r"\bзакрыть\b", re.IGNORECASE)),
    ("target_infinitive_as_finite:съесть", re.compile(r"\bсъесть\b", re.IGNORECASE)),
    ("target_infinitive_as_finite:произойти", re.compile(r"\bпроизойти\b", re.IGNORECASE)),
    ("target_infinitive_as_finite:содержать", re.compile(r"\bсодержать\b", re.IGNORECASE)),
    ("bad_adjective_form:громкее", re.compile(r"\bгромкее\b", re.IGNORECASE)),
    ("bad_adjective_form:тихюю", re.compile(r"\bтихюю\b", re.IGNORECASE)),
    ("bad_np_agreement:городской_цитата", re.compile(r"\bгородской\s+цитата\b", re.IGNORECASE)),
    ("bad_adjective_noun_pair:письменная_соседка", re.compile(r"\bписьменная\s+соседка\b", re.IGNORECASE)),
    ("bad_adjective_noun_pair:письменный_студент", re.compile(r"\bписьменный\s+студент\b", re.IGNORECASE)),
    ("bad_adjective_noun_pair:внимательный_банк", re.compile(r"\bвнимательный\s+банк\b", re.IGNORECASE)),
    ("bad_adjective_noun_pair:краткое_министерство", re.compile(r"\bкраткое\s+министерство\b", re.IGNORECASE)),
    ("bad_adjective_noun_pair:личная_редакция", re.compile(r"\bличная\s+редакция\b", re.IGNORECASE)),
    ("bad_dash_pair:городская_цитата_сообщение", re.compile(r"\bгородская\s+цитата\s+[—-]\s+сообщение\b", re.IGNORECASE)),
    ("bad_dash_pair:заявка_заключение", re.compile(r"\bзаявка\s+[—-]\s+заключение\b", re.IGNORECASE)),
    ("bad_dash_pair:план_главная_сводка", re.compile(r"\bплан\s+[—-]\s+главная\s+сводка\b", re.IGNORECASE)),
    ("bad_dash_pair:заключение_личный_принцип", re.compile(r"\bзаключение\s+[—-]\s+личный\s+принцип\b", re.IGNORECASE)),
    ("bad_object_pair:подписал_абзац", re.compile(r"\bподписал[аи]?\s+(?:[а-яё-]+\s+){0,2}абзац\b", re.IGNORECASE)),
    ("bad_object_pair:исправил_инцидент", re.compile(r"\bисправил[аи]?\s+(?:[а-яё-]+\s+){0,2}инцидент\b", re.IGNORECASE)),
    ("bad_object_pair:открыл_справку", re.compile(r"\bоткрыл[аи]?\s+(?:[а-яё-]+\s+){0,2}справку\b", re.IGNORECASE)),
    (
        "uninflected_object_noun",
        re.compile(
            r"\b(?:содержит|содержал|содержала|содержали|отправил|отправила|отправили|"
            r"открыл|открыла|открыли|проверил|проверила|проверили|утвердил|утвердила|"
            r"утвердили|подписал|подписала|подписали|рассчитал|рассчитала|рассчитали|"
            r"исправил|исправила|исправили)\s+(?:[а-яё-]+\s+){0,2}"
            r"(?:ошибка|таблица|справка|инструкция|формула|сводка|проблема)\b",
            re.IGNORECASE,
        ),
    ),
)
VO_WHITELIST = ("во дворе",)
MERGE_OR_HYPHEN_LABELS = frozenset(
    {
        "MERGE_TAK_ZHE_TO_TAKZHE",
        "MERGE_TO_ZHE_TO_TOZHE",
        "MERGE_ZA_TO_TO_ZATO",
        "HYPHENATE_PARTICLE_TO",
        "HYPHENATE_PARTICLE_LIBO",
        "HYPHENATE_PARTICLE_NIBUD",
        "HYPHENATE_KOE",
        "HYPHENATE_PO_ADVERB",
    }
)
SINGLE_TOKEN_EDIT_LABELS = frozenset(
    {
        "DELETE",
        "LOWERCASE",
        "UPPERCASE",
        "CAPITALIZE",
        "SPLIT_NE_VERB",
        "SPLIT_TAKZHE_TO_TAK_ZHE",
        "SPLIT_TOZHE_TO_TO_ZHE",
        "SPLIT_ZATO_TO_ZA_TO",
        "FIX_TSYA_TO_TTSYA",
        "FIX_TTSYA_TO_TSYA",
        "DICT_REPLACE",
        "SPAN_REPLACE_BY_LEXICON",
    }
)


def validate_surface(text: str) -> list[str]:
    reasons: list[str] = []
    if "{" in text or "}" in text:
        reasons.append("template_brace")
    if "  " in text:
        reasons.append("double_space")
    if BROKEN_PUNCTUATION_SPACING_RE.search(text):
        reasons.append("broken_punctuation_spacing")
    if LATIN_RE.search(text):
        reasons.append("latin_letters")
    if _has_bad_vo_phrase(text.lower()):
        reasons.append("bad_vo_phrase")
    if text and not _starts_with_uppercase_after_opening_wrappers(text):
        reasons.append("sentence_start_not_uppercase")
    if not FINAL_PUNCTUATION_RE.search(text):
        reasons.append("missing_final_punctuation")
    if REPEATED_PUNCTUATION_RE.search(text):
        reasons.append("repeated_punctuation")

    reasons.extend(check_known_bad_phrases(text))

    return _dedupe(reasons)


def check_known_bad_phrases(text: str) -> list[str]:
    lowered = text.lower()
    reasons: list[str] = []
    for reason, pattern in BAD_PAIR_REASONS:
        if pattern.search(lowered):
            reasons.append(reason)
    return _dedupe(reasons)


def validate_generated_pair(example: GeneratedExample) -> list[str]:
    reasons: list[str] = []
    source_reasons = validate_surface(example.source_text)
    target_reasons = validate_surface(example.target_text)
    target_reasons.extend(_validate_target_morphology(example))
    allowed_source = allowed_source_surface_failures(example)
    real_source_reasons = [reason for reason in source_reasons if reason not in allowed_source]
    reasons.extend(real_source_reasons)
    reasons.extend(target_reasons)
    reasons.extend(_validate_metadata_json_safety(example.metadata))
    reasons.extend(_validate_construction_metadata(example))
    reasons.extend(_validate_token_edit_counts(example))
    reasons.extend(_validate_safety_clause_metadata(example))
    if not bool(example.metadata.get("allow_repeated_content_words", False)):
        reasons.extend(f"source_{reason}" for reason in _repeated_content_word_reasons(example.source_text))
        reasons.extend(f"target_{reason}" for reason in _repeated_content_word_reasons(example.target_text))
    return _dedupe(reasons)


def assert_json_safe_metadata(value: Any) -> None:
    if _json_safety_reason(value, "$") is not None:
        reason = _json_safety_reason(value, "$")
        raise ValueError(reason or "metadata is not JSON-safe")


def _repeated_content_word_reasons(text: str) -> list[str]:
    words = [word.lower().replace("ё", "е") for word in CONTENT_WORD_RE.findall(text)]
    if len(words) > 14:
        return []
    seen: set[str] = set()
    repeated: list[str] = []
    for word in words:
        if len(word) < 4 or word in CONTENT_WORD_STOPLIST:
            continue
        if word in seen and word not in repeated:
            repeated.append(word)
        seen.add(word)
    return [f"repeated_content_word:{word}" for word in repeated]


def count_logical_token_edits(token_labels: list[str] | tuple[str, ...]) -> int:
    count = 0
    expecting_skip_merged = False

    for label in token_labels:
        if label == "KEEP":
            expecting_skip_merged = False
            continue

        if label == "SKIP_MERGED":
            if not expecting_skip_merged:
                raise ValueError("SKIP_MERGED without preceding merge/hyphen operation")
            expecting_skip_merged = False
            continue

        if label in MERGE_OR_HYPHEN_LABELS:
            count += 1
            expecting_skip_merged = True
            continue

        if label in SINGLE_TOKEN_EDIT_LABELS:
            count += 1
            expecting_skip_merged = False
            continue

        raise ValueError(f"Unknown token edit label for logical edit count: {label}")

    return count


def allowed_source_surface_failures(example: GeneratedExample) -> set[str]:
    if (
        example.primary_rule_id in {"final_punctuation", "punct_final_marks"}
        and example.mode == "positive"
        and example.metadata.get("expected_error") == "missing_final_punctuation"
        and validate_surface(example.source_text) == ["missing_final_punctuation"]
        and validate_surface(example.target_text) == []
        and _is_valid_final_punctuation_positive_source(example)
    ):
        return {"missing_final_punctuation"}
    if (
        example.primary_rule_id.startswith("dialogue_")
        and example.mode == "positive"
        and "missing_final_punctuation" in validate_surface(example.source_text)
        and "missing_final_punctuation" not in validate_surface(example.target_text)
        and example.gap_labels
        and "DOT" in example.gap_labels
        and int(example.metadata.get("expected_gap_edit_count") or 0) > 0
    ):
        return {"missing_final_punctuation"}
    if (
        example.primary_rule_id == "casing_sentence_start"
        and example.mode == "positive"
        and validate_surface(example.source_text) == ["sentence_start_not_uppercase"]
        and "sentence_start_not_uppercase" not in validate_surface(example.target_text)
    ):
        return {"sentence_start_not_uppercase"}
    return set()


def _validate_target_morphology(example: GeneratedExample) -> list[str]:
    reasons: list[str] = []
    lowered = example.target_text.lower()
    for reason, pattern in BAD_TARGET_MORPHOLOGY_REASONS:
        if pattern.search(lowered):
            reasons.append(reason)
    reasons.extend(_validate_safety_clause_rendered_target(example))
    return _dedupe(reasons)


def validate_ast_sentence(ast: Any, rendered_text: str) -> list[str]:
    reasons = validate_surface(rendered_text)
    reasons.extend(check_predicate_rendering(ast, rendered_text))
    reasons.extend(check_subject_verb_agreement(ast, rendered_text))
    reasons.extend(check_np_agreement(ast, rendered_text))
    reasons.extend(check_preposition_case_compatibility(ast, rendered_text))
    for clause in _clauses_for_ast(ast):
        reasons.extend(_validate_clause_semantics(clause))
    return _dedupe(reasons)


def assert_valid_or_raise(ast: Any, rendered_text: str) -> None:
    reasons = validate_ast_sentence(ast, rendered_text)
    if reasons:
        raise ValueError(f"Invalid generated sentence: {', '.join(reasons)}")


def validate_target_ast_or_raise(ast: Any, target_text: str, example: GeneratedExample | None = None) -> None:
    reasons = validate_ast_sentence(ast, target_text)
    if example is not None:
        reasons.extend(validate_generated_pair(example))
    reasons = _dedupe(reasons)
    if reasons:
        raise ValueError(f"Invalid generated target: {', '.join(reasons)}")


def safety_clauses_for_ast(ast: Any, lexicon: Lexicon | None = None) -> list[dict[str, Any]]:
    lexicon = lexicon or _default_lexicon()
    morphology = _default_morphology()
    clauses: list[dict[str, Any]] = []
    for clause in _clauses_for_ast(ast):
        frame = _resolve_frame(lexicon, clause.predicate)
        if frame is None:
            raise ValueError("Clause predicate must have a known frame_id.")
        clauses.append(build_safety_clause_from_clause(clause, frame, morphology))
    return clauses


def build_safety_clause_from_clause(
    clause: Clause,
    frame: VerbFrame,
    morphology: MorphologyEngine | None = None,
) -> dict[str, Any]:
    morphology = morphology or _default_morphology()
    object_payload: dict[str, Any] | None = None
    prep_slots: list[dict[str, str]] = []

    if clause.predicate.object_np is not None:
        object_np = clause.predicate.object_np
        if object_np.preposition is None:
            object_payload = _np_payload(object_np, morphology)
        else:
            slot = _matching_prep_slot(frame, object_np.preposition, object_np.case, object_np.semantic_class)
            prep_slots.append(
                {
                    "preposition": object_np.preposition,
                    "case": _canonical_case(object_np.case),
                    "noun_lemma": object_np.noun_lemma,
                    "semantic_class": object_np.semantic_class,
                    "allowed_semantic_class": slot.semantic_class if slot is not None else object_np.semantic_class,
                    "surface": _render_np_surface(object_np, morphology),
                }
            )

    return {
        "frame_id": frame.frame_id,
        "verb_lemma": clause.predicate.verb_lemma,
        "predicate": {
            "tense": clause.predicate.tense,
            "verb_lemma": clause.predicate.verb_lemma,
            "rendered_verb": _render_predicate_verb(clause.predicate, clause.subject, morphology),
        },
        "frame_family": frame.frame_family,
        "subject": _np_payload(clause.subject, morphology),
        "object": object_payload,
        "allowed_subject_classes": list(frame.subject_classes),
        "allowed_object_classes": list(frame.object_classes),
        "prep_slots": prep_slots,
    }


def check_predicate_rendering(ast: Any, rendered_text: str) -> list[str]:
    reasons: list[str] = []
    morphology = _default_morphology()
    lowered = rendered_text.lower()

    for clause in _clauses_for_ast(ast):
        predicate = clause.predicate
        if predicate.tense == "present":
            expected = morphology.inflect_verb_present(predicate.verb_lemma, number=clause.subject.number)
        elif predicate.tense == "past":
            expected = morphology.inflect_verb_past(predicate.verb_lemma, clause.subject.gender, clause.subject.number)
        else:
            continue

        if expected and not _contains_word(lowered, expected):
            reasons.append("predicate_verb_form_missing")
        if predicate.verb_lemma != expected and _contains_word(lowered, predicate.verb_lemma):
            reasons.append("predicate_infinitive_as_finite")

    return _dedupe(reasons)


def check_subject_verb_agreement(ast: Any, rendered_text: str) -> list[str]:
    reasons: list[str] = []
    morphology = _default_morphology()
    lowered = rendered_text.lower()

    for clause in _clauses_for_ast(ast):
        predicate = clause.predicate
        subject = clause.subject
        if predicate.tense == "present":
            sing = morphology.inflect_verb_present(predicate.verb_lemma, number="sing")
            plur = morphology.inflect_verb_present(predicate.verb_lemma, number="plur")
            if subject.number == "plur" and sing != plur and _contains_word(lowered, sing):
                reasons.append("subject_verb_number_agreement")
            elif subject.number != "plur" and sing != plur and _contains_word(lowered, plur):
                reasons.append("subject_verb_number_agreement")
            continue

        if predicate.tense != "past":
            continue

        masc = morphology.inflect_verb_past(predicate.verb_lemma, "masc", "sing")
        fem = morphology.inflect_verb_past(predicate.verb_lemma, "fem", "sing")
        neut = morphology.inflect_verb_past(predicate.verb_lemma, "neut", "sing")
        plur = morphology.inflect_verb_past(predicate.verb_lemma, "plur", "plur")

        if subject.number == "plur":
            if any(_contains_word(lowered, form) for form in (masc, fem, neut)):
                reasons.append("subject_verb_number_agreement")
            continue

        if subject.gender == "fem" and _contains_word(lowered, masc):
            reasons.append("subject_verb_gender_agreement")
        elif subject.gender == "masc" and _contains_word(lowered, fem):
            reasons.append("subject_verb_gender_agreement")
        elif subject.gender == "neut" and any(_contains_word(lowered, form) for form in (masc, fem)):
            reasons.append("subject_verb_gender_agreement")

    return _dedupe(reasons)


def check_np_agreement(ast: Any, rendered_text: str) -> list[str]:
    reasons: list[str] = []
    morphology = _default_morphology()
    lowered = rendered_text.lower()

    for np in _noun_phrases_for_ast(ast):
        expected_noun = morphology.inflect_noun(np.noun_lemma, np.case, np.number)
        if expected_noun and not _contains_word(lowered, expected_noun):
            reasons.append("np_noun_form")
        for adjective in np.adjective_lemmas:
            expected = morphology.inflect_adjective(adjective, np.gender, _adjective_case(np), np.number)
            if expected and not _contains_word(lowered, expected):
                reasons.append("np_adjective_agreement")

    return _dedupe(reasons)


def check_preposition_case_compatibility(ast: Any, rendered_text: str) -> list[str]:
    reasons: list[str] = []
    lexicon = _default_lexicon()
    frame_checked_prepositional_objects: set[int] = set()

    if _has_bad_vo_phrase(rendered_text.lower()):
        reasons.append("bad_vo_phrase")

    for clause in _clauses_for_ast(ast):
        object_np = clause.predicate.object_np
        if object_np is None or object_np.preposition is None:
            continue
        frame_checked_prepositional_objects.add(id(object_np))
        frame = _resolve_frame(lexicon, clause.predicate)
        noun_entry = _resolve_noun(lexicon, object_np)
        if frame is None or noun_entry is None:
            reasons.append("invalid_prep_slot_semantics")
            continue
        if not lexicon.frames.validate_prep_slot(frame, object_np.preposition, noun_entry, object_np.case):
            reasons.append("invalid_prep_slot_semantics")

    for np in _noun_phrases_for_ast(ast):
        if np.preposition is None:
            continue
        if id(np) in frame_checked_prepositional_objects:
            continue
        noun_entry = _resolve_noun(lexicon, np)
        if noun_entry is None:
            reasons.append("unknown_preposition_np")
            continue
        if not is_valid_prepositional_phrase(np.preposition, noun_entry, np.case):
            reasons.append("invalid_preposition_case_or_semantics")

    return _dedupe(reasons)


def _clauses_for_ast(ast: Any) -> tuple[Clause, ...]:
    if isinstance(ast, SimpleSentence):
        return (ast.clause,)
    if isinstance(ast, ComplexSentence):
        return (ast.main, ast.subordinate)
    if isinstance(ast, IntroductorySentence):
        return (ast.clause,)
    if isinstance(ast, HomogeneousSentence):
        return tuple(Clause(subject=ast.subject, predicate=predicate) for predicate in ast.predicates)
    if isinstance(ast, DashSubjectPredicateSentence):
        return ()
    return ()


def _validate_clause_semantics(clause: Clause) -> list[str]:
    lexicon = _default_lexicon()
    frame = _resolve_frame(lexicon, clause.predicate)
    subject = _resolve_noun(lexicon, clause.subject)
    object_entry = _semantic_object_entry(lexicon, frame, clause.predicate.object_np)

    payload = {"frame": frame, "subject": subject, "object_np": object_entry}
    reasons = validate_clause_semantics(payload)

    if not clause.predicate.frame_id:
        reasons.append("missing_frame_id")

    if frame is not None and subject is not None and clause.predicate.verb_lemma != frame.verb_lemma:
        reasons.append("verb_frame_mismatch")

    if frame is not None and clause.predicate.object_np is not None:
        reasons.extend(_validate_object_np_semantics(lexicon, frame, clause.predicate.object_np))

    return reasons


def _semantic_object_entry(
    lexicon: Lexicon,
    frame: VerbFrame | None,
    object_np: NounPhrase | None,
) -> NounEntry | None:
    if object_np is None:
        return None
    if frame is not None and object_np.preposition is not None and not frame.object_classes:
        return None
    return _resolve_noun(lexicon, object_np)


def _validate_object_np_semantics(lexicon: Lexicon, frame: VerbFrame, object_np: NounPhrase) -> list[str]:
    object_entry = _resolve_noun(lexicon, object_np)
    if object_entry is None:
        return ["unknown_object_np"]

    if object_np.preposition is None:
        return []

    if not lexicon.frames.validate_prep_slot(frame, object_np.preposition, object_entry, object_np.case):
        return ["invalid_prep_slot_semantics"]
    return []


def _resolve_frame(lexicon: Lexicon, predicate: VerbPhrase) -> VerbFrame | None:
    if predicate.frame_id:
        for frame in lexicon.frames.frames:
            if frame.frame_id == predicate.frame_id:
                return frame
    return None


def _resolve_noun(lexicon: Lexicon, np: NounPhrase | None) -> NounEntry | None:
    if np is None:
        return None
    for noun in lexicon.nouns:
        if noun.lemma == np.noun_lemma:
            return noun
    return None


def _validate_metadata_json_safety(metadata: dict[str, Any]) -> list[str]:
    reason = _json_safety_reason(metadata, "$")
    return ["metadata_not_json_safe"] if reason is not None else []


def _validate_construction_metadata(example: GeneratedExample) -> list[str]:
    if example.metadata.get("production") is not True:
        return []

    reasons: list[str] = []
    if example.metadata.get("uses_construction_bank") is not True:
        reasons.append("missing_construction_bank_usage")

    construction_id = str(example.metadata.get("construction_id") or "").strip()
    construction_family = str(example.metadata.get("construction_family") or "").strip()
    if not construction_id:
        reasons.append("missing_construction_id")
    if not construction_family:
        reasons.append("missing_construction_family")

    raw_clauses = example.metadata.get("safety_clauses")
    if not isinstance(raw_clauses, list):
        reasons.append("missing_safety_clauses")
        raw_clauses = []

    if not construction_id:
        return _dedupe(reasons)

    try:
        from src.grammar_gen.constructions import ConstructionBank

        pattern = ConstructionBank.default().pattern(construction_id)
    except Exception:
        reasons.append("unknown_construction_id")
        pattern = None

    if pattern is None:
        return _dedupe(reasons)
    if construction_family and pattern.family != construction_family:
        reasons.append("construction_family_mismatch")
    if example.primary_rule_id not in pattern.allowed_rule_ids:
        reasons.append("construction_rule_not_allowed")

    for clause in raw_clauses:
        reasons.extend(_validate_safety_clause(clause))

    return _dedupe(reasons)


def _json_safety_reason(value: Any, path: str) -> str | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return None
    if isinstance(value, list):
        for index, item in enumerate(value):
            reason = _json_safety_reason(item, f"{path}[{index}]")
            if reason is not None:
                return reason
        return None
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                return f"{path} contains non-string key {key!r}"
            reason = _json_safety_reason(item, f"{path}.{key}")
            if reason is not None:
                return reason
        return None
    return f"{path} contains non-JSON-safe {type(value).__name__}"


def _validate_token_edit_counts(example: GeneratedExample) -> list[str]:
    reasons: list[str] = []
    try:
        logical_token_edits = count_logical_token_edits(example.token_edit_labels)
    except ValueError:
        return ["invalid_token_edit_sequence"]

    expected_token = _optional_int(example.metadata.get("expected_token_edit_count"))
    if expected_token is None and "expected_token_edit_count" in example.metadata:
        reasons.append("invalid_expected_token_edit_count")
    elif expected_token is not None and expected_token != logical_token_edits:
        reasons.append("expected_token_edit_count_mismatch")

    expected_gap = _optional_int(example.metadata.get("expected_gap_edit_count"))
    if expected_gap is None and "expected_gap_edit_count" in example.metadata:
        reasons.append("invalid_expected_gap_edit_count")

    boundary_edits = _logical_boundary_edit_count(example)
    expected_boundary = _optional_int(example.metadata.get("expected_boundary_edit_count"))
    if expected_boundary is None and "expected_boundary_edit_count" in example.metadata:
        reasons.append("invalid_expected_boundary_edit_count")
    elif expected_boundary is not None and expected_boundary != boundary_edits:
        reasons.append("expected_boundary_edit_count_mismatch")

    expected_total = _optional_int(example.metadata.get("expected_edit_count"))
    if expected_total is None and "expected_edit_count" in example.metadata:
        reasons.append("invalid_expected_edit_count")
    elif expected_total is not None:
        gap_component = expected_gap if expected_gap is not None else 0
        boundary_component = expected_boundary if expected_boundary is not None else 0
        if expected_total != logical_token_edits + gap_component + boundary_component:
            reasons.append("expected_edit_count_mismatch")

    expected_total_alias = _optional_int(example.metadata.get("expected_total_edit_count"))
    if expected_total_alias is None and "expected_total_edit_count" in example.metadata:
        reasons.append("invalid_expected_total_edit_count")
    elif expected_total_alias is not None:
        gap_component = expected_gap if expected_gap is not None else 0
        boundary_component = expected_boundary if expected_boundary is not None else 0
        if expected_total_alias != logical_token_edits + gap_component + boundary_component:
            reasons.append("expected_total_edit_count_mismatch")

    return reasons


def _logical_boundary_edit_count(example: GeneratedExample) -> int:
    return sum(1 for label in example.boundary_before_labels if label != "NONE") + sum(
        1 for label in example.boundary_after_labels if label != "NONE"
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    if result < 0:
        return None
    return result


def _validate_safety_clause_metadata(example: GeneratedExample) -> list[str]:
    uses_safety_clauses = example.metadata.get("uses_safety_clauses")
    if uses_safety_clauses is not True:
        return []

    raw_clauses = example.metadata.get("safety_clauses")
    if not isinstance(raw_clauses, list) or not raw_clauses:
        return ["missing_safety_clauses"]

    reasons: list[str] = []
    for clause in raw_clauses:
        reasons.extend(_validate_safety_clause(clause))
    return _dedupe(reasons)


def _validate_safety_clause_rendered_target(example: GeneratedExample) -> list[str]:
    raw_clauses = example.metadata.get("safety_clauses")
    if not isinstance(raw_clauses, list):
        return []

    reasons: list[str] = []
    target = example.target_text
    for clause in raw_clauses:
        if not isinstance(clause, Mapping):
            continue
        predicate = clause.get("predicate")
        if isinstance(predicate, Mapping):
            tense = str(predicate.get("tense") or "")
            verb_lemma = str(predicate.get("verb_lemma") or "")
            rendered_verb = str(predicate.get("rendered_verb") or "")
            if tense in {"past", "present"} and rendered_verb and not _contains_word(target, rendered_verb):
                reasons.append("target_predicate_form_missing")
            if (
                tense in {"past", "present"}
                and verb_lemma
                and rendered_verb
                and verb_lemma != rendered_verb
                and _contains_word(target, verb_lemma)
            ):
                reasons.append("target_predicate_lemma_rendered")

        for role in ("subject", "object"):
            payload = clause.get(role)
            if not isinstance(payload, Mapping):
                continue
            surface = str(payload.get("surface") or "")
            if surface and not _contains_phrase(target, surface):
                reasons.append(f"target_{role}_surface_missing")
            adjective_surfaces = _string_list(payload.get("adjective_surfaces"))
            for adjective_surface in adjective_surfaces:
                if adjective_surface and not _contains_word(target, adjective_surface):
                    reasons.append(f"target_{role}_adjective_missing")
    return _dedupe(reasons)


def _validate_safety_clause(clause: Any) -> list[str]:
    if not isinstance(clause, Mapping):
        return ["invalid_safety_clause"]

    lexicon = _default_lexicon()
    frame_id = str(clause.get("frame_id") or "").strip()
    verb_lemma = str(clause.get("verb_lemma") or "").strip()
    reasons: list[str] = []

    if not frame_id:
        reasons.append("missing_frame_id")
        frame = None
    else:
        frame = _frame_by_id(lexicon, frame_id)
        if frame is None:
            reasons.append("unknown_frame_id")

    if not verb_lemma:
        reasons.append("missing_verb_lemma")
    if frame is None:
        return reasons
    if verb_lemma and verb_lemma != frame.verb_lemma:
        reasons.append("verb_frame_mismatch")

    subject = clause.get("subject")
    if not isinstance(subject, Mapping):
        reasons.append("missing_subject")
    else:
        subject_class = str(subject.get("semantic_class") or "")
        allowed_subject_classes = _string_list(clause.get("allowed_subject_classes")) or list(frame.subject_classes)
        if subject_class not in frame.subject_classes or subject_class not in allowed_subject_classes:
            reasons.append("invalid_subject_semantics")

    obj = clause.get("object")
    allowed_object_classes = _string_list(clause.get("allowed_object_classes")) or list(frame.object_classes)
    if frame.object_classes:
        if obj is None:
            reasons.append("missing_object")
        elif not isinstance(obj, Mapping):
            reasons.append("invalid_object_np")
        else:
            object_class = str(obj.get("semantic_class") or "")
            if object_class not in frame.object_classes or object_class not in allowed_object_classes:
                reasons.append("invalid_object_semantics")
            object_lemma = str(obj.get("lemma") or "")
            if object_lemma and not object_lemma_allowed_for_frame(frame.frame_id, object_lemma):
                reasons.append("invalid_object_lemma_for_frame")
    elif obj is not None:
        reasons.append("unexpected_object")

    reasons.extend(_validate_safety_prep_slots(clause.get("prep_slots"), frame))
    return _dedupe(reasons)


def _validate_safety_prep_slots(raw_slots: Any, frame: VerbFrame) -> list[str]:
    if raw_slots is None:
        return []
    if not isinstance(raw_slots, list):
        return ["invalid_prep_slots"]

    reasons: list[str] = []
    for slot in raw_slots:
        if not isinstance(slot, Mapping):
            reasons.append("invalid_prep_slot")
            continue
        preposition = str(slot.get("preposition") or "")
        case = _canonical_case(str(slot.get("case") or ""))
        semantic_class = str(slot.get("semantic_class") or "")
        allowed_semantic_class = str(slot.get("allowed_semantic_class") or semantic_class)
        if semantic_class != allowed_semantic_class:
            reasons.append("invalid_prep_slot_semantics")
            continue
        if _matching_prep_slot(frame, preposition, case, semantic_class) is None:
            reasons.append("invalid_prep_slot_semantics")
    return reasons


def _frame_by_id(lexicon: Lexicon, frame_id: str) -> VerbFrame | None:
    for frame in lexicon.frames.frames:
        if frame.frame_id == frame_id:
            return frame
    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value]


def _render_predicate_verb(predicate: VerbPhrase, subject: NounPhrase, morphology: MorphologyEngine) -> str:
    if predicate.tense == "present":
        return morphology.inflect_verb_present(predicate.verb_lemma, number=subject.number)
    if predicate.tense == "past":
        return morphology.inflect_verb_past(predicate.verb_lemma, subject.gender, subject.number)
    return morphology.infinitive(predicate.verb_lemma)


def _render_np_surface(np: NounPhrase, morphology: MorphologyEngine) -> str:
    parts: list[str] = []
    if np.preposition:
        parts.append(np.preposition)
    parts.extend(
        morphology.inflect_adjective(adjective, np.gender, _adjective_case(np), np.number)
        for adjective in np.adjective_lemmas
    )
    parts.append(morphology.inflect_noun(np.noun_lemma, np.case, np.number))
    return " ".join(part for part in parts if part).strip()


def _np_payload(np: NounPhrase, morphology: MorphologyEngine) -> dict[str, Any]:
    return {
        "lemma": np.noun_lemma,
        "semantic_class": np.semantic_class,
        "gender": np.gender,
        "number": np.number,
        "animacy": np.animacy,
        "case": _canonical_case(np.case),
        "surface": _render_np_surface(np, morphology),
        "adjective_lemmas": list(np.adjective_lemmas),
        "adjective_surfaces": [
            morphology.inflect_adjective(adjective, np.gender, _adjective_case(np), np.number)
            for adjective in np.adjective_lemmas
        ],
    }


def _noun_phrases_for_ast(ast: Any) -> tuple[NounPhrase, ...]:
    phrases: list[NounPhrase] = []
    for clause in _clauses_for_ast(ast):
        phrases.append(clause.subject)
        if clause.predicate.object_np is not None:
            phrases.append(clause.predicate.object_np)

    if isinstance(ast, DashSubjectPredicateSentence):
        phrases.extend((ast.subject, ast.predicate_nominal))
    return tuple(phrases)


def _adjective_case(np: NounPhrase) -> str:
    if np.case != "accs" or np.animacy != "inanim":
        return np.case
    if np.number == "plur" or np.gender in {"masc", "neut"}:
        return "nomn"
    return np.case


def _matching_prep_slot(
    frame: VerbFrame,
    preposition: str,
    case: str,
    semantic_class: str,
) -> PrepSlot | None:
    normalized_case = _canonical_case(case)
    for slot in frame.prep_slots:
        if (
            slot.preposition == preposition
            and _canonical_case(slot.case) == normalized_case
            and slot.semantic_class == semantic_class
        ):
            return slot
    return None


def _canonical_case(case: str) -> str:
    return {
        "nom": "nomn",
        "nomn": "nomn",
        "gen": "gent",
        "gent": "gent",
        "dat": "datv",
        "datv": "datv",
        "acc": "accs",
        "accs": "accs",
        "ins": "ablt",
        "ablt": "ablt",
        "prep": "loct",
        "loct": "loct",
    }.get(case, case)


def _contains_word(text: str, word: str) -> bool:
    if not word:
        return False
    normalized_word = _default_morphology().normalize_yo(word.lower())
    pattern = rf"(?<![А-Яа-яЁё-]){re.escape(normalized_word)}(?![А-Яа-яЁё-])"
    if re.search(pattern, _default_morphology().normalize_yo(text.lower()), re.IGNORECASE):
        return True
    return False


def _contains_phrase(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    normalized_text = _default_morphology().normalize_yo(text.lower())
    normalized_phrase = _default_morphology().normalize_yo(phrase.lower())
    return normalized_phrase in normalized_text


def _has_bad_vo_phrase(text: str) -> bool:
    for match in VO_RE.finditer(text):
        start = match.start() + len(match.group(1) or "")
        suffix = text[start:]
        if not any(_starts_with_whitelist_phrase(suffix, phrase) for phrase in VO_WHITELIST):
            return True
    return False


def _starts_with_uppercase_after_opening_wrappers(text: str) -> bool:
    stripped = text.lstrip()
    while stripped and stripped[0] in "«\"'“„([":
        stripped = stripped[1:].lstrip()
    return bool(stripped and (stripped[0].isupper() or stripped[0].isdigit()))


def _is_valid_final_punctuation_positive_source(example: GeneratedExample) -> bool:
    target_mark = _final_mark(example.target_text)
    if target_mark is None:
        return False
    if _final_mark(example.source_text) is not None:
        return False
    if example.source_text != example.target_text[: -len(target_mark)]:
        return False
    if not example.gap_labels:
        return False
    if example.gap_labels[-1] != FINAL_PUNCTUATION_LABELS[target_mark]:
        return False
    return all(label == "KEEP" for label in example.token_edit_labels)


def _final_mark(text: str) -> str | None:
    stripped = text.rstrip()
    if stripped.endswith("..."):
        return "..."
    if stripped.endswith("\u2026"):
        return "\u2026"
    if stripped and stripped[-1] in ".!?":
        return stripped[-1]
    return None


def _starts_with_whitelist_phrase(text: str, phrase: str) -> bool:
    if not text.startswith(phrase):
        return False
    if len(text) == len(phrase):
        return True
    return text[len(phrase)] in " \t\r\n,.!?;:"


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


@lru_cache(maxsize=1)
def _default_lexicon() -> Lexicon:
    return Lexicon.default()


@lru_cache(maxsize=1)
def _default_morphology() -> MorphologyEngine:
    return MorphologyEngine(lexicon=_default_lexicon(), critical=True)
