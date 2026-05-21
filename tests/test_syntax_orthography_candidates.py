from __future__ import annotations

from src.candidates.candidate_generator import Candidate, CandidateGenerator
from src.inference.corrector import Corrector
from src.validation.strict_validator import StrictValidator


SYNTAX_RULE_IDS = {
    "ne_verb",
    "ne_adjective",
    "ne_adverb",
    "ne_participle",
    "ne_short_form",
    "ne_predicative",
    "ni_stable_expression",
    "ni_particle_context",
    "n_nn_adjective",
    "n_nn_participle",
    "n_nn_deverbal_adjective",
    "n_nn_short_form",
    "tsya_soft_insert",
    "tsya_soft_delete",
    "context_tak_zhe",
    "context_to_zhe",
    "context_chto_by",
    "context_za_to",
    "context_vsledstvie",
    "context_nesmotrya",
}


def _edit_candidates(text: str) -> list[Candidate]:
    return [
        candidate
        for candidate in CandidateGenerator().generate(text)
        if candidate.edit_type != "keep" and candidate.source != candidate.replacement
    ]


def _find_candidate(text: str, source: str, replacement: str, rule_id: str) -> Candidate:
    candidates = _edit_candidates(text)
    return next(
        candidate
        for candidate in candidates
        if candidate.source.lower() == source.lower()
        and candidate.replacement.lower() == replacement.lower()
        and candidate.rule_id == rule_id
    )


def test_canonical_syntax_orthography_module_exports_active_rule_ids():
    from src.rules.registry import rule_by_id
    from src.rules.syntax_orthography import syntax_orthography_rules

    exported_ids = {rule.spec.id for rule in syntax_orthography_rules()}

    assert SYNTAX_RULE_IDS <= exported_ids
    assert all(rule_by_id(rule_id) is not None for rule_id in SYNTAX_RULE_IDS)


def test_ne_candidates_include_syntax_metadata_and_model_gate():
    cases = [
        ("Я незнал ответа.", "незнал", "не знал", "ne_verb", "verb_split"),
        ("Он сделал это не случайно.", "не случайно", "неслучайно", "ne_adverb", "adverb_join"),
        ("Не проверенный вовремя документ вернули.", "Не проверенный", "Непроверенный", "ne_participle", "participle_join"),
        ("Он не согласен.", "не согласен", "несогласен", "ne_short_form", "short_form_join"),
        ("Это не возможно.", "не возможно", "невозможно", "ne_predicative", "predicative_join"),
    ]

    for text, source, replacement, rule_id, subtype in cases:
        candidate = _find_candidate(text, source, replacement, rule_id)

        assert candidate.edit_type == "split_join"
        assert candidate.mode == "model_required"
        assert candidate.requires_model is True
        assert candidate.requires_scoring is True
        assert candidate.requires == ("morphology", "syntax", "model")
        assert candidate.syntax_family == "ne_with_parts_of_speech"
        assert candidate.subtype == subtype
        assert "syntax" in candidate.evidence
        assert candidate.metadata and candidate.metadata["syntax_family"] == "ne_with_parts_of_speech"


def test_ni_candidates_are_bounded_and_model_required():
    stable = _find_candidate("Он не разу не ошибся.", "не разу", "ни разу", "ni_stable_expression")
    particle = _find_candidate("Что бы он не сказал, решение принято.", "не", "ни", "ni_particle_context")

    for candidate in (stable, particle):
        assert candidate.mode == "model_required"
        assert candidate.requires_model is True
        assert candidate.requires == ("morphology", "syntax", "model")
        assert candidate.syntax_family == "ni_context"
        assert candidate.edit_domain == "orthography"
        assert "bounded" in candidate.evidence


def test_n_nn_and_tsya_candidates_include_metadata():
    cases = [
        ("Длиный путь занял день.", "Длиный", "Длинный", "n_nn_adjective", "n_nn_context"),
        ("Жареный на масле картофель остыл.", "Жареный", "Жаренный", "n_nn_participle", "n_nn_context"),
        ("Жаренный картофель остыл.", "Жаренный", "Жареный", "n_nn_deverbal_adjective", "n_nn_context"),
        ("Ошибка исправленна.", "исправленна", "исправлена", "n_nn_short_form", "n_nn_context"),
        ("Они могут появится завтра.", "появится", "появиться", "tsya_soft_insert", "tsya_tsya_context"),
    ]

    for text, source, replacement, rule_id, syntax_family in cases:
        candidate = _find_candidate(text, source, replacement, rule_id)

        assert candidate.mode == "model_required"
        assert candidate.requires_model is True
        assert candidate.syntax_family == syntax_family
        assert candidate.edit_domain == "orthography"
        assert candidate.evidence


def test_context_pair_candidates_are_bounded_and_syntax_marked():
    cases = [
        ("Он сделал так же как коллега.", "так же", "также", "context_tak_zhe"),
        ("Он сделал то же что коллега.", "то же", "тоже", "context_to_zhe"),
        ("Чтобы закончить работу нужно согласование.", "Чтобы", "Что бы", "context_chto_by"),
        ("Он отвечает за то решение.", "за то", "зато", "context_za_to"),
        ("Вследствие ошибки отчёт вернули.", "Вследствие", "В следствие", "context_vsledstvie"),
        ("Несмотря на дождь встреча началась.", "Несмотря на", "Не смотря на", "context_nesmotrya"),
    ]

    for text, source, replacement, rule_id in cases:
        candidate = _find_candidate(text, source, replacement, rule_id)

        assert candidate.mode == "model_required"
        assert candidate.requires_model is True
        assert candidate.requires == ("syntax", "morphology", "model")
        assert candidate.syntax_family == "context_pairs"
        assert "bounded context pair" in candidate.evidence


def test_plain_corrector_does_not_auto_apply_syntax_orthography_candidates():
    examples = [
        "Я незнал ответа.",
        "Они могут появится завтра.",
        "Он сделал так же как коллега.",
        "Чтобы закончить работу нужно согласование.",
        "Вследствие ошибки отчёт вернули.",
    ]

    for source in examples:
        assert Corrector().correct(source).corrected_text == source


def test_validator_uses_syntax_orthography_rejection_reasons():
    validator = StrictValidator(tsya_threshold=0.98)

    n_nn = validator.validate(
        "Власти намерены добиваться компенсации.",
        "Власти намеренны добиваться компенсации.",
        trusted_edits=[
            Candidate(
                "намерены",
                "намеренны",
                "spelling",
                start=7,
                end=15,
                confidence=0.999,
                requires_model=True,
                rule_id="n_nn_short_form",
                syntax_family="n_nn_context",
            )
        ],
    )
    ne = validator.validate(
        "Это не случайно важно.",
        "Это неслучайно важно.",
        trusted_edits=[
            Candidate(
                "не случайно",
                "неслучайно",
                "split_join",
                start=4,
                end=15,
                confidence=0.999,
                requires_model=True,
                rule_id="ne_adverb",
                syntax_family="ne_with_parts_of_speech",
            )
        ],
    )
    tsya = validator.validate(
        "Он учится каждый день.",
        "Он учиться каждый день.",
        trusted_edits=[
            Candidate(
                "учится",
                "учиться",
                "spelling",
                start=3,
                end=9,
                confidence=0.999,
                requires_model=True,
                rule_id="tsya_soft_insert",
                syntax_family="tsya_tsya_context",
            )
        ],
    )

    assert any(edit.status == "rejected" and edit.reason == "n_nn_unsafe" for edit in n_nn.edits)
    assert any(edit.status == "rejected" and edit.reason == "ne_ni_unsafe" for edit in ne.edits)
    assert any(edit.status == "rejected" and edit.reason == "tsya_unsafe" for edit in tsya.edits)


def test_validator_allows_useful_trusted_syntax_orthography_repairs():
    validator = StrictValidator(tsya_threshold=0.98)

    tsya_source = "Они могут появится завтра."
    tsya_target = "Они могут появиться завтра."
    tsya_start = tsya_source.index("появится")
    tsya_result = validator.validate(
        tsya_source,
        tsya_target,
        trusted_edits=[
            Candidate(
                "появится",
                "появиться",
                "spelling",
                start=tsya_start,
                end=tsya_start + len("появится"),
                confidence=0.99,
                requires_model=True,
                rule_id="tsya_soft_insert",
                syntax_family="tsya_tsya_context",
            )
        ],
    )

    nn_source = "Длиный путь занял день."
    nn_target = "Длинный путь занял день."
    nn_result = validator.validate(
        nn_source,
        nn_target,
        trusted_edits=[
            Candidate(
                "Длиный",
                "Длинный",
                "spelling",
                start=0,
                end=len("Длиный"),
                confidence=0.99,
                requires_model=True,
                rule_id="n_nn_adjective",
                syntax_family="n_nn_context",
            )
        ],
    )

    assert tsya_result.apply_accepted() == tsya_target
    assert nn_result.apply_accepted() == nn_target
