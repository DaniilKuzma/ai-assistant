from src.candidates.candidate_generator import Candidate
import pytest
import torch

from src.inference.corrector import CorrectionResult
from src.inference.model_corrector import (
    BatchedFeatureModelScores,
    ModelCandidatePrediction,
    ModelPunctuationPrediction,
    TorchCandidateModelBackend,
    TrainedModelCorrector,
    _prepare_heads_state_dict_for_module,
    _select_candidates,
)
from src.memory.correction_memory import CorrectionMemory
from src.model.heads import build_linear_heads
from src.training.tensorization import DebugTokenizer, build_training_feature


class FakeBackend:
    def __init__(self, scores: dict[str, float], punctuation=None):
        self.scores = scores
        self.punctuation = punctuation or []

    def score_candidates(self, text: str, candidates: list[Candidate]) -> list[ModelCandidatePrediction]:
        return [
            ModelCandidatePrediction(
                candidate=candidate,
                score=self.scores.get(candidate.replacement, 0.0),
                confidence=self.scores.get(candidate.replacement, 0.0),
            )
            for candidate in candidates
        ]

    def predict_punctuation(self, text: str):
        return self.punctuation


def _generated_candidate(corrector: TrainedModelCorrector, text: str, replacement: str) -> Candidate:
    for candidate in corrector.candidates.generate(text):
        if candidate.replacement == replacement and candidate.edit_type != "keep":
            return candidate
    raise AssertionError(f"Candidate with replacement {replacement!r} was not generated")


def _generated_punctuation_candidate(
    corrector: TrainedModelCorrector,
    text: str,
    *,
    rule_id: str,
    label: str,
    action: str,
) -> Candidate:
    for candidate in corrector.candidates.generate(text):
        if candidate.rule_id == rule_id and candidate.label == label and candidate.action == action:
            return candidate
    raise AssertionError(f"Punctuation candidate {rule_id!r}/{label!r}/{action!r} was not generated")


def _decision_for(result: CorrectionResult, replacement: str) -> dict:
    for decision in result.candidate_decisions:
        if decision.get("replacement") == replacement:
            return decision
    raise AssertionError(f"Decision for replacement {replacement!r} was not recorded")


def _punctuation_decision_for(result: CorrectionResult, *, rule_id: str, replacement: str) -> dict:
    for decision in result.candidate_decisions:
        if (
            decision.get("rule_id") == rule_id
            and decision.get("replacement") == replacement
            and decision.get("action")
        ):
            return decision
    raise AssertionError(f"Punctuation decision for {rule_id!r}/{replacement!r} was not recorded")


def test_trained_model_corrector_does_not_apply_whitelist_candidate_when_model_score_is_low():
    corrector = TrainedModelCorrector(FakeBackend({}), thresholds={"split_join_threshold": 0.9})

    result = corrector.correct("Я незнаю что делать")

    assert result.corrected_text == "Я незнаю что делать"


def test_trained_model_corrector_applies_candidate_when_model_score_passes_threshold():
    corrector = TrainedModelCorrector(FakeBackend({"не знаю": 0.99}), thresholds={"split_join_threshold": 0.9})

    result = corrector.correct("Я незнаю что делать")

    assert result.corrected_text == "Я не знаю что делать"
    assert any(edit.edit_type == "split_word" and edit.status == "accepted" for edit in result.edits)


def test_rejected_memory_suppresses_candidate_even_when_score_passes_threshold():
    text = "Я незнаю что делать"
    memory = CorrectionMemory(storage_path=None)
    corrector = TrainedModelCorrector(
        FakeBackend({"не знаю": 0.99}),
        thresholds={"split_join_threshold": 0.9},
        correction_memory=memory,
        doc_id="doc-1",
    )
    memory.remember_candidate(text, _generated_candidate(corrector, text, "не знаю"), "rejected", doc_id="doc-1")

    result = corrector.correct(text)

    assert result.corrected_text == text
    decision = _decision_for(result, "не знаю")
    assert decision["memory_decision"] == "rejected"
    assert decision["memory_applied"] is True
    assert decision["memory_reason"] == "exact_context_key"
    assert decision["memory_key"]
    assert decision["selected"] is False


def test_ignored_memory_suppresses_candidate_even_when_score_passes_threshold():
    text = "Я незнаю что делать"
    memory = CorrectionMemory(storage_path=None)
    corrector = TrainedModelCorrector(
        FakeBackend({"не знаю": 0.99}),
        thresholds={"split_join_threshold": 0.9},
        correction_memory=memory,
        doc_id="doc-1",
    )
    memory.remember_candidate(text, _generated_candidate(corrector, text, "не знаю"), "ignored", doc_id="doc-1")

    result = corrector.correct(text)

    assert result.corrected_text == text
    decision = _decision_for(result, "не знаю")
    assert decision["memory_decision"] == "ignored"
    assert decision["memory_applied"] is True
    assert decision["selected"] is False


def test_accepted_memory_reuses_candidate_even_when_score_is_below_threshold():
    text = "Я незнаю что делать"
    memory = CorrectionMemory(storage_path=None)
    corrector = TrainedModelCorrector(
        FakeBackend({"не знаю": 0.1}),
        thresholds={"split_join_threshold": 0.9},
        correction_memory=memory,
        doc_id="doc-1",
    )
    memory.remember_candidate(text, _generated_candidate(corrector, text, "не знаю"), "accepted", doc_id="doc-1")

    result = corrector.correct(text)

    assert result.corrected_text == "Я не знаю что делать"
    decision = _decision_for(result, "не знаю")
    assert decision["memory_decision"] == "accepted"
    assert decision["memory_applied"] is True
    assert decision["selected_by_memory"] is True
    assert decision["threshold_passed"] is False
    assert decision["selected"] is True


def test_accepted_memory_selected_candidate_still_goes_through_strict_validator():
    text = "Они могут появиться завтра."
    memory = CorrectionMemory(storage_path=None)
    corrector = TrainedModelCorrector(
        FakeBackend({"появится": 0.1}),
        thresholds={"spelling_threshold": 0.9, "tsya_threshold": 0.98},
        correction_memory=memory,
        doc_id="doc-1",
    )
    memory.remember_candidate(text, _generated_candidate(corrector, text, "появится"), "accepted", doc_id="doc-1")

    result = corrector.correct(text)

    assert result.corrected_text == text
    assert any(edit.source == "появиться" and edit.status == "rejected" and edit.reason == "tsya_unsafe" for edit in result.edits)
    decision = _decision_for(result, "появится")
    assert decision["selected_by_memory"] is True
    assert decision["validator_status"] == "rejected"
    assert decision["validator_reason"] == "tsya_unsafe"


def test_without_memory_existing_threshold_behavior_is_unchanged_and_trace_has_memory_fields():
    corrector = TrainedModelCorrector(FakeBackend({}), thresholds={"split_join_threshold": 0.9})

    result = corrector.correct("Я незнаю что делать")

    assert result.corrected_text == "Я незнаю что делать"
    decision = _decision_for(result, "не знаю")
    assert decision["memory_decision"] == ""
    assert decision["memory_applied"] is False
    assert decision["memory_reason"] == ""
    assert decision["selected_by_memory"] is False
    assert decision["memory_key"] == ""


def test_correction_result_exposes_candidate_decisions_and_defaults_to_empty_list():
    default_result = CorrectionResult("source", "target", [])
    corrector = TrainedModelCorrector(FakeBackend({"не знаю": 0.99}), thresholds={"split_join_threshold": 0.9})

    result = corrector.correct("Я незнаю что делать")

    assert default_result.candidate_decisions == []
    assert result.candidate_decisions == corrector.last_candidate_decisions
    assert any(decision.get("replacement") == "не знаю" for decision in result.candidate_decisions)


def test_from_config_leaves_memory_disabled_by_default(monkeypatch, tmp_path):
    lexicon_path = tmp_path / "russian_lexicon.txt"
    lexicon_path.write_text("библиотека\n", encoding="utf-8")
    memory_path = tmp_path / "memory.jsonl"
    backend = FakeBackend({})
    monkeypatch.setattr(TorchCandidateModelBackend, "from_config", classmethod(lambda cls, config: backend))

    corrector = TrainedModelCorrector.from_config(
        {
            "dictionary": {
                "enabled": True,
                "lexicon_path": str(lexicon_path),
                "max_candidates": 2,
                "min_score": 85,
            },
            "thresholds": {"mode": "balanced", "balanced": {}},
            "correction_memory": {
                "enabled": False,
                "storage_path": str(memory_path),
                "context_window_chars": 48,
                "accepted_reuse": True,
                "rejected_suppress": True,
            },
        }
    )

    assert corrector.correction_memory is None
    assert not memory_path.exists()


def test_trained_model_corrector_applies_final_punctuation_only_when_scorer_accepts_candidate():
    low_confidence = TrainedModelCorrector(FakeBackend({"." : 0.4}), thresholds={"final_punctuation_threshold": 0.9})
    high_confidence = TrainedModelCorrector(FakeBackend({"." : 0.99}), thresholds={"final_punctuation_threshold": 0.9})

    low_result = low_confidence.correct("Проект готов")
    high_result = high_confidence.correct("Проект готов")

    assert low_result.corrected_text == "Проект готов"
    assert high_result.corrected_text == "Проект готов."
    assert any(
        edit.edit_type == "final_punctuation"
        and edit.status == "accepted"
        and edit.rule_id == "final_punctuation_default"
        for edit in high_result.edits
    )


def test_trained_model_corrector_applies_candidate_with_rule_threshold():
    corrector = TrainedModelCorrector(
        FakeBackend({"Жизнь": 0.80}),
        thresholds={"frequent_errors_threshold": 0.70, "spelling_threshold": 0.95},
    )

    result = corrector.correct("Жызнь прекрасна.")

    assert result.corrected_text == "Жизнь прекрасна."
    assert any(edit.rule_id and edit.source == "Жызнь" and edit.status == "accepted" for edit in result.edits)


def test_candidate_threshold_precedence_uses_rule_family_edit_type_and_default():
    exact = Candidate(
        "жызнь",
        "жизнь",
        "spelling",
        0,
        5,
        confidence=0.94,
        rule_id="pattern_жы_жи",
        mode="deterministic",
    )
    family = Candidate(
        "учится",
        "учиться",
        "spelling",
        0,
        6,
        confidence=0.9,
        requires_model=True,
        rule_id="tsya_soft_insert",
        mode="model_required",
        edit_domain="orthography",
        syntax_family="tsya_tsya_context",
        subtype="soft_insert",
        evidence="syntax/POS verb context evidence",
        implementation_group="syntax_tsya_context",
        metadata={"syntax_family": "tsya_tsya_context"},
    )
    spelling = Candidate("чясто", "часто", "spelling", 0, 5, confidence=0.94, mode="deterministic")
    edit_type = Candidate("что то", "что-то", "hyphen", 0, 6, confidence=0.95, mode="candidate_only")
    default = Candidate("ошыпка", "ошибка", "unknown_edit", 0, 6, confidence=0.5)

    assert not _select_candidates(
        [ModelCandidatePrediction(exact, score=0.89, confidence=0.89)],
        {"pattern_жы_жи_threshold": 0.90, "deterministic_threshold": 0.70, "spelling_threshold": 0.50},
    )
    assert not _select_candidates(
        [ModelCandidatePrediction(family, score=0.97, confidence=0.97)],
        {"tsya_threshold": 0.98, "model_required_threshold": 0.90, "spelling_threshold": 0.50},
    )
    assert _select_candidates(
        [ModelCandidatePrediction(spelling, score=0.71, confidence=0.71)],
        {"spelling_threshold": 0.70, "default_threshold": 0.95},
    )
    assert _select_candidates(
        [ModelCandidatePrediction(edit_type, score=0.86, confidence=0.86)],
        {"hyphen_threshold": 0.85, "candidate_only_threshold": 0.90},
    )
    assert not _select_candidates(
        [ModelCandidatePrediction(default, score=0.84, confidence=0.84)],
        {"default_threshold": 0.85},
    )


def test_model_selection_treats_zero_length_punctuation_at_same_gap_as_conflicting():
    comma = Candidate(
        "",
        ",",
        "punctuation_insert",
        7,
        7,
        confidence=0.9,
        requires_model=True,
        rule_id="comma_subordinate",
        mode="model_required",
    )
    colon = Candidate(
        "",
        ":",
        "punctuation_insert",
        7,
        7,
        confidence=0.9,
        requires_model=True,
        rule_id="enumeration_colon",
        mode="model_required",
    )

    selected = _select_candidates(
        [
            ModelCandidatePrediction(comma, score=0.99, confidence=0.99),
            ModelCandidatePrediction(colon, score=0.98, confidence=0.98),
        ],
        {"punctuation_insert_threshold": 0.9, "model_required_threshold": 0.9},
    )

    assert len(selected) == 1
    assert selected[0].replacement == ","
    assert selected[0].rule_id == "comma_subordinate"


def test_model_selection_preserves_candidate_metadata():
    candidate = Candidate(
        "учится",
        "учиться",
        "spelling",
        3,
        9,
        confidence=0.9,
        requires_model=True,
        rule_id="tsya_soft_insert",
        mode="model_required",
        edit_domain="orthography",
        syntax_family="tsya_tsya_context",
        subtype="soft_insert",
        evidence="syntax/POS verb context evidence",
        implementation_group="syntax_tsya_context",
        metadata={"syntax_family": "tsya_tsya_context"},
    )

    selected = _select_candidates(
        [ModelCandidatePrediction(candidate, score=0.99, confidence=0.98)],
        {"tsya_threshold": 0.9},
    )

    assert len(selected) == 1
    assert selected[0].rule_id == "tsya_soft_insert"
    assert selected[0].mode == "model_required"
    assert selected[0].requires_model is True
    assert selected[0].requires_scoring is True
    assert selected[0].edit_domain == "orthography"
    assert selected[0].syntax_family == "tsya_tsya_context"
    assert selected[0].subtype == "soft_insert"
    assert selected[0].evidence == "syntax/POS verb context evidence"
    assert selected[0].implementation_group == "syntax_tsya_context"
    assert selected[0].metadata == {"syntax_family": "tsya_tsya_context"}


def test_trained_model_corrector_preserves_accepted_punctuation_rule_id():
    corrector = TrainedModelCorrector(
        FakeBackend(
            {},
            punctuation=[ModelPunctuationPrediction(1, "COMMA", 0.99, action="INSERT", rule_id="comma_subordinate")],
        ),
        thresholds={"punctuation_threshold": 0.9},
    )

    result = corrector.correct("Я думаю что важно.")

    assert result.corrected_text == "Я думаю, что важно."
    assert any(
        edit.edit_type == "punctuation_insert"
        and edit.status == "accepted"
        and edit.rule_id == "comma_subordinate"
        for edit in result.edits
    )


def test_trained_model_corrector_ignores_free_punctuation_prediction_without_candidate():
    corrector = TrainedModelCorrector(
        FakeBackend(
            {},
            punctuation=[ModelPunctuationPrediction(0, "DASH", 0.99999, action="INSERT")],
        ),
        thresholds={"punctuation_threshold": 0.9, "dash_threshold": 0.9},
    )

    result = corrector.correct("Получается это решение подходит группе альфа.")

    assert result.corrected_text == "Получается это решение подходит группе альфа."
    assert not any(edit.edit_type == "punctuation_insert" and edit.replacement == "—" for edit in result.edits)


def test_trained_model_corrector_rejects_context_dependent_pair_without_strict_context():
    corrector = TrainedModelCorrector(FakeBackend({"так же": 0.99}), thresholds={"split_join_threshold": 0.9})

    result = corrector.correct("Он также пришел.")

    assert result.corrected_text == "Он также пришел."
    assert any(edit.source.lower() == "также" and edit.status == "rejected" for edit in result.edits)


def test_trained_model_corrector_can_apply_context_dependent_pair_with_high_confidence_and_strict_context():
    corrector = TrainedModelCorrector(
        FakeBackend({"так же": 0.99}),
        thresholds={"split_join_threshold": 0.9, "context_pair_threshold": 0.98},
    )

    result = corrector.correct("Он сделал также как я.")

    assert result.corrected_text == "Он сделал так же как я."
    assert any(edit.source.lower() == "также" and edit.status == "accepted" for edit in result.edits)


def test_trained_model_corrector_keeps_existing_punctuation_when_model_predicts_none_for_gap():
    corrector = TrainedModelCorrector(
        FakeBackend({}, punctuation=[ModelPunctuationPrediction(1, "NONE", 0.99)]),
        thresholds={"punctuation_threshold": 0.9},
    )

    result = corrector.correct("Я думаю, это важно.")

    assert result.corrected_text == "Я думаю, это важно."
    assert not any(edit.edit_type == "punctuation_delete" and edit.status == "accepted" for edit in result.edits)


def test_trained_model_corrector_deletes_existing_punctuation_only_with_delete_action_and_threshold():
    low_confidence = TrainedModelCorrector(
        FakeBackend({}, punctuation=[ModelPunctuationPrediction(1, "NONE", 0.91, action="DELETE")]),
        thresholds={"punctuation_threshold": 0.5, "punctuation_delete_threshold": 0.95},
    )
    high_confidence = TrainedModelCorrector(
        FakeBackend({}, punctuation=[ModelPunctuationPrediction(1, "NONE", 0.97, action="DELETE")]),
        thresholds={"punctuation_threshold": 0.5, "punctuation_delete_threshold": 0.95},
    )

    assert low_confidence.correct("Я думаю,, что это важно.").corrected_text == "Я думаю,, что это важно."
    result = high_confidence.correct("Я думаю,, что это важно.")

    assert result.corrected_text == "Я думаю, что это важно."
    assert any(edit.edit_type == "punctuation_delete" and edit.status == "accepted" for edit in result.edits)


def test_trained_model_corrector_ignores_unsupported_free_punctuation_replace():
    corrector = TrainedModelCorrector(
        FakeBackend({}, punctuation=[ModelPunctuationPrediction(1, "COLON", 0.99, action="REPLACE")]),
        thresholds={"punctuation_threshold": 0.9},
    )

    result = corrector.correct("Он сказал, привет.")

    assert result.corrected_text == "Он сказал, привет."
    assert not any(edit.edit_type == "punctuation_replace" and edit.status == "accepted" for edit in result.edits)


def test_trained_model_corrector_does_not_insert_sentence_final_mark_inside_sentence():
    corrector = TrainedModelCorrector(
        FakeBackend({}, punctuation=[ModelPunctuationPrediction(2, "DOT", 0.99, action="INSERT")]),
        thresholds={"punctuation_threshold": 0.9},
    )

    result = corrector.correct("Он предназначен для роботов поисковых систем.")

    assert result.corrected_text == "Он предназначен для роботов поисковых систем."


def test_trained_model_corrector_inserts_dash_with_spacing():
    corrector = TrainedModelCorrector(
        FakeBackend({}, punctuation=[ModelPunctuationPrediction(0, "DASH", 0.99, action="INSERT")]),
        thresholds={"punctuation_threshold": 0.9},
    )

    result = corrector.correct("Москва это столица.")

    assert result.corrected_text == "Москва — это столица."


def test_trained_model_corrector_supports_candidate_backed_direct_speech_quotes_only():
    corrector = TrainedModelCorrector(
        FakeBackend(
            {},
            punctuation=[
                ModelPunctuationPrediction(1, "COLON", 0.99, action="INSERT"),
                ModelPunctuationPrediction(1, "QUOTE_OPEN", 0.99),
                ModelPunctuationPrediction(3, "QUOTE_CLOSE", 0.99),
                ModelPunctuationPrediction(3, "BRACKET_OPEN", 0.99),
                ModelPunctuationPrediction(4, "BRACKET_CLOSE", 0.99),
            ],
        ),
        thresholds={"punctuation_threshold": 0.9},
    )

    result = corrector.correct("Он сказал проект готов.")

    assert result.corrected_text == "Он сказал: «проект готов»."
    assert any(edit.rule_id == "direct_speech_colon" and edit.status == "accepted" for edit in result.edits)
    assert any(edit.rule_id == "direct_speech_quotes" and edit.status == "accepted" for edit in result.edits)
    assert not any(edit.replacement in {"(", ")"} and edit.status == "accepted" for edit in result.edits)


def test_direct_speech_dash_after_closing_quote():
    corrector = TrainedModelCorrector(
        FakeBackend(
            {},
            punctuation=[
                ModelPunctuationPrediction(1, "DASH", 0.99, action="INSERT"),
            ],
        ),
        thresholds={"dash_threshold": 0.9},
    )

    result = corrector.correct("«Команда справилась» сказала Мария.")

    assert result.corrected_text == "«Команда справилась» — сказала Мария."
    assert result.corrected_text != "«Команда справилась —» сказала Мария."
    assert any(edit.rule_id == "direct_speech_dash" and edit.status == "accepted" for edit in result.edits)


def test_trained_model_corrector_uses_label_specific_punctuation_thresholds():
    corrector = TrainedModelCorrector(
        FakeBackend(
            {},
            punctuation=[
                ModelPunctuationPrediction(1, "COMMA", 0.86, action="INSERT"),
                ModelPunctuationPrediction(1, "COLON", 0.86, action="INSERT"),
            ],
        ),
        thresholds={"punctuation_threshold": 0.5, "comma_threshold": 0.9, "colon_threshold": 0.8},
    )

    result = corrector.correct("Он сказал привет дальше.")

    assert result.corrected_text == "Он сказал: привет дальше."


def test_trained_model_corrector_uses_rule_specific_punctuation_threshold_before_label():
    corrector = TrainedModelCorrector(
        FakeBackend(
            {},
            punctuation=[ModelPunctuationPrediction(1, "COMMA", 0.89, action="INSERT", rule_id="comma_subordinate")],
        ),
        thresholds={"punctuation_threshold": 0.5, "comma_threshold": 0.95, "comma_subordinate_threshold": 0.88},
    )

    result = corrector.correct("Я думаю что важно.")

    assert result.corrected_text == "Я думаю, что важно."


def test_punctuation_rejected_memory_suppresses_comma_insertion_even_when_confident():
    text = "Я думаю что важно."
    memory = CorrectionMemory(storage_path=None)
    corrector = TrainedModelCorrector(
        FakeBackend(
            {},
            punctuation=[ModelPunctuationPrediction(1, "COMMA", 0.99, action="INSERT", rule_id="comma_subordinate")],
        ),
        thresholds={"punctuation_threshold": 0.9},
        correction_memory=memory,
        doc_id="doc-1",
    )
    memory.remember_candidate(
        text,
        _generated_punctuation_candidate(corrector, text, rule_id="comma_subordinate", label="COMMA", action="INSERT"),
        "rejected",
        doc_id="doc-1",
    )

    result = corrector.correct(text)

    assert result.corrected_text == text
    decision = _punctuation_decision_for(result, rule_id="comma_subordinate", replacement=",")
    assert decision["rule_id"] == "comma_subordinate"
    assert decision["memory_decision"] == "rejected"
    assert decision["memory_applied"] is True
    assert decision["memory_reason"] == "exact_context_key"
    assert decision["selected"] is False


def test_punctuation_accepted_memory_allows_comma_insertion_below_threshold():
    text = "Я думаю что важно."
    memory = CorrectionMemory(storage_path=None)
    corrector = TrainedModelCorrector(
        FakeBackend(
            {},
            punctuation=[ModelPunctuationPrediction(1, "COMMA", 0.85, action="INSERT", rule_id="comma_subordinate")],
        ),
        thresholds={"punctuation_threshold": 0.9},
        correction_memory=memory,
        doc_id="doc-1",
    )
    memory.remember_candidate(
        text,
        _generated_punctuation_candidate(corrector, text, rule_id="comma_subordinate", label="COMMA", action="INSERT"),
        "accepted",
        doc_id="doc-1",
    )

    result = corrector.correct(text)

    assert result.corrected_text == "Я думаю, что важно."
    decision = _punctuation_decision_for(result, rule_id="comma_subordinate", replacement=",")
    assert decision["rule_id"] == "comma_subordinate"
    assert decision["memory_decision"] == "accepted"
    assert decision["memory_applied"] is True
    assert decision["selected_by_memory"] is True
    assert decision["threshold_passed"] is False
    assert decision["selected"] is True


def test_punctuation_accepted_memory_still_goes_through_strict_validator():
    text = "Он понял одно проект готов."
    memory = CorrectionMemory(storage_path=None)
    corrector = TrainedModelCorrector(
        FakeBackend(
            {},
            punctuation=[ModelPunctuationPrediction(2, "COLON", 0.85, action="INSERT", rule_id="explanation_colon")],
        ),
        thresholds={"punctuation_threshold": 0.9},
        correction_memory=memory,
        doc_id="doc-1",
    )
    memory.remember_candidate(
        text,
        _generated_punctuation_candidate(corrector, text, rule_id="explanation_colon", label="COLON", action="INSERT"),
        "accepted",
        doc_id="doc-1",
    )

    result = corrector.correct(text)

    assert result.corrected_text == text
    assert any(edit.rule_id == "explanation_colon" and edit.status == "rejected" for edit in result.edits)
    decision = _punctuation_decision_for(result, rule_id="explanation_colon", replacement=":")
    assert decision["rule_id"] == "explanation_colon"
    assert decision["selected_by_memory"] is True
    assert decision["validator_status"] == "rejected"
    assert decision["validator_reason"] == "unsafe_colon_candidate"


def test_trained_model_corrector_does_not_select_low_score_tsya_candidate():
    corrector = TrainedModelCorrector(
        FakeBackend({"появиться": 0.4}),
        thresholds={"spelling_threshold": 0.9, "tsya_threshold": 0.98},
    )

    result = corrector.correct("Они могут появится завтра.")

    assert result.corrected_text == "Они могут появится завтра."
    assert not any(edit.source.lower() == "появится" and edit.replacement.lower() == "появиться" for edit in result.edits)


def test_trained_model_corrector_rejects_high_confidence_dangerous_tsya_prediction():
    corrector = TrainedModelCorrector(
        FakeBackend({"появится": 0.99}),
        thresholds={"spelling_threshold": 0.5, "tsya_threshold": 0.98},
    )

    result = corrector.correct("Они могут появиться завтра.")

    assert result.corrected_text == "Они могут появиться завтра."
    assert any(edit.source == "появиться" and edit.status == "rejected" and edit.reason == "tsya_unsafe" for edit in result.edits)


def test_trained_model_corrector_accepts_high_confidence_useful_tsya_prediction():
    corrector = TrainedModelCorrector(
        FakeBackend({"появиться": 0.99}),
        thresholds={"spelling_threshold": 0.5, "tsya_threshold": 0.98},
    )

    result = corrector.correct("Они могут появится завтра.")

    assert result.corrected_text == "Они могут появиться завтра."
    assert any(edit.source == "появится" and edit.status == "accepted" and edit.rule_id == "tsya_soft_insert" for edit in result.edits)


def test_trained_model_corrector_does_not_append_dot_after_ellipsis():
    corrector = TrainedModelCorrector(FakeBackend({}))

    result = corrector.correct("Мы ждали файл…")

    assert result.corrected_text == "Мы ждали файл…"


@pytest.mark.parametrize(
    "source",
    [
        "Я очень рада, что вам мои посты нравятся :)",
        "Слезяться глаза и плачет дождь,",
    ],
)
def test_trained_model_corrector_does_not_append_final_dot_after_unsafe_tail(source):
    corrector = TrainedModelCorrector(FakeBackend({".": 0.999}), thresholds={"final_punctuation_threshold": 0.5})

    result = corrector.correct(source)

    assert result.corrected_text == source
    assert not any(edit.edit_type == "final_punctuation" and edit.status == "accepted" for edit in result.edits)


def test_torch_backend_passes_candidate_replacement_tokens_to_model():
    tokenizer = FakeTokenizer()
    module = CapturingModule(max_candidates=3)
    backend = TorchCandidateModelBackend(
        tokenizer=tokenizer,
        module=module,
        device=torch.device("cpu"),
        punctuation_labels={},
        max_length=8,
        max_candidates=3,
    )
    candidates = [
        Candidate("незнаю", "незнаю", "keep", 2, 8),
        Candidate("незнаю", "не знаю", "split_join", 2, 8),
    ]

    backend.score_candidates("Я незнаю", candidates)

    replacement_ids = module.last_kwargs["candidate_replacement_ids"]
    replacement_mask = module.last_kwargs["candidate_replacement_mask"]
    assert replacement_ids.shape[0:2] == torch.Size([1, 3])
    assert replacement_mask[0, 0].any()
    assert replacement_ids[0, 0].sum().item() > 0


def test_torch_backend_candidate_budget_prefers_edit_over_keep():
    tokenizer = FakeTokenizer()
    module = CapturingModule(max_candidates=1)
    backend = TorchCandidateModelBackend(
        tokenizer=tokenizer,
        module=module,
        device=torch.device("cpu"),
        punctuation_labels={},
        max_length=32,
        max_candidates=1,
    )
    candidates = [
        Candidate("слово", "слово", "keep", 0, 5),
        Candidate("недумаю", "не думаю", "split_join", 6, 13, confidence=0.97),
    ]

    predictions = backend.score_candidates("слово недумаю", candidates)

    assert [prediction.candidate.replacement for prediction in predictions] == ["не думаю"]


def test_torch_backend_passes_word_gap_indices_to_model_for_punctuation():
    tokenizer = FakeTokenizer()
    module = CapturingModule(max_candidates=1, punctuation_label_count=3)
    backend = TorchCandidateModelBackend(
        tokenizer=tokenizer,
        module=module,
        device=torch.device("cpu"),
        punctuation_labels={"NONE": 0, "COMMA": 1, "DOT": 2},
        punctuation_action_labels={"KEEP_NONE": 0, "KEEP_EXISTING": 1, "INSERT": 2, "DELETE": 3, "REPLACE": 4},
        max_length=8,
        max_candidates=1,
    )

    backend.predict_punctuation("Я думаю, что")

    gap_indices = module.last_kwargs["punctuation_gap_indices"]
    right_gap_indices = module.last_kwargs["punctuation_right_gap_indices"]
    assert gap_indices.shape == torch.Size([1, 8])
    assert right_gap_indices.shape == torch.Size([1, 8])
    assert gap_indices.tolist()[0][:3] == [0, 5, 7]
    assert right_gap_indices.tolist()[0][:3] == [1, 7, 7]


def test_torch_backend_uses_punctuation_confidence_head_for_prediction_confidence():
    tokenizer = FakeTokenizer()
    module = CapturingModule(max_candidates=1, punctuation_label_count=3, punctuation_confidence_logit=2.0)
    backend = TorchCandidateModelBackend(
        tokenizer=tokenizer,
        module=module,
        device=torch.device("cpu"),
        punctuation_labels={"NONE": 0, "COMMA": 1, "DOT": 2},
        punctuation_action_labels={"KEEP_NONE": 0, "KEEP_EXISTING": 1, "INSERT": 2, "DELETE": 3, "REPLACE": 4},
        max_length=8,
        max_candidates=1,
    )

    predictions = backend.predict_punctuation("Я думаю что")

    assert predictions
    assert predictions[0].confidence == pytest.approx(torch.sigmoid(torch.tensor(2.0)).item())
    assert predictions[0].action in {"KEEP_NONE", "KEEP_EXISTING", "INSERT", "DELETE", "REPLACE"}


def test_torch_backend_scores_training_features_in_batches():
    feature = build_training_feature(
        "Я незнаю что делать",
        "Я не знаю, что делать.",
        tokenizer=DebugTokenizer(),
        punctuation_label_map={"NONE": 0, "COMMA": 1, "DOT": 2},
        punctuation_action_label_map={"KEEP_NONE": 0, "KEEP_EXISTING": 1, "INSERT": 2, "DELETE": 3, "REPLACE": 4},
        error_type_label_map={"keep": 0, "split_join": 1, "punctuation": 2, "final_punctuation": 3},
        max_length=12,
        max_candidates=4,
    )
    module = CapturingModule(max_candidates=4, punctuation_label_count=3)
    backend = TorchCandidateModelBackend(
        tokenizer=DebugTokenizer(),
        module=module,
        device=torch.device("cpu"),
        punctuation_labels={"NONE": 0, "COMMA": 1, "DOT": 2},
        punctuation_action_labels={"KEEP_NONE": 0, "KEEP_EXISTING": 1, "INSERT": 2, "DELETE": 3, "REPLACE": 4},
        max_length=12,
        max_candidates=4,
    )

    scores, forward_time = backend.score_features_batched([feature, feature], batch_size=2, mixed_precision=True)

    assert len(scores) == 2
    assert all(isinstance(item, BatchedFeatureModelScores) for item in scores)
    assert scores[0].candidate_scores == [0.5, 0.5, 0.5, 0.5]
    assert len(scores[0].punctuation_label_ids) == 12
    assert module.last_kwargs["input_ids"].shape == torch.Size([2, 12])
    assert forward_time >= 0.0


def test_legacy_candidate_projection_heads_are_expanded_for_current_model_shape():
    heads = torch.nn.ModuleDict(build_linear_heads(hidden_size=4, punctuation_labels=3, error_types=2))
    legacy_state = heads.state_dict()
    legacy_weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    legacy_state["candidate_projection.weight"] = legacy_weight

    prepared = _prepare_heads_state_dict_for_module(legacy_state, heads)

    assert prepared["candidate_projection.weight"].shape == torch.Size([4, 12])
    assert torch.equal(prepared["candidate_projection.weight"][:, :4], legacy_weight)
    assert torch.equal(prepared["candidate_projection.weight"][:, 4:], torch.zeros(4, 8))
    heads.load_state_dict(prepared)


class FakeTokenizer:
    def __call__(
        self,
        text,
        *,
        return_offsets_mapping=True,
        truncation=True,
        padding="max_length",
        max_length=8,
        add_special_tokens=True,
    ):
        tokens = [(char, index, index + 1) for index, char in enumerate(text) if not char.isspace()]
        tokens = tokens[:max_length]
        input_ids = [ord(char) % 97 + 2 for char, _start, _end in tokens]
        attention_mask = [1] * len(input_ids)
        offsets = [(start, end) for _char, start, end in tokens]
        pad = max_length - len(input_ids)
        input_ids += [0] * pad
        attention_mask += [0] * pad
        offsets += [(0, 0)] * pad
        result = {"input_ids": input_ids, "attention_mask": attention_mask}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


class CapturingModule:
    def __init__(
        self,
        max_candidates: int,
        punctuation_label_count: int = 0,
        punctuation_action_count: int = 5,
        punctuation_confidence_logit: float = 0.0,
    ):
        self.max_candidates = max_candidates
        self.punctuation_label_count = punctuation_label_count
        self.punctuation_action_count = punctuation_action_count
        self.punctuation_confidence_logit = punctuation_confidence_logit
        self.last_kwargs = None

    def __call__(self, **kwargs):
        self.last_kwargs = kwargs
        batch_size = kwargs.get("input_ids", torch.zeros(1, 0)).shape[0]
        gap_count = kwargs.get("punctuation_gap_indices", torch.zeros(1, 0)).shape[1]
        return {
            "candidate_scores": torch.zeros(batch_size, self.max_candidates),
            "confidence_logits": torch.zeros(batch_size, self.max_candidates),
            "punctuation_logits": torch.zeros(batch_size, gap_count, self.punctuation_label_count),
            "punctuation_action_logits": torch.zeros(batch_size, gap_count, self.punctuation_action_count),
            "punctuation_confidence_logits": torch.full((batch_size, gap_count), self.punctuation_confidence_logit),
        }
