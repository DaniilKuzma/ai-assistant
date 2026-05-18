import re

from pymorphy3 import MorphAnalyzer

from src.candidates.candidate_generator import CandidateGenerator
from src.inference.corrector import Corrector
from src.inference.model_corrector import ModelCandidatePrediction, TrainedModelCorrector


MORPH = MorphAnalyzer()
WORD_RE = re.compile(r"[а-я]+")


class ScoringBackend:
    def __init__(self, accepted_replacements: set[str]):
        self.accepted_replacements = accepted_replacements

    def score_candidates(self, text, candidates):
        return [
            ModelCandidatePrediction(
                candidate=candidate,
                score=0.99 if candidate.replacement.lower() in self.accepted_replacements else 0.0,
                confidence=0.99 if candidate.replacement.lower() in self.accepted_replacements else 0.0,
            )
            for candidate in candidates
        ]

    def predict_punctuation(self, text):
        return []


def test_candidate_generator_emits_ne_plus_verb_split_candidates_for_hundreds_of_forms():
    generator = CandidateGenerator()
    forms = _known_forms(
        [
            "думать",
            "знать",
            "работать",
            "понимать",
            "хотеть",
            "делать",
            "читать",
            "писать",
            "говорить",
            "видеть",
            "слышать",
            "помнить",
            "любить",
            "играть",
            "спать",
            "идти",
            "ехать",
            "решать",
            "смотреть",
            "отвечать",
        ],
        poses={"VERB", "INFN"},
        limit=240,
    )
    cases = [(f"не{form}", f"не {form}") for form in forms]

    found = _candidate_pairs_for_cases(generator, cases)

    assert len(cases) >= 200
    assert set(cases).issubset(found)


def test_rule_registry_has_unique_ids():
    from src.rules.registry import all_rules

    ids = [rule.spec.id for rule in all_rules()]

    assert len(ids) == len(set(ids))
    assert "ne_verb" in ids
    assert "tsya_soft_insert" in ids


def test_tsya_rules_are_model_required_bidirectional_rules():
    from src.rules.registry import rule_by_id

    insert_rule = rule_by_id("tsya_soft_insert")
    delete_rule = rule_by_id("tsya_soft_delete")

    assert insert_rule.spec.mode == "model_required"
    assert delete_rule.spec.mode == "model_required"
    assert any(candidate.replacement == "учиться" and candidate.requires_model for candidate in insert_rule.generate_candidates("учится"))
    assert any(candidate.replacement == "учится" and candidate.requires_model for candidate in delete_rule.generate_candidates("учиться"))


def test_candidate_generator_does_not_split_known_ne_exceptions():
    generator = CandidateGenerator()
    candidates = generator.generate("Я ненавижу шум, недоумеваю и недооценил риск.")

    replacements = {candidate.replacement.lower() for candidate in candidates}

    assert "не навижу" not in replacements
    assert "не доумеваю" not in replacements
    assert "не дооценил" not in replacements


def test_candidate_generator_emits_spelling_pattern_candidates_for_hundreds_of_forms():
    generator = CandidateGenerator()
    buckets = {
        "жи/ши/ча/ща/чу/щу": _corrupt_forms_with_replacements(
            _combo_forms(),
            {"жи": "жы", "ши": "шы", "ча": "чя", "ща": "щя", "чу": "чю", "щу": "щю"},
            limit=220,
        ),
        "ь/ъ": _hard_sign_cases(limit=200),
        "з/с prefixes": _prefix_z_s_cases(limit=220),
        "и/ы после ц": _corrupt_forms_with_replacements(_ci_forms(), {"ци": "цы"}, limit=200),
        "о/е после шипящих": _corrupt_forms_with_replacements(
            _hissing_o_e_forms(),
            {"же": "жо", "ше": "шо", "че": "чо", "ще": "що"},
            limit=200,
        ),
    }
    cases = [case for bucket_cases in buckets.values() for case in bucket_cases]

    found = _candidate_pairs_for_cases(generator, cases)

    assert all(len(bucket_cases) >= 200 for bucket_cases in buckets.values())
    assert len(cases) >= 1000
    assert set(cases).issubset(found)


def test_candidate_generator_emits_tsya_candidates_as_model_scored_when_context_is_needed():
    generator = CandidateGenerator()
    candidates = generator.generate("Он учится и хочет учиться.")

    values = {
        (
            candidate.source.lower(),
            candidate.replacement.lower(),
            candidate.edit_type,
            candidate.mode,
            candidate.requires_model,
            candidate.requires_scoring,
        )
        for candidate in candidates
    }

    assert ("учится", "учиться", "spelling", "model_required", True, True) in values
    assert ("учиться", "учится", "spelling", "model_required", True, True) in values


def test_plain_corrector_does_not_apply_candidate_only_ne_verb_without_scorer():
    result = Corrector().correct("Я недумаю об этом.")

    assert result.corrected_text == "Я недумаю об этом."
    assert not any(edit.source.lower() == "недумаю" and edit.replacement.lower() == "не думаю" for edit in result.edits)


def test_model_corrector_can_apply_generated_ne_verb_candidate_after_scoring():
    corrector = TrainedModelCorrector(
        ScoringBackend({"не думаю"}),
        thresholds={"split_join_threshold": 0.9, "spelling_threshold": 0.9},
    )

    result = corrector.correct("Я недумаю об этом.")

    assert result.corrected_text == "Я не думаю об этом."
    assert any(edit.source.lower() == "недумаю" and edit.status == "accepted" for edit in result.edits)


def _candidate_pairs_for_cases(generator: CandidateGenerator, cases: list[tuple[str, str]]) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for start in range(0, len(cases), 40):
        chunk = cases[start : start + 40]
        text = " ".join(wrong for wrong, _correct in chunk)
        found.update(
            (candidate.source.lower(), candidate.replacement.lower())
            for candidate in generator.generate(text)
            if candidate.edit_type != "keep"
        )
    return found


def _known_forms(lemmas: list[str], poses: set[str] | None = None, limit: int = 300) -> list[str]:
    forms: list[str] = []
    seen: set[str] = set()
    for lemma in lemmas:
        for parsed in MORPH.parse(lemma)[0].lexeme:
            word = parsed.word.replace("ё", "е")
            if word in seen or not WORD_RE.fullmatch(word):
                continue
            if poses and parsed.tag.POS not in poses:
                continue
            seen.add(word)
            forms.append(word)
            if len(forms) >= limit:
                return forms
    return forms


def _combo_forms() -> list[str]:
    return _known_forms(
        [
            "жизнь",
            "живой",
            "животное",
            "широкий",
            "ширина",
            "машина",
            "частый",
            "часто",
            "защита",
            "чаща",
            "чудо",
            "чувство",
            "щука",
            "искать",
            "писать",
            "держать",
            "сказать",
            "хотеть",
            "молчать",
            "тащить",
        ],
        limit=500,
    )


def _ci_forms() -> list[str]:
    return _known_forms(
        [
            "цифра",
            "цирк",
            "цитата",
            "цивилизация",
            "цикл",
            "циркуль",
            "цистерна",
            "цилиндр",
            "циничный",
            "цинга",
            "циновка",
            "цифровой",
            "медицина",
            "акация",
            "станция",
            "операция",
            "лекция",
            "традиция",
            "полиция",
            "нация",
        ],
        limit=500,
    )


def _hissing_o_e_forms() -> list[str]:
    return _known_forms(
        [
            "шел",
            "пришел",
            "нашел",
            "желтый",
            "черный",
            "дешевый",
            "печеный",
            "тушеный",
            "сгущенный",
            "жесткий",
            "шелковый",
        ],
        limit=500,
    )


def _hard_sign_cases(limit: int) -> list[tuple[str, str]]:
    correct_forms = _known_forms(
        [
            "подъезд",
            "объект",
            "объявление",
            "съезд",
            "въезд",
            "изъян",
            "разъяснение",
            "объяснение",
            "предъявление",
            "съемка",
            "адъютант",
            "конъюнктура",
            "субъект",
            "инъекция",
            "объятие",
            "объединение",
        ],
        limit=400,
    )
    cases = []
    for correct in correct_forms:
        if "ъ" not in correct:
            continue
        cases.append((correct.replace("ъ", ""), correct))
        cases.append((correct.replace("ъ", "ь"), correct))
        if len(cases) >= limit:
            return cases
    return cases


def _prefix_z_s_cases(limit: int) -> list[tuple[str, str]]:
    correct_forms = _known_forms(
        [
            "бесполезный",
            "бесплатный",
            "беспокойный",
            "бесконечный",
            "бесшумный",
            "безвкусный",
            "безграмотный",
            "бездарный",
            "безбрежный",
            "разбить",
            "рассказать",
            "расписать",
            "исписать",
            "избить",
            "воспитать",
            "возвратить",
            "вспомнить",
            "взбить",
        ],
        limit=700,
    )
    cases = []
    for correct in correct_forms:
        wrong = _toggle_prefix_z_s(correct)
        if wrong and wrong != correct:
            cases.append((wrong, correct))
            if len(cases) >= limit:
                return cases
    return cases


def _toggle_prefix_z_s(word: str) -> str | None:
    for z_prefix, s_prefix in [
        ("без", "бес"),
        ("раз", "рас"),
        ("из", "ис"),
        ("воз", "вос"),
        ("вз", "вс"),
    ]:
        if word.startswith(z_prefix):
            return s_prefix + word[len(z_prefix) :]
        if word.startswith(s_prefix):
            return z_prefix + word[len(s_prefix) :]
    return None


def _corrupt_forms_with_replacements(
    correct_forms: list[str],
    correct_to_wrong: dict[str, str],
    limit: int,
) -> list[tuple[str, str]]:
    cases = []
    for correct in correct_forms:
        for correct_part, wrong_part in correct_to_wrong.items():
            if correct_part not in correct:
                continue
            wrong = correct.replace(correct_part, wrong_part, 1)
            cases.append((wrong, correct))
            break
        if len(cases) >= limit:
            return cases
    return cases
