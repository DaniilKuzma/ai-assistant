from collections import Counter
import math
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from candidate_generator import CandidateGenerator
from context_reranker import ContextReranker, ContextScore
from data_preparation import DatasetGenerator, safe_normalize_raw_spacing
from edit_labels import build_training_example
from entity_guard import ProtectedLexicon
from hybrid_corrector import CorrectionResult, HybridCorrector, RuntimeCandidate
from hybrid_preprocessor import HybridPreprocessor
from morphology_guard import is_morphological_dictionary_word, is_same_lemma_inflection
from quality_guard import QualityGuard
from text_utils import (
    choose_punctuation_label,
    deterministic_spacing_correction,
    extract_word_slots,
    is_protected_punctuation_gap,
    rebuild_preserving_layout,
)
from training_augmentation import augment_keep_candidates, populate_top_k_candidates


class ConservativeRuntimeTests(unittest.TestCase):
    def test_rebuild_preserves_structural_punctuation(self):
        original = "Российская Intel — «Инел А/О» № 143."
        slots = extract_word_slots(original)
        words = [slot.word for slot in slots]
        words[2] = "Интел"

        rebuilt = rebuild_preserving_layout(
            original,
            slots,
            words,
            [choose_punctuation_label(slot.punct_after) for slot in slots],
            punctuation_mode="conservative",
        )

        self.assertIn("Intel — «Интел А/О» № 143.", rebuilt)
        self.assertIn("—", rebuilt)
        self.assertIn("«", rebuilt)
        self.assertIn("»", rebuilt)
        self.assertIn("/", rebuilt)
        self.assertIn("№", rebuilt)

    def test_candidate_generator_is_conservative(self):
        generator = CandidateGenerator.from_texts(
            ["Интел выпустила процессор.", "Также это важно.", "Это большое дело."],
            min_freq=1,
            max_distance=1,
        )

        intel_candidates = [c.text.lower() for c in generator.get_candidates("Инел")]
        split_candidates = [c.text.lower() for c in generator.get_candidates("говориться")]
        short_candidates = [c.text.lower() for c in generator.get_candidates("гель")]

        self.assertIn("интел", intel_candidates)
        self.assertNotIn("говорить ся", split_candidates)
        self.assertNotIn("дело", short_candidates)

    def test_service_token_split_candidates_beat_dictionary_deletion(self):
        generator = CandidateGenerator.from_texts(
            [
                "Об этом сообщается на странице.",
                "Пока Microsoft не заставляет потребителей.",
                "Чтобы избежать их поражения.",
                "которые не только могут помочь.",
                "От проекта отказались.",
                "Избавиться от засора можно быстро.",
                "На прошлой неделе цены выросли.",
                "По словам мужчины, все спокойно.",
            ],
            min_freq=1,
            max_distance=1,
            allow_split_candidates=True,
        )

        cases = {
            "сообщаетсяна": "сообщается на",
            "незаставляет": "не заставляет",
            "ихпоражения": "их поражения",
            "которыене": "которые не",
            "отпроекта": "от проекта",
            "Избавитьсяот": "Избавиться от",
            "Напрошлой": "На прошлой",
            "Пословам": "По словам",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                candidates = generator.get_candidates(source)
                self.assertGreater(len(candidates), 0)
                self.assertEqual(candidates[0].text, expected)
                self.assertEqual(candidates[0].source, "split")

    def test_deterministic_spacing_layer_fixes_safe_cases(self):
        cases = {
            "Привет ,мир": "Привет, мир",
            "Привет  мир": "Привет мир",
            "слово—слово": "слово — слово",
        }

        for source, expected in cases.items():
            with self.subTest(source=source):
                corrected, changed = deterministic_spacing_correction(source)
                self.assertTrue(changed)
                self.assertEqual(corrected, expected)

    def test_deterministic_spacing_layer_keeps_protected_fragments(self):
        unchanged = [
            "Сайт https://example.com,тест",
            "Почта test@example.com,тест",
            "Цена 6,30",
            "Дата 02.05.2026,тест",
            "Intel,Core",
            "№  143",
            "«Привет»  (мир)",
            "А/О  тест",
        ]

        for text in unchanged:
            with self.subTest(text=text):
                corrected, changed = deterministic_spacing_correction(text)
                self.assertFalse(changed)
                self.assertEqual(corrected, text)

    def test_runtime_reports_space_edit_only_for_spacing_fix(self):
        corrector = HybridCorrector(
            model_path="missing.keras",
            preprocessor_path="missing.pkl",
            candidate_generator_path="missing.pkl",
            context_reranker_enabled=False,
            deterministic_spacing=True,
        )

        result = corrector.correct("Привет ,мир", return_details=True)

        self.assertEqual(result.corrected, "Привет, мир")
        self.assertTrue(result.accepted)
        self.assertEqual([edit.action for edit in result.edits], ["SPACE"])
        self.assertEqual(result.edits[0].candidate_source, "deterministic_spacing")

        unchanged = corrector.correct("Привет, мир", return_details=True)
        self.assertEqual(unchanged.edits, [])

    def test_source_punctuation_is_vectorized(self):
        example = build_training_example(
            "Я думаю что это важно.",
            "Я думаю, что это важно.",
        )
        self.assertIsNotNone(example)
        self.assertIn("", example.source_punct_labels)
        self.assertIn(",", example.target_punct_labels)

        preprocessor = HybridPreprocessor(max_length=16, max_vocab_size=100)
        preprocessor.fit([example])
        inputs, outputs, weights = preprocessor.vectorize_examples([example])
        infer = preprocessor.vectorize_inference(
            example.source_words,
            example.candidate_words,
            example.source_punct_labels,
        )

        self.assertIn("source_punct_ids", inputs)
        self.assertIn("source_punct_ids", infer)
        self.assertEqual(inputs["source_punct_ids"].shape, (1, 16))
        self.assertEqual(inputs["candidate_ids"].shape, (1, 16, 8))
        self.assertEqual(outputs["punct"].shape, (1, 16))
        self.assertEqual(weights["punct"].shape, (1, 16))
        comma_index = example.target_punct_labels.index(",")
        self.assertEqual(weights["punct"][0, comma_index], 8.0)

        replacement = build_training_example("Инцедент произошел.", "Инцидент произошел.")
        self.assertIsNotNone(replacement)
        repl_inputs, repl_outputs, repl_weights = preprocessor.vectorize_examples([replacement])
        self.assertEqual(repl_inputs["candidate_ids"].shape, (1, 16, 8))
        self.assertEqual(repl_outputs["action"][0, 0], 2)
        self.assertEqual(repl_weights["action"][0, 0], 3.0)
        self.assertEqual(repl_weights["action"][0, 1], 1.0)

        clean_example = build_training_example("Это важно.", "Это важно.")
        self.assertIsNotNone(clean_example)
        preprocessor.fit([clean_example])
        _, _, clean_weights = preprocessor.vectorize_examples([clean_example])
        self.assertEqual(clean_weights["punct"][0, 0], 2.0)
        self.assertEqual(clean_weights["action"][0, 0], 2.0)

    def test_model_builds_with_source_punctuation_input(self):
        try:
            from hybrid_model import build_hybrid_model
        except ImportError as exc:
            self.skipTest(f"TensorFlow is unavailable in this interpreter: {exc}")

        model = build_hybrid_model(
            vocab_size=32,
            max_length=8,
            action_classes=3,
            punct_classes=8,
            d_model=16,
            num_heads=2,
            ff_dim=32,
            num_layers=1,
        )
        input_names = {inp.name.split(":")[0].split("/")[-1] for inp in model.inputs}

        self.assertIn("source_punct_ids", input_names)
        self.assertEqual(len(model.inputs), 3)
        candidate_shape = [inp.shape for inp in model.inputs if inp.name.split(":")[0].split("/")[-1] == "candidate_ids"][0]
        self.assertEqual(candidate_shape[-1], 8)

    def test_runtime_rejects_risky_dictionary_candidates(self):
        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.thresholds = HybridCorrector._thresholds("strict")
        corrector.min_dictionary_score = 1.0
        corrector.use_morphology_guard = True
        corrector.candidate_generator = CandidateGenerator.from_texts(
            ["Волос был рядом.", "интел работает."],
            min_freq=1,
            max_distance=1,
        )

        self.assertFalse(
            corrector._can_apply_candidate(
                "Волов",
                RuntimeCandidate("Волос", source="dictionary", distance=1, score=10.0),
                confidence=0.99,
                margin=0.50,
            )
        )

        self.assertFalse(
            corrector._can_apply_candidate(
                "интеа",
                RuntimeCandidate("интел", source="dictionary", distance=1, score=0.9),
                confidence=0.99,
                margin=0.50,
            )
        )
        self.assertTrue(
            corrector._can_apply_candidate(
                "интеа",
                RuntimeCandidate("интел", source="dictionary", distance=1, score=10.0),
                confidence=0.99,
                margin=0.50,
            )
        )

    def test_morphology_guard_blocks_inflection_swaps_only(self):
        self.assertTrue(is_same_lemma_inflection("летнем", "летним"))
        self.assertTrue(is_same_lemma_inflection("блокпосты", "блокпосту"))
        self.assertFalse(is_same_lemma_inflection("добисться", "добиться"))
        self.assertFalse(is_same_lemma_inflection("квртире", "квартире"))
        self.assertTrue(is_morphological_dictionary_word("перевез"))
        self.assertTrue(is_morphological_dictionary_word("цепью"))
        self.assertFalse(is_morphological_dictionary_word("инцедент"))
        self.assertFalse(is_morphological_dictionary_word("добисться"))

        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.thresholds = HybridCorrector._thresholds("strict")
        corrector.min_dictionary_score = 1.0
        corrector.use_morphology_guard = True
        corrector.candidate_generator = CandidateGenerator.from_texts(
            ["летним днем можно добиться результата."],
            min_freq=1,
            max_distance=1,
        )

        self.assertFalse(
            corrector._can_apply_candidate(
                "летнем",
                RuntimeCandidate("летним", source="dictionary", distance=1, score=10.0),
                confidence=0.99,
                margin=0.50,
            )
        )
        self.assertTrue(
            corrector._can_apply_candidate(
                "добисться",
                RuntimeCandidate("добиться", source="dictionary", distance=1, score=10.0),
                confidence=0.99,
                margin=0.50,
            )
        )

    def test_dictionary_gate_rejects_valid_dictionary_words(self):
        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.thresholds = HybridCorrector._thresholds("strict")
        corrector.min_dictionary_score = 1.0
        corrector.use_morphology_guard = True
        corrector.candidate_generator = CandidateGenerator.from_texts(
            ["перевел инцидент квартире"],
            min_freq=1,
            max_distance=1,
        )

        self.assertFalse(
            corrector._can_apply_candidate(
                "перевез",
                RuntimeCandidate("перевел", source="dictionary", distance=1, score=10.0),
                confidence=0.99,
                margin=0.50,
            )
        )
        self.assertTrue(
            corrector._can_apply_candidate(
                "инцедент",
                RuntimeCandidate("инцидент", source="dictionary", distance=1, score=10.0),
                confidence=0.99,
                margin=0.50,
            )
        )

    def test_dictionary_candidate_dropping_glued_service_token_is_blocked(self):
        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.thresholds = HybridCorrector._thresholds("strict")
        corrector.min_dictionary_score = 0.25
        corrector.use_morphology_guard = False
        corrector.use_entity_guard = False
        corrector.candidate_generator = CandidateGenerator.from_texts(
            [
                "сообщается заставляет поражения которые проекта избавиться прошлой словам "
                "ссылкой Сейчас Когда квадратных ездили важно"
            ],
            min_freq=1,
            max_distance=1,
        )

        blocked_cases = [
            ("сообщаетсяна", "сообщается"),
            ("незаставляет", "заставляет"),
            ("ихпоражения", "поражения"),
            ("которыене", "которые"),
            ("отпроекта", "проекта"),
            ("Избавитьсяот", "Избавиться"),
            ("Напрошлой", "Прошлой"),
            ("Пословам", "Словам"),
        ]
        for source, target in blocked_cases:
            with self.subTest(source=source):
                reason = corrector._candidate_block_reason(
                    source,
                    RuntimeCandidate(target, source="dictionary", distance=2, score=3.0, rank=0),
                    confidence=0.999,
                    margin=0.50,
                    skip_context=True,
                )
                self.assertEqual(reason, "dictionary_drops_glued_service_token")

        allowed_cases = [
            ("сссылкой", "ссылкой"),
            ("ССейчас", "Сейчас"),
            ("ККогда", "Когда"),
            ("кквадратных", "квадратных"),
            ("вездили", "ездили"),
            ("вважно", "важно"),
        ]
        for source, target in allowed_cases:
            with self.subTest(source=source):
                reason = corrector._candidate_block_reason(
                    source,
                    RuntimeCandidate(target, source="dictionary", distance=1, score=3.0, rank=0),
                    confidence=0.999,
                    margin=0.90,
                    skip_context=True,
                )
                self.assertEqual(reason, "")

    def test_dirty_dictionary_recovery_keeps_strict_action_floor(self):
        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.thresholds = HybridCorrector._thresholds("strict")
        corrector.min_dictionary_score = 1.0
        corrector.use_morphology_guard = False
        corrector.use_entity_guard = True
        corrector.entity_guard = ProtectedLexicon.empty()
        corrector.candidate_generator = CandidateGenerator.from_texts(["которые"], min_freq=1, max_distance=1)

        candidate = RuntimeCandidate("которые", source="dictionary", distance=1, score=2.0, rank=0)
        self.assertTrue(
            corrector._can_apply_candidate(
                "готорые",
                candidate,
                confidence=0.83,
                margin=0.20,
            )
        )
        self.assertFalse(
            corrector._can_apply_candidate(
                "готорые",
                candidate,
                confidence=0.81,
                margin=0.20,
            )
        )

    def test_short_dictionary_recovery_and_low_score_context_gate(self):
        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.thresholds = HybridCorrector._thresholds("strict")
        corrector.min_dictionary_score = 0.25
        corrector.use_morphology_guard = False
        corrector.use_entity_guard = True
        corrector.entity_guard = ProtectedLexicon.empty()
        corrector.candidate_generator = CandidateGenerator.from_texts(
            ["может инцидент"],
            min_freq=1,
            max_distance=1,
        )

        self.assertTrue(
            corrector._can_apply_candidate(
                "ожет",
                RuntimeCandidate("может", source="dictionary", distance=1, score=2.0, rank=0),
                confidence=0.96,
                margin=0.55,
            )
        )
        self.assertFalse(
            corrector._can_apply_candidate(
                "ожет",
                RuntimeCandidate("может", source="dictionary", distance=1, score=2.0, rank=0),
                confidence=0.94,
                margin=0.55,
            )
        )

        low_score = RuntimeCandidate("инцидент", source="dictionary", distance=1, score=-1.0, rank=0)
        self.assertFalse(
            corrector._can_apply_candidate(
                "инцедент",
                low_score,
                confidence=0.96,
                margin=0.55,
            )
        )
        self.assertTrue(
            corrector._can_apply_candidate(
                "инцедент",
                low_score,
                confidence=0.96,
                margin=0.55,
                context_margin=0.80,
            )
        )

    def test_protected_short_rule_candidate_is_blocked(self):
        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.thresholds = HybridCorrector._thresholds("strict")
        corrector.min_dictionary_score = 0.25
        corrector.use_morphology_guard = False
        corrector.use_entity_guard = True
        corrector.entity_guard = ProtectedLexicon.from_texts(["ДДля"])
        corrector.candidate_generator = CandidateGenerator.from_texts(["для"], min_freq=1, max_distance=1)

        self.assertFalse(
            corrector._can_apply_candidate(
                "ДДля",
                RuntimeCandidate("Для", source="rule", distance=1, score=2.0, rank=0),
                confidence=0.99,
                margin=0.55,
            )
        )

    def test_candidate_generator_defaults_to_min_freq_one(self):
        from train_hybrid import train_hybrid
        import inspect

        self.assertEqual(CandidateGenerator().min_freq, 1)
        self.assertEqual(inspect.signature(train_hybrid).parameters["candidate_min_freq"].default, 1)

    def test_keep_candidate_augmentation_preserves_keep_label(self):
        example = build_training_example("летнем вечером было тихо.", "летнем вечером было тихо.")
        self.assertIsNotNone(example)
        generator = CandidateGenerator.from_texts(["летним вечером было тихо."], min_freq=1, max_distance=1)

        augmented, stats = augment_keep_candidates(
            [example],
            generator,
            probability=1.0,
            dirty_probability=1.0,
            max_per_example=1,
            seed=1,
        )

        self.assertEqual(stats["augmented_tokens"], 1)
        flattened = [item for candidates in augmented[0].candidate_words for item in candidates]
        self.assertIn("летним", flattened)
        idx = next(i for i, candidates in enumerate(augmented[0].candidate_words) if "летним" in candidates)
        self.assertEqual(augmented[0].action_labels[idx], "KEEP")

    def test_top_k_candidate_population_uses_generated_target_rank(self):
        example = build_training_example("отеты были красными.", "одеты были красными.")
        self.assertIsNotNone(example)
        generator = CandidateGenerator.from_texts(
            ["одеты были красными."] * 2 + ["ответы были неправильными."] * 8,
            min_freq=1,
            max_distance=1,
        )

        populated, stats = populate_top_k_candidates([example], generator, candidate_top_k=8)

        self.assertEqual(stats["oracle_replace_tokens"], 1)
        self.assertEqual(stats["target_candidate_hits"], 1)
        self.assertEqual(stats["oracle_injected_tokens"], 0)
        self.assertEqual(populated[0].candidate_words[0][1], "одеты")
        self.assertEqual(populated[0].action_labels[0], "REPLACE_1")
        self.assertEqual(populated[0].target_candidate_ranks[0], 1)
        self.assertLessEqual(len(populated[0].candidate_words[0]), 8)

    def test_safe_split_candidates_are_narrow(self):
        generator = CandidateGenerator.from_texts(
            [
                "тот же из них в них к ним с ним крупные европейские машину частями "
                "говорить ся News ru мат часть супер ячейки трехсот летней из бюджета "
                "сказать об зданий от из женщин"
            ],
            min_freq=1,
            max_distance=1,
            allow_split_candidates=True,
        )

        self.assertIn("тот же", [c.text for c in generator.get_candidates("тотже")])
        self.assertIn("из них", [c.text for c in generator.get_candidates("изних")])
        self.assertIn("в них", [c.text for c in generator.get_candidates("вних")])
        self.assertIn("к ним", [c.text for c in generator.get_candidates("кним")])
        self.assertIn("с ним", [c.text for c in generator.get_candidates("сним")])
        self.assertIn("крупные европейские", [c.text for c in generator.get_candidates("крупныеевропейские")])
        self.assertIn("из бюджета", [c.text for c in generator.get_candidates("избюджета")])
        self.assertIn("сказать об", [c.text for c in generator.get_candidates("сказатьоб")])
        self.assertIn("зданий от", [c.text for c in generator.get_candidates("зданийот")])
        self.assertIn("из женщин", [c.text for c in generator.get_candidates("изженщин")])
        self.assertNotIn("мат часть", [c.text for c in generator.get_candidates("матчасть")])
        self.assertNotIn("супер ячейки", [c.text for c in generator.get_candidates("суперячейки")])
        self.assertNotIn("трехсот летней", [c.text for c in generator.get_candidates("трехсотлетней")])
        self.assertNotIn("говорить ся", [c.text for c in generator.get_candidates("говориться")])
        self.assertEqual(generator.get_candidates("Newsru"), [])

    def test_short_explicit_typo_rules(self):
        generator = CandidateGenerator.from_texts(["все что быть есть дело лет уже"], min_freq=1, max_distance=1)

        self.assertIn("все", [c.text for c in generator.get_candidates("фсе")])
        self.assertIn("что", [c.text for c in generator.get_candidates("щто")])
        self.assertIn("что", [c.text for c in generator.get_candidates("што")])
        self.assertIn("быть", [c.text for c in generator.get_candidates("ыть")])
        self.assertIn("есть", [c.text for c in generator.get_candidates("эсть")])
        self.assertIn("что", [c.text for c in generator.get_candidates("чтто")])
        self.assertIn("что", [c.text for c in generator.get_candidates("ччто")])
        self.assertIn("дело", [c.text for c in generator.get_candidates("дэло")])
        self.assertIn("лет", [c.text for c in generator.get_candidates("ллет")])
        self.assertIn("уже", [c.text for c in generator.get_candidates("ууже")])

    def test_short_repeated_first_letter_rules_are_dynamic(self):
        generator = CandidateGenerator.from_texts(["Для как уже"], min_freq=1, max_distance=1)

        self.assertIn("Для", [c.text for c in generator.get_candidates("ДДля")])
        self.assertIn("Как", [c.text for c in generator.get_candidates("ККак")])
        self.assertIn("уже", [c.text for c in generator.get_candidates("ууже")])

    def test_context_device_is_cpu_by_default_and_propagated(self):
        reranker = ContextReranker(enabled=False)
        self.assertEqual(reranker.device, "cpu")

        corrector = HybridCorrector(
            model_path="missing.keras",
            preprocessor_path="missing.pkl",
            candidate_generator_path="missing.pkl",
            context_reranker_enabled=False,
            context_device="cpu",
        )

        self.assertEqual(corrector.context_device, "cpu")
        self.assertEqual(corrector.context_reranker.device, "cpu")

    def test_entity_guard_blocks_clean_rare_words(self):
        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.thresholds = HybridCorrector._thresholds("strict")
        corrector.min_dictionary_score = 0.25
        corrector.use_morphology_guard = False
        corrector.use_entity_guard = True
        corrector.entity_guard = ProtectedLexicon.from_texts(
            [
                "В Подпоржья открыли музей.",
                "Суднев рассказал о проекте.",
                "Перерва проходит через район.",
                "Редкое слово дергота есть в корпусе.",
            ]
        )
        corrector.candidate_generator = CandidateGenerator.from_texts(
            ["подполья судне перерыва дер гота"],
            min_freq=1,
            max_distance=1,
        )

        cases = [
            ("Подпоржья", RuntimeCandidate("Подполья", source="dictionary", distance=1, score=2.0)),
            ("Суднев", RuntimeCandidate("Судне", source="dictionary", distance=1, score=2.0)),
            ("Перерва", RuntimeCandidate("Перерыва", source="dictionary", distance=1, score=2.0)),
            ("дергота", RuntimeCandidate("дер гота", source="split", distance=1, score=2.0)),
        ]
        for source, candidate in cases:
            with self.subTest(source=source):
                reason = corrector._candidate_block_reason(
                    source,
                    candidate,
                    confidence=0.999,
                    margin=0.90,
                    skip_context=True,
                )
                self.assertIn(reason, {"protected_lexicon_word", "entity_context_guard"})

    def test_context_source_veto_relaxes_for_clear_lowercase_oov_typo(self):
        class FakeReranker:
            def score_candidates(self, words, index, source, candidates):
                scores = {"готорые": -1.0, "которые": -1.4}
                return [ContextScore(text=c, score=scores[c], available=True) for c in candidates]

        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.context_reranker_enabled = True
        corrector.context_margin = 0.25
        corrector.context_reranker = FakeReranker()
        corrector.runtime_stats = Counter()
        corrector.thresholds = HybridCorrector._thresholds("strict")
        corrector.min_dictionary_score = 0.25
        corrector.use_morphology_guard = False
        corrector.use_entity_guard = True
        corrector.entity_guard = ProtectedLexicon.from_texts(["Правильный корпус без этой опечатки."])
        corrector.candidate_generator = CandidateGenerator.from_texts(["которые"], min_freq=1, max_distance=1)

        outcome = corrector._maybe_rerank_candidate(
            ["люди", "готорые", "пришли"],
            1,
            "готорые",
            RuntimeCandidate("которые", source="dictionary", distance=1, score=2.0, rank=0),
            [RuntimeCandidate("которые", source="dictionary", distance=1, score=2.0, rank=0)],
            confidence=0.99,
            margin=0.50,
        )

        self.assertIsNotNone(outcome.selected)
        self.assertEqual(outcome.reason, "context_source_veto_relaxed")
        self.assertEqual(corrector.runtime_stats["context_source_veto_relaxed_count"], 1)

    def test_context_source_veto_still_blocks_protected_word(self):
        class FakeReranker:
            def score_candidates(self, words, index, source, candidates):
                scores = {"дергота": -1.0, "дерготе": -1.4}
                return [ContextScore(text=c, score=scores[c], available=True) for c in candidates]

        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.context_reranker_enabled = True
        corrector.context_margin = 0.25
        corrector.context_reranker = FakeReranker()
        corrector.runtime_stats = Counter()
        corrector.thresholds = HybridCorrector._thresholds("strict")
        corrector.min_dictionary_score = 0.25
        corrector.use_morphology_guard = False
        corrector.use_entity_guard = True
        corrector.entity_guard = ProtectedLexicon.from_texts(["дергота"])
        corrector.candidate_generator = CandidateGenerator.from_texts(["дерготе"], min_freq=1, max_distance=1)

        outcome = corrector._maybe_rerank_candidate(
            ["дергота"],
            0,
            "дергота",
            RuntimeCandidate("дерготе", source="dictionary", distance=1, score=2.0, rank=0),
            [RuntimeCandidate("дерготе", source="dictionary", distance=1, score=2.0, rank=0)],
            confidence=0.99,
            margin=0.50,
        )

        self.assertIsNone(outcome.selected)
        self.assertEqual(outcome.reason, "context_prefers_source")

    def test_cpu_reranker_scores_after_tensorflow_model_load(self):
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception as exc:
            self.skipTest(f"RuBERT dependencies are unavailable: {exc}")

        model_path = ROOT / "models" / "hybrid_corrector.keras"
        preprocessor_path = ROOT / "models" / "hybrid_preprocessor.pkl"
        candidate_path = ROOT / "models" / "candidate_generator.pkl"
        if not model_path.exists() or not preprocessor_path.exists() or not candidate_path.exists():
            self.skipTest("trained hybrid artifacts are not available")

        HybridCorrector(
            model_path=str(model_path),
            preprocessor_path=str(preprocessor_path),
            candidate_generator_path=str(candidate_path),
            context_reranker_enabled=False,
        )
        reranker = ContextReranker(enabled=True, device="cpu")
        scores = reranker.score_candidates(
            ["дом", "кубца", "первой", "гильдии"],
            1,
            "кубца",
            ["кубца", "кубка", "купца"],
        )
        if not reranker.available:
            self.skipTest(f"RuBERT model is unavailable: {reranker.disabled_reason}")

        self.assertTrue(any(math.isfinite(score.score) for score in scores))

    def test_runtime_can_apply_rank_above_zero(self):
        class FakeModel:
            def predict(self, inputs, verbose=0):
                action = np.zeros((1, 8, 10), dtype=np.float32)
                punct = np.zeros((1, 8, 8), dtype=np.float32)
                action[:, :, 0] = 0.99
                punct[:, :, 0] = 0.99
                action[0, 0, 0] = 0.01
                action[0, 0, 3] = 0.99
                return {"action": action, "punct": punct}

        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.preprocessor = HybridPreprocessor(max_length=8, candidate_top_k=8)
        corrector.model = FakeModel()
        corrector.thresholds = HybridCorrector._thresholds("strict")
        corrector.candidate_generator = CandidateGenerator.from_texts(["ответы одеты"], min_freq=1, max_distance=1)
        corrector.min_dictionary_score = 1.0
        corrector.use_morphology_guard = False
        corrector.context_reranker_enabled = False
        corrector.punctuation_mode = "conservative"

        words, puncts, edits, _, _, _ = corrector._model_correct(
            ["отеты", "были"],
            [
                [
                    RuntimeCandidate("ответы", source="dictionary", distance=1, score=3.0, rank=0),
                    RuntimeCandidate("одеты", source="dictionary", distance=1, score=3.0, rank=1),
                ],
                [RuntimeCandidate("были", rank=0)],
            ],
            ["", "."],
            [" ", "."],
        )

        self.assertEqual(words[0], "одеты")
        self.assertEqual(edits[0].candidate_rank, 1)
        self.assertEqual(edits[0].action, "REPLACE_1")

    def test_runtime_can_apply_service_token_split_candidate(self):
        class FakeModel:
            def predict(self, inputs, verbose=0):
                action = np.zeros((1, 8, 10), dtype=np.float32)
                punct = np.zeros((1, 8, 8), dtype=np.float32)
                action[:, :, 0] = 0.99
                punct[:, :, 0] = 0.99
                action[0, 2, 0] = 0.01
                action[0, 2, 2] = 0.99
                return {"action": action, "punct": punct}

        class FakeReranker:
            def score_candidates(self, words, index, source, candidates):
                scores = {"сообщаетсяна": -5.0, "сообщается на": -1.0, "сообщается": -4.0}
                return [ContextScore(text=c, score=scores.get(c, -3.0), available=True) for c in candidates]

        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.preprocessor = HybridPreprocessor(max_length=8, candidate_top_k=8)
        corrector.model = FakeModel()
        corrector.guard = QualityGuard()
        corrector.thresholds = HybridCorrector._thresholds("strict")
        corrector.candidate_top_k = 8
        corrector.candidate_generator = CandidateGenerator.from_texts(
            ["Об этом сообщается на странице."],
            min_freq=1,
            max_distance=1,
            allow_split_candidates=True,
        )
        corrector.min_dictionary_score = 0.25
        corrector.use_morphology_guard = False
        corrector.use_entity_guard = False
        corrector.context_reranker_enabled = True
        corrector.context_margin = 0.25
        corrector.context_reranker = FakeReranker()
        corrector.runtime_stats = Counter()
        corrector.punctuation_mode = "conservative"
        corrector.deterministic_spacing = True

        result = corrector._correct_segment("Об этом сообщаетсяна странице.")

        self.assertTrue(result.accepted)
        self.assertEqual(result.corrected, "Об этом сообщается на странице.")
        self.assertEqual(result.edits[0].candidate_source, "split")

    def test_runtime_can_apply_titlecase_service_token_split_candidate(self):
        class FakeModel:
            def predict(self, inputs, verbose=0):
                action = np.zeros((1, 8, 10), dtype=np.float32)
                punct = np.zeros((1, 8, 8), dtype=np.float32)
                action[:, :, 0] = 0.99
                punct[:, :, 0] = 0.99
                action[0, 0, 0] = 0.01
                action[0, 0, 2] = 0.99
                return {"action": action, "punct": punct}

        class FakeReranker:
            def score_candidates(self, words, index, source, candidates):
                return [
                    ContextScore(
                        text=c,
                        score=-1.0 if " " in c else (-5.0 if c == source else -3.0),
                        available=True,
                    )
                    for c in candidates
                ]

        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.preprocessor = HybridPreprocessor(max_length=8, candidate_top_k=8)
        corrector.model = FakeModel()
        corrector.guard = QualityGuard()
        corrector.thresholds = HybridCorrector._thresholds("strict")
        corrector.candidate_top_k = 8
        corrector.candidate_generator = CandidateGenerator.from_texts(
            ["На прошлой неделе цены выросли. По словам мужчины, все спокойно."],
            min_freq=1,
            max_distance=1,
            allow_split_candidates=True,
        )
        corrector.min_dictionary_score = 0.25
        corrector.use_morphology_guard = False
        corrector.use_entity_guard = False
        corrector.context_reranker_enabled = True
        corrector.context_margin = 0.25
        corrector.context_reranker = FakeReranker()
        corrector.runtime_stats = Counter()
        corrector.punctuation_mode = "conservative"
        corrector.deterministic_spacing = True

        cases = {
            "Напрошлой неделе цены выросли.": "На прошлой неделе цены выросли.",
            "Пословам мужчины все спокойно.": "По словам мужчины все спокойно.",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                result = corrector._correct_segment(source)
                self.assertTrue(result.accepted)
                self.assertEqual(result.corrected, expected)
                self.assertEqual(result.edits[0].candidate_source, "split")

    def test_context_reranker_can_change_top_candidate(self):
        class FakeReranker:
            def score_candidates(self, words, index, source, candidates):
                scores = {"кубца": -4.0, "кубка": -3.0, "купца": -1.0}
                return [ContextScore(text=c, score=scores[c], available=True) for c in candidates]

        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.context_reranker_enabled = True
        corrector.context_margin = 0.25
        corrector.context_reranker = FakeReranker()
        corrector.runtime_stats = Counter()
        corrector.thresholds = HybridCorrector._thresholds("strict")
        corrector.min_dictionary_score = 1.0
        corrector.use_morphology_guard = False
        corrector.candidate_generator = CandidateGenerator.from_texts(["кубка купца"], min_freq=1, max_distance=1)

        outcome = corrector._maybe_rerank_candidate(
            ["дом", "кубца", "первой", "гильдии"],
            1,
            "кубца",
            RuntimeCandidate("кубка", source="dictionary", distance=1, score=3.0, rank=0),
            [
                RuntimeCandidate("кубка", source="dictionary", distance=1, score=3.0, rank=0),
                RuntimeCandidate("купца", source="dictionary", distance=1, score=3.0, rank=1),
            ],
            confidence=0.99,
            margin=0.50,
        )

        self.assertEqual(outcome.selected.text, "купца")
        self.assertTrue(outcome.reranked)
        self.assertEqual(outcome.reason, "context_changed_top1")
        self.assertGreater(outcome.context_margin, 0.25)

    def test_context_reranker_can_block_source_preferred_split(self):
        class FakeReranker:
            def score_candidates(self, words, index, source, candidates):
                scores = {"матчасть": -1.0, "мат часть": -3.0}
                return [ContextScore(text=c, score=scores[c], available=True) for c in candidates]

        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.context_reranker_enabled = True
        corrector.context_margin = 0.25
        corrector.context_reranker = FakeReranker()
        corrector.runtime_stats = Counter()
        corrector.thresholds = HybridCorrector._thresholds("strict")
        corrector.min_dictionary_score = 1.0
        corrector.use_morphology_guard = False
        corrector.candidate_generator = CandidateGenerator.from_texts(["мат часть"], min_freq=1, max_distance=1)

        outcome = corrector._maybe_rerank_candidate(
            ["матчасть", "надо", "знать"],
            0,
            "матчасть",
            RuntimeCandidate("мат часть", source="split", distance=1, score=3.0, rank=0),
            [RuntimeCandidate("мат часть", source="split", distance=1, score=3.0, rank=0)],
            confidence=0.99,
            margin=0.50,
        )

        self.assertIsNone(outcome.selected)
        self.assertEqual(outcome.reason, "context_prefers_source")
        self.assertEqual(corrector.runtime_stats["split_blocked_by_context_count"], 1)

    def test_split_candidate_fails_closed_on_non_finite_context(self):
        class FakeReranker:
            def score_candidates(self, words, index, source, candidates):
                return [ContextScore(text=c, score=float("-inf"), available=True) for c in candidates]

        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.context_reranker_enabled = True
        corrector.context_margin = 0.25
        corrector.context_reranker = FakeReranker()
        corrector.runtime_stats = Counter()
        corrector.thresholds = HybridCorrector._thresholds("strict")
        corrector.min_dictionary_score = 1.0
        corrector.use_morphology_guard = False
        corrector.candidate_generator = CandidateGenerator.from_texts(["тот же"], min_freq=1, max_distance=1)

        outcome = corrector._maybe_rerank_candidate(
            ["тотже"],
            0,
            "тотже",
            RuntimeCandidate("тот же", source="split", distance=1, score=3.0, rank=0),
            [RuntimeCandidate("тот же", source="split", distance=1, score=3.0, rank=0)],
            confidence=0.99,
            margin=0.50,
        )

        self.assertIsNone(outcome.selected)
        self.assertEqual(outcome.reason, "split_context_non_finite")
        self.assertEqual(corrector.runtime_stats["reranker_non_finite_count"], 2)

    def test_context_reranker_fail_open(self):
        reranker = ContextReranker(enabled=False)
        scores = reranker.score_candidates(["дом", "кубца"], 1, "кубца", ["кубца", "купца"])

        self.assertFalse(scores[0].available)
        self.assertEqual(scores[0].reason, "disabled")

    def test_phrase_guard_blocks_collapsed_duplicate_next_word(self):
        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.thresholds = HybridCorrector._thresholds("strict")
        corrector.min_dictionary_score = 0.25
        corrector.use_morphology_guard = True
        corrector.candidate_generator = CandidateGenerator.from_texts(["причем также"], min_freq=1, max_distance=1)

        self.assertFalse(
            corrector._can_apply_candidate(
                "При",
                RuntimeCandidate("Причем", source="phrase", distance=0, score=100.0),
                confidence=0.99,
                margin=0.50,
                next_word="чем",
            )
        )
        self.assertFalse(
            corrector._can_apply_candidate(
                "так",
                RuntimeCandidate("также", source="phrase", distance=0, score=100.0),
                confidence=0.99,
                margin=0.50,
                next_word="же",
            )
        )

    def test_punctuation_gate_keeps_final_and_protected_context(self):
        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.thresholds = HybridCorrector._thresholds("strict")
        corrector.punctuation_mode = "conservative"

        self.assertFalse(
            corrector._can_apply_punctuation(
                ".",
                "",
                ".",
                "",
                is_final=True,
                confidence=0.999,
                margin=0.90,
            )
        )
        self.assertFalse(
            corrector._can_apply_punctuation(
                ",",
                "",
                ", ",
                "Intel",
                is_final=False,
                confidence=0.999,
                margin=0.90,
            )
        )
        self.assertTrue(
            corrector._can_apply_punctuation(
                "",
                ",",
                " ",
                "что",
                is_final=False,
                confidence=0.999,
                margin=0.90,
            )
        )

    def test_clean_comma_delete_is_extra_conservative(self):
        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.thresholds = HybridCorrector._thresholds("strict")
        corrector.punctuation_mode = "conservative"

        self.assertFalse(
            corrector._can_apply_punctuation(
                ",",
                "",
                ", ",
                "который",
                is_final=False,
                confidence=1.0,
                margin=1.0,
            )
        )
        self.assertFalse(
            corrector._can_apply_punctuation(
                ",",
                "",
                ", ",
                "представленный",
                is_final=False,
                confidence=1.0,
                margin=1.0,
            )
        )
        self.assertFalse(
            corrector._punctuation_decision(
                ",",
                "",
                ", ",
                "Брянской",
                "Орловской",
                is_final=False,
                confidence=1.0,
                margin=1.0,
            )[0]
        )
        self.assertFalse(
            corrector._can_apply_punctuation(
                ",",
                "",
                ", ",
                "пожалуй",
                is_final=False,
                confidence=1.0,
                margin=1.0,
            )
        )
        self.assertFalse(
            corrector._can_apply_punctuation(
                "",
                ".",
                " ",
                "Wrigley",
                is_final=False,
                confidence=0.999,
                margin=0.90,
            )
        )
        self.assertTrue(
            corrector._can_apply_punctuation(
                ".",
                ",",
                ". ",
                "что",
                is_final=False,
                confidence=0.999,
                margin=0.90,
            )
        )
        self.assertFalse(
            corrector._can_apply_punctuation(
                ".",
                ",",
                ".",
                "",
                is_final=True,
                confidence=0.999,
                margin=0.90,
            )
        )
        self.assertTrue(
            corrector._can_apply_punctuation(
                "",
                ".",
                "",
                "",
                is_final=True,
                confidence=0.999,
                margin=0.90,
            )
        )

    def test_comma_insert_guards(self):
        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.thresholds = HybridCorrector._thresholds("strict")
        corrector.punctuation_mode = "conservative"

        self.assertFalse(
            corrector._punctuation_decision(
                "",
                ",",
                " ",
                "актуален",
                "другой",
                is_final=False,
                confidence=0.999,
                margin=0.90,
            )[0]
        )
        self.assertFalse(
            corrector._punctuation_decision(
                "",
                ",",
                " ",
                "слепок",
                "с",
                is_final=False,
                confidence=0.999,
                margin=0.90,
            )[0]
        )
        self.assertFalse(
            corrector._punctuation_decision(
                "",
                ",",
                " ",
                "эксперта",
                "на",
                is_final=False,
                confidence=0.999,
                margin=0.90,
            )[0]
        )
        self.assertFalse(
            corrector._punctuation_decision(
                "",
                ",",
                " ",
                "Известно",
                "также",
                is_final=False,
                confidence=0.999,
                margin=0.90,
            )[0]
        )
        self.assertFalse(
            corrector._punctuation_decision(
                "",
                ",",
                " ",
                "сообщалось",
                "о",
                is_final=False,
                confidence=0.999,
                margin=0.90,
            )[0]
        )
        self.assertFalse(
            corrector._punctuation_decision(
                "",
                ",",
                " ",
                "сообщалось",
                "ранее",
                is_final=False,
                confidence=0.999,
                margin=0.90,
            )[0]
        )
        self.assertTrue(
            corrector._punctuation_decision(
                "",
                ",",
                " ",
                "сообщалось",
                "что",
                is_final=False,
                confidence=0.999,
                margin=0.90,
            )[0]
        )
        self.assertTrue(
            corrector._punctuation_decision(
                "",
                ",",
                " ",
                "также",
                "что",
                is_final=False,
                confidence=0.999,
                margin=0.90,
            )[0]
        )

    def test_protected_punctuation_gaps_block_technical_tokens(self):
        self.assertTrue(is_protected_punctuation_gap(". ", word="ч", next_word="2"))
        self.assertTrue(is_protected_punctuation_gap(". ", word="ст", next_word="286"))
        self.assertTrue(is_protected_punctuation_gap(".", word="News", next_word="ru"))
        self.assertTrue(is_protected_punctuation_gap(".", word="T", next_word="злоумышленники"))
        self.assertTrue(is_protected_punctuation_gap(":", word="4", next_word="2"))

        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.thresholds = HybridCorrector._thresholds("strict")
        corrector.punctuation_mode = "conservative"

        self.assertFalse(
            corrector._can_apply_punctuation(
                ".",
                ",",
                ". ",
                "2",
                is_final=False,
                confidence=0.999,
                margin=0.90,
            )
        )

    def test_rule_guards_block_names_and_infinitive_context(self):
        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.thresholds = HybridCorrector._thresholds("strict")
        corrector.min_dictionary_score = 1.0
        corrector.use_morphology_guard = True
        corrector.candidate_generator = CandidateGenerator.from_texts(["говорится рис отразится"], min_freq=1, max_distance=1)

        self.assertFalse(
            corrector._can_apply_candidate(
                "Риз",
                RuntimeCandidate("Рис", source="rule", distance=1, score=2.0),
                confidence=0.99,
                margin=0.50,
                next_word="Уизерспун",
            )
        )
        self.assertFalse(
            corrector._can_apply_candidate(
                "отразиться",
                RuntimeCandidate("отразится", source="rule", distance=1, score=2.0),
                confidence=0.99,
                margin=0.50,
                prev_word="может",
            )
        )
        self.assertTrue(
            corrector._can_apply_candidate(
                "говориться",
                RuntimeCandidate("говорится", source="rule", distance=1, score=2.0),
                confidence=0.99,
                margin=0.50,
            )
        )

    def test_rebuild_does_not_change_layout_without_punctuation_edit(self):
        original = "Обвинения идут по двум статьям: ч. 2 ст. 286."
        slots = extract_word_slots(original)
        words = [slot.word for slot in slots]
        source_puncts = [choose_punctuation_label(slot.punct_after) for slot in slots]

        rebuilt = rebuild_preserving_layout(
            original,
            slots,
            words,
            source_puncts,
            punctuation_mode="conservative",
            source_puncts=source_puncts,
            rewrite_punct_indices=[],
        )

        self.assertEqual(rebuilt, original)

    def test_window_ranges_overlap_long_lines(self):
        ranges = HybridCorrector._window_ranges(260, 128, overlap=24)

        self.assertEqual(ranges[0], (0, 128))
        self.assertEqual(ranges[1], (104, 232))
        self.assertEqual(ranges[-1], (208, 260))

    def test_rejected_segments_do_not_normalize_spacing(self):
        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.guard = QualityGuard()

        def fake_segment(segment):
            return CorrectionResult(segment, segment, 0.99, False, "protected_value_changed", [])

        corrector._correct_segment = fake_segment
        original = "Что услышали спасатели?1 марта 2024Впервые стало известно."

        result = corrector.correct_with_details(original)

        self.assertEqual(result.corrected, original)
        self.assertFalse(result.accepted)

    def test_global_guard_rejection_returns_original_text(self):
        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.guard = QualityGuard()

        def fake_segment(segment):
            return CorrectionResult(segment, segment.replace("«", "").replace("»", ""), 0.99, True, "accepted", [])

        corrector._correct_segment = fake_segment
        original = "Он сказал: «пример»."

        result = corrector.correct_with_details(original)

        self.assertEqual(result.corrected, original)
        self.assertFalse(result.accepted)

    def test_safe_raw_spacing_normalizer_is_narrow(self):
        self.assertEqual(
            safe_normalize_raw_spacing("Что случилось?1 марта 2024Впервые сообщили."),
            "Что случилось? 1 марта 2024 Впервые сообщили.",
        )
        self.assertEqual(
            safe_normalize_raw_spacing("Дом,который стоит у дороги."),
            "Дом,который стоит у дороги.",
        )

    def test_guard_rejects_structural_punctuation_loss(self):
        guard = QualityGuard()
        original = "Intel — «Интел А/О» № 143."
        corrected = "Intel Интел А О 143."

        decision = guard.check(original, corrected, confidence=0.99, edit_count=4)

        self.assertFalse(decision.accepted)
        self.assertIn(
            decision.reason,
            {"protected_value_changed", "structural_punctuation_changed"},
        )

    def test_curriculum_keeps_exact_clean_ratio(self):
        generator = DatasetGenerator()
        texts = [f"Это простой тестовый текст номер {i}." for i in range(10)]

        df = generator.generate_dataset_curriculum(
            texts,
            samples_per_text=1,
            clean_ratio=0.30,
            curriculum=[{"name": "spelling_light", "profile": "spelling_light", "p": 1.0}],
        )
        clean_ratio = (df["difficulty"] == "clean").mean()

        self.assertEqual(len(df), 10)
        self.assertAlmostEqual(clean_ratio, 0.30)


if __name__ == "__main__":
    unittest.main()
