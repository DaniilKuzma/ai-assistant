from collections import Counter
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from candidate_generator import CandidateGenerator
from context_reranker import ContextReranker, ContextScore
from data_preparation import DatasetGenerator, safe_normalize_raw_spacing
from edit_labels import build_examples_from_dataframe, build_training_example
from entity_guard import ProtectedLexicon
from external_datasets import (
    AI_FOREVER_TEST_FILES,
    AI_FOREVER_TRAIN_FILES,
    ExternalPair,
    classify_real_error_types,
    extract_pair,
    filter_real_pairs,
    read_json_records,
    real_pair_filter_reason,
)
from hybrid_corrector import CorrectionEdit, CorrectionResult, HybridCorrector, PunctuationDecision, RuntimeCandidate
from hybrid_preprocessor import HybridPreprocessor
from morphology_guard import is_morphological_dictionary_word, is_same_lemma_inflection
from quality_guard import QualityGuard
from text_utils import (
    choose_punctuation_label,
    extract_word_slots,
    is_protected_punctuation_gap,
    rebuild_preserving_layout,
)
from training_augmentation import augment_keep_candidates, populate_top_k_candidates
import evaluate_hybrid as evaluate_hybrid_module


class ConservativeRuntimeTests(unittest.TestCase):
    def test_external_jsonl_loader_reads_real_pair_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.json"
            path.write_text(
                '{"source":"давольно милый","correction":"довольно милый","domain":"web"}\n',
                encoding="utf-8",
            )

            rows = list(read_json_records(path))
            pair = extract_pair(rows[0], "test/source")

        self.assertEqual(len(rows), 1)
        self.assertIsNotNone(pair)
        self.assertEqual(pair.error_text, "давольно милый")
        self.assertEqual(pair.correct_text, "довольно милый")
        self.assertEqual(pair.source_domain, "web")

    def test_real_pair_filter_accepts_local_spell_and_punct_but_rejects_spacing(self):
        pairs = [
            ExternalPair("Это давольно важно.", "Это довольно важно.", "unit"),
            ExternalPair("Я думаю что это важно.", "Я думаю, что это важно.", "unit"),
            ExternalPair("Привет , мир.", "Привет, мир.", "unit"),
        ]

        df, stats = filter_real_pairs(pairs)

        self.assertEqual(stats["input"], 3)
        self.assertEqual(stats["kept"], 2)
        self.assertEqual(stats["spacing_only"], 1)
        self.assertEqual(set(df["source_kind"]), {"real"})
        self.assertIn("real_spelling", set("|".join(df["error_types"]).split("|")))
        self.assertIn("real_punctuation", set("|".join(df["error_types"]).split("|")))
        self.assertNotIn("real_spacing", set("|".join(df["error_types"]).split("|")))

    def test_real_pair_filter_rejects_bad_pairs(self):
        self.assertEqual(real_pair_filter_reason("", "Текст."), "empty")
        self.assertEqual(real_pair_filter_reason("Текст.", "Текст."), "identical")
        self.assertEqual(real_pair_filter_reason("а" * 241, "а" * 240), "too_long")
        self.assertEqual(
            real_pair_filter_reason(
                "Мама мыла раму.",
                "Вчера горел музей.",
            ),
            "heavy_rewrite",
        )

    def test_external_pair_normalizes_yo(self):
        pair = extract_pair(
            {"source": "Ежик идет.", "correction": "Ёжик идёт.", "domain": "literature"},
            "unit",
        )

        self.assertIsNotNone(pair)
        self.assertEqual(pair.error_text, "Ежик идет.")
        self.assertEqual(pair.correct_text, "Ежик идет.")
        self.assertEqual(real_pair_filter_reason(pair.error_text, pair.correct_text), "identical")

    def test_external_test_files_are_not_train_files(self):
        train_paths = {path for _, _, path in AI_FOREVER_TRAIN_FILES}
        test_paths = {path for _, _, path in AI_FOREVER_TEST_FILES}

        self.assertTrue(train_paths)
        self.assertTrue(test_paths)
        self.assertTrue(all("train.json" in path for path in train_paths))
        self.assertTrue(all("test.json" in path for path in test_paths))
        self.assertTrue(train_paths.isdisjoint(test_paths))

    def test_real_source_columns_do_not_break_label_building(self):
        df = pd.DataFrame(
            [
                {
                    "error_text": "Я думаю что это важно.",
                    "correct_text": "Я думаю, что это важно.",
                    "difficulty": "real_spellcheck_punctuation",
                    "error_types": "real_punctuation",
                    "source_kind": "real",
                    "source_dataset": "unit",
                    "source_domain": "web",
                }
            ]
        )

        examples, stats = build_examples_from_dataframe(df)

        self.assertEqual(stats["kept"], 1)
        self.assertEqual(len(examples), 1)
        self.assertFalse(examples[0].is_clean)

    def test_real_error_type_classifier_does_not_emit_spacing_label(self):
        labels = classify_real_error_types("Привет , мир.", "Привет, мир.")

        self.assertNotIn("real_spacing", labels)

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

    def test_candidate_generator_caches_runtime_candidates(self):
        generator = CandidateGenerator.from_texts(
            ["инцидент произошел", "интересный инцидент"],
            min_freq=1,
            max_distance=1,
        )

        first = generator.get_candidates("инцедент", max_candidates=8)
        second = generator.get_candidates("инцедент", max_candidates=8)

        self.assertEqual(first, second)
        self.assertTrue(any(key[:4] == ("инцедент", 8, False, True) for key in generator._candidate_cache))

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

    def test_v10_curated_examples_cover_new_error_types_without_spacing(self):
        generator = DatasetGenerator()
        df = generator.curated_v10_examples("test")
        labels = set("|".join(df["error_types"]).split("|"))
        expected = {
            "spelling_compound_joining",
            "spelling_capitalization",
            "spelling_abbreviation_case",
            "spelling_borrowed_word",
            "punct_quote_style",
            "punct_remove_quotes",
            "punct_direct_speech_dash_missing",
            "punct_direct_speech_inner_punct",
            "punct_dash_missing",
            "punct_dash_extra",
            "punct_bracket_missing_close",
            "punct_bracket_extra",
            "punct_ellipsis_extra",
            "punct_ellipsis_missing",
            "punct_list_missing_colon",
            "punct_list_item_punctuation",
            "punct_quote_punct_order",
        }

        self.assertTrue(expected.issubset(labels))
        self.assertFalse(any(label.startswith("space_") for label in labels))

    def test_synthetic_generation_does_not_emit_space_error_labels(self):
        generator = DatasetGenerator()
        texts = [
            "Москва — столица России.",
            "Он сказал: «Привет!»",
            "Мы решили действовать по-новому.",
        ]

        df = generator.generate_dataset_curriculum(
            texts,
            samples_per_text=4,
            clean_ratio=0.0,
            curriculum=[
                {"name": "orthography_rules", "profile": "orthography_rules", "p": 0.5},
                {"name": "punctuation_structural", "profile": "punctuation_structural", "p": 0.5},
            ],
        )
        labels = set("|".join(df["error_types"]).split("|"))

        self.assertFalse(any(label.startswith("space_") for label in labels))

    def test_build_training_example_handles_case_phrase_and_hyphen_pairs(self):
        cases = [
            ("москва стала центром.", "Москва стала центром."),
            ("Мы сделали по новому.", "Мы сделали по-новому."),
            ("Мини футбол популярен.", "Мини-футбол популярен."),
            ("Железно-дорожный вокзал открыт.", "Железнодорожный вокзал открыт."),
        ]

        for error_text, correct_text in cases:
            with self.subTest(error_text=error_text):
                example = build_training_example(error_text, correct_text)
                self.assertIsNotNone(example)
                self.assertIn("REPLACE_0", example.action_labels)

    def test_v10_candidate_generator_covers_rule_phrase_and_case_candidates(self):
        generator = CandidateGenerator.from_texts(
            [
                "Москва стала центром.",
                "Президент Российской Федерации подписал указ.",
                "Этот ВУЗ открыл лабораторию.",
                "Мягкий поролон лежал рядом.",
                "Нужно кардинально изменить подход.",
                "Железнодорожный вокзал открыт.",
            ],
            min_freq=1,
            max_distance=1,
        )

        cases = {
            "москва": "Москва",
            "президент": "Президент",
            "вуз": "ВУЗ",
            "паралон": "поролон",
            "координально": "кардинально",
            "железно-дорожный": "железнодорожный",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                candidates = [candidate.text for candidate in generator.get_candidates(source)]
                self.assertIn(expected, candidates)
        self.assertEqual(generator.phrase_confusions["по новому"][0], "по-новому")
        self.assertEqual(generator.phrase_confusions["в виду"][0], "ввиду")
        self.assertEqual(generator.phrase_confusions["мини футбол"][0], "мини-футбол")

    def test_v11_candidate_generator_covers_safety_first_sources(self):
        generator = CandidateGenerator.from_texts(
            [
                "Нужно рассказать познавательно.",
                "Утренняя зарядка помогает.",
                "В общем, это работает.",
                "Как будто все спокойно.",
                "Почему-то он пришел.",
            ],
            min_freq=1,
            max_distance=1,
            allow_split_candidates=True,
        )

        cases = {
            "расказать": "рассказать",
            "позновательно": "познавательно",
            "зарадка": "зарядка",
            "вобщем": "в общем",
            "какбудто": "как будто",
            "почемуто": "почему-то",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                candidates = generator.get_candidates(source, max_candidates=16)
                self.assertIn(expected, [candidate.text for candidate in candidates])

        source_by_text = {
            candidate.text: candidate.source
            for candidate in generator.get_candidates("позновательно", max_candidates=16)
        }
        self.assertEqual(source_by_text["познавательно"], "orthographic")
        source_by_text = {
            candidate.text: candidate.source
            for candidate in generator.get_candidates("вобщем", max_candidates=16)
        }
        self.assertEqual(source_by_text["в общем"], "phrase_safe")

    def test_v11_safe_split_blocks_repeated_first_and_short_vowel_service_splits(self):
        generator = CandidateGenerator.from_texts(
            ["Сейчас важный момент. Это винные сорта. Бывают иные варианты."],
            min_freq=1,
            max_distance=1,
            allow_split_candidates=True,
        )

        self.assertIn("сейчас", [candidate.text.lower() for candidate in generator.get_candidates("ссейчас")])
        self.assertNotIn("с сейчас", [candidate.text.lower() for candidate in generator.get_candidates("ссейчас")])
        self.assertNotIn("в иные", [candidate.text.lower() for candidate in generator.get_candidates("виные")])

    def test_v11_mined_confusions_use_train_pairs_only(self):
        generator = CandidateGenerator.from_texts(
            ["довольно познавательно рассказать зарядка"],
            min_freq=1,
            max_distance=1,
        )

        stats = generator.fit_mined_confusions(
            ["давольно интересно", "давольно полезно", "чистый текст"],
            ["довольно интересно", "довольно полезно", "чистый текст"],
            min_count=2,
        )

        self.assertEqual(stats["accepted"], 1)
        candidates = generator.get_candidates("давольно", max_candidates=16)
        self.assertIn("довольно", [candidate.text for candidate in candidates])
        self.assertEqual(
            next(candidate.source for candidate in candidates if candidate.text == "довольно"),
            "mined",
        )

    def test_structural_punctuation_labels_vectorize_and_rebuild(self):
        example = build_training_example('Он сказал: "Привет".', "Он сказал: «Привет».")
        self.assertIsNotNone(example)
        self.assertIn(':"', example.source_punct_labels)
        self.assertIn(":«", example.target_punct_labels)
        self.assertIn('".', example.source_punct_labels)
        self.assertIn("».", example.target_punct_labels)

        preprocessor = HybridPreprocessor(max_length=8)
        inputs, targets, _ = preprocessor.vectorize_examples([example])
        self.assertGreater(inputs["source_punct_ids"].max(), 0)
        self.assertGreater(targets["punct"].max(), 0)

        slots = extract_word_slots('Он сказал: "Привет".')
        rebuilt = rebuild_preserving_layout(
            'Он сказал: "Привет".',
            slots,
            example.target_words,
            example.target_punct_labels,
            punctuation_mode="conservative",
            source_puncts=example.source_punct_labels,
            rewrite_punct_indices=[
                idx
                for idx, (source, target) in enumerate(zip(example.source_punct_labels, example.target_punct_labels))
                if source != target
            ],
        )
        self.assertEqual(rebuilt, "Он сказал: «Привет».")

    def test_president_case_candidate_requires_official_context(self):
        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.thresholds = HybridCorrector._thresholds("strict")
        corrector.min_dictionary_score = 0.25
        corrector.use_morphology_guard = False
        corrector.use_entity_guard = False
        corrector.candidate_generator = CandidateGenerator.from_texts(["Президент Российской Федерации."], min_freq=1)
        corrector.runtime_stats = Counter()

        candidate = RuntimeCandidate("Президент", source="rule", distance=1, score=1.0)

        self.assertEqual(
            corrector._candidate_block_reason("президент", candidate, 0.99, 0.50, next_word="сказал"),
            "president_case_context_guard",
        )
        self.assertEqual(
            corrector._candidate_block_reason("президент", candidate, 0.99, 0.50, next_word="РФ"),
            "",
        )

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
        batch_infer = preprocessor.vectorize_inference_batch(
            [example.source_words],
            [example.candidate_words],
            [example.source_punct_labels],
        )

        self.assertIn("source_punct_ids", inputs)
        self.assertIn("source_punct_ids", infer)
        self.assertEqual(inputs["source_punct_ids"].shape, (1, 16))
        self.assertEqual(inputs["candidate_ids"].shape, (1, 16, preprocessor.candidate_top_k))
        np.testing.assert_array_equal(batch_infer["token_ids"], infer["token_ids"])
        np.testing.assert_array_equal(batch_infer["candidate_ids"], infer["candidate_ids"])
        np.testing.assert_array_equal(batch_infer["source_punct_ids"], infer["source_punct_ids"])
        self.assertEqual(outputs["punct"].shape, (1, 16))
        self.assertEqual(weights["punct"].shape, (1, 16))
        comma_index = example.target_punct_labels.index(",")
        self.assertEqual(weights["punct"][0, comma_index], 8.0)

        replacement = build_training_example("Инцедент произошел.", "Инцидент произошел.")
        self.assertIsNotNone(replacement)
        repl_inputs, repl_outputs, repl_weights = preprocessor.vectorize_examples([replacement])
        self.assertEqual(repl_inputs["candidate_ids"].shape, (1, 16, preprocessor.candidate_top_k))
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
        self.assertEqual(candidate_shape[-1], 16)

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

    def test_low_action_dictionary_recovery_allows_safe_top1_typos(self):
        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.thresholds = HybridCorrector._thresholds("strict")
        corrector.min_dictionary_score = 0.25
        corrector.use_morphology_guard = True
        corrector.use_entity_guard = True
        corrector.entity_guard = ProtectedLexicon.empty()
        corrector.runtime_stats = Counter()
        corrector.candidate_generator = CandidateGenerator.from_texts(
            [
                "ближайшее структуры Финляндии популярности оперативное "
                "официальный авиакомпаний добиваться временное"
            ],
            min_freq=1,
            max_distance=1,
        )

        allowed_cases = [
            ("ближуайшее", "ближайшее", 1, 3.7, 0.76, 0.54),
            ("струхтуры", "структуры", 1, 2.5, 0.58, 0.31),
            ("Финлятндии", "Финляндии", 1, 3.3, 0.64, 0.29),
            ("популянрости", "популярности", 2, 1.3, 0.75, 0.51),
            ("операттивное", "оперативное", 1, 1.3, 0.50, 0.06),
        ]
        for source, target, distance, score, confidence, margin in allowed_cases:
            with self.subTest(source=source):
                reason = corrector._candidate_block_reason(
                    source,
                    RuntimeCandidate(target, source="dictionary", distance=distance, score=score, rank=0),
                    confidence=confidence,
                    margin=margin,
                )
                self.assertEqual(reason, "")

        self.assertEqual(corrector.runtime_stats["low_action_dictionary_recovery_count"], len(allowed_cases))

    def test_low_action_dictionary_recovery_keeps_risky_candidates_blocked(self):
        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.thresholds = HybridCorrector._thresholds("strict")
        corrector.min_dictionary_score = 0.25
        corrector.use_morphology_guard = True
        corrector.use_entity_guard = True
        corrector.entity_guard = ProtectedLexicon.from_texts(["Финлятндии"])
        corrector.runtime_stats = Counter()
        corrector.candidate_generator = CandidateGenerator.from_texts(
            [
                "структуры Финляндии официальных авиакомпаний добраться временно "
                "перевел инцидент"
            ],
            min_freq=1,
            max_distance=1,
        )

        blocked_cases = [
            ("струхтуры", RuntimeCandidate("структуры", source="dictionary", distance=1, score=1.19, rank=0)),
            ("струхтуры", RuntimeCandidate("структуры", source="dictionary", distance=1, score=2.5, rank=1)),
            ("офисиальный", RuntimeCandidate("официальных", source="dictionary", distance=2, score=0.61, rank=0)),
            ("авиакмпании", RuntimeCandidate("авиакомпаний", source="dictionary", distance=2, score=-0.1, rank=0)),
            ("добиаться", RuntimeCandidate("добраться", source="dictionary", distance=2, score=2.4, rank=1)),
            ("времемное", RuntimeCandidate("временно", source="dictionary", distance=2, score=0.7, rank=0)),
            ("СТРУХТУРЫ", RuntimeCandidate("СТРУКТУРЫ", source="dictionary", distance=1, score=2.5, rank=0)),
            ("strukh", RuntimeCandidate("strukt", source="dictionary", distance=1, score=2.5, rank=0)),
            ("Финлятндии", RuntimeCandidate("Финляндии", source="dictionary", distance=1, score=3.3, rank=0)),
            ("перевез", RuntimeCandidate("перевел", source="dictionary", distance=1, score=3.3, rank=0)),
        ]
        for source, candidate in blocked_cases:
            with self.subTest(source=source, target=candidate.text):
                reason = corrector._candidate_block_reason(
                    source,
                    candidate,
                    confidence=0.60,
                    margin=0.12,
                )
                self.assertNotEqual(reason, "")

        self.assertEqual(corrector.runtime_stats["low_action_dictionary_recovery_count"], 0)

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

    def test_safe_comma_delete_recovery_for_service_words(self):
        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.thresholds = HybridCorrector._thresholds("strict")
        corrector.punctuation_mode = "conservative"
        corrector.runtime_stats = Counter()

        cases = [
            ("на", "мирных"),
            ("в", "эксплуатацию"),
            ("до", "приезда"),
            ("от", "стоимости"),
            ("по", "фамилии"),
            ("без", "претензии"),
        ]
        for word, next_word in cases:
            with self.subTest(word=word, next_word=next_word):
                can_apply, reason = corrector._punctuation_decision(
                    ",",
                    "",
                    ", ",
                    word,
                    next_word,
                    is_final=False,
                    confidence=0.990,
                    margin=0.960,
                )
                self.assertTrue(can_apply)
                self.assertEqual(reason, "safe_comma_delete_recovery")

        self.assertEqual(corrector.runtime_stats["safe_comma_delete_recovery_count"], len(cases))

    def test_safe_service_comma_delete_recovery_for_lowercase_service_words(self):
        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.thresholds = HybridCorrector._thresholds("strict")
        corrector.punctuation_mode = "conservative"
        corrector.runtime_stats = Counter()

        cases = [
            ("в", "начале"),
            ("на", "место"),
            ("с", "октября"),
            ("о", "проверке"),
            ("за", "решение"),
        ]
        for word, next_word in cases:
            with self.subTest(word=word, next_word=next_word):
                can_apply, reason = corrector._punctuation_decision(
                    ",",
                    "",
                    ", ",
                    word,
                    next_word,
                    is_final=False,
                    confidence=0.950,
                    margin=0.900,
                )
                self.assertTrue(can_apply)
                self.assertEqual(reason, "safe_service_comma_delete_recovery")

        self.assertEqual(corrector.runtime_stats["safe_comma_delete_recovery_count"], len(cases))
        self.assertEqual(corrector.runtime_stats["safe_service_comma_delete_recovery_count"], len(cases))

    def test_safe_date_comma_delete_recovery_for_day_month(self):
        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.thresholds = HybridCorrector._thresholds("strict")
        corrector.punctuation_mode = "conservative"
        corrector.runtime_stats = Counter()

        for day, month in [("11", "января"), ("27", "марта"), ("1", "мая")]:
            with self.subTest(day=day, month=month):
                can_apply, reason = corrector._punctuation_decision(
                    ",",
                    "",
                    ", ",
                    day,
                    month,
                    is_final=False,
                    confidence=0.860,
                    margin=0.700,
                )
                self.assertTrue(can_apply)
                self.assertEqual(reason, "safe_date_comma_delete_recovery")

        self.assertEqual(corrector.runtime_stats["safe_comma_delete_recovery_count"], 3)
        self.assertEqual(corrector.runtime_stats["safe_date_comma_delete_recovery_count"], 3)

    def test_safe_comma_delete_recovery_keeps_clean_guards(self):
        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.thresholds = HybridCorrector._thresholds("strict")
        corrector.punctuation_mode = "conservative"
        corrector.runtime_stats = Counter()

        blocked_cases = [
            ("на", "который", False, 0.990, 0.960),
            ("на", "что", False, 0.990, 0.960),
            ("на", "пожалуй", False, 0.990, 0.960),
            ("на", "представленный", False, 0.990, 0.960),
            ("Брянской", "Орловской", False, 0.990, 0.960),
            ("на", "Intel", False, 0.990, 0.960),
            ("в", "2008", False, 0.990, 0.960),
            ("О", "война", False, 0.990, 0.960),
            ("на", "мирных", True, 0.990, 0.960),
            ("на", "мирных", False, 0.949, 0.960),
            ("на", "мирных", False, 0.990, 0.899),
            ("4", "и", False, 0.990, 0.960),
            ("32", "января", False, 0.990, 0.960),
            ("11", "января", False, 0.859, 0.750),
            ("11", "января", False, 0.900, 0.699),
        ]
        for word, next_word, is_final, confidence, margin in blocked_cases:
            with self.subTest(word=word, next_word=next_word, confidence=confidence, margin=margin):
                can_apply, reason = corrector._punctuation_decision(
                    ",",
                    "",
                    ", ",
                    word,
                    next_word,
                    is_final=is_final,
                    confidence=confidence,
                    margin=margin,
                    protected_gap=next_word == "Intel",
                )
                self.assertFalse(can_apply)
                self.assertNotEqual(reason, "safe_comma_delete_recovery")

        self.assertEqual(corrector.runtime_stats["safe_comma_delete_recovery_count"], 0)
        self.assertEqual(corrector.runtime_stats["safe_service_comma_delete_recovery_count"], 0)
        self.assertEqual(corrector.runtime_stats["safe_date_comma_delete_recovery_count"], 0)

    def test_safe_final_period_recovery_for_final_gap(self):
        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.thresholds = HybridCorrector._thresholds("strict")
        corrector.punctuation_mode = "conservative"
        corrector.runtime_stats = Counter()

        cases = [
            ("", ".", "", 0.820, 0.700),
            (",", ".", ",", 0.820, 0.700),
        ]
        for source_punct, predicted_punct, gap, confidence, margin in cases:
            with self.subTest(source_punct=source_punct):
                can_apply, reason = corrector._punctuation_decision(
                    source_punct,
                    predicted_punct,
                    gap,
                    "текст",
                    "",
                    is_final=True,
                    confidence=confidence,
                    margin=margin,
                )
                self.assertTrue(can_apply)
                self.assertEqual(reason, "safe_final_period_recovery")

        self.assertEqual(corrector.runtime_stats["safe_final_period_recovery_count"], len(cases))

    def test_safe_final_period_recovery_keeps_guards(self):
        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.thresholds = HybridCorrector._thresholds("strict")
        corrector.punctuation_mode = "conservative"
        corrector.runtime_stats = Counter()

        blocked_cases = [
            ("", ".", " ", "текст", "дальше", False, 0.900, 0.800, False),
            ("", ".", "", "Elec", "", True, 0.900, 0.800, True),
            ("", "!", "", "текст", "", True, 0.900, 0.800, False),
            (":", ".", ":", "текст", "", True, 0.900, 0.800, False),
            ("", ".", "", "текст", "", True, 0.819, 0.750, False),
            ("", ".", "", "текст", "", True, 0.900, 0.699, False),
            (",", ".", ",", "текст", "", True, 0.819, 0.750, False),
            (",", ".", ",", "текст", "", True, 0.900, 0.699, False),
        ]
        for source_punct, predicted_punct, gap, word, next_word, is_final, confidence, margin, protected_gap in blocked_cases:
            with self.subTest(source_punct=source_punct, predicted_punct=predicted_punct, confidence=confidence, margin=margin):
                can_apply, reason = corrector._punctuation_decision(
                    source_punct,
                    predicted_punct,
                    gap,
                    word,
                    next_word,
                    is_final=is_final,
                    confidence=confidence,
                    margin=margin,
                    protected_gap=protected_gap,
                )
                self.assertFalse(can_apply)
                self.assertNotEqual(reason, "safe_final_period_recovery")

        self.assertEqual(corrector.runtime_stats["safe_final_period_recovery_count"], 0)

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
        blocked_pairs = [
            ("словам", "мужчины"),
            ("информации", "источника"),
            ("того", "или"),
            ("того", "чтобы"),
            ("того", "самого"),
            ("того", "стоит"),
            ("тому", "времени"),
            ("конечно", "же"),
            ("заявил", "сегодня"),
            ("добавил", "он"),
            ("напомнил", "ей"),
            ("напомнил", "об"),
            ("рассказал", "женам"),
            ("спросила", "у"),
            ("договорились", "о"),
            ("холодных", "стран"),
            ("баллистических", "ракет"),
            ("подписных", "листов"),
            ("эффективное", "средство"),
            ("утверждают", "производители"),
            ("утверждалось", "о"),
            ("башни", "не"),
            ("почерковеды", "работали"),
            ("очередь", "министр"),
            ("Ответьте", "же"),
            ("апрель", "2008"),
            ("Oxu", "az"),
            ("традицию", "предавать"),
            ("Семенова", "выразила"),
            ("Ласточкин", "добавил"),
            ("ответьте", "мне"),
            ("контекст", "достаточно"),
            ("средство", "освоения"),
        ]
        for word, next_word in blocked_pairs:
            with self.subTest(word=word, next_word=next_word):
                self.assertFalse(
                    corrector._punctuation_decision(
                        "",
                        ",",
                        " ",
                        word,
                        next_word,
                        is_final=False,
                        confidence=0.999,
                        margin=0.90,
                    )[0]
                )
        can_apply, reason = corrector._punctuation_decision(
            "",
            ",",
            " ",
            "того",
            "стоит",
            is_final=False,
            confidence=0.999,
            margin=0.90,
        )
        self.assertFalse(can_apply)
        self.assertEqual(reason, "comma_insert_fixed_phrase_guard")
        for word in ["сообщил", "сообщалось", "заявил"]:
            with self.subTest(word=word, next_word="что"):
                self.assertTrue(
                    corrector._punctuation_decision(
                        "",
                        ",",
                        " ",
                        word,
                        "что",
                        is_final=False,
                        confidence=0.999,
                        margin=0.90,
                    )[0]
                )
        self.assertFalse(
            corrector._punctuation_decision(
                ";",
                ":",
                "; ",
                "контрабас",
                "Андрей",
                is_final=False,
                confidence=0.999,
                margin=0.90,
            )[0]
        )
        self.assertFalse(
            corrector._punctuation_decision(
                ":",
                ";",
                ": ",
                "народы",
                "Как",
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

    def test_evaluate_no_diagnostics_skips_large_csvs(self):
        class FakeCorrector:
            is_trained = True

            def __init__(self, **kwargs):
                self.runtime_stats = Counter()
                self.candidate_generator = object()
                self.context_reranker = type("FakeReranker", (), {"device": "cpu"})()
                self.entity_guard = type("FakeEntityGuard", (), {"size": 0})()

            def correct(self, text, return_details=False):
                return CorrectionResult(text, text, 1.0, True, "accepted", [])

        original_corrector = evaluate_hybrid_module.HybridCorrector
        evaluate_hybrid_module.HybridCorrector = FakeCorrector
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                dataset_path = tmp_path / "dataset.csv"
                output_dir = tmp_path / "report"
                pd.DataFrame(
                    [
                        {
                            "error_text": "Это тест.",
                            "correct_text": "Это тест.",
                            "difficulty": "clean",
                            "error_types": "clean",
                            "split": "test",
                        }
                    ]
                ).to_csv(dataset_path, index=False, encoding="utf-8")

                summary = evaluate_hybrid_module.evaluate_hybrid(
                    dataset_path=str(dataset_path),
                    sample_size=None,
                    output_dir=str(output_dir),
                    write_diagnostics=False,
                    context_reranker_enabled=False,
                )

                self.assertEqual(summary["runtime_version"], "11.0")
                self.assertEqual(summary["low_action_dictionary_recovery_count"], 0)
                self.assertEqual(summary["safe_comma_delete_recovery_count"], 0)
                self.assertEqual(summary["safe_service_comma_delete_recovery_count"], 0)
                self.assertEqual(summary["safe_date_comma_delete_recovery_count"], 0)
                self.assertEqual(summary["safe_final_period_recovery_count"], 0)
                self.assertEqual(summary["punct_target_change_count"], 0)
                self.assertEqual(summary["punct_target_predicted_count"], 0)
                self.assertEqual(summary["punct_target_applied_count"], 0)
                self.assertEqual(summary["punct_target_predicted_but_blocked_count"], 0)
                self.assertIsNone(summary["punct_target_predicted_but_blocked_rate"])
                self.assertFalse(summary["write_diagnostics"])
                self.assertTrue((output_dir / "summary.json").exists())
                self.assertTrue((output_dir / "error_analysis.csv").exists())
                self.assertTrue((output_dir / "error_metrics_by_type.csv").exists())
                self.assertTrue((output_dir / "worst_cases.csv").exists())
                self.assertFalse((output_dir / "word_edit_diagnostics.csv").exists())
                self.assertFalse((output_dir / "word_decision_diagnostics.csv").exists())
                self.assertFalse((output_dir / "punct_edit_diagnostics.csv").exists())
        finally:
            evaluate_hybrid_module.HybridCorrector = original_corrector

    def test_evaluate_counts_punctuation_targets(self):
        class FakeCorrector:
            is_trained = True

            def __init__(self, **kwargs):
                self.runtime_stats = Counter()
                self.candidate_generator = object()
                self.context_reranker = type("FakeReranker", (), {"device": "cpu"})()
                self.entity_guard = type("FakeEntityGuard", (), {"size": 0})()

            def correct(self, text, return_details=False):
                return CorrectionResult(
                    text,
                    text,
                    1.0,
                    True,
                    "accepted",
                    [],
                    punctuation_diagnostics=[
                        PunctuationDecision(
                            index=1,
                            word="думаю",
                            source_punct="",
                            predicted_punct=",",
                            confidence=0.90,
                            margin=0.10,
                            applied=False,
                            blocked_reason="low_confidence_or_margin",
                            is_final=False,
                        )
                    ],
                )

        original_corrector = evaluate_hybrid_module.HybridCorrector
        evaluate_hybrid_module.HybridCorrector = FakeCorrector
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                dataset_path = tmp_path / "dataset.csv"
                output_dir = tmp_path / "report"
                pd.DataFrame(
                    [
                        {
                            "error_text": "Я думаю что это важно.",
                            "correct_text": "Я думаю, что это важно.",
                            "difficulty": "punctuation_only",
                            "error_types": "punct_remove_comma_before_clause",
                            "split": "test",
                        }
                    ]
                ).to_csv(dataset_path, index=False, encoding="utf-8")

                summary = evaluate_hybrid_module.evaluate_hybrid(
                    dataset_path=str(dataset_path),
                    sample_size=None,
                    output_dir=str(output_dir),
                    context_reranker_enabled=False,
                )
                punct_df = pd.read_csv(output_dir / "punct_edit_diagnostics.csv")

            self.assertEqual(summary["runtime_version"], "11.0")
            self.assertEqual(summary["safe_service_comma_delete_recovery_count"], 0)
            self.assertEqual(summary["safe_date_comma_delete_recovery_count"], 0)
            self.assertEqual(summary["safe_final_period_recovery_count"], 0)
            self.assertEqual(summary["punct_target_change_count"], 1)
            self.assertEqual(summary["punct_target_predicted_count"], 1)
            self.assertEqual(summary["punct_target_applied_count"], 0)
            self.assertEqual(summary["punct_target_predicted_but_blocked_count"], 1)
            self.assertEqual(summary["punct_target_predicted_but_blocked_rate"], 1.0)
            for column in [
                "expected_source_punct",
                "expected_target_punct",
                "punct_target_changed",
                "punct_target_predicted",
                "punct_target_applied",
                "punct_target_predicted_but_blocked",
            ]:
                self.assertIn(column, punct_df.columns)
        finally:
            evaluate_hybrid_module.HybridCorrector = original_corrector

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

    def test_guard_rejects_balanced_parenthesis_damage_from_clean_cases(self):
        guard = QualityGuard()
        cases = [
            (
                'Его итогом стал концертный альбом "Burning Japan Live" (1994).',
                'Его итогом стал концертный альбом "Burning Japan Live" (1994.',
            ),
            (
                "Об этом сообщается на сайте региональное управление Федеральной службы судебных приставов (ФССП).",
                "Об этом сообщается на сайте региональное управление Федеральной службы судебных приставов (ФССП.",
            ),
            (
                "По факту возбуждено уголовное дело по статье 213 (хулиганство) УК РФ.",
                "По факту возбуждено уголовное дело по статье 213 хулиганство) УК РФ.",
            ),
            (
                "Похищенные из храма иконы (более 20) и ценности у них изъяты.",
                "Похищенные из храма иконы (более 20 и ценности у них изъяты.",
            ),
        ]

        for original, corrected in cases:
            with self.subTest(original=original):
                decision = guard.check(original, corrected, confidence=0.99, edit_count=1)
                self.assertFalse(decision.accepted)
                self.assertEqual(decision.reason, "structural_punctuation_changed")

    def test_guard_allows_unmatched_closing_parenthesis_cleanup(self):
        guard = QualityGuard()

        decision = guard.check(
            "Он пришел) домой.",
            "Он пришел домой.",
            confidence=0.99,
            edit_count=1,
        )

        self.assertTrue(decision.accepted)

    def test_segment_guard_rejects_high_confidence_structural_punct_deletion(self):
        corrector = HybridCorrector.__new__(HybridCorrector)
        corrector.guard = QualityGuard()
        corrector.punctuation_mode = "conservative"
        original = "По факту возбуждено дело по статье 213 (хулиганство) УК РФ."
        slots = extract_word_slots(original)
        source_words = [slot.word for slot in slots]
        source_puncts = [choose_punctuation_label(slot.punct_after) for slot in slots]
        corrected_puncts = list(source_puncts)
        target_index = source_words.index("213")
        corrected_puncts[target_index] = ""

        result = corrector._finish_segment_correction(
            original,
            slots,
            source_puncts,
            source_words,
            corrected_puncts,
            [CorrectionEdit(target_index, "(", "", "PUNCT", 0.999)],
            [],
            [],
            0.999,
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.corrected, original)
        self.assertEqual(result.guard_reason, "structural_punctuation_changed")

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
