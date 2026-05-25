# Functions Map

Карта ниже перечисляет важные публичные классы и функции. Это не полный код и не замена точечному чтению модулей.

## Inference

- `src.inference.corrector.Corrector` — безопасный fallback corrector.
- `src.inference.corrector.CorrectionResult` — `source_text`, `corrected_text`, `edits`.
- `src.inference.model_corrector.TrainedModelCorrector` — inference facade для trained adapters/heads.
- `src.inference.model_corrector.TorchCandidateModelBackend` — PyTorch backend для scoring candidates и punctuation gaps.
- `src.inference.model_corrector.ModelCandidatePrediction` — score/confidence для candidate.
- `src.inference.model_corrector.ModelPunctuationPrediction` — label/action/confidence для punctuation gap.
- `src.inference.edit_realizer.apply_candidate` — применяет один candidate к тексту.
- `src.inference.postprocess.normalize_spacing` — нормализует пробелы после edits.

## Candidates And Rules

- `src.candidates.candidate_generator.CandidateGenerator` — генерирует bounded candidates из правил, словаря и punctuation layer.
- `src.candidates.candidate_generator.Candidate` — единица возможного исправления.
- `src.candidates.candidate_ranking.rank_candidates_for_budget` — ограничивает candidates под `max_candidates`.
- `src.candidates.dictionary_candidates.dictionary_candidate_specs` — fuzzy/dictionary candidates.
- `src.rules.registry.all_rules`, `orthography_rules`, `punctuation_rules`, `rule_by_id` — registry правил.
- `src.rules.orthography.orthography_rules` — набор орфографических правил.
- `src.rules.punctuation.generate_punctuation_candidates` — пунктуационные candidates.
- `src.rules.coverage_matrix.validate_project_rules_coverage` — проверка `configs/rules.yaml`.

## Validation

- `src.validation.diff_analyzer.DiffAnalyzer.analyze` — классифицирует различия source/target в edit taxonomy.
- `src.validation.diff_analyzer.Edit` — edit with spans, source/replacement, type, status, reason.
- `src.validation.strict_validator.StrictValidator.validate` — принимает/rejects edits в strict scope.
- `src.validation.strict_validator.ValidationResult.apply_accepted` — применяет accepted edits.
- `src.validation.edit_classifier.is_allowed_edit_type` — проверка допустимых edit types.

## Model And Training

- `src.model.encoder.load_tokenizer`, `load_encoder` — загрузка encoder-only Transformers backend.
- `src.model.edit_model.CandidateAwareEditModel` — multitask model: candidate score, punctuation, confidence, error type.
- `src.model.heads.build_linear_heads` — custom linear heads.
- `src.model.losses.multitask_loss` — combined loss.
- `src.training.tensorization.build_training_feature` — строит feature для одного row.
- `src.training.tensorization.build_features_from_rows` — batch feature builder.
- `src.training.tensorization.EditBatchCollator` — collate tensors for training.
- `src.training.trainer.EditModelTrainer.train_epoch` — training loop.
- `src.training.train.train` — основной training/evaluation entrypoint.
- `src.training.train.evaluate_trained_model` — evaluation saved/fine-tuned corrector.
- `src.training.save_load.save_training_artifacts` — сохраняет config/labels/thresholds.

## Data And Evaluation

- `src.config.candidate_dataset_config.get_candidate_dataset_config` — normalized canonical view of `data.candidate_opportunity`; primary source for dataset settings.
- `src.config.candidate_dataset_config.validate_candidate_dataset_config` — rejects legacy `data.training_dataset` / `data.training_dataset_core`, duplicate top-level dataset keys, invalid totals/quotas/activation/thresholds.
- `src.config.candidate_dataset_config.candidate_dataset_paths`, `candidate_dataset_totals`, `candidate_dataset_rule_quota`, `candidate_dataset_rule_activation`, `candidate_dataset_audit`, `candidate_dataset_composition` — typed section helpers used by build/preflight/setup modules.
- `src.data.full_dataset_builder.build_dataset_from_config` — сборка полного dataset по config.
- `src.data.dataset_contract.ensure_contract_columns` — добавляет contract/layer/weight/quota columns к rows или DataFrame.
- `src.data.dataset_contract.row_dataset_layer` — определяет слой строки по explicit metadata или legacy source type.
- `src.data.dataset_contract.stable_dataset_hash` — стабильный hash по source/target/rule ids для manifest/report freshness.
- `src.data.operator_dataset_builder.build_operator_training_dataset_from_config` — canonical builder для `candidate_opportunity`.
- `src.data.operator_dataset_builder.generate_atomic_positive_rows_from_clean_pool` — генерирует verified atomic positives из clean pool opportunities.
- `src.data.training_quality_audit.audit_training_dataset` — собирает audit frames и summaries для layered dataset gates.
- `src.data.external_sources.load_external_pair_sources` — загрузка внешних пар.
- `src.data.clean_corpus_sources.load_clean_corpus_sentences` — clean corpus source loader.
- `src.data.synthetic_generator.SyntheticGenerator` — synthetic corruptions/identity examples.
- `src.evaluation.evaluate.evaluate_rows_detailed` — метрики + подробные edit logs.
- `src.evaluation.metrics.compute_metrics` — core metrics.
- `src.evaluation.reports.write_required_evaluation_reports` — обязательные CSV/MD reports.
- `src.evaluation.threshold_sweep.threshold_sweep` — precision/recall by threshold.

## App And DOCX

- `src.app.streamlit_app.build_streamlit_corrector` — выбирает trained model или fallback.
- `src.app.streamlit_app.render_highlighted_diff` — HTML diff для UI.
- `src.app.streamlit_app.main` — Streamlit entrypoint.
- `src.docx.docx_corrector.correct_docx` — исправление `.docx`.
- `src.docx.docx_reader.read_paragraphs` — чтение параграфов.
- `src.docx.docx_writer.write_paragraphs_like` — запись документа с сохранением runs.

