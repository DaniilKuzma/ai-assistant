# Pretraining Readiness Report

Generated: 2026-05-18, Europe/Moscow.

Scope: final preflight before large dataset generation and training. No full training was run. No full dataset was generated. Commands were run with `.venv/bin/python` because bare `python` is not available in this shell unless the virtualenv is activated.

Taxonomy reference:
- Orfogrammka orthography: https://orfogrammka.ru/орфография/
- Orfogrammka punctuation: https://orfogrammka.ru/пунктуация/

## Verdict

`BLOCKED`

The test suite, syntax layer, DOCX checks, model smoke test, dataset dry-run, and no-training pipeline now pass. The remaining blocker is clean overcorrection in the no-training evaluation with the existing checkpoint: `clean_overcorrection_rate = 0.28`. Because clean overcorrection remains, the project must not be marked `READY_FOR_DATASET_BUILD` or `READY_FOR_SHORT_TRAINING`.

## Status Matrix

| Area | Status | Evidence |
| --- | --- | --- |
| Full pytest | PASS | `.venv/bin/python -m pytest -q`: `474 passed in 172.24s` |
| Syntax | PASS | `.venv/bin/python -m pytest -q tests/test_syntax_layer.py tests/test_docx_io.py tests/test_encoder_loading.py`: `12 passed in 8.50s`; direct `parse_syntax("Мы ждали файл…")` returned 4 tokens |
| DOCX | PASS | `tests/test_docx_io.py` passed in the targeted suite; DOCX flow keeps injected/model-capable correctors for model-required edits |
| Model loading | PASS | `TorchCandidateModelBackend.from_config(config)` loaded; candidate scoring and punctuation prediction smoke passed with no TensorFlow/Keras crash |
| CandidateGenerator metadata | PASS | Full pytest is green; candidate metadata checks cover preserved `rule_id`, `mode`, `requires_model`, config/injected dictionary lexicon, and separate `frequent_errors` |
| Validator | PASS | Full pytest is green for numbers, percents, decimals, URL/email, dangerous `-тся/-ться`, punctuation noise, quote/bracket balance, and capitalization false positives |
| Clean overcorrection | BLOCKER | Plain deterministic smoke kept 9/9 control clean cases unchanged, including `69-летний`, `Мы ждали файл…`, valid hyphen forms, and valid punctuation. No-training evaluation with `existing_checkpoint` still reports `clean_overcorrection_rate = 0.28`; examples are in `reports/clean_overcorrection_examples.csv` |
| rules.yaml taxonomy | PASS | `orthography` and `punctuation` sections exist; implemented rule ids are reportable; planned taxonomy entries do not require duplicate executable `RuleSpec`; `comma_conjunction` reports under `homogeneous_commas` |
| Required reports | PASS | Required reports exist, including `candidate_recall_by_rule.csv`; punctuation gap report `gap_label_coverage_by_rule.csv` also exists |
| Dataset dry-run | PASS | `RUSSIAN_CORRECTOR_DATASET_LIMIT=5000`: `total=5000`, `clean=500`, `synthetic=4389`, `real=111`, `hard_negative_count=125`, spelling and punctuation rows present, `rule_id_counts` non-empty, train/val/test splits present |
| No-training pipeline | PASS with blocker | `RUSSIAN_CORRECTOR_DISABLE_MODEL_TRAINING=1`: `model_training_ran=False`, `evaluation_backend=existing_checkpoint`, `checkpoint_load_error=""`; required training/evaluation reports were written |
| Candidate recall report | PASS | `reports/candidate_recall_by_rule.csv` exists and is non-empty (`38` lines) |

## Dataset Dry-Run Manifest

- `total`: 5000
- `final_total_rows`: 5000
- `composition.clean`: 500
- `composition.synthetic`: 4389
- `composition.real`: 111
- `hard_negative_count`: 125
- `error_type_counts.spelling`: 1512
- `error_type_counts.punctuation`: 1905
- `rule_id_counts`: 50 non-empty entries
- `splits`: `train=4483`, `val=259`, `test=258`

## No-Training Pipeline

- `feature_count`: 4483
- `evaluation_split`: `test`
- `evaluation_count`: 258
- `model_training_ran`: `False`
- `model_training_disabled`: `True`
- `model_training_disabled_source`: `RUSSIAN_CORRECTOR_DISABLE_MODEL_TRAINING`
- `evaluation_backend`: `existing_checkpoint`
- `checkpoint_load_error`: empty
- `clean_overcorrection_rate`: 0.28

Allowed backend naming is now respected: no-training does not report `freshly_trained_model`.

## Fixed Blockers

- Plain `Corrector` now only applies safe model-required sentence-start capitalization without a scorer; hyphen, punctuation, context-pair, and other model-required edits remain gated.
- `comma_conjunction` is no longer a duplicate executable owner in `rules.yaml`; it is reported under `homogeneous_commas`.
- Candidate-derived hyphen and split/join labels are alignment-aware, so repeated identical candidates are labeled only at the actually changed span.
- No-training report metadata uses allowed backend names and does not pretend a fresh ruRoberta model was trained.

## Remaining Blockers

- Clean overcorrection remains in the no-training evaluation for the existing checkpoint: `clean_overcorrection_rate = 0.28`. Examples include quote normalization, unwanted comma insertion, unwanted capitalization, and lexical changes such as `намерены` -> `намеренны`.

## Readiness

- `READY_FOR_DATASET_BUILD`: no.
- `READY_FOR_SHORT_TRAINING`: no.
- Current verdict: `BLOCKED`.

The dataset builder itself passed the 5000-row dry-run, and the no-training pipeline executes correctly. Overall readiness remains blocked until clean overcorrection is reduced or explicitly handled by stricter thresholds/validation/checkpoint selection.

## Commands Run

```bash
.venv/bin/python -m pytest -q tests/test_inference_smoke.py::test_plain_corrector_does_not_apply_candidate_only_hyphen_or_model_punctuation_without_scorer tests/test_rule_metrics.py::test_real_rules_config_maps_rule_ids_to_report_groups tests/test_training_features.py::test_training_feature_uses_aligned_edit_span_for_repeated_candidates
.venv/bin/python -m pytest -q tests/test_train_entrypoint.py::test_no_training_existing_checkpoint_backend_uses_allowed_name tests/test_train_entrypoint.py::test_train_uses_trained_corrector_for_reports_when_model_training_runs tests/test_training_pipeline.py::test_pipeline_runs_with_model_training_disabled
.venv/bin/python -m pytest -q tests/test_inference_smoke.py tests/test_rule_metrics.py tests/test_training_features.py tests/test_train_entrypoint.py tests/test_training_pipeline.py tests/test_rules_coverage.py tests/test_negative_rule_suites.py tests/test_candidate_generator.py
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q tests/test_syntax_layer.py tests/test_docx_io.py tests/test_encoder_loading.py
RUSSIAN_CORRECTOR_DATASET_LIMIT=5000 .venv/bin/python - <<'PY'
from src.config.load_config import load_config
from src.data.full_dataset_builder import build_dataset_from_config

config = load_config("configs/config.yaml")
print(build_dataset_from_config(config, force=True))
PY
RUSSIAN_CORRECTOR_DISABLE_MODEL_TRAINING=1 .venv/bin/python -m src.training.train configs/config.yaml
```
