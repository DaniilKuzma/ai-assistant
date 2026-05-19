# Pretraining Readiness Report

Generated: 2026-05-19 00:55, Europe/Moscow.

Scope: final preflight before short training. The project remains edit-based and candidate-aware: rules generate candidates, the existing checkpoint scores them, thresholds resolve rule-specific decisions, and `StrictValidator` blocks unsafe edits. No full model training was run in this preflight.

## Verdict

`READY_FOR_SHORT_TRAINING`

The previous blocker is cleared. No-training evaluation with `existing_checkpoint` now uses the conservative threshold profile and reports `clean_overcorrection_rate = 0.0` on the 50k medium dry-run dataset. Active candidate recall passes: every active/included rule except the non-target `unknown` bucket is at least `0.904`, and all punctuation target groups used by synthetic generation are at `1.0`.

## Status Matrix

| Area | Status | Evidence |
| --- | --- | --- |
| Full pytest | PASS | `.venv/bin/python -m pytest -q`: `488 passed in 173.42s` |
| Lexicon | PASS | `data/processed/russian_lexicon.txt` exists, `3,063,822` entries, `72,808,893` bytes |
| Medium dataset dry-run | PASS | `RUSSIAN_CORRECTOR_DATASET_LIMIT=50000`: `total=50000`, `clean=5000`, `synthetic=43889`, `real=1111`, `train=44762`, `val=2626`, `test=2612` |
| No-training pipeline | PASS | `RUSSIAN_CORRECTOR_DISABLE_MODEL_TRAINING=1`: `status=evaluation_prepared`, `model_training_ran=False`, `evaluation_backend=existing_checkpoint`, `threshold_profile=conservative`, `feature_build_skipped=True` |
| Clean overcorrection | PASS | `clean_overcorrection_rate=0.0`, `clean_exact_match=1.0`, `reports/clean_overcorrection_examples.csv` has 0 rows |
| Candidate recall | PASS | Active/included min recall excluding `unknown` is `0.904255`; `unknown=0.300813` is not an active synthetic target |
| Model loading | PASS | Existing adapter/head artifacts loaded for `TrainedModelCorrector`; `checkpoint_load_error` is empty |
| Reports | PASS | Required evaluation, rule, gap coverage, candidate recall, accepted/rejected edit, dataset, and training reports were regenerated |

## Key Metrics

- `exact_match`: 0.426876
- `dirty_improved_rate`: 0.407705
- `dirty_worse_rate`: 0.040220
- `edit_precision`: 0.909851
- `edit_recall`: 0.375816
- `punctuation_precision`: 0.971198
- `punctuation_recall`: 0.730503
- `combined_score`: 2.285541

## Candidate Recall

Active/included rules below 0.80: none.

The only below-threshold bucket is `unknown`: `gold_count=123`, `candidate_present_count=37`, `candidate_recall=0.300813`. It is a residual reporting bucket for unsupported/real edits, not an active synthetic/evaluation target.

Notable active recalls:

- `frequent_error_exact`: 0.904255
- `comma_subordinate`, `comma_conjunction`, `introductory_comma`, `detached_adverbial_comma`: 1.0
- `direct_speech_colon`, `direct_speech_dash`, `direct_speech_quotes`: 1.0
- `subject_predicate_dash`, `enumeration_colon`, `final_punctuation_default`: 1.0
- `n_nn_*`, `ne_*`, context-pair, hyphen, prefix, and hissing-vowel spelling groups present in the report: 1.0

Unsupported low-recall synthetic groups are inactive or absent from generated targets until candidate recall is fixed: `quote_open`, `quote_close`, `bracket_pair_balance`, `semicolon`, broad `quotes_brackets`, broad `delete_replace`, and `punctuation_noise`.

## Overcorrection Fixes

The observed clean false-positive families are now blocked by thresholds, candidate constraints, generation narrowing, or validator guards:

- straight quote normalization (`"` -> `«/»`);
- direct speech punctuation on existing clean quote/name patterns such as `А.Белоусов`;
- NER capitalization on clean common noun spans such as `налоговой службы`;
- sentence-start capitalization after abbreviation-like contexts;
- N/NN lexical rewrites such as `намерены` -> `намеренны`;
- unsafe `не` split/join pairs such as `не случайно` -> `неслучайно`, `небольшой` -> `не большой`, and adjective-like `недавнем` -> `не давнем`;
- unwanted dash insertion after discourse markers such as `Получается` and `Значит`;
- unwanted comma insertion in `что если ... то` and inside sentence-initial gerundial phrases before prepositional complements.

## Readiness

- `READY_FOR_DATASET_BUILD`: yes.
- `READY_FOR_SHORT_TRAINING`: yes.
- Current verdict: `READY_FOR_SHORT_TRAINING`.

## Commands Run

```bash
.venv/bin/python scripts/build_russian_lexicon.py
.venv/bin/python -m pytest -q
RUSSIAN_CORRECTOR_DATASET_LIMIT=50000 .venv/bin/python - <<'PY'
from src.config.load_config import load_config
from src.data.full_dataset_builder import build_dataset_from_config

config = load_config("configs/config.yaml")
print(build_dataset_from_config(config, force=True))
PY
RUSSIAN_CORRECTOR_DISABLE_MODEL_TRAINING=1 .venv/bin/python -m src.training.train configs/config.yaml
```
