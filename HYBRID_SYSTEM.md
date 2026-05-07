# Hybrid Correction System

This project now uses one main correction strategy:

```text
HybridCorrector = top-k candidate generator + edit-based Transformer + source-aware punctuation head + RuBERT context reranker + quality guard
```

The model does not rewrite the whole sentence. It predicts local edit actions for
tokens and punctuation labels after tokens. The action space is `KEEP`,
`DELETE`, and `REPLACE_0 ... REPLACE_7`, where each replace action selects one
of the generated candidates.

The runtime no longer has a deterministic whitespace correction layer or a
separate spacing error class. URL, email, decimal/date-like fragments, Latin
text, slashes, `№`, numbers and unknown structural punctuation are protected.
The punctuation head can change simple trailing marks and whitelisted
structural gap labels for quotes, dashes, brackets, ellipsis, direct speech and
lists in `punctuation_mode="conservative"`.
The runtime now works on full lines, falling back to overlapping windows for
long inputs. The current model format has `source_punct_ids` and top-k
`candidate_ids`; stale model artifacts are intentionally treated as obsolete
and should be retrained. The current training/evaluation scenario is V10.5:
`model_version=10`, `runtime_version="10.5"`, real spellcheck/punctuation pairs
in train/val, and external real-pair evaluation kept outside the synthetic
test split. Spacing-only real rows are filtered out. Ambiguous dictionary/split edits are rescored by
`DeepPavlov/rubert-base-cased` with `context-device auto`, using CUDA when the
local PyTorch stack can run it and falling back to CPU on scoring errors; split
edits fail closed when context is not reliable. Dictionary corrections use `candidate_min_freq=1`,
`min_dictionary_score=0.25`, a `pymorphy2` morphology guard, and an
entity/noise guard loaded from `data/raw/texts.txt` and
`data/processed/*.csv` `correct_text` values to avoid changing valid clean/raw
words, names, toponyms and rare terms.

V10.5 keeps the cautious dirty-aware recall gates, adds curated compound/case/
abbreviation/borrowed-word spelling errors, and adds structural punctuation
labels. AI Forever spellcheck/punctuation train rows are filtered to local
edit-compatible non-spacing pairs and repeated in train. Runtime guards reject
structural punctuation edits that worsen bracket or quote balance. A narrow
low-action recovery can apply safe top-1 dictionary typos without relaxing
punctuation, entity, protected-word or morphology guards. V10.5 also adds narrow
comma-delete recovery for extra commas after short service words, a lowercase
service-word extension, day-month date comma cleanup, safe final-period recovery
for the last gap, plus punctuation target diagnostics in evaluator summaries.
Held-out AI Forever
and RuSpellGold test files stay separate. Keyboard-neighbor typo generation is
still intentionally not implemented.

## Main Files

```text
src/hybrid_corrector.py      runtime corrector
src/candidate_generator.py   dictionary and rule candidates
src/context_reranker.py      RuBERT/RuRoBERTa candidate reranker
src/entity_guard.py          protected clean/raw lexicon and entity context
src/external_datasets.py     external real-pair loading and filtering
src/morphology_guard.py      morphology guard for dictionary candidates
src/training_augmentation.py candidate-aware KEEP negatives and clean audit
src/edit_labels.py           error_text -> token edit labels
src/hybrid_preprocessor.py   vocabularies and vectorization
src/hybrid_model.py          Keras Transformer model
src/train_hybrid.py          training entry point
src/evaluate_hybrid.py       evaluation entry point
src/quality_guard.py         safety checks
src/text_utils.py            tokenization and reconstruction
```

The legacy `src/train.py` and `src/evaluate.py` scripts now delegate to the
hybrid trainer/evaluator when run directly.

## Training

```bash
PYTHONPATH=src .venv/bin/python src/train.py \
  --dataset data/processed/dataset.csv \
  --output-dir models \
  --model-version 10 \
  --max-length 128 \
  --max-vocab-size 80000 \
  --candidate-min-freq 1 \
  --candidate-max-distance 1 \
  --candidate-top-k 16 \
  --long-oov-max-distance 2 \
  --long-oov-min-length 8 \
  --min-dictionary-score 0.25 \
  --clean-action-keep-weight 2.0 \
  --dirty-action-keep-weight 0.8 \
  --action-change-weight 4.0 \
  --punct-change-weight 8.0 \
  --final-punct-weight 8.0 \
  --clean-punct-keep-weight 4.0 \
  --dirty-punct-keep-weight 1.0 \
  --context-model-name DeepPavlov/rubert-base-cased \
  --context-device auto \
  --d-model 128 \
  --num-layers 2 \
  --ff-dim 256 \
  --dropout 0.30 \
  --learning-rate 1e-4 \
  --batch-size 64 \
  --epochs 30
```

Training writes:

```text
models/hybrid_corrector.keras
models/hybrid_preprocessor.pkl
models/candidate_generator.pkl
models/hybrid_config.json
models/hybrid_training_log.csv
```

## Evaluation

```bash
PYTHONPATH=src .venv/bin/python src/evaluate.py \
  --dataset data/processed/test.csv \
  --sample-size 1000 \
  --output-dir report/synthetic_v11 \
  --strictness strict \
  --punctuation-mode conservative \
  --candidate-top-k 16 \
  --context-device auto \
  --min-dictionary-score 0.25 \
  --no-diagnostics \
  --inference-batch-size 512
```

Smoke evaluation writes:

```text
report/synthetic_v11/summary.json
report/synthetic_v11/error_analysis.csv
report/synthetic_v11/error_metrics_by_type.csv
report/synthetic_v11/worst_cases.csv
```

Full diagnostic runs without `--no-diagnostics` also write:

```text
report/word_edit_diagnostics.csv
report/word_decision_diagnostics.csv
report/punct_edit_diagnostics.csv
report/clean_candidate_audit.csv
report/clean_noise_filter.csv
```

## V10.5 Evaluation

The main notebook evaluation defaults to a synthetic-compatible smoke report:

```text
report/synthetic_v11/summary.json
report/synthetic_v11/error_analysis.csv
report/synthetic_v11/error_metrics_by_type.csv
report/synthetic_v11/worst_cases.csv
```

External real-pair reports are written separately:

```text
report/external_ai_forever_v11/summary.json
report/external_ruspellgold_v11/summary.json
```

Acceptance targets after full evaluation:

```text
clean_overcorrection_rate <= 0.002
worse_rate <= 0.01
overall exact_match above 0.5699
dirty exact_match above 0.3398
punct_applied_count above the current 856 level
source_kind_slices and source_dataset_slices present in summary.json
low_action_dictionary_recovery_count present in summary.json
safe_comma_delete_recovery_count present in summary.json
safe_service_comma_delete_recovery_count present in summary.json
safe_date_comma_delete_recovery_count present in summary.json
safe_final_period_recovery_count present in summary.json
safe_final_period_recovery_count at least 74
punct_target_applied_count present in summary.json
punct_target_predicted_but_blocked_rate present in summary.json
```

## Runtime

```python
from hybrid_corrector import HybridCorrector

corrector = HybridCorrector()
print(corrector.correct("Машиное обучение это интерестно."))
```

If `models/hybrid_corrector.keras` is not trained yet, the runtime can still
start in a conservative dictionary mode so the GUI remains usable. The final
project model is the hybrid model saved as `models/hybrid_corrector.keras`.

## Checks

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```
