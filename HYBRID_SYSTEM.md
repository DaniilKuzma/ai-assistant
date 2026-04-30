# Hybrid Correction System

This project now uses one main correction strategy:

```text
HybridCorrector = top-k candidate generator + edit-based Transformer + source-aware punctuation head + RuBERT context reranker + quality guard
```

The model does not rewrite the whole sentence. It predicts local edit actions for
tokens and punctuation labels after tokens. The action space is `KEEP`,
`DELETE`, and `REPLACE_0 ... REPLACE_4`, where each replace action selects one
of the generated candidates.

The runtime now preserves structural punctuation from the source text by
default. Quotes, dashes, brackets, slashes, `№`, numbers and Latin fragments are
protected; the punctuation head is only allowed to change simple trailing marks
`, . ? ! : ;` in `punctuation_mode="conservative"`.
The runtime now works on full lines, falling back to overlapping windows for
long inputs. The current model format has `source_punct_ids` and top-k
`candidate_ids`; old v5 and earlier models are intentionally treated as
obsolete and should be retrained. The current runtime expects v6 models trained
with rank-consistent `REPLACE_0 ... REPLACE_4` labels, safe split-candidates,
candidate-aware KEEP examples, clean-noise filtering and stronger
punctuation-change sample weights. Ambiguous dictionary/split edits are
optionally rescored by `DeepPavlov/rubert-base-cased`. Dictionary corrections use
`candidate_min_freq=1`, `min_dictionary_score=1.0`, and a `pymorphy2` morphology
guard to avoid changing valid Russian dictionary words and inflected forms.

Keyboard-neighbor typo generation is intentionally not implemented.

## Main Files

```text
src/hybrid_corrector.py      runtime corrector
src/candidate_generator.py   dictionary and rule candidates
src/context_reranker.py      RuBERT/RuRoBERTa candidate reranker
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
  --max-length 128 \
  --max-vocab-size 80000 \
  --candidate-min-freq 1 \
  --candidate-max-distance 1 \
  --candidate-top-k 5 \
  --min-dictionary-score 1.0 \
  --context-model-name DeepPavlov/rubert-base-cased \
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
  --sample-size 500 \
  --output-dir report \
  --strictness strict \
  --punctuation-mode conservative \
  --candidate-top-k 5
```

Evaluation writes:

```text
report/summary.json
report/error_analysis.csv
report/error_metrics_by_type.csv
report/word_edit_diagnostics.csv
report/punct_edit_diagnostics.csv
report/clean_candidate_audit.csv
report/clean_noise_filter.csv
report/worst_cases.csv
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
