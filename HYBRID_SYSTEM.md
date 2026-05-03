# Hybrid Correction System

This project now uses one main correction strategy:

```text
HybridCorrector = deterministic spacing + top-k candidate generator + edit-based Transformer + source-aware punctuation head + RuBERT context reranker + quality guard
```

The model does not rewrite the whole sentence. It predicts local edit actions for
tokens and punctuation labels after tokens. The action space is `KEEP`,
`DELETE`, and `REPLACE_0 ... REPLACE_7`, where each replace action selects one
of the generated candidates.

The runtime now includes a deterministic spacing layer before the final
`QualityGuard` decision. It only fixes high-precision whitespace cases:
duplicate spaces between Cyrillic text, spaces before simple punctuation,
missing spaces after simple punctuation in Cyrillic context, and spacing around
`—` between words. URL, email, decimal/date-like fragments, Latin text,
quotes, brackets, slashes, `№` and structural punctuation are protected.

The runtime preserves structural punctuation from the source text by
default. Quotes, dashes, brackets, slashes, `№`, numbers and Latin fragments are
protected; the punctuation head is only allowed to change simple trailing marks
`, . ? ! : ;` in `punctuation_mode="conservative"`.
The runtime now works on full lines, falling back to overlapping windows for
long inputs. The current model format has `source_punct_ids` and top-k
`candidate_ids`; old v6 and earlier models are intentionally treated as
obsolete and should be retrained. The current runtime is v8.1. It remains
compatible with v7 model artifacts, but the next trained model should write
`model_version=8` and use the V8 action sample weights. Ambiguous
dictionary/split edits are rescored by `DeepPavlov/rubert-base-cased` on CPU by
default; split edits fail closed when context is not reliable. Dictionary
corrections use `candidate_min_freq=1`, `min_dictionary_score=0.25`, a
`pymorphy2` morphology guard, and an entity/noise guard loaded from
`data/raw/texts.txt` and `data/processed/*.csv` `correct_text` values to avoid
changing valid clean/raw words, names, toponyms and rare terms.

V8 adds cautious dirty-aware recall gates: obvious lowercase OOV dictionary
typos can pass the stricter dictionary threshold when `distance=1`,
`score>=1.0` and rank is top-2; low-score recovery needs very high model
confidence and strong context reranker confirmation; short OOV typos of length
3-4 are allowed only in high-confidence, non-entity contexts. Dynamic short
rules cover repeated first-letter cases such as `ДДля -> Для` and `ККак -> Как`.

Keyboard-neighbor typo generation is intentionally not implemented.

## Main Files

```text
src/hybrid_corrector.py      runtime corrector
src/candidate_generator.py   dictionary and rule candidates
src/context_reranker.py      RuBERT/RuRoBERTa candidate reranker
src/entity_guard.py          protected clean/raw lexicon and entity context
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
  --candidate-top-k 8 \
  --long-oov-max-distance 2 \
  --long-oov-min-length 8 \
  --min-dictionary-score 0.25 \
  --clean-action-keep-weight 2.0 \
  --dirty-action-keep-weight 1.0 \
  --action-change-weight 3.0 \
  --context-model-name DeepPavlov/rubert-base-cased \
  --context-device cpu \
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
  --all \
  --output-dir report \
  --strictness strict \
  --punctuation-mode conservative \
  --candidate-top-k 8 \
  --context-device cpu \
  --min-dictionary-score 0.25
```

For an A/B run without the deterministic spacing layer, add:

```bash
--disable-deterministic-spacing
```

Evaluation writes:

```text
report/summary.json
report/error_analysis.csv
report/error_metrics_by_type.csv
report/word_edit_diagnostics.csv
report/word_decision_diagnostics.csv
report/punct_edit_diagnostics.csv
report/clean_candidate_audit.csv
report/clean_noise_filter.csv
report/worst_cases.csv
```

## Latest Full-Test Snapshot

The latest full notebook evaluation was written to `report/` with
`runtime_version="8.1"` after the next evaluation run.

```text
examples: 5913
exact_match: 0.475731
clean exact_match: 0.998555
dirty exact_match: 0.192859
mean_cer_delta: 0.003985
dirty_slice.mean_cer_delta: 0.006150
worse_rate: 0.003044
clean_overcorrection_rate: 0.001445
word_overcorrection_rate: 0.0
punct_overcorrection_rate: 0.001445
space_overcorrection_rate: 0.0
target_present_but_not_applied_rate: 0.491918
candidate_coverage_rate: 0.802004
deterministic_spacing_applied_count: 518
space_edit_count: 508
reranker_non_finite_count: 0
entity_guard_blocked_count: 1831
context_source_veto_relaxed_count: 5
punct_input_similarity: 0.887929
punct_pred_similarity: 0.892349
```

Interpretation:

```text
V8 keeps clean-safety within the target range.
Deterministic spacing is useful and has no clean spacing overcorrection.
The remaining bottleneck is dirty recall, especially punctuation and glued-word splits.
```

Current V8.1.1 runtime fix:

```text
1. Prefer safe service-token splits for glued words such as сообщаетсяна and незаставляет.
2. Block dictionary candidates that delete a glued service token.
3. Guard reporting-verb comma insertions before о/об/обо/ранее.
4. Watch clean_overcorrection_rate, worse_rate, target_present_but_not_applied_rate,
   deterministic_spacing_applied_count, space_edit_count and space_overcorrection_rate.
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
