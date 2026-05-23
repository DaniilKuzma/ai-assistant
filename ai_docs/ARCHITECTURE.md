# Architecture

## High-Level Flow

1. Text preprocessing: токенизация, sentence split, protected spans.
2. Candidate generation: орфография, split/join, hyphen, dictionary, punctuation candidates.
3. `CorrectionMemory` lookup: поиск ранее принятого, отклоненного или проигнорированного решения для candidate в том же документе и контексте.
4. Model scoring: ruRoberta encoder + LoRA + custom heads оценивают candidates и punctuation gaps.
5. Thresholding: rule/family/edit-type thresholds из `configs/config.yaml`; memory can reuse accepted candidates or suppress rejected/ignored candidates according to config.
6. Validation: `StrictValidator` принимает только изменения в strict scope.
7. Realization: accepted edits применяются к тексту.
8. `CorrectionMemory` feedback update: UI/service явно записывает `accepted/rejected/ignored/manual`; это не training и не изменение весов модели.
9. Optional incremental correction: `IncrementalCorrector` или DOCX incremental path переиспользуют кеш неизмененных сегментов/абзацев.
10. Iteration: correction loop повторяется до стабилизации или `decoder.max_passes`.

## Runtime Paths

Plain fallback:

- `src.inference.corrector.Corrector`
- применяет только conservative deterministic edits;
- не должен применять candidate-only/model-required исправления без scorer.

Model-backed path:

- `src.inference.model_corrector.TrainedModelCorrector`
- грузит adapters из `models/current/adapters`;
- грузит heads из `models/current/heads/heads.pt`;
- выбирает candidates по thresholds;
- может учитывать `CorrectionMemory` при selection, но не меняет веса модели;
- передает trusted edits в validator.

Streamlit path:

- `src/app/streamlit_app.py`
- сначала пытается загрузить trained model;
- при ошибке или отсутствии артефактов показывает/использует rule fallback.
- может включать контекстную память решений и инкрементальную проверку текста через session state.

DOCX path:

- `src/docx/docx_corrector.py`
- читает параграфы;
- исправляет каждый параграф выбранным corrector;
- пишет новый `.docx`, сохраняя базовую структуру и runs.
- `correct_docx_incremental` переиспользует кеш неизмененных абзацев и проверяет только новые/измененные абзацы.

## Training Flow

Entry point: `src/training/train.py`.

Основные шаги:

- загрузить `configs/config.yaml`;
- canonical dataset build читает `data.dataset_contract: candidate_opportunity`;
- `scripts/build_dataset.py` / `scripts/rebuild_training_dataset.py` собирают layered atomic dataset без запуска model training;
- `src.data.operator_dataset_builder` строит слои `atomic_positive`, `atomic_hard_negative`, `clean_identity`, `real_atomic`, `stress_multi_error`;
- atomic positives требуют candidate coverage и `StrictValidator`, а stress rows получают пониженный `loss_weight`;
- manifest пишет `dataset_hash`, `verdict`, `audit_errors`, `layer_counts` и отчеты quota/audit gates;
- старые `training_dataset` / `training_dataset_core` ключи остаются fallback-совместимыми, но top-level `data` keys имеют приоритет;
- training entrypoint берет готовый layered dataset и строит training features;
- при `training.run_model_training: true` обучить encoder+heads;
- сохранить adapters, heads, config, labels, thresholds;
- запустить evaluation и записать reports.

Переменная `RUSSIAN_CORRECTOR_DISABLE_MODEL_TRAINING=1` отключает обучение и оставляет pipeline в no-training/evaluation режиме.

## Validation Boundary

`StrictValidator` — обязательная защита. Даже model-backed prediction не должен напрямую менять текст без validation. Особо чувствительные зоны:

- URL, email, code-like spans;
- числа, проценты, десятичные дроби;
- аббревиатуры;
- кавычки и скобки;
- `-тся/-ться`;
- контекстные пары вроде `также/так же`.

