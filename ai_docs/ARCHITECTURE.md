# Architecture

## High-Level Flow

1. Text preprocessing: токенизация, sentence split, protected spans.
2. Candidate generation: орфография, split/join, hyphen, dictionary, punctuation candidates.
3. Model scoring: ruRoberta encoder + LoRA + custom heads.
4. Thresholding: rule/family/edit-type thresholds из `configs/config.yaml`.
5. Validation: `StrictValidator` принимает только изменения в strict scope.
6. Realization: accepted edits применяются к тексту.
7. Iteration: correction loop повторяется до стабилизации или `decoder.max_passes`.

## Runtime Paths

Plain fallback:

- `src.inference.corrector.Corrector`
- применяет только conservative deterministic edits;
- не должен применять candidate-only/model-required исправления без scorer.

Model-backed path:

- `src.inference.model_corrector.TrainedModelCorrector`
- грузит adapters из `models/adapters/latest`;
- грузит heads из `models/heads/latest/heads.pt`;
- выбирает candidates по thresholds;
- передает trusted edits в validator.

Streamlit path:

- `src/app/streamlit_app.py`
- сначала пытается загрузить trained model;
- при ошибке или отсутствии артефактов показывает/использует rule fallback.

DOCX path:

- `src/docx/docx_corrector.py`
- читает параграфы;
- исправляет каждый параграф выбранным corrector;
- пишет новый `.docx`, сохраняя базовую структуру и runs.

## Training Flow

Entry point: `src/training/train.py`.

Основные шаги:

- загрузить `configs/config.yaml`;
- взять готовый dataset или собрать synthetic fallback;
- построить training features;
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

