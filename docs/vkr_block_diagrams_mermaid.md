# Блок-схемы для ВКР в формате Mermaid

Все схемы ниже используют только стандартные элементы блок-схем:
терминатор, ввод/вывод, процесс, условие, предопределённый процесс и
соединитель. Оформление задано чёрно-белым стилем.

## Схема 1А. GUI: проверка текста

Файл и функция: `src/app/streamlit_app.py`, `main()`.

```mermaid
flowchart TD
    A([Начало]) --> B[[ _import_streamlit() ]]
    B --> C[Создать cached_streamlit_corrector]
    C --> D[/Вывод: page_config, title, tabs Text и DOCX/]
    D --> E[/Ввод: use_incremental, source из text_area/]
    E --> F{Нажата кнопка<br/>и source.strip()?}
    F -- Нет --> R[[ _render_latest_text_correction() ]]
    F -- Да --> G[Получить cache_key и session_state caches]
    G --> H{use_incremental?}
    H -- Да --> I[Получить previous_cache]
    I --> J[[ IncrementalCorrector(corrector).correct_incremental(source, previous_cache) ]]
    J --> K[Обновить previous_text_segment_cache]
    H -- Нет --> L[[ corrector.correct(source) ]]
    K --> M[Записать last_source_text и last_corrected_text]
    L --> M
    M --> N[Собрать latest_text_correction]
    N --> O{incremental_summary есть?}
    O -- Да --> P[Добавить incremental_summary]
    O -- Нет --> Q[Оставить latest_text_correction без summary]
    P --> R
    Q --> R
    R --> S([Конец])

    classDef default fill:#fff,stroke:#000,color:#000;
    linkStyle default stroke:#000,color:#000;
```

## Схема 1Б. GUI: проверка DOCX

Файл и функция: `src/app/streamlit_app.py`, `main()`.

```mermaid
flowchart TD
    A([Начало]) --> B[/Ввод: uploaded из file_uploader/]
    B --> C{uploaded is not None?}
    C -- Нет --> Z([Конец])
    C -- Да --> D[/Ввод: use_docx_incremental/]
    D --> E{docx_correction_requested?}
    E -- Нет --> Z
    E -- Да --> F[[ cached_streamlit_corrector(DEFAULT_CONFIG_PATH) ]]
    F --> G[Создать TemporaryDirectory]
    G --> H[Записать uploaded.getvalue() в input.docx]
    H --> I{use_docx_incremental?}
    I -- Да --> J[Получить paragraph_cache_by_upload и cache_key]
    J --> K[[ correct_docx_incremental(input_path, output_path, corrector, previous_cache) ]]
    K --> L[Обновить paragraph_cache_by_upload]
    L --> M[[ build_docx_incremental_summary(result) ]]
    I -- Нет --> N[[ read_paragraphs(input_path) ]]
    N --> O[[ correct_docx(input_path, output_path, corrector) ]]
    O --> P[Собрать docx_summary]
    M --> Q[/Вывод: st.info(summary)/]
    P --> Q
    Q --> R[/Вывод: download_button(corrected.docx)/]
    R --> S[[ build_edit_rows(edits) ]]
    S --> T[/Вывод: st.dataframe(rows)/]
    T --> Z([Конец])

    classDef default fill:#fff,stroke:#000,color:#000;
    linkStyle default stroke:#000,color:#000;
```

## Схема 2. Online-генерация примера

Файл и метод: `src/grammar_gen/generator.py`,
`OnlineExampleGenerator._sample_with_rng()`.

```mermaid
flowchart TD
    A([Начало]) --> B[/Ввод: rng, builder, requested_mode, rule_id, generation_index, generation_seed/]
    B --> C[retries = _max_generation_retries()]
    C --> D[Инициализировать failures и last_*]
    D --> E{rule_id задан<br/>и правило не найдено?}
    E -- Да --> X([Конец: GenerationError])
    E -- Нет --> F[attempt = 1]
    F --> G{attempt <= retries?}
    G -- Нет --> H[Собрать details из failures]
    H --> X
    G -- Да --> I[[ _select_rule_and_mode(requested_mode, rule_id, rng) ]]
    I --> J[Сохранить last_rule_id и last_mode]
    J --> K[[ rule.generate(builder, realizer, rng, selected_mode) ]]
    K --> L[[ _validate_example(example, rule, selected_mode) ]]
    L --> M[[ _with_generation_metadata(validated, generation_index, generation_seed) ]]
    M --> N[/Вывод: GeneratedExample/]
    N --> O([Конец])
    I -. _GeneratedPairValidationError .-> P[Сохранить last source, target и validation failures]
    K -. Exception .-> Q[Добавить type и текст exception в failures]
    L -. _GeneratedPairValidationError .-> P
    P --> R[attempt = attempt + 1]
    Q --> R
    R --> G

    classDef default fill:#fff,stroke:#000,color:#000;
    linkStyle default stroke:#000,color:#000;
```

## Схема 2Б. Выбор правила и режима

Файл и метод: `src/grammar_gen/generator.py`,
`OnlineExampleGenerator._select_rule_and_mode()`.

```mermaid
flowchart TD
    A([Начало]) --> B[/Ввод: requested_mode, rule_id, rng/]
    B --> C{rule_id is not None?}
    C -- Да --> D[[ registry.get_rule(rule_id) ]]
    D --> E{rule is None?}
    E -- Да --> X([Конец: GenerationError])
    E -- Нет --> F{requested_mode is not None?}
    F -- Да --> G{rule.can_generate(requested_mode)?}
    G -- Нет --> X
    G -- Да --> H[/Вывод: rule, requested_mode/]
    F -- Нет --> I[[ _sample_mode_for_rule(rule, rng) ]]
    I --> J[/Вывод: rule, sampled_mode/]
    C -- Нет --> K{requested_mode is not None?}
    K -- Да --> L[[ _select_rule_for_mode(requested_mode, family=None, rng) ]]
    L --> M[/Вывод: rule, requested_mode/]
    K -- Нет --> N[[ _sample_mode_from_mix(rng) ]]
    N --> O[[ _select_rule_for_mode(mode, family, rng) ]]
    O --> P[/Вывод: rule, mode/]
    H --> Z([Конец])
    J --> Z
    M --> Z
    P --> Z

    classDef default fill:#fff,stroke:#000,color:#000;
    linkStyle default stroke:#000,color:#000;
```

## Схема 3А. Формализация GeneratedExample в признаки модели

Файл и функция: `src/training/tensorization.py`,
`build_direct_training_feature()`.

```mermaid
flowchart TD
    A([Начало]) --> B[/Ввод: GeneratedExample, tokenizer, max_length/]
    B --> C[[ _encode(tokenizer, example.source_text, max_length) ]]
    C --> D[Создать массивы word_token_indices, word_token_mask, token_edit_label_ids, rule_tag_ids]
    D --> E[ source_tokens = example.source_tokens[:max_words] ]
    E --> F[index = 0]
    F --> G{index < len(source_tokens)?}
    G -- Нет --> H[Создать массивы gap_left_indices, gap_right_indices, gap_mask, gap_label_ids]
    G -- Да --> I[[ _first_overlapping_token_index(offsets, word.start, word.end) ]]
    I --> J{token_index is None?}
    J -- Да --> K[index = index + 1]
    J -- Нет --> L[Записать word_token_index, mask, token_edit_label_id, rule_tag_id]
    L --> K
    K --> G
    H --> M[gap_index = 0]
    M --> N{gap_index < len(source_tokens)?}
    N -- Нет --> Y[/Вывод: DirectTrainingFeature/]
    Y --> Z([Конец])
    N -- Да --> O{word_token_mask[gap_index]?}
    O -- Нет --> P[gap_index = gap_index + 1]
    O -- Да --> Q[gap_left_indices = word_token_indices]
    Q --> R{Есть следующий размеченный word token?}
    R -- Да --> S[gap_right_indices = word_token_indices[next]]
    R -- Нет --> T[gap_right_indices = -1]
    S --> U[gap_mask = True; gap_label_id = gap_label_to_id]
    T --> U
    U --> P
    P --> N

    classDef default fill:#fff,stroke:#000,color:#000;
    linkStyle default stroke:#000,color:#000;
```

## Схема 3Б. OnlineGrammarDataset

Файл и метод: `src/training/online_dataset.py`,
`OnlineGrammarDataset.__iter__()`.

```mermaid
flowchart TD
    A([Начало]) --> B[[ get_worker_info() ]]
    B --> C[Вычислить worker_id и worker_count]
    C --> D[[ online_generator_from_config(config, seed) ]]
    D --> E{samples_per_epoch is None?}
    E -- Да --> F[index = worker_id]
    F --> G{while True}
    G --> H[[ generator.sample_by_index(index) ]]
    H --> I[[ build_direct_training_feature(example, tokenizer, max_length) ]]
    I --> J[/Вывод: yield feature/]
    J --> K[index = index + worker_count]
    K --> G
    E -- Нет --> L[epoch_start = _epoch * samples_per_epoch]
    L --> M[local_index = worker_id]
    M --> N{local_index < samples_per_epoch?}
    N -- Нет --> O[_epoch = _epoch + 1]
    O --> Z([Конец])
    N -- Да --> P[[ generator.sample_by_index(epoch_start + local_index) ]]
    P --> Q[[ build_direct_training_feature(example, tokenizer, max_length) ]]
    Q --> R[/Вывод: yield feature/]
    R --> S[local_index = local_index + worker_count]
    S --> N

    classDef default fill:#fff,stroke:#000,color:#000;
    linkStyle default stroke:#000,color:#000;
```

## Схема 4А. Генерация через YAML RuleLayer

Файл и метод: `src/rule_layers/direct_cases.py`,
`DirectCasesLayer.generate()`.

```mermaid
flowchart TD
    A([Начало]) --> B[/Ввод: builder, realizer, rng, mode/]
    B --> C[Удалить builder; выбрать cases по mode]
    C --> D{cases пуст?}
    D -- Да --> X([Конец: ValueError])
    D -- Нет --> E[[ rng.weighted_choice(cases с весами) ]]
    E --> F[iteration = 0]
    F --> G{iteration < 6?}
    G -- Нет --> H[[ build_generated_example_from_case(selected, realizer, layer) ]]
    H --> I[/Вывод: GeneratedExample/]
    I --> Z([Конец])
    G -- Да --> J[[ contextualize_case(selected, rng) ]]
    J --> K[[ build_generated_example_from_case(varied, realizer, layer) ]]
    K --> L[[ validate_generated_pair(example) ]]
    L --> M{Причин валидации нет?}
    M -- Да --> N[/Вывод: GeneratedExample/]
    N --> Z
    M -- Нет --> O[iteration = iteration + 1]
    O --> G
    K -. ValueError .-> P{varied == selected?}
    P -- Да --> X
    P -- Нет --> O

    classDef default fill:#fff,stroke:#000,color:#000;
    linkStyle default stroke:#000,color:#000;
```

## Схема 4Б. Сборка GeneratedExample из LayerDirectCase

Файл и функция: `src/rule_layers/example_builders.py`,
`build_generated_example_from_case()`.

```mermaid
flowchart TD
    A([Начало]) --> B[/Ввод: case, realizer, layer/]
    B --> C[[ realizer.tokenize_words_with_offsets(case.source_text) ]]
    C --> D{tokens пуст?}
    D -- Да --> X([Конец: ValueError])
    D -- Нет --> E[[ _initial_token_labels(case, tokens) ]]
    E --> F[[ _initial_gap_labels(case, tokens) ]]
    F --> G[rule_ids = список none]
    G --> H[token_op_index = 0]
    H --> I{token_op_index < len(case.token_operations)?}
    I -- Да --> J[[ _apply_token_operation(operation, case, realizer, tokens, token_labels, rule_ids) ]]
    J --> K[Добавить operation name, source_pattern, target_pattern]
    K --> L[token_op_index = token_op_index + 1]
    L --> I
    I -- Нет --> M[gap_op_index = 0]
    M --> N{gap_op_index < len(case.gap_operations)?}
    N -- Да --> O[[ _apply_gap_operation(operation, case, realizer, tokens, gap_labels, rule_ids) ]]
    O --> P[Добавить operation name, source_pattern, target_pattern]
    P --> Q[gap_op_index = gap_op_index + 1]
    Q --> N
    N -- Нет --> R[[ _apply_direct_rule_ids(case, token_labels, gap_labels, rule_ids) ]]
    R --> S{case.rule_id отсутствует в rule_ids?}
    S -- Да --> T[rule_ids[0] = case.rule_id]
    S -- Нет --> U[[ _validate_lengths(tokens, token_labels, gap_labels, rule_ids) ]]
    T --> U
    U --> V[[ _metadata(case, layer, operation_names, source_patterns, target_patterns) ]]
    V --> W[/Вывод: GeneratedExample/]
    W --> Z([Конец])

    classDef default fill:#fff,stroke:#000,color:#000;
    linkStyle default stroke:#000,color:#000;
```

## Схема 5. Морфемный слой

Файл и метод: `src/orthography_gen/compiler.py`,
`OrthographicScenarioCompiler.compile_example()`.

```mermaid
flowchart TD
    A([Начало]) --> B[/Ввод: rule_id, mode, rng/]
    B --> C[[ _coerce_mode(mode) ]]
    C --> D[[ _cards_for(rule_id, generation_mode) ]]
    D --> E[[ rng.choice(cards) ]]
    E --> F[[ rng.choice(available_forms(card)) ]]
    F --> G{generation_mode == HARD_NEGATIVE?}
    G -- Да --> H[[ injector.make_hard_negative(card, form_key) ]]
    H --> I[[ _choose_context(card.hard_negative_contexts, rng) ]]
    G -- Нет --> J[[ injector.make_positive(card, form_key) ]]
    J --> K[[ _choose_context(card.safe_contexts, rng) ]]
    I --> L[Заполнить context.setdefault(...)]
    K --> L
    L --> M[[ wrapper.wrap(source_word, target_word, pos, context) ]]
    M --> N[labels = KEEP по числу wrapped.source_tokens]
    N --> O{generation_mode == POSITIVE?}
    O -- Да --> P[labels[edited_token_index] = DICT_REPLACE]
    O -- Нет --> Q[[ _metadata(card, context, construction_id, source, target, form_key) ]]
    P --> Q
    Q --> R[Собрать GeneratedExample]
    R --> S[[ contextualize_example(example, rng) ]]
    S --> T[/Вывод: GeneratedExample/]
    T --> Z([Конец])

    classDef default fill:#fff,stroke:#000,color:#000;
    linkStyle default stroke:#000,color:#000;
```

## Схема 6. Forward модели прямой разметки

Файл и метод: `src/model/edit_model.py`,
`DirectEditTaggerModel._build_module()._Module.forward()`.

```mermaid
flowchart TD
    A([Начало]) --> B[/Ввод: input_ids, attention_mask, word/gap indices и masks/]
    B --> C[[ encoder(input_ids, attention_mask) ]]
    C --> D[hidden = output.last_hidden_state; batch_size = hidden.shape[0]]
    D --> E{word_token_indices is None?}
    E -- Да --> F[Создать пустой tensor word_token_indices]
    E -- Нет --> G[[ _gather_hidden(hidden, word_token_indices) ]]
    F --> G
    G --> H[[ head token_edit(word_representations) ]]
    H --> I[[ head token_confidence(word_representations) ]]
    I --> J[[ head rule(word_representations) ]]
    J --> K[[ _mask_or_ones(word_token_mask, word_token_indices, hidden.device) ]]
    K --> L[Занулить token_edit_logits, token_confidence_logits, rule_logits по mask]
    L --> M{gap_left_indices is None?}
    M -- Да --> N[Создать пустой tensor gap_left_indices]
    M -- Нет --> O[[ _gap_representations(hidden, gap_left_indices, gap_right_indices) ]]
    N --> O
    O --> P[[ head gap_punctuation(gap_representations) ]]
    P --> Q[[ head gap_confidence(gap_representations) ]]
    Q --> R[[ _mask_or_ones(gap_mask, gap_left_indices, hidden.device) ]]
    R --> S[Занулить gap logits по mask]
    S --> T[/Вывод: hidden_states, token/gap/rule/confidence logits/]
    T --> Z([Конец])

    classDef default fill:#fff,stroke:#000,color:#000;
    linkStyle default stroke:#000,color:#000;
```

## Схема 7А. Runtime-корректор

Файл и метод: `src/runtime/corrector.py`, `Corrector.correct()`.

```mermaid
flowchart TD
    A([Начало]) --> B[/Ввод: text/]
    B --> C[[ _runtime_metadata() ]]
    C --> D{not text?}
    D -- Да --> E[/Вывод: CorrectionResult без правок/]
    E --> Z([Конец])
    D -- Нет --> F[source_text = text; current = text; edits = empty]
    F --> G[[ _apply_deterministic(current) ]]
    G --> H[edits.extend(deterministic_edits)]
    H --> I[[ _apply_deterministic_lexicon(current) ]]
    I --> J[edits.extend(lexicon_edits)]
    J --> K{neural_backend is not None?}
    K -- Да --> L[[ _apply_neural_token_edits(current) ]]
    L --> M[edits.extend(token_edits)]
    M --> N[[ _apply_neural_gap_edits(current) ]]
    N --> O[edits.extend(gap_edits)]
    K -- Нет --> P[[ scope_guard.validate_result(source_text, current, edits) ]]
    O --> P
    P --> Q{ok?}
    Q -- Нет --> R[metadata scope_guard_rejections = reasons]
    R --> S[/Вывод: CorrectionResult с исходным текстом и пустыми edits/]
    S --> Z
    Q -- Да --> T[[ attach_explanations(edits) ]]
    T --> U{reasons есть?}
    U -- Да --> V[metadata scope_guard_warnings = reasons]
    U -- Нет --> W[metadata без warnings]
    V --> X[/Вывод: CorrectionResult(source_text, current, edits, metadata)/]
    W --> X
    X --> Z

    classDef default fill:#fff,stroke:#000,color:#000;
    linkStyle default stroke:#000,color:#000;
```

## Схема 7Б. Детерминированный проход

Файл и метод: `src/runtime/corrector.py`, `Corrector._apply_deterministic()`.

```mermaid
flowchart TD
    A([Начало]) --> B[/Ввод: text/]
    B --> C[current = text; accepted = empty]
    C --> D[pass_index = 0]
    D --> E{pass_index < max_passes?}
    E -- Нет --> Y[/Вывод: current, accepted/]
    Y --> Z([Конец])
    E -- Да --> F[[ deterministic_engine.propose_edits(current) ]]
    F --> G[[ scope_guard.validate_edit(current, edit) для каждого edit ]]
    G --> H[proposed = прошедшие validate_edit]
    H --> I{proposed пуст?}
    I -- Да --> Y
    I -- Нет --> J[[ apply_runtime_edits(current, proposed) ]]
    J --> K[accepted.extend(proposed)]
    K --> L{updated == current?}
    L -- Да --> Y
    L -- Нет --> M[current = updated]
    M --> N[pass_index = pass_index + 1]
    N --> E

    classDef default fill:#fff,stroke:#000,color:#000;
    linkStyle default stroke:#000,color:#000;
```

## Схема 7В. Отбор нейросетевых token-меток

Файл и метод: `src/runtime/corrector.py`, `Corrector._gate_token_labels()`.

```mermaid
flowchart TD
    A([Начало]) --> B[/Ввод: text, tokens, labels, confidences, margins, rule_ids/]
    B --> C[accepted = KEEP по числу tokens; consumed = empty]
    C --> D[index = 0]
    D --> E{index < len(labels[:len(tokens)])?}
    E -- Нет --> Y[/Вывод: accepted/]
    Y --> Z([Конец])
    E -- Да --> F{index in consumed<br/>или label KEEP/SKIP_MERGED?}
    F -- Да --> G[index = index + 1]
    F -- Нет --> H[rule_id = _rule_id; edit_type = token_edit_type_for_label]
    H --> I[[ thresholds.should_apply(confidence, margin, rule_id, edit_type) ]]
    I --> J{Применять?}
    J -- Нет --> G
    J -- Да --> K[Создать candidate_labels с одной активной меткой]
    K --> L{_label_consumes_next(label)<br/>и есть следующий token?}
    L -- Да --> M[candidate_labels[index+1] = SKIP_MERGED]
    L -- Нет --> N[[ apply_token_edit_labels(...) ]]
    M --> N
    N --> O{candidate_edits есть<br/>и все validate_edit?}
    O -- Нет --> G
    O -- Да --> P[accepted[index] = label]
    P --> Q[[ _accepted_consumed_indexes(tokens, index, label, candidate_edits) ]]
    Q --> R[consumed.update(edit_consumed)]
    R --> S[Для consumed_index != index поставить SKIP_MERGED]
    S --> G
    G --> E

    classDef default fill:#fff,stroke:#000,color:#000;
    linkStyle default stroke:#000,color:#000;
```

## Схема 8А. Применение token edit labels

Файл и функция: `src/runtime/edit_realizer.py`,
`apply_token_edit_labels()`.

```mermaid
flowchart TD
    A([Начало]) --> B[/Ввод: text, tokens, labels, confidences, threshold, rule_ids, orthographic_lexicon/]
    B --> C[edits = empty; consumed = empty; count = min(lengths)]
    C --> D[index = 0]
    D --> E{index < count?}
    E -- Нет --> F[[ apply_runtime_edits(text, edits) ]]
    F --> G[/Вывод: corrected_text, edits/]
    G --> Z([Конец])
    E -- Да --> H{index in consumed?}
    H -- Да --> I[index = index + 1]
    H -- Нет --> J[label = labels[index]]
    J --> K{label KEEP или SKIP_MERGED?}
    K -- Да --> I
    K -- Нет --> L[confidence = confidences[index]]
    L --> M{confidence < threshold?}
    M -- Да --> I
    M -- Нет --> N[[ _token_edit_for_label(...) ]]
    N --> O{edit is None?}
    O -- Да --> I
    O -- Нет --> P[edits.append(edit)]
    P --> Q[[ _consumed_indexes(tokens, index, label, edit) ]]
    Q --> R[consumed.update(...)]
    R --> I
    I --> E

    classDef default fill:#fff,stroke:#000,color:#000;
    linkStyle default stroke:#000,color:#000;
```

## Схема 8Б. Применение gap labels

Файл и функция: `src/runtime/edit_realizer.py`, `apply_gap_labels()`.

```mermaid
flowchart TD
    A([Начало]) --> B[/Ввод: text, tokens, gap_labels, confidences, threshold, rule_ids/]
    B --> C[edits = empty; count = min(lengths)]
    C --> D[index = 0]
    D --> E{index < count?}
    E -- Нет --> F[[ apply_runtime_edits(text, edits) ]]
    F --> G[/Вывод: corrected_text, edits/]
    G --> Z([Конец])
    E -- Да --> H[label = gap_labels[index]]
    H --> I{label == NONE?}
    I -- Да --> J[index = index + 1]
    I -- Нет --> K[confidence = confidences[index]]
    K --> L{confidence < threshold?}
    L -- Да --> J
    L -- Нет --> M{label == DELETE_PUNCTUATION?}
    M -- Да --> N[[ _delete_punctuation_edit(text, tokens[index], confidence, rule_id) ]]
    N --> O{edit is not None?}
    O -- Да --> P[edits.append(edit)]
    O -- Нет --> J
    P --> J
    M -- Нет --> Q[punctuation = PUNCTUATION_BY_LABEL.get(label)]
    Q --> R{punctuation is None?}
    R -- Да --> J
    R -- Нет --> S[[ _gap_edit_for_label(text, token, label, punctuation, confidence, rule_id) ]]
    S --> T{edit is not None?}
    T -- Да --> U[edits.append(edit)]
    T -- Нет --> J
    U --> J
    J --> E

    classDef default fill:#fff,stroke:#000,color:#000;
    linkStyle default stroke:#000,color:#000;
```

## Схема 8В. Применение RuntimeEdit к строке

Файл и функция: `src/runtime/edit_realizer.py`, `apply_runtime_edits()`.

```mermaid
flowchart TD
    A([Начало]) --> B[/Ввод: text, edits/]
    B --> C[result = text]
    C --> D[sorted_edits = edits, отсортированные по start/end/index в обратном порядке]
    D --> E[index = 0]
    E --> F{index < len(sorted_edits)?}
    F -- Нет --> Y[/Вывод: result/]
    Y --> Z([Конец])
    F -- Да --> G[item = sorted_edits[index]]
    G --> H[start = граница 0..len(result)]
    H --> I[end = граница start..len(result)]
    I --> J[result = result[:start] + item.replacement + result[end:]]
    J --> K[index = index + 1]
    K --> F

    classDef default fill:#fff,stroke:#000,color:#000;
    linkStyle default stroke:#000,color:#000;
```

## Схема 9А. Проверка отдельной правки ScopeGuard

Файл и метод: `src/runtime/scope_guard.py`, `ScopeGuard.validate_edit()`.

```mermaid
flowchart TD
    A([Начало]) --> B[/Ввод: source_text, edit/]
    B --> C{edit_type не разрешён?}
    C -- Да --> N[/Вывод: False/]
    C -- Нет --> D{start/end вне границ?}
    D -- Да --> N
    D -- Нет --> E{edit.source задан<br/>и span не совпадает?}
    E -- Да --> N
    E -- Нет --> F{source == replacement?}
    F -- Да --> N
    F -- Нет --> G{Удаляется слово<br/>и edit_type не punctuation?}
    G -- Да --> N
    G -- Нет --> H{Добавляется слово<br/>из пустого source?}
    H -- Да --> N
    H -- Нет --> I{rule_id == spacing_normalization?}
    I -- Да --> O[/Вывод: word_sequence(source) == word_sequence(replacement)/]
    I -- Нет --> J{edit_type split_join или hyphen?}
    J -- Да --> P[/Вывод: letters_only(source) == letters_only(replacement)/]
    J -- Нет --> K{edit_type == casing?}
    K -- Да --> Q[/Вывод: source.lower() == replacement.lower()/]
    K -- Нет --> L{edit_type == punctuation?}
    L -- Да --> R[/Вывод: not _adds_word(edit)/]
    L -- Нет --> M{edit_type == spelling?}
    M -- Да --> S[[ _local_spelling_replacement_is_safe(edit) ]]
    S --> T[/Вывод: результат проверки/]
    M -- Нет --> U[/Вывод: True/]
    N --> Z([Конец])
    O --> Z
    P --> Z
    Q --> Z
    R --> Z
    T --> Z
    U --> Z

    classDef default fill:#fff,stroke:#000,color:#000;
    linkStyle default stroke:#000,color:#000;
```

## Схема 9Б. Проверка результата ScopeGuard

Файл и метод: `src/runtime/scope_guard.py`, `ScopeGuard.validate_result()`.

```mermaid
flowchart TD
    A([Начало]) --> B[/Ввод: source_text, corrected_text, edits/]
    B --> C[reasons = empty]
    C --> D[index = 0]
    D --> E{index < len(edits)?}
    E -- Да --> F[[ validate_edit(source_text, edit) ]]
    F --> G{edit валиден?}
    G -- Нет --> H[Добавить invalid_edit reason]
    G -- Да --> I[index = index + 1]
    H --> I
    I --> E
    E -- Нет --> J{corrected_text слишком длинный?}
    J -- Да --> K[Добавить result_too_long]
    J -- Нет --> L{semantic word delta слишком большой?}
    K --> L
    L -- Да --> M[Добавить semantic_word_delta]
    L -- Нет --> N{reasons пуст?}
    M --> N
    N -- Да --> O[/Вывод: True, empty reasons/]
    O --> Z([Конец])
    N -- Нет --> P[hard_reasons = фильтр reasons]
    P --> Q{hard_reasons есть?}
    Q -- Да --> R[/Вывод: False, reasons/]
    R --> Z
    Q -- Нет --> S[[ apply_runtime_edits(source_text, edits) ]]
    S --> T{projected == corrected_text?}
    T -- Да --> U[/Вывод: True, reasons/]
    T -- Нет --> V[/Вывод: True, reasons + projection_mismatch_stage_relative_offsets/]
    U --> Z
    V --> Z

    classDef default fill:#fff,stroke:#000,color:#000;
    linkStyle default stroke:#000,color:#000;
```

## Схема 9В. Валидация GeneratedExample

Файл и функция: `src/grammar_gen/safety.py`, `validate_generated_pair()`.

```mermaid
flowchart TD
    A([Начало]) --> B[/Ввод: example/]
    B --> C[reasons = empty]
    C --> D[[ validate_surface(example.source_text) ]]
    D --> E[[ validate_surface(example.target_text) ]]
    E --> F[[ _validate_target_morphology(example) ]]
    F --> G[[ allowed_source_surface_failures(example) ]]
    G --> H[real_source_reasons = source_reasons без allowed_source]
    H --> I[Добавить real_source_reasons и target_reasons в reasons]
    I --> J[[ _validate_metadata_json_safety(example.metadata) ]]
    J --> K[[ _validate_construction_metadata(example) ]]
    K --> L[[ _validate_token_edit_counts(example) ]]
    L --> M[[ _validate_safety_clause_metadata(example) ]]
    M --> N{allow_repeated_content_words == True?}
    N -- Да --> O[[ _dedupe(reasons) ]]
    N -- Нет --> P[[ _repeated_content_word_reasons(source_text) ]]
    P --> Q[[ _repeated_content_word_reasons(target_text) ]]
    Q --> R[Добавить source_ и target_ причины повторов]
    R --> O
    O --> S[/Вывод: список причин/]
    S --> Z([Конец])

    classDef default fill:#fff,stroke:#000,color:#000;
    linkStyle default stroke:#000,color:#000;
```
