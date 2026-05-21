# Syntax Required Inventory

This audit reads `configs/rules.yaml` only. It does not build a dataset, train, change thresholds, or change checkpoints.

## Summary

- total syntax_required entries: 57
- total leaf syntax_required entries: 57
- syntax families: 25 total, 5 nonzero
- IMPLEMENT_NOW: 0
- IMPLEMENT_WITH_BOUNDED_PATTERN: 12
- blocked/deferred: 45
- expected training_eligible gain: 12

## Family Summary

| syntax_family | total | implement_now | bounded | blocked | priority |
| --- | ---: | ---: | ---: | ---: | ---: |
| subordinate_clause_comma | 0 | 0 | 0 | 0 | 4 |
| coordinating_conjunction_comma | 0 | 0 | 0 | 0 | 4 |
| introductory_words | 0 | 0 | 0 | 0 | 4 |
| address_comma | 0 | 0 | 0 | 0 | 4 |
| homogeneous_members | 0 | 0 | 0 | 0 | 4 |
| detached_adverbial_phrases | 0 | 0 | 0 | 0 | 4 |
| detached_participial_phrases | 0 | 0 | 0 | 0 | 4 |
| detached_applications | 0 | 0 | 0 | 0 | 4 |
| clarification_members | 0 | 0 | 0 | 0 | 4 |
| comparative_turnovers | 2 | 0 | 0 | 2 | 4 |
| subject_predicate_dash | 0 | 0 | 0 | 0 | 4 |
| asyndetic_complex_sentence | 0 | 0 | 0 | 0 | 4 |
| consequence_dash | 0 | 0 | 0 | 0 | 4 |
| explanation_colon | 0 | 0 | 0 | 0 | 4 |
| enumeration_colon_dash | 0 | 0 | 0 | 0 | 4 |
| semicolon | 0 | 0 | 0 | 0 | 4 |
| direct_speech_syntax | 0 | 0 | 0 | 0 | 4 |
| quote_bracket_balance | 0 | 0 | 0 | 0 | 4 |
| punctuation_combinations | 11 | 0 | 0 | 11 | 4 |
| ne_with_parts_of_speech | 15 | 0 | 7 | 8 | 3 |
| ni_stable_and_context | 0 | 0 | 0 | 0 | 4 |
| n_nn_context | 0 | 0 | 0 | 0 | 4 |
| tsya_tsya_context | 0 | 0 | 0 | 0 | 4 |
| context_pairs | 22 | 0 | 5 | 17 | 3 |
| grammatical_endings_context | 7 | 0 | 0 | 7 | 4 |

## Inventory

| matrix_key | title | syntax_family | recommended_action | reason |
| --- | --- | --- | --- | --- |
| orthography_1_2_7_1 | Написание букв в окончаниях по формам слов с тем же окончанием | grammatical_endings_context | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| orthography_1_2_7_2 | Написание в форме им. п. ед. ч. муж. рода прилагательных / слов, склоняющихся как прилагательные | grammatical_endings_context | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| orthography_1_2_7_3 | Окончания, содержащие гласные, не проверяемые ударной позицией | grammatical_endings_context | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| orthography_1_2_7_4 | Безударные окончания у существительных с некоторыми суффиксами | grammatical_endings_context | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| orthography_1_2_7_5 | Падежные формы существительных на -ий, -ие, -ия | grammatical_endings_context | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| orthography_1_2_7_6 | Предлоги «в», «на», «по» с существительными на -ие, -ии | grammatical_endings_context | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| orthography_3_2_3_1 | Наречия и прилагательные или причастия | context_pairs | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| orthography_3_4_1 | Слитное написание местоименных слов | context_pairs | IMPLEMENT_WITH_BOUNDED_PATTERN | context_pairs can be represented as candidate-backed local edits with explicit syntax features, model scoring, validator guards, and no broad rewrite. |
| orthography_3_4_3_2 | Сочетания с местоименными словами | context_pairs | IMPLEMENT_WITH_BOUNDED_PATTERN | context_pairs can be represented as candidate-backed local edits with explicit syntax features, model scoring, validator guards, and no broad rewrite. |
| orthography_3_5_1_1 | Наречия, образованные с помощью приставок от наречий | context_pairs | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| orthography_3_5_1_2 | Наречия, образованные от имён прилагательных | context_pairs | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| orthography_3_5_1_3 | Наречия, образованные от местоименных слов | context_pairs | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| orthography_3_5_1_4 | Наречия, образованные с помощью приставок «в», «на» от собирательных числительных | context_pairs | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| orthography_3_5_1_5 | Наречия с первой частью «впол-» | context_pairs | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| orthography_3_5_1_6 | Наречия с пространственным и временным значениями | context_pairs | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| orthography_3_5_1_7 | Слитное написание сочетаний с предлогами-приставками | context_pairs | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| orthography_3_5_2_1 | Раздельное написание сочетаний с предлогами-приставками | context_pairs | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| orthography_3_5_2_2 | Наречные сочетания из предлога-приставки «с» и формы род. п. существительного на «-у (-а)» | context_pairs | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| orthography_3_5_2_3 | Наречные сочетания предлогов-приставок «в», «с» со второй частью, начинающейся с гласных | context_pairs | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| orthography_3_5_2_4 | Наречные сочетания, образованные повторением существительных или числительных с предлогом | context_pairs | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| orthography_3_5_2_5 | Наречные сочетания с существительным в разных падежах с предлогами | context_pairs | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| orthography_3_5_3_1 | Дефисное написание наречий | context_pairs | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| orthography_3_5_4 | Написание отрицательных наречий | ne_with_parts_of_speech | IMPLEMENT_WITH_BOUNDED_PATTERN | ne_with_parts_of_speech can be represented as candidate-backed local edits with explicit syntax features, model scoring, validator guards, and no broad rewrite. |
| orthography_3_6_1_2 | Союзы и частицы из сочетаний предлогов и союзов с местоименными словами | context_pairs | IMPLEMENT_WITH_BOUNDED_PATTERN | context_pairs can be represented as candidate-backed local edits with explicit syntax features, model scoring, validator guards, and no broad rewrite. |
| orthography_3_6_1_4 | Слитное написание междометий и звукоподражаний | context_pairs | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| orthography_3_6_2_3 | Слитное написание междометий и звукоподражаний | context_pairs | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| orthography_3_6_2_4 | Написание междометий и звукоподражаний через дефис | context_pairs | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| orthography_3_6_4_1 | Дефисное написание сочетаний с частицами | context_pairs | IMPLEMENT_WITH_BOUNDED_PATTERN | context_pairs can be represented as candidate-backed local edits with explicit syntax features, model scoring, validator guards, and no broad rewrite. |
| orthography_3_6_4_2 | Раздельное написание сочетаний с частицами | context_pairs | IMPLEMENT_WITH_BOUNDED_PATTERN | context_pairs can be represented as candidate-backed local edits with explicit syntax features, model scoring, validator guards, and no broad rewrite. |
| orthography_3_7_1_3 | Приставки «недо-» и «небез- (небес-)» | ne_with_parts_of_speech | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| orthography_3_7_1_4_2 | Непринадлежность к разряду лиц или явлений в сочетаниях с «не» | ne_with_parts_of_speech | NEEDS_SEMANTIC_MODEL | Local syntax can propose probes, but production candidates need semantic disambiguation beyond bounded patterns. |
| orthography_3_7_1_4_3 | Определение или предлог перед существительным с «не» | ne_with_parts_of_speech | NEEDS_SEMANTIC_MODEL | Local syntax can propose probes, but production candidates need semantic disambiguation beyond bounded patterns. |
| orthography_3_7_1_4_4 | Наличие слов типа «очень», «крайне», «весьма» | ne_with_parts_of_speech | IMPLEMENT_WITH_BOUNDED_PATTERN | ne_with_parts_of_speech can be represented as candidate-backed local edits with explicit syntax features, model scoring, validator guards, and no broad rewrite. |
| orthography_3_7_1_4_5 | Наличие уточняющих наречий | ne_with_parts_of_speech | IMPLEMENT_WITH_BOUNDED_PATTERN | ne_with_parts_of_speech can be represented as candidate-backed local edits with explicit syntax features, model scoring, validator guards, and no broad rewrite. |
| orthography_3_7_1_5 | Слитное написание с полными формами причастий | ne_with_parts_of_speech | IMPLEMENT_WITH_BOUNDED_PATTERN | ne_with_parts_of_speech can be represented as candidate-backed local edits with explicit syntax features, model scoring, validator guards, and no broad rewrite. |
| orthography_3_7_2_2 | Написание с числительными и счётными существительными | ne_with_parts_of_speech | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| orthography_3_7_2_3 | Написание с местоименными словами | ne_with_parts_of_speech | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| orthography_3_7_2_4 | Написание со всегда краткими прилагательными | ne_with_parts_of_speech | NEEDS_SEMANTIC_MODEL | Local syntax can propose probes, but production candidates need semantic disambiguation beyond bounded patterns. |
| orthography_3_7_2_5 | Написание с наречиями и неизменяемыми словами в роли сказуемого | ne_with_parts_of_speech | NEEDS_SEMANTIC_MODEL | Local syntax can propose probes, but production candidates need semantic disambiguation beyond bounded patterns. |
| orthography_3_7_2_6 | Написание с любыми словами, пишущимися через дефис | ne_with_parts_of_speech | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| orthography_3_7_2_8 | Конструкции с противопоставлением | ne_with_parts_of_speech | IMPLEMENT_WITH_BOUNDED_PATTERN | ne_with_parts_of_speech can be represented as candidate-backed local edits with explicit syntax features, model scoring, validator guards, and no broad rewrite. |
| orthography_3_7_2_9 | Конструкции, усиливающие отрицание | ne_with_parts_of_speech | IMPLEMENT_WITH_BOUNDED_PATTERN | ne_with_parts_of_speech can be represented as candidate-backed local edits with explicit syntax features, model scoring, validator guards, and no broad rewrite. |
| orthography_3_7_2_10 | Сочетания «едва ли не…, чуть ли не…, разве не…, не далее как…, не позже чем…, не раньше чем…» | ne_with_parts_of_speech | IMPLEMENT_WITH_BOUNDED_PATTERN | ne_with_parts_of_speech can be represented as candidate-backed local edits with explicit syntax features, model scoring, validator guards, and no broad rewrite. |
| orthography_4_11_2 | Местоимения «Вы», «Ваш» | grammatical_endings_context | BLOCK_FOR_NOW | This capitalization choice is stylistic/formality-sensitive and is not safe as strict spelling or punctuation correction. |
| punctuation_5_7_1 | Неразложимые сочетания с подчинительными союзами и союзными словами | punctuation_combinations | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| punctuation_5_7_2 | Неразложимые сочетания с частицами | punctuation_combinations | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| punctuation_5_7_3 | Выражения с глаголом «хотеть» | punctuation_combinations | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| punctuation_5_7_4 | Неразложимые сочетания с местоименными и наречными словами | punctuation_combinations | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| punctuation_5_7_5 | Неразложимые сочетания с союзом «чем» | punctuation_combinations | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| punctuation_5_7_6 | Неразложимые сочетания «не кто иной, как» и «не что иное, как» | punctuation_combinations | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| punctuation_5_7_7 | Запятая внутри выражений | punctuation_combinations | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| punctuation_5_7_8 | Сочетания типа «кто ни на есть» | punctuation_combinations | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| punctuation_5_7_9 | Союз «что» в выражении «только и…что» | punctuation_combinations | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| punctuation_5_7_10 | Тире в цельных по смыслу сочетаниях | punctuation_combinations | NEEDS_DICTIONARY | Syntax alone is insufficient; safe candidates require a lexical, stable-expression, or morphology dictionary first. |
| punctuation_5_8_7 | Сравнительный оборот с «как» | comparative_turnovers | NEEDS_SEMANTIC_MODEL | Local syntax can propose probes, but production candidates need semantic disambiguation beyond bounded patterns. |
| punctuation_5_8_10 | Устойчивые сочетания с союзом «как» | comparative_turnovers | NEEDS_SEMANTIC_MODEL | Local syntax can propose probes, but production candidates need semantic disambiguation beyond bounded patterns. |
| punctuation_9_7 | Оформление сноски | punctuation_combinations | KEEP_METADATA_ONLY | Footnote formatting is document-layout metadata and should remain outside the spelling/punctuation correction path. |
