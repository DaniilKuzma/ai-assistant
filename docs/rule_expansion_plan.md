# План расширения правил Орфограммки

Источник классификации: https://orfogrammka.ru/орфография/ и https://orfogrammka.ru/пунктуация/.

`configs/rules.yaml` теперь используется как coverage matrix: строка может быть полноценным исполняемым правилом, частичным candidate layer или только честной taxonomy-заглушкой. Статусы `planned`, `model_required`, `syntax_required`, `dictionary_model_required` и `ner_required` не объявляют RuleSpec реализованным.

## Уже реализовано

- Орфография: базовые буквенные паттерны после шипящих и ц (`letter_basic_hissing_vowel_patterns`): жи/ши, ча/ща, чу/щу, ци/цы, о/е после шипящих, словарно проверяемые candidates.
- Орфография: приставки на з-/с- (`consonants_prefix_z_s`): bounded dictionary candidates для `сделать`-подобных форм и приставок без-/бес-, раз-/рас-, из-/ис-, воз-/вос-, вз-/вс-.
- Орфография: прописная буква в начале текста (`capitalization_sentence_start`): легкий fallback только для первого токена.
- Пунктуация: финальная точка (`sentence_final_default_dot`): candidate для точки, если в конце нет `.`, `?`, `!` или `...`.

## Частично реализовано или candidate-only

- `dictionary_model_required` spelling layer: есть bounded lexicon-backed candidates для fuzzy/edit-distance, двойных согласных, клавиатурной соседней клавиши, перестановки соседних букв, пропуска и лишней буквы; все требуют model scorer и validator.
- `ё/е`: policy disabled by default. `yo_e_candidate` генерируется только при явном `dictionary.yo_e.enabled: true`, поддержке обеих форм в lexicon/provider и дальнейшем model scoring.
- `hard_soft_signs`: есть `ъ` после приставки и замена ошибочного `ь` на `ъ`; остальные функции `ь` требуют морфологии, словаря или модели.
- `typical_dictionary_words`: есть ограниченный exact whitelist частых ошибок; это не полноценная словарная орфография.
- `ne_ni_particles`: есть candidates для слитного `не` с глаголами; выбор требует модели, `ни` не покрыто.
- `solid_hyphen_separate_pronouns_adverbs_particles`: есть candidates для `-то/-либо/-нибудь`, `кое-/кой-`, `по-...` и legacy whitelist.
- `solid_hyphen_separate_service_words`: есть контекстные пары вроде `также/так же`, `тоже/то же`, `чтобы/что бы`; применять только через scorer.
- `solid_hyphen_separate_general`: есть bounded candidates для `пол-/полу-`; остальные общие слитные/дефисные случаи остаются roadmap.
- `punctuation_combinations`: есть удаление очевидных дублей и простых лишних знаков; сложные сочетания остаются отдельной задачей.

## Требуемые зависимости

- Syntax: падежные окончания, Н/НН, именные группы со слитным/дефисным/раздельным написанием, `не` с причастиями и противопоставлениями, тире между членами предложения, лишняя запятая в управлении, однородные члены, обособления, сравнительные обороты, ССП/СПП/БСП.
- Dictionary: корневые, приставочные и суффиксальные гласные, проверяемые/непроверяемые согласные, двойные согласные, аббревиатуры, заимствования, словарные слова, общий edit-distance typo layer, словарь сокращений.
- NER: личные имена, география, организации, документы, события, награды, инициалы, часть обращений и приложений.
- Model: `-тся/-ться`, контекстные пары, омонимия вводных слов, прямая речь, кавычки/скобки, сложная пунктуация и нейропунктуация.

## Рекомендуемый порядок

1. Укреплять coverage contract: валидный YAML, обязательные `orfogrammka_id` и `parent_group`, реальные registry ids только для implemented/partial/candidate-only групп.
2. Расширять dictionary generator: частотный лексикон, edit-distance candidates, словари сокращений, заимствований и словарных слов.
3. Закрывать безопасные deterministic/candidate-only орфограммы только при наличии общего правила, словарной проверки и synthetic corruption.
4. Подключать syntax и NER для правил, где без структуры предложения высокий риск ложных исправлений.
5. Подключать model scoring и validator для контекстных пар, сложной пунктуации, прямой речи и нейропунктуации.

## Тесты по группам

- Implemented: registry test, candidate generation test, test markers в `configs/rules.yaml`, synthetic corruption test там, где правило умеет `generate_corruptions`, negative tests против overcorrection.
- Partial/candidate-only: проверять, что listed `rule_id` существует в `RuleRegistry`, candidates имеют `rule_id`, `mode`, `requires_model/requires_scoring` и не применяются plain corrector без trusted scorer.
- Metadata-only: `planned`, `model_required`, `syntax_required`, `dictionary_model_required`, `ner_required` могут иметь пустой `rules` или report-only ids, но не считаются implemented.
- Syntax-required: unit tests с замоканным `src.nlp.syntax.parse_syntax`, затем integration tests с Natasha, если модель доступна локально.
- Dictionary-model-required: tests на словарные candidates, лимиты, частотную фильтрацию, protected spans и отсутствие частных пар как основного механизма.
- NER-required: tests на mocked NER spans и негативные примеры для обычных строчных слов.
- Model-required: scorer/validator tests на принятие trusted candidates и отклонение низкой уверенности.
## Matrix Eval Findings

- Full matrix audit artifacts are under `reports/matrix_eval/`.
- Dedicated eval corpus is under `data/processed/matrix_eval/`.
- Use `reports/matrix_eval/next_dataset_activation_plan.md` for the next dataset cycle.
## Matrix Eval Matrix Findings

- Matrix artifacts are under `reports/matrix_eval/`.
- Matrix eval corpus is under `data/processed/matrix_eval/`.
- Use `reports/matrix_eval/activation_activation_plan.md` and `reports/matrix_eval/rule_expansion_backlog_core.md` for `training_dataset` planning.
- Planned and metadata-only matrix entries remain backlog items until executable support exists.
