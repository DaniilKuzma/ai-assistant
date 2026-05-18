# План расширения правил Орфограммки

Источник классификации: https://orfogrammka.ru/орфография/ и https://orfogrammka.ru/пунктуация/.

Документ фиксирует roadmap покрытия типов ошибок. Он не объявляет planned/model/syntax/dictionary/NER-группы реализованными и не требует для них исполняемых rule objects.

## Уже реализовано

- Орфография: базовые буквенные паттерны после шипящих и ц (`letter_basic_hissing_vowel_patterns`): жи/ши, ча/ща, чу/щу, ци/цы, о/е после шипящих, словарно проверяемые candidates.
- Орфография: приставки на з-/с- (`consonants_prefix_z_s`): bounded dictionary candidates для `сделать`-подобных форм и приставок без-/бес-, раз-/рас-, из-/ис-, воз-/вос-, вз-/вс-.
- Орфография: прописная буква в начале текста (`capitalization_sentence_start`): легкий fallback только для первого токена.
- Пунктуация: финальная точка (`sentence_final_default_dot`): deterministic fallback, если в конце нет `.`, `?`, `!` или `...`.

## Частично реализовано или candidate-only

- `hard_soft_signs`: есть `ъ` после приставки и замена ошибочного `ь` на `ъ`; остальные функции `ь` требуют морфологии.
- `typical_dictionary_words`: есть ограниченный whitelist частых ошибок; это не полноценный словарный исправитель.
- `ne_ni_particles`: есть candidates для слитного `не` с глаголами; выбор требует модели, `ни` не покрыто.
- `solid_hyphen_separate_pronouns_adverbs_particles`: есть candidates для `-то/-либо/-нибудь`, `кое-/кой-`, `по-...` и legacy whitelist; применять только через scorer.
- `punctuation_combinations`: есть удаление очевидных дублей и простых лишних знаков; сложные сочетания остаются отдельной задачей.

## Требуемые зависимости

- Natasha/syntax: падежные окончания, слитное/дефисное/раздельное написание именных групп, `не` с причастиями и противопоставлениями, тире между членами предложения, лишняя запятая в управлении, однородные члены, обособления, сравнительные обороты, ССП/СПП/БСП, сложное взаимодействие знаков.
- Dictionary candidate generator: безударные корневые, приставочные и суффиксальные гласные, парные/непроизносимые/двойные согласные, аббревиатуры, иностранные слова, общий edit-distance typo layer, словарь сокращений.
- NER: личные имена, география, организации, документы, события, награды, часть обращений и приложений.
- Model scoring: `-тся/-ться`, контекстные пары (`также/так же`, `чтобы/что бы` и т.п.), омонимия вводных слов, прямая речь, сложная пунктуация, нейропунктуация.

## Рекомендуемый порядок

1. Укрепить инфраструктуру: держать `configs/rules.yaml` валидным, связывать все `rule_id` с `RuleRegistry` и rule-level reports, поддерживать тестовые markers для implemented групп.
2. Расширять dictionary generator: общий словарный candidate layer, частотный лексикон, edit-distance candidates, словари сокращений и заимствований.
3. Закрывать безопасные deterministic/candidate-only орфограммы: новые группы добавлять только при наличии общего правила, словарной проверки и synthetic corruption.
4. Подключать syntax/NER-слой для правил, где без структуры предложения высокий риск ложных исправлений.
5. Подключать model scoring и validator для контекстных пар, сложной пунктуации и нейропунктуации.

## Тесты по группам

- Implemented: registry test, candidate generation test, synthetic corruption test там, где правило умеет `generate_corruptions`, и negative tests против overcorrection.
- Partial/candidate-only: проверять, что candidates имеют `rule_id`, `mode`, `requires_model/requires_scoring` и не применяются plain corrector без trusted scorer.
- Syntax-required: unit tests с замоканным `src.nlp.syntax.parse_syntax`, затем integration tests с Natasha, если модель доступна локально.
- Dictionary-model-required: tests на словарные candidates, лимиты, частотную фильтрацию, protected spans и отсутствие частных пар как основного механизма.
- NER-required: tests на mocked NER spans и негативные примеры для обычных строчных слов.
- Model-required: scorer/validator tests на принятие trusted candidates и отклонение низкой уверенности.
