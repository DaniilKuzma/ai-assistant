from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import re

from src.candidates.frequent_errors import HYPHEN_WHITELIST, REVERSE_SYNTHETIC_ERRORS, REVERSE_SYNTHETIC_ERROR_TYPES
from src.preprocessing.protected_spans import find_protected_spans
from src.preprocessing.tokenizer import tokenize_words
from src.rules.orthography import orthography_rules


PUNCTUATION_BALANCE_GROUPS = (
    "comma_subordinate",
    "comma_conjunction",
    "introductory",
    "address_comma",
    "homogeneous_members",
    "detached_members",
    "comparative_turnover",
    "colon",
    "subject_predicate_dash",
    "direct_speech",
    "final_punctuation",
)
DEFAULT_PUNCTUATION_BALANCE = {
    "comma_subordinate": 20_000,
    "comma_conjunction": 15_000,
    "introductory": 10_000,
    "address_comma": 6_000,
    "homogeneous_members": 8_000,
    "detached_members": 8_000,
    "comparative_turnover": 6_000,
    "colon": 8_000,
    "subject_predicate_dash": 8_000,
    "direct_speech": 8_000,
    "final_punctuation": 20_000,
}
ORTHOGRAPHY_BALANCE_GROUPS = (
    "ne_verb",
    "ne_pos",
    "tsya",
    "combo",
    "hard_sign",
    "prefix_z_s",
    "prefix_pre_pri",
    "ci",
    "hissing_o_e",
    "n_nn",
    "context_pairs",
    "dictionary_fuzzy",
)
DEFAULT_ORTHOGRAPHY_BALANCE = {
    "ne_verb": 8_000,
    "ne_pos": 6_000,
    "tsya": 6_000,
    "combo": 8_000,
    "hard_sign": 6_000,
    "prefix_z_s": 6_000,
    "prefix_pre_pri": 4_000,
    "ci": 4_000,
    "hissing_o_e": 5_000,
    "n_nn": 5_000,
    "context_pairs": 5_000,
    "dictionary_fuzzy": 5_000,
}
INACTIVE_SYNTHETIC_PUNCTUATION_GROUPS = frozenset(
    {
        "comma",
        "dash",
        "delete_replace",
        "punctuation_noise",
        "quotes_brackets",
        "semicolon",
        "unsupported_colon",
    }
)

DICTIONARY_FUZZY_SYNTHETIC_ERRORS = {
    "библиотека": "библеотека",
    "корова": "карова",
    "молоко": "малако",
    "собака": "сабака",
    "территория": "територия",
}

TARGETED_BACKFILL_TOPICS = (
    "рабочий отчет",
    "новый документ",
    "точный план",
    "городской архив",
    "утренний выпуск",
    "важный договор",
    "короткая заметка",
    "итоговая таблица",
    "свежая сводка",
    "открытая заявка",
    "письменный ответ",
    "подробная справка",
    "районная комиссия",
    "финальный список",
    "новая редакция",
    "архивная запись",
    "дежурный редактор",
    "общий график",
    "плановая встреча",
    "осенняя проверка",
    "сводный отчет",
    "важная подпись",
    "рабочая группа",
    "городская служба",
    "новый раздел",
    "учебная таблица",
    "вечерний протокол",
    "проверенный файл",
    "согласованный договор",
    "подготовленный список",
    "закрытая заявка",
    "длинная запись",
    "обновленный журнал",
    "северный филиал",
    "зимнее расписание",
    "летняя проверка",
    "районный протокол",
    "входящее письмо",
    "старый договор",
    "новая таблица",
    "закрытый раздел",
    "утренний приказ",
    "служебная записка",
    "городской проект",
    "важное решение",
    "общий список",
    "рабочий кабинет",
    "архивный номер",
    "открытый вопрос",
    "плановый отчет",
    "новый маршрут",
    "вечерняя смена",
    "срочная заявка",
    "письменная просьба",
    "короткий протокол",
    "дневной выпуск",
    "точная дата",
    "готовый макет",
    "основной договор",
    "временный график",
    "дежурная группа",
    "финальная версия",
    "контрольная таблица",
    "принятый отчет",
    "закрытое письмо",
    "служебный ответ",
    "городская программа",
    "важная поправка",
    "новый порядок",
    "итоговый документ",
    "отдельная папка",
    "рабочая переписка",
    "городской журнал",
    "основная сводка",
    "ранний выпуск",
    "новая заявка",
    "закрытый протокол",
    "письменный договор",
    "дежурный список",
    "точный маршрут",
    "служебный журнал",
    "важный раздел",
    "рабочий приказ",
    "архивная таблица",
    "общая справка",
    "принятая заявка",
    "вечерний отчет",
    "короткий договор",
    "осенний график",
    "проверенная запись",
    "срочный ответ",
    "финальная подпись",
)

TARGETED_BACKFILL_PAIR_TEMPLATES: dict[str, tuple[tuple[str, str], ...]] = {
    "comma_subordinate": (
        ("Я думаю что {topic} готов к отправке.", "Я думаю, что {topic} готов к отправке."),
        ("Команда решила что {topic} нужно проверить.", "Команда решила, что {topic} нужно проверить."),
        ("Редактор заметил что {topic} изменился утром.", "Редактор заметил, что {topic} изменился утром."),
        ("Мы проверили что {topic} открыт для правок.", "Мы проверили, что {topic} открыт для правок."),
        ("Он объяснил что {topic} задержали ненадолго.", "Он объяснил, что {topic} задержали ненадолго."),
        ("Проект готов потому что {topic} уже согласован.", "Проект готов, потому что {topic} уже согласован."),
        ("Отчет задержали потому что {topic} вернули позже.", "Отчет задержали, потому что {topic} вернули позже."),
        ("Если команда успеет то {topic} отправят сегодня.", "Если команда успеет, то {topic} отправят сегодня."),
    ),
    "comma_conjunction": (
        ("Мы пришли но встреча уже закончилась.", "Мы пришли, но встреча уже закончилась."),
        ("Документ готов но его нужно проверить.", "Документ готов, но его нужно проверить."),
        ("Команда ждала а руководитель уточнял сроки.", "Команда ждала, а руководитель уточнял сроки."),
        ("План простой зато надежный.", "План простой, зато надежный."),
        ("{topic} готов но требует подписи.", "{topic} готов, но требует подписи."),
        ("Редактор ждал а {topic} оставался в архиве.", "Редактор ждал, а {topic} оставался в архиве."),
        ("Комиссия собралась однако {topic} еще проверяли.", "Комиссия собралась, однако {topic} еще проверяли."),
        ("Автор торопился зато {topic} сохранили вовремя.", "Автор торопился, зато {topic} сохранили вовремя."),
    ),
    "introductory_comma": (
        ("Конечно {topic} требует проверки.", "Конечно, {topic} требует проверки."),
        ("Например {topic} можно отправить завтра.", "Например, {topic} можно отправить завтра."),
        ("Возможно {topic} будет готов утром.", "Возможно, {topic} будет готов утром."),
        ("По-видимому {topic} изменился вечером.", "По-видимому, {topic} изменился вечером."),
        ("Следовательно {topic} нужно сохранить отдельно.", "Следовательно, {topic} нужно сохранить отдельно."),
        ("Кажется {topic} уже лежит в архиве.", "Кажется, {topic} уже лежит в архиве."),
    ),
    "address_comma": (
        ("Коллеги проверим {topic}.", "Коллеги, проверим {topic}."),
        ("Иван отправь {topic}.", "Иван, отправь {topic}."),
        ("Мария уточни {topic}.", "Мария, уточни {topic}."),
        ("Коллеги исправим {topic}.", "Коллеги, исправим {topic}."),
        ("Иван открой {topic}.", "Иван, открой {topic}."),
        ("Мария посмотри {topic}.", "Мария, посмотри {topic}."),
    ),
    "comparative_turnover_comma": (
        ("Он говорил словно все уже решено.", "Он говорил, словно все уже решено."),
        ("Команда работала будто времени не осталось.", "Команда работала, будто времени не осталось."),
        ("Отчет выглядел как будто его готовили ночью.", "Отчет выглядел, как будто его готовили ночью."),
        ("Редактор замер будто услышал важную новость.", "Редактор замер, будто услышал важную новость."),
        ("{topic} выглядел словно готовый итог.", "{topic} выглядел, словно готовый итог."),
        ("{topic} лежал как будто его забыли утром.", "{topic} лежал, как будто его забыли утром."),
    ),
    "detached_adverbial_comma": (
        ("Закончив проверку команда отправила отчет.", "Закончив проверку, команда отправила отчет."),
        ("Получив письмо редактор уточнил детали.", "Получив письмо, редактор уточнил детали."),
        ("Проверив данные аналитик обновил таблицу.", "Проверив данные, аналитик обновил таблицу."),
        ("Сделав правки автор сохранил документ.", "Сделав правки, автор сохранил документ."),
        ("Прочитав отчет комиссия приняла решение.", "Прочитав отчет, комиссия приняла решение."),
        ("Закончив работу группа закрыла заявку.", "Закончив работу, группа закрыла заявку."),
    ),
    "homogeneous_comma": (
        ("Мы проверили и отчет и заявку.", "Мы проверили и отчет, и заявку."),
        ("В архиве лежали и договоры и счета.", "В архиве лежали и договоры, и счета."),
        ("Редактор исправил и заголовки и даты.", "Редактор исправил и заголовки, и даты."),
        ("Команда открыла и таблицу и сводку.", "Команда открыла и таблицу, и сводку."),
        ("Автор сохранил ни план ни отчет.", "Автор сохранил ни план, ни отчет."),
        ("В списке есть и подписи и сроки.", "В списке есть и подписи, и сроки."),
    ),
    "subject_predicate_dash": (
        ("Москва это столица России.", "Москва — это столица России."),
        ("Проект это сложная задача.", "Проект — это сложная задача."),
        ("Главная цель это проверка отчета.", "Главная цель — это проверка отчета."),
        ("Документ это важный результат.", "Документ — это важный результат."),
        ("Итоговый вывод это рабочий вариант.", "Итоговый вывод — это рабочий вариант."),
        ("Новая редакция это основа договора.", "Новая редакция — это основа договора."),
    ),
    "enumeration_colon": (
        ("В отчете указано следующее ошибки задержки и риски.", "В отчете указано следующее: ошибки задержки и риски."),
        ("Команда проверила следующие документы сроки и подписи.", "Команда проверила следующие: документы сроки и подписи."),
        ("В списке осталось следующее отчет договор и заявка.", "В списке осталось следующее: отчет договор и заявка."),
        ("Редактор отметил следующие заголовки подписи и даты.", "Редактор отметил следующие: заголовки подписи и даты."),
    ),
    "explanation_colon": (
        ("Он понял одно проект готов.", "Он понял одно: проект готов."),
        ("Команда решила одно отчет нужно вернуть.", "Команда решила одно: отчет нужно вернуть."),
        ("Редактор заметил одно подпись отсутствует.", "Редактор заметил одно: подпись отсутствует."),
    ),
    "consequence_dash": (
        ("Начался дождь мы остались дома.", "Начался дождь — мы остались дома."),
        ("Начался дождь встречу перенесли утром.", "Начался дождь — встречу перенесли утром."),
        ("Начался дождь команда закрыла окна.", "Начался дождь — команда закрыла окна."),
    ),
    "asyndetic_dash": (
        ("Солнце село стало холодно.", "Солнце село — стало холодно."),
        ("Солнце село город затих.", "Солнце село — город затих."),
        ("Солнце село встреча закончилась.", "Солнце село — встреча закончилась."),
    ),
    "direct_speech_colon": (
        ("Он сказал проект готов.", "Он сказал: «проект готов»."),
        ("Редактор сообщил отчет принят.", "Редактор сообщил: «отчет принят»."),
        ("Руководитель сказал команда справилась.", "Руководитель сказал: «команда справилась»."),
        ("Автор ответил документ готов.", "Автор ответил: «документ готов»."),
    ),
    "direct_speech_quotes": (
        ("Он сказал проект готов.", "Он сказал: «проект готов»."),
        ("Редактор сообщил отчет принят.", "Редактор сообщил: «отчет принят»."),
        ("Руководитель сказал команда справилась.", "Руководитель сказал: «команда справилась»."),
        ("Автор ответил документ готов.", "Автор ответил: «документ готов»."),
    ),
    "direct_speech_dash": (
        ("«Проект готов» сказал автор.", "«Проект готов» — сказал автор."),
        ("«Отчет принят» сообщил редактор.", "«Отчет принят» — сообщил редактор."),
        ("«Команда справилась» сказала Мария.", "«Команда справилась» — сказала Мария."),
        ("«Документ открыт» ответил Иван.", "«Документ открыт» — ответил Иван."),
    ),
    "context_tak_zhe": (
        ("Он сделал также как коллега.", "Он сделал так же как коллега."),
        ("Редактор поступил также как автор.", "Редактор поступил так же как автор."),
        ("Команда решила также как комиссия.", "Команда решила так же как комиссия."),
    ),
    "context_to_zhe": (
        ("Я тоже упражнение проверил утром.", "Я то же упражнение проверил утром."),
        ("Он тоже задание отправил вечером.", "Он то же задание отправил вечером."),
        ("Редактор тоже замечание оставил в тексте.", "Редактор то же замечание оставил в тексте."),
    ),
    "context_chto_by": (
        ("Чтобы он ни сказал решение принято.", "Что бы он ни сказал решение принято."),
        ("Чтобы команда ни решила отчет останется.", "Что бы команда ни решила отчет останется."),
        ("Чтобы редактор ни отметил срок сохранится.", "Что бы редактор ни отметил срок сохранится."),
        ("Что бы закончить работу нужно согласование.", "Чтобы закончить работу нужно согласование."),
    ),
    "context_za_to": (
        ("За то команда успела подготовить отчет.", "Зато команда успела подготовить отчет."),
        ("Зато решение отвечал руководитель.", "За то решение отвечал руководитель."),
        ("За то редактор быстро закрыл заявку.", "Зато редактор быстро закрыл заявку."),
    ),
    "context_vsledstvie": (
        ("В следствие ошибки отчет вернули.", "Вследствие ошибки отчет вернули."),
        ("Вследствие внесли новые материалы.", "В следствие внесли новые материалы."),
        ("В следствие задержки встречу перенесли.", "Вследствие задержки встречу перенесли."),
    ),
    "context_nesmotrya": (
        ("Не смотря на задержку проект завершили.", "Несмотря на задержку проект завершили."),
        ("Несмотря в документы он отвечал на вопросы.", "Не смотря в документы он отвечал на вопросы."),
        ("Не смотря на дождь команда пришла вовремя.", "Несмотря на дождь команда пришла вовремя."),
    ),
    "hyphen_particles": (
        ("Кто то проверил {topic}.", "Кто-то проверил {topic}."),
        ("Где либо сохранился {topic}.", "Где-либо сохранился {topic}."),
        ("Когда нибудь появится {topic}.", "Когда-нибудь появится {topic}."),
        ("Что то изменилось в архиве.", "Что-то изменилось в архиве."),
    ),
    "hyphen_koe_koy": (
        ("Кое кто проверил {topic}.", "Кое-кто проверил {topic}."),
        ("Кое где открыт {topic}.", "Кое-где открыт {topic}."),
        ("Кой кто отправил {topic}.", "Кой-кто отправил {topic}."),
        ("Кое как работает {topic}.", "Кое-как работает {topic}."),
    ),
    "hyphen_po_adverbs": (
        ("Он говорит по русски о проекте.", "Он говорит по-русски о проекте."),
        ("Автор сделал по новому вариант отчета.", "Автор сделал по-новому вариант отчета."),
        ("Редактор ответил по дружески утром.", "Редактор ответил по-дружески утром."),
        ("Команда решила по старому плану.", "Команда решила по-старому плану."),
    ),
    "pol_polu_compounds": (
        ("Он купил пол лимона для опыта.", "Он купил пол-лимона для опыта."),
        ("На столе лежит пол яблока.", "На столе лежит пол-яблока."),
        ("Команда ждала полу финал проекта.", "Команда ждала полуфинал проекта."),
        ("Редактор проверил пол Москвы за день.", "Редактор проверил пол-Москвы за день."),
    ),
    "hyphen_whitelist": (
        ("Он говорит по русски о документе.", "Он говорит по-русски о документе."),
        ("Кто нибудь отправит отчет утром.", "Кто-нибудь отправит отчет утром."),
        ("Кое кто сохранил таблицу вечером.", "Кое-кто сохранил таблицу вечером."),
    ),
    "ni_stable_expression": (
        ("Не разу не ошибся редактор.", "Ни разу не ошибся редактор."),
        ("Не в коем случае не меняйте срок.", "Ни в коем случае не меняйте срок."),
        ("Во что бы то не стало отчет отправят.", "Во что бы то ни стало отчет отправят."),
    ),
}

TARGETED_BACKFILL_TERM_BANK: dict[str, tuple[str, ...]] = {
    "dictionary_fuzzy": ("библиотека", "корова", "молоко", "собака", "территория"),
    "double_consonant_candidate": ("грамматика", "территория", "профессия", "комиссия"),
    "keyboard_typo_candidate": ("молоко", "корова", "грамматика", "собака"),
    "swapped_letters_candidate": ("корова", "библиотека", "молоко", "собака"),
    "missing_letter_candidate": ("молоко", "корова", "библиотека", "собака"),
    "extra_letter_candidate": ("собака", "молоко", "корова", "библиотека"),
    "cy_exception": ("цыган", "цыпленок", "цыпочки", "цыкнуть"),
    "ne_adjective": ("несложный", "неинтересный", "непонятный", "некрасивый"),
    "ne_adverb": ("неслучайно", "недолго", "несложно", "небыстро"),
    "ne_participle": ("непроверенный", "непрочитанный", "неподписанный", "незаконченный"),
    "pattern_щю_щу": ("щука", "щуплый", "щуриться", "щупальце"),
    "pattern_чю_чу": ("чудо", "чувство", "чужой", "чулан"),
    "n_nn_deverbal_adjective": ("жареный", "ветреный", "раненый"),
}

TARGETED_BACKFILL_UNSUPPORTED_PROBES: dict[str, tuple[tuple[str, str], ...]] = {
    "subject_predicate_dash": (("Москва столица России.", "Москва — столица России."),),
    "enumeration_colon": (
        (
            "В отчете указаны три проблемы ошибки задержки и риски.",
            "В отчете указаны три проблемы: ошибки задержки и риски.",
        ),
    ),
    "homogeneous_comma": (
        ("Команда проверила отчеты письма и заявки.", "Команда проверила отчеты, письма и заявки."),
    ),
}


@dataclass(frozen=True)
class SyntheticTransformation:
    start: int
    end: int
    replacement: str
    error_type: str
    group: str = ""
    rule_id: str = ""

    def apply(self, text: str) -> str:
        return text[: self.start] + self.replacement + text[self.end :]


def synthetic_transformations(text: str) -> list[SyntheticTransformation]:
    protected = tuple((span.start, span.end) for span in find_protected_spans(text))
    return [
        *_final_punctuation_transformations(text),
        *_punctuation_transformations(text, protected),
        *_paired_punctuation_transformations(text, protected),
        *_orthography_transformations(text, protected),
        *_dictionary_fuzzy_transformations(text, protected),
        *_lexical_transformations(text, protected),
    ]


def synthetic_variant_sources(text: str) -> list[tuple[str, list[str], list[str], list[str]]]:
    variants: list[tuple[str, list[str], list[str], list[str]]] = []

    transformations = synthetic_transformations(text)
    for transformation in transformations:
        variants.append(
            (
                transformation.apply(text),
                [transformation.error_type],
                [transformation.group],
                [transformation.rule_id],
            )
        )

    for first, second in combinations(transformations[:18], 2):
        if _overlaps_any(first, [second]):
            continue
        variants.append(
            (
                apply_transformations(text, [first, second]),
                [first.error_type, second.error_type],
                [first.group, second.group],
                [first.rule_id, second.rule_id],
            )
        )

    return variants


def source_dataset_for_groups(groups: list[str]) -> str:
    unique_groups = sorted({group for group in groups if group})
    if len(unique_groups) == 1:
        return f"synthetic_open_corpus_{unique_groups[0]}"
    return "synthetic_open_corpus_mixed"


def apply_transformations(text: str, transformations: list[SyntheticTransformation]) -> str:
    source = text
    for transformation in sorted(transformations, key=lambda item: (item.start, item.end), reverse=True):
        source = transformation.apply(source)
    return source


def _lexical_transformations(text: str, protected: tuple[tuple[int, int], ...]) -> list[SyntheticTransformation]:
    transformations: list[SyntheticTransformation] = []
    lower = text.lower()
    for clean, dirty in REVERSE_SYNTHETIC_ERRORS.items():
        start = lower.find(clean)
        if start < 0:
            continue
        if _span_overlaps_protected(start, start + len(clean), protected):
            continue
        source = text[start : start + len(clean)]
        error_type = REVERSE_SYNTHETIC_ERROR_TYPES.get(clean, "spelling")
        transformations.append(
            SyntheticTransformation(
                start,
                start + len(clean),
                _match_case(source, dirty),
                error_type,
                _lexical_group(error_type),
                "hyphen_whitelist" if error_type == "hyphen" else "frequent_error_exact",
            )
        )
    return transformations


def _dictionary_fuzzy_transformations(text: str, protected: tuple[tuple[int, int], ...]) -> list[SyntheticTransformation]:
    transformations: list[SyntheticTransformation] = []
    lower = text.lower()
    for clean, dirty in DICTIONARY_FUZZY_SYNTHETIC_ERRORS.items():
        start = lower.find(clean)
        if start < 0:
            continue
        end = start + len(clean)
        if _span_overlaps_protected(start, end, protected):
            continue
        transformations.append(
            SyntheticTransformation(
                start,
                end,
                _match_case(text[start:end], dirty),
                "spelling",
                "dictionary_fuzzy",
                "dictionary_fuzzy",
            )
        )
    return transformations


def _orthography_transformations(text: str, protected: tuple[tuple[int, int], ...]) -> list[SyntheticTransformation]:
    transformations: list[SyntheticTransformation] = []
    words = tuple(tokenize_words(text))

    for index, _word in enumerate(words):
        for rule in orthography_rules():
            if not hasattr(rule, "generate_corruptions"):
                continue
            for dirty in rule.generate_corruptions(text, words, index, protected):
                transformations.append(
                    SyntheticTransformation(
                        dirty.start,
                        dirty.end,
                        dirty.replacement,
                        dirty.error_type,
                        dirty.group,
                        dirty.rule_id,
                    )
                )
    return transformations


def _punctuation_transformations(text: str, protected: tuple[tuple[int, int], ...]) -> list[SyntheticTransformation]:
    transformations: list[SyntheticTransformation] = []
    for index, char in enumerate(text):
        if _span_overlaps_protected(index, index + 1, protected):
            continue
        if _is_final_punctuation_position(text, index) or _is_decimal_comma(text, index):
            continue
        if char in ",:;—":
            group = _punctuation_group(text, index, char)
            if group in INACTIVE_SYNTHETIC_PUNCTUATION_GROUPS:
                continue
            transformations.append(
                SyntheticTransformation(index, index + 1, "", "punctuation", group, _punctuation_rule_id(group))
            )
    return transformations


def _paired_punctuation_transformations(text: str, protected: tuple[tuple[int, int], ...]) -> list[SyntheticTransformation]:
    del text, protected
    return []


def _final_punctuation_transformations(text: str) -> list[SyntheticTransformation]:
    stripped = text.rstrip()
    if not stripped or stripped[-1] != ".":
        return []
    start = len(stripped) - 1
    return [
        SyntheticTransformation(start, start + 1, "", "final_punctuation", "final_punctuation", "final_punctuation_default"),
    ]


def _punctuation_group(text: str, index: int, char: str) -> str:
    if char == ",":
        following = text[index + 1 : index + 24].lower()
        prefix = text[max(0, index - 20) : index + 1].lower()
        before = text[:index].lower()
        after = text[index + 1 :].lower()
        if re.match(r"\s*(что|чтобы|если|когда)\b", following):
            return "comma_subordinate"
        if re.match(r"\s*(но|а)\b", following):
            return "comma_conjunction"
        if re.search(r"\b(конечно|например|однако|во-первых),$", prefix):
            return "introductory"
        if re.search(r"^\s*(коллеги|иван|мария)$", before):
            return "address_comma"
        if re.search(r"\b(и|ни)\s+[а-яё]+$", before) and re.match(r"\s*(и|ни)\b", following):
            return "homogeneous_members"
        if re.search(r"^(закончив|сделав|прочитав)\s+[а-яё]+$", before.strip()):
            return "detached_members"
        if re.match(r"\s*(словно|будто|как\s+будто)\b", following):
            return "comparative_turnover"
        return "comma"
    if char == ":":
        if re.search(r"\b(сказал|сказала|спросил|ответил)\s*:$", text[: index + 1].lower()):
            return "direct_speech"
        if re.search(r"\b(следующее|следующие)\s*:$", text[: index + 1].lower()):
            return "colon"
        return "unsupported_colon"
    if char == "—":
        if re.match(r"\s*это\b", text[index + 1 :].lower()):
            return "subject_predicate_dash"
        return "dash"
    if char == ";":
        return "semicolon"
    return "punctuation"


def _punctuation_rule_id(group: str) -> str:
    return {
        "comma_subordinate": "comma_subordinate",
        "comma_conjunction": "comma_conjunction",
        "introductory": "introductory_comma",
        "address_comma": "address_comma",
        "homogeneous_members": "homogeneous_comma",
        "detached_members": "detached_adverbial_comma",
        "comparative_turnover": "comparative_turnover_comma",
        "colon": "enumeration_colon",
        "dash": "subject_predicate_dash",
        "subject_predicate_dash": "subject_predicate_dash",
        "direct_speech": "direct_speech_colon",
        "semicolon": "semicolon",
        "comma": "comma",
        "punctuation_noise": "punctuation_delete_replace",
    }.get(group, "punctuation_delete_replace")


def _lexical_group(error_type: str) -> str:
    if error_type in {"split_join", "hyphen"}:
        return error_type
    return "frequent_spelling"


def _punctuation_replacements_for(char: str) -> list[str]:
    return {
        ",": [":", ";"],
        ":": [",", ";"],
        ";": [","],
        "—": [","],
    }.get(char, [])


def _looks_like_direct_speech_quote(text: str, quote_start: int) -> bool:
    prefix = text[max(0, quote_start - 24) : quote_start].lower()
    return bool(re.search(r"\b(сказал|сказала|спросил|ответил)\s*:\s*$", prefix))


def _overlaps_any(transformation: SyntheticTransformation, selected: list[SyntheticTransformation]) -> bool:
    return any(_overlaps(transformation, item) for item in selected)


def _overlaps(left: SyntheticTransformation, right: SyntheticTransformation) -> bool:
    if left.start == left.end or right.start == right.end:
        return left.start == right.start
    return max(left.start, right.start) < min(left.end, right.end)


def _span_overlaps_protected(start: int, end: int, protected: tuple[tuple[int, int], ...]) -> bool:
    return any(start < protected_end and protected_start < end for protected_start, protected_end in protected)


def _is_decimal_comma(text: str, index: int) -> bool:
    if text[index] != ",":
        return False
    previous_char = text[index - 1] if index > 0 else ""
    next_char = text[index + 1] if index + 1 < len(text) else ""
    return previous_char.isdigit() and next_char.isdigit()


def _is_final_punctuation_position(text: str, index: int) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and index == len(stripped) - 1 and stripped[-1] in ".!?…"


def _match_case(source: str, replacement: str) -> str:
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement
