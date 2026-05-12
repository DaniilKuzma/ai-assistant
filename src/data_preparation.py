"""
Модуль для подготовки датасета.

Функции:
- Загрузка исходных текстов
- Генерация синтетических ошибок
- Создание обучающих пар (текст_с_ошибкой, правильный_текст)
- Сохранение подготовленных данных
"""

import os
import random
import re
import json
from typing import Any, Dict, List, Tuple
import pandas as pd
from tqdm import tqdm

from strict_error_taxonomy import ERROR_SCOPE_CLEAN, error_types_are_strict


def split_texts(texts: List[str], train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """Сплитит *чистые* тексты на train/val/test до аугментации, чтобы не было утечки."""
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio должно быть = 1.0")

    # Убираем пустые и дубликаты (иначе можно случайно 'увидеть' одно и то же предложение в разных сплитах)
    texts = [t.strip() for t in texts if t and t.strip()]
    texts = list(dict.fromkeys(texts))  # preserve order

    rng = random.Random(seed)
    rng.shuffle(texts)

    n = len(texts)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_texts = texts[:n_train]
    val_texts = texts[n_train:n_train + n_val]
    test_texts = texts[n_train + n_val:]

    return train_texts, val_texts, test_texts


MONTH_RE = (
    "января|февраля|марта|апреля|мая|июня|июля|августа|"
    "сентября|октября|ноября|декабря"
)


def safe_normalize_raw_spacing(text: str) -> str:
    """Fix only obvious raw-source spacing joins without rewriting the text."""
    text = str(text).strip()
    text = re.sub(rf"([.!?])(?=\d{{1,2}}\s+(?:{MONTH_RE})\b)", r"\1 ", text, flags=re.IGNORECASE)
    text = re.sub(r"([.!?])(?=\d{4}\b)", r"\1 ", text)
    text = re.sub(r"(\b\d{4})(?=[A-ZА-ЯЁ])", r"\1 ", text)
    text = re.sub(r"([.!?])(?=[A-ZА-ЯЁ])", r"\1 ", text)
    return text


class DatasetGenerator:
    """Класс для генерации датасета с ошибками."""
    
    def __init__(self, error_config=None):
        """Инициализация генератора датасета."""
        # Типичные орфографические ошибки в русском языке.
        # Осознанно не добавляем ё -> е, клавиатурные опечатки и ошибки окончаний.
        self.common_mistakes = {
            'а': ['о', 'я'],
            'о': ['а'],
            'е': ['и', 'э'],
            'и': ['е', 'ы', 'й'],
            'ы': ['и'],
            'у': ['ю', 'о'],
            'ю': ['у'],
            'я': ['а', 'е'],
            'э': ['е'],
            'ж': ['ш', 'з'],
            'ш': ['ж', 'щ'],
            'ч': ['щ', 'ц'],
            'щ': ['ч', 'ш'],
            'ц': ['с', 'ч'],
            'з': ['с', 'ж'],
            'с': ['з', 'ц'],
            'б': ['п'],
            'п': ['б'],
            'в': ['ф'],
            'ф': ['в'],
            'г': ['к', 'х'],
            'к': ['г', 'х'],
            'д': ['т'],
            'т': ['д'],
            'н': ['м'],
            'м': ['н'],
            'тся': ['ться', 'ца'],
            'ться': ['тся', 'ца'],
            'ого': ['ово'],
            'его': ['ево'],
        }
        
        # Знаки препинания для простых пунктуационных ошибок.
        # Кавычки, тире, слэши и скобки не удаляем в augmentation по умолчанию.
        self.punctuation = ['.', ',', '!', '?', ':', ';']

        self.word_re = re.compile(r'^([^A-Za-zА-Яа-яЁё]*)([A-Za-zА-Яа-яЁё]+)([^A-Za-zА-Яа-яЁё]*)$')
        self.letter_re = re.compile(r'[A-Za-zА-Яа-яЁё]')

        # Единый конфиг типов ошибок - включаем все типы
        default_config = {
            'replace': False,
            'delete': False,
            'swap': False,
            'double': False,
            'extra': False,
        
            'punct_remove': True,   # можно, но редко (см. ниже)
            'punct_wrong': True,
            'ensure_error': True,
            'max_errors_per_sentence': 5,  # 5 = издевательство над языком
            'strict_scope': True,
        }


        if error_config is None:
            self.error_config = default_config
        else:
            self.error_config = {**default_config, **error_config}
        for removed_typo_option in ("delete", "swap", "extra", "double"):
            self.error_config[removed_typo_option] = False

        self.error_profiles = {
            "spelling_light": {
                "word_k_range": (1, 1),
                "word_error_rate": 0.18,
                "max_word_errors": 1,
                "punct_ops": (0, 0),
                "orthography_ops": (0, 0),
                "structural_punct_ops": (0, 0),
            },
            "spelling_medium": {
                "word_k_range": (1, 3),
                "word_k_weights": [0.55, 0.35, 0.10],
                "word_error_rate": 0.30,
                "max_word_errors": 3,
                "punct_ops": (0, 0),
                "orthography_ops": (0, 1),
                "structural_punct_ops": (0, 0),
            },
            "orthography_rules": {
                "word_k_range": (0, 0),
                "word_error_rate": 0.0,
                "max_word_errors": 0,
                "punct_ops": (0, 0),
                "orthography_ops": (1, 2),
                "structural_punct_ops": (0, 0),
            },
            "punctuation_only": {
                "word_k_range": (0, 0),
                "word_error_rate": 0.0,
                "max_word_errors": 0,
                "punct_ops": (1, 1),
                "orthography_ops": (0, 0),
                "structural_punct_ops": (0, 0),
            },
            "punctuation_structural": {
                "word_k_range": (0, 0),
                "word_error_rate": 0.0,
                "max_word_errors": 0,
                "punct_ops": (0, 0),
                "orthography_ops": (0, 0),
                "structural_punct_ops": (1, 1),
            },
            "mixed": {
                "word_k_range": (1, 3),
                "word_k_weights": [0.55, 0.35, 0.10],
                "word_error_rate": 0.28,
                "max_word_errors": 3,
                "punct_ops": (1, 1),
                "orthography_ops": (0, 1),
                "structural_punct_ops": (0, 1),
            },
            "hard": {
                "word_k_range": (3, 5),
                "word_k_weights": [0.55, 0.30, 0.15],
                "word_error_rate": 0.45,
                "max_word_errors": 5,
                "punct_ops": (1, 3),
                "orthography_ops": (1, 2),
                "structural_punct_ops": (1, 2),
            },
        }

        self.profile_weights = {
            "orthography_rules": 0.32,
            "punctuation_only": 0.34,
            "punctuation_structural": 0.24,
            "mixed": 0.08,
            "hard": 0.02,
        }
        
    def load_texts_from_file(self, filepath: str) -> List[str]:
        """
        Загрузка текстов из файла.
        
        Args:
            filepath: Путь к файлу с текстами
            
        Returns:
            Список строк текста
        """
        if not os.path.exists(filepath):
            print(f"Файл {filepath} не найден. Создаю примеры текстов...")
            return self.generate_sample_texts()
        
        with open(filepath, 'r', encoding='utf-8') as f:
            texts = f.readlines()
        
        # Очистка и фильтрация
        texts = [safe_normalize_raw_spacing(t) for t in texts if t.strip()]
        return texts
    
    def generate_sample_texts(self) -> List[str]:
        """
        Генерация примеров текстов для демонстрации.
        
        Returns:
            Список примеров текстов
        """
        sample_texts = [
            "Машинное обучение - это раздел искусственного интеллекта.",
            "Нейронные сети позволяют решать сложные задачи.",
            "Python является популярным языком программирования.",
            "Глубокое обучение требует большого количества данных.",
            "Нейросетевые модели хорошо работают с последовательностями.",
            "Обработка естественного языка - интересная область.",
            "Курсовая работа требует серьезного подхода.",
            "Анализ данных помогает находить закономерности.",
            "Важно правильно подготовить данные для обучения.",
            "Качество модели зависит от качества данных.",
        ]
        
        # Расширяем датасет, повторяя тексты с вариациями
        extended_texts = []
        for text in sample_texts:
            extended_texts.append(text)
            # Добавляем варианты с дополнительными предложениями
            for other_text in sample_texts:
                if text != other_text:
                    extended_texts.append(f"{text} {other_text}")
        
        return extended_texts
    
    def introduce_spelling_error(
        self,
        word: str,
        error_type: str = 'replace',
        return_label: bool = False
    ) -> str | Tuple[str, str | None]:
        """
        Args:
            word: Слово для внесения ошибки
            error_type: Тип ошибки ('replace', 'delete', 'swap')

        Returns:
            Слово с внесенной ошибкой (гарантированно изменённое, если это возможно)
        """
        if len(word) < 3:
            return (word, None) if return_label else word

        wlow = word.lower()
        editable_end = max(1, len(word) - (3 if len(word) >= 6 else 2))
        label = None

        if error_type == 'replace':
            # 1) Сначала пробуем "подстрочные" правила (тся/ться и т.п.)
            multi_keys = [k for k in self.common_mistakes.keys() if len(k) > 1 and k in wlow]
            if multi_keys and random.random() < 0.7:
                # несколько попыток, чтобы точно поменять (если вдруг replacement == key)
                for _ in range(5):
                    key = random.choice(multi_keys)
                    repl = random.choice(self.common_mistakes[key])
                    if repl == key:
                        continue
                    start = wlow.find(key)
                    if start != -1:
                        new_word = word[:start] + repl + word[start + len(key):]
                        if new_word != word:
                            if key in ('ого', 'его'):
                                label = 'spelling_suffix_pronunciation'
                            elif key in ('тся', 'ться'):
                                label = 'spelling_tsya'
                            else:
                                label = 'spelling_replace'
                            return (new_word, label) if return_label else new_word

            # 2) Иначе замена одного символа, но ТОЛЬКО на позициях, где есть правило
            eligible = [i for i, ch in enumerate(wlow)
                        if i < editable_end and ch in self.common_mistakes and len(ch) == 1]

            if eligible:
                for _ in range(10):
                    pos = random.choice(eligible)
                    ch = wlow[pos]
                    repl = random.choice(self.common_mistakes[ch])
                    new_word = word[:pos] + repl + word[pos+1:]
                    if new_word != word:
                        return (new_word, 'spelling_replace') if return_label else new_word

            return (word, None) if return_label else word

        elif error_type in {'delete', 'swap', 'extra', 'double'}:
            return (word, None) if return_label else word

        return (word, label) if return_label else word

    def _choose_profile(self, profile_weights: Dict[str, float] | None = None) -> str:
        """Выбрать профиль ошибок по весам."""
        weights = profile_weights or self.profile_weights
        profiles = list(weights.keys())
        probs = [weights[p] for p in profiles]
        return random.choices(profiles, weights=probs, k=1)[0]

    @staticmethod
    def _clean_slot_indices(total: int, clean_ratio: float) -> set[int]:
        """Return exact slots that should stay clean for a generated split."""
        if total <= 0 or clean_ratio <= 0:
            return set()
        clean_count = round(total * float(clean_ratio))
        clean_count = max(1, min(total, clean_count))
        return set(random.sample(range(total), clean_count))

    def _sample_op_count(self, value: Tuple[int, int] | int | None) -> int:
        """Получить количество операций из int или диапазона."""
        if value is None:
            return 0
        if isinstance(value, int):
            return max(0, value)
        lo, hi = value
        lo = max(0, int(lo))
        hi = max(lo, int(hi))
        return random.randint(lo, hi)

    def _word_candidates(self, words: List[str]) -> List[int]:
        """Индексы токенов, в которых можно безопасно портить буквенную часть."""
        candidates = []
        for i, tok in enumerate(words):
            m = self.word_re.match(tok)
            if not m:
                continue
            core = m.group(2)
            if len(core) > 2 and core.isalpha():
                candidates.append(i)
        return candidates

    def _apply_word_errors(
        self,
        text: str,
        error_rate: float = 0.25,
        k_range: Tuple[int, int] | None = None,
        k_weights: List[float] | None = None,
        max_errors_override: int | None = None,
    ) -> Tuple[str, List[str]]:
        """Внести словные/орфографические ошибки и вернуть их типы."""
        words = text.split()
        if not words:
            return text, []

        candidates = self._word_candidates(words)
        if not candidates:
            return text, []

        max_errors_cfg = int(self.error_config.get('max_errors_per_sentence', 5))
        max_errors = int(max_errors_override) if max_errors_override is not None else max_errors_cfg

        base_weights = {'replace': 1.0}
        word_error_types = [t for t in base_weights if self.error_config.get(t, False)]
        if not word_error_types:
            return text, []
        weights = [base_weights[t] for t in word_error_types]

        max_by_rate = max(1, round(len(candidates) * float(error_rate))) if error_rate > 0 else 0
        k_cap = min(max_errors, max_by_rate, len(candidates))
        if k_cap <= 0:
            return text, []

        if k_range is None:
            k = 1
            if max_errors >= 2 and random.random() < 0.35:
                k = 2
            if max_errors >= 3 and random.random() < 0.10:
                k = 3
        else:
            k_min, k_max = k_range
            k_min = max(0, int(k_min))
            k_max = max(k_min, int(k_max))
            ks = list(range(k_min, k_max + 1))
            if k_weights is not None and len(k_weights) == len(ks):
                k = random.choices(ks, weights=k_weights, k=1)[0]
            else:
                k = random.choice(ks)

        k = min(k, k_cap)
        if k <= 0:
            return text, []

        error_types = []
        for idx in random.sample(candidates, k):
            tok = words[idx]
            m = self.word_re.match(tok)
            if not m:
                continue

            prefix, word, suffix = m.groups()
            if len(word) <= 2 or not word.isalpha():
                continue

            et = random.choices(word_error_types, weights=weights, k=1)[0]
            if et in ('replace', 'delete', 'swap'):
                new, label = self.introduce_spelling_error(word, error_type=et, return_label=True)
            elif et == 'double':
                new = self.introduce_double_char(word)
                label = 'spelling_double'
            else:
                new = self.introduce_extra_char(word)
                label = 'spelling_extra'

            if word and word[0].isupper():
                new = new[:1].upper() + new[1:]

            if new != word:
                words[idx] = prefix + new + suffix
                error_types.append(label or f"spelling_{et}")

        return ' '.join(words), error_types

    def _replace_one_regex(self, text: str, pattern: str, repl, flags: int = 0) -> Tuple[str, bool]:
        """Заменить одно случайное вхождение regex."""
        matches = list(re.finditer(pattern, text, flags))
        if not matches:
            return text, False
        m = random.choice(matches)
        replacement = repl(m) if callable(repl) else repl
        return text[:m.start()] + replacement + text[m.end():], True

    def _remove_one_punctuation(self, text: str) -> Tuple[str, str | None]:
        positions = [(i, ch) for i, ch in enumerate(text) if ch in self.punctuation]
        if not positions:
            return text, None
        i, ch = random.choice(positions)
        label = {
            ',': 'punct_remove_comma',
            '.': 'punct_remove_period',
            '?': 'punct_remove_question',
            '!': 'punct_remove_exclamation',
            ':': 'punct_remove_colon',
            ';': 'punct_remove_semicolon',
            '-': 'punct_remove_dash',
            '—': 'punct_remove_dash',
        }.get(ch, 'punct_remove')
        return text[:i] + text[i + 1:], label

    def _remove_some_commas(self, text: str) -> Tuple[str, str | None]:
        positions = [i for i, ch in enumerate(text) if ch == ',']
        if not positions:
            return text, None
        max_remove = 1
        remove_positions = set(random.sample(positions, max_remove))
        new = ''.join(ch for i, ch in enumerate(text) if i not in remove_positions)
        label = 'punct_remove_all_commas' if max_remove == len(positions) else 'punct_remove_comma'
        return new, label

    def _remove_final_punctuation(self, text: str) -> Tuple[str, str | None]:
        stripped = text.rstrip()
        if stripped and stripped[-1] in '.?!…':
            return stripped[:-1] + text[len(stripped):], 'punct_remove_final'
        return text, None

    def _wrong_punctuation(self, text: str) -> Tuple[str, str | None]:
        repl = {',': '.', '.': ',', '!': '.', '?': '.', ':': ';', ';': ':'}
        names = {
            ',': 'comma',
            '.': 'period',
            '!': 'exclamation',
            '?': 'question',
            ':': 'colon',
            ';': 'semicolon',
        }
        positions = [(i, ch) for i, ch in enumerate(text) if ch in repl]
        if not positions:
            return text, None
        i, ch = random.choice(positions)
        return text[:i] + repl[ch] + text[i + 1:], f"punct_wrong_{names[ch]}_to_{names[repl[ch]]}"

    def _wrong_period_to_comma(self, text: str) -> Tuple[str, str | None]:
        positions = [
            m.start()
            for m in re.finditer(r"\.(?=\s+[а-яё])", text)
            if m.start() < len(text.rstrip()) - 1
        ]
        if not positions:
            return text, None
        i = random.choice(positions)
        return text[:i] + "," + text[i + 1:], "punct_wrong_period_to_comma"

    def _remove_internal_comma(self, text: str) -> Tuple[str, str | None]:
        positions = [
            m.start()
            for m in re.finditer(r",(?=\s+\S)", text)
            if m.start() < len(text.rstrip()) - 1
        ]
        if not positions:
            return text, None
        i = random.choice(positions)
        return text[:i] + text[i + 1:], "punct_remove_internal_comma"

    def _remove_comma_before_clause_marker(self, text: str) -> Tuple[str, str | None]:
        markers = (
            'что', 'чтобы', 'если', 'когда', 'который', 'которая', 'которое',
            'которые', 'которого', 'которому', 'которым', 'которой', 'которую',
            'которых', 'которыми'
        )
        pattern = r',\s+(' + '|'.join(re.escape(m) for m in markers) + r')\b'
        new, changed = self._replace_one_regex(text, pattern, lambda m: ' ' + m.group(1), flags=re.IGNORECASE)
        return (new, 'punct_remove_comma_before_clause') if changed else (text, None)

    def _remove_comma_after_intro(self, text: str) -> Tuple[str, str | None]:
        intros = (
            'конечно', 'однако', 'например', 'впрочем', 'значит', 'кстати',
            'во-первых', 'во-вторых', 'наконец', 'вероятно', 'возможно',
            'поэтому', 'следовательно'
        )
        pattern = r'\b(' + '|'.join(re.escape(w) for w in intros) + r'),\s+'
        new, changed = self._replace_one_regex(text, pattern, lambda m: m.group(1) + ' ', flags=re.IGNORECASE)
        return (new, 'punct_remove_intro_comma') if changed else (text, None)

    def _remove_homogeneous_comma(self, text: str) -> Tuple[str, str | None]:
        pattern = r'\b([А-Яа-яЁё]{3,}),\s+([а-яё]{3,})\s+и\s+([а-яё]{3,})\b'
        new, changed = self._replace_one_regex(
            text,
            pattern,
            lambda m: f"{m.group(1)} {m.group(2)} и {m.group(3)}",
        )
        return (new, 'punct_remove_homogeneous_comma') if changed else (text, None)

    def _insert_comma_before_single_i(self, text: str) -> Tuple[str, str | None]:
        pattern = r'\b([А-Яа-яЁё]{3,})\s+и\s+([а-яё]{3,})\b'

        def repl(m):
            before = text[max(0, m.start() - 2):m.start()]
            if "," in before:
                return m.group(0)
            return f"{m.group(1)}, и {m.group(2)}"

        new, changed = self._replace_one_regex(text, pattern, repl)
        return (new, 'punct_extra_comma_before_single_i') if changed and new != text else (text, None)

    def _insert_extra_comma(self, text: str) -> Tuple[str, str | None]:
        markers = ('и', 'а', 'но', 'что', 'как', 'если', 'когда', 'или')
        pattern = r'(?<![,;:])\s+(' + '|'.join(markers) + r')\b'
        new, changed = self._replace_one_regex(text, pattern, lambda m: ', ' + m.group(1), flags=re.IGNORECASE)
        if changed:
            return new, 'punct_extra_comma'

        words = list(re.finditer(r'\s+', text))
        if len(words) > 2:
            m = random.choice(words[1:-1])
            return text[:m.start()] + ', ' + text[m.end():], 'punct_extra_comma'
        return text, None

    def _bsp_colon_missing_error(self, text: str) -> Tuple[str, str | None]:
        predicates = (
            'понял', 'поняла', 'поняли', 'знал', 'знала', 'знали',
            'увидел', 'увидела', 'увидели', 'заметил', 'заметила',
            'заметили', 'решил', 'решила', 'решили', 'ясно', 'важно'
        )
        pattern = r'\b(' + '|'.join(predicates) + r'):\s+([а-яё])'
        new, changed = self._replace_one_regex(text, pattern, lambda m: f"{m.group(1)} {m.group(2)}")
        return (new, 'punct_bsp_colon_missing') if changed else (text, None)

    def _dash_error(self, text: str) -> Tuple[str, str | None]:
        if '—' in text:
            if random.random() < 0.55:
                return text.replace('—', '-', 1), 'punct_dash_to_hyphen'
            return text.replace(' — ', ' ', 1).replace('—', '', 1), 'punct_remove_dash'
        if ' - ' in text and random.random() < 0.5:
            return text.replace(' - ', ' ', 1), 'punct_remove_dash'
        return text, None

    def _remove_quote(self, text: str) -> Tuple[str, str | None]:
        quote_pairs = [('«', '»'), ('"', '"'), ('“', '”'), ('„', '“')]
        available = [(l, r) for l, r in quote_pairs if l in text and r in text]
        if not available:
            return text, None
        left, right = random.choice(available)
        return text.replace(left, '', 1).replace(right, '', 1), 'punct_remove_quotes'

    def _apply_punctuation_errors(self, text: str, op_count: int) -> Tuple[str, List[str]]:
        """Внести пунктуационные ошибки."""
        if op_count <= 0:
            return text, []

        operations = [
            self._remove_comma_before_clause_marker,
            self._remove_comma_before_clause_marker,
            self._remove_comma_before_clause_marker,
            self._remove_comma_after_intro,
            self._remove_comma_after_intro,
            self._remove_homogeneous_comma,
            self._remove_homogeneous_comma,
            self._insert_comma_before_single_i,
            self._insert_comma_before_single_i,
            self._remove_final_punctuation,
            self._remove_final_punctuation,
            self._remove_final_punctuation,
            self._bsp_colon_missing_error,
            self._bsp_colon_missing_error,
        ]

        error_types = []
        current = text
        for _ in range(op_count):
            changed = False
            for op in random.sample(operations, len(operations)):
                new, label = op(current)
                if label and new != current:
                    current = new
                    error_types.append(label)
                    changed = True
                    break
            if not changed:
                break
        return current, error_types

    def _orthography_compound_joining_error(self, text: str) -> Tuple[str, str | None]:
        operations = [
            (r'\bпо[-‑]новому\b', lambda m: self._match_case(m.group(0), "по новому")),
            (r'\bввиду\b', lambda m: self._match_case(m.group(0), "в виду")),
            (r'\bжелезнодорожный\b', lambda m: self._match_case(m.group(0), "железно-дорожный")),
            (r'\bмини[-‑]футбол\b', lambda m: self._match_case(m.group(0), "мини футбол")),
        ]
        for pattern, repl in random.sample(operations, len(operations)):
            new, changed = self._replace_one_regex(text, pattern, repl, flags=re.IGNORECASE)
            if changed:
                return new, "spelling_compound_joining"
        return text, None

    def _orthography_tsya_error(self, text: str) -> Tuple[str, str | None]:
        operations = [
            (r'\b([А-Яа-яЁё]+?)тся\b', lambda m: self._match_case(m.group(0), m.group(1) + "ться")),
            (r'\b([А-Яа-яЁё]+?)ться\b', lambda m: self._match_case(m.group(0), m.group(1) + "тся")),
        ]
        for pattern, repl in random.sample(operations, len(operations)):
            new, changed = self._replace_one_regex(text, pattern, repl, flags=re.IGNORECASE)
            if changed:
                return new, "spelling_tsya"
        return text, None

    def _orthography_suffix_pronunciation_error(self, text: str) -> Tuple[str, str | None]:
        operations = [
            (r'\b([А-Яа-яЁё]{3,})ого\b', lambda m: self._match_case(m.group(0), m.group(1) + "ово")),
            (r'\b([А-Яа-яЁё]{3,})его\b', lambda m: self._match_case(m.group(0), m.group(1) + "ево")),
        ]
        for pattern, repl in random.sample(operations, len(operations)):
            new, changed = self._replace_one_regex(text, pattern, repl, flags=re.IGNORECASE)
            if changed:
                return new, "spelling_suffix_pronunciation"
        return text, None

    def _orthography_prefix_error(self, text: str) -> Tuple[str, str | None]:
        operations = [
            (r'\bбес(?=[кпстфхцчшщ])', lambda m: self._match_case(m.group(0), "без")),
            (r'\bрас(?=[кпстфхцчшщ])', lambda m: self._match_case(m.group(0), "раз")),
            (r'\bис(?=[кпстфхцчшщ])', lambda m: self._match_case(m.group(0), "из")),
            (r'\bпре(?=красн)', lambda m: self._match_case(m.group(0), "при")),
            (r'\bпри(?=ех)', lambda m: self._match_case(m.group(0), "пре")),
        ]
        for pattern, repl in random.sample(operations, len(operations)):
            new, changed = self._replace_one_regex(text, pattern, repl, flags=re.IGNORECASE)
            if changed:
                return new, "spelling_prefix"
        return text, None

    def _orthography_n_nn_error(self, text: str) -> Tuple[str, str | None]:
        operations = [
            (r'\bдлинный\b', lambda m: self._match_case(m.group(0), "длиный")),
            (r'\bценный\b', lambda m: self._match_case(m.group(0), "ценый")),
            (r'\bискусственный\b', lambda m: self._match_case(m.group(0), "искуственный")),
        ]
        for pattern, repl in random.sample(operations, len(operations)):
            new, changed = self._replace_one_regex(text, pattern, repl, flags=re.IGNORECASE)
            if changed:
                return new, "spelling_n_nn"
        return text, None

    def _orthography_soft_hard_sign_error(self, text: str) -> Tuple[str, str | None]:
        operations = [
            (r'\bподъезд\b', lambda m: self._match_case(m.group(0), "подезд")),
            (r'\bобъект\b', lambda m: self._match_case(m.group(0), "обект")),
            (r'\bсемья\b', lambda m: self._match_case(m.group(0), "семя")),
        ]
        for pattern, repl in random.sample(operations, len(operations)):
            new, changed = self._replace_one_regex(text, pattern, repl, flags=re.IGNORECASE)
            if changed:
                return new, "spelling_soft_hard_sign"
        return text, None

    def _orthography_ne_ni_error(self, text: str) -> Tuple[str, str | None]:
        operations = [
            (r'\bне\s+(?=был|была|было|были|готов|знал|видел)', lambda m: self._match_case(m.group(0), "ни ")),
            (r'\bникогда\b', lambda m: self._match_case(m.group(0), "некогда")),
            (r'\bникто\b', lambda m: self._match_case(m.group(0), "некто")),
        ]
        for pattern, repl in random.sample(operations, len(operations)):
            new, changed = self._replace_one_regex(text, pattern, repl, flags=re.IGNORECASE)
            if changed:
                return new, "spelling_ne_ni"
        return text, None

    def _orthography_sibilant_vowel_error(self, text: str) -> Tuple[str, str | None]:
        operations = [
            (r'\bшорох\b', lambda m: self._match_case(m.group(0), "шерох")),
            (r'\bжёлтый\b', lambda m: self._match_case(m.group(0), "жолтый")),
            (r'\bчёрный\b', lambda m: self._match_case(m.group(0), "чорный")),
            (r'\bшёпот\b', lambda m: self._match_case(m.group(0), "шопот")),
        ]
        for pattern, repl in random.sample(operations, len(operations)):
            new, changed = self._replace_one_regex(text, pattern, repl, flags=re.IGNORECASE)
            if changed:
                return new, "spelling_vowel_after_sibilant"
        return text, None

    def _orthography_ts_i_y_error(self, text: str) -> Tuple[str, str | None]:
        operations = [
            (r'\bцифра\b', lambda m: self._match_case(m.group(0), "цыфра")),
            (r'\bцирк\b', lambda m: self._match_case(m.group(0), "цырк")),
            (r'\bцыган\b', lambda m: self._match_case(m.group(0), "циган")),
        ]
        for pattern, repl in random.sample(operations, len(operations)):
            new, changed = self._replace_one_regex(text, pattern, repl, flags=re.IGNORECASE)
            if changed:
                return new, "spelling_i_y_after_ts"
        return text, None

    def _orthography_hyphen_error(self, text: str) -> Tuple[str, str | None]:
        operations = [
            (r'\bиз[-‑]за\b', lambda m: self._match_case(m.group(0), "из за")),
            (r'\bкое[-‑]что\b', lambda m: self._match_case(m.group(0), "кое что")),
            (r'\bкто[-‑]то\b', lambda m: self._match_case(m.group(0), "кто то")),
            (r'\bпо[-‑]русски\b', lambda m: self._match_case(m.group(0), "по русски")),
        ]
        for pattern, repl in random.sample(operations, len(operations)):
            new, changed = self._replace_one_regex(text, pattern, repl, flags=re.IGNORECASE)
            if changed:
                return new, "spelling_hyphen"
        return text, None

    def _orthography_dictionary_word_error(self, text: str) -> Tuple[str, str | None]:
        operations = [
            (r'\bрассказать\b', lambda m: self._match_case(m.group(0), "расказать")),
            (r'\bрассчитать\b', lambda m: self._match_case(m.group(0), "расчитать")),
            (r'\bаккуратный\b', lambda m: self._match_case(m.group(0), "акуратный")),
        ]
        for pattern, repl in random.sample(operations, len(operations)):
            new, changed = self._replace_one_regex(text, pattern, repl, flags=re.IGNORECASE)
            if changed:
                return new, "spelling_dictionary_word"
        return text, None

    def _orthography_capitalization_error(self, text: str) -> Tuple[str, str | None]:
        operations = [
            (r'\bМосква\b', "москва"),
            (r'\bПрезидент(?=\s+(?:Российской\s+Федерации|РФ)\b)', "президент"),
        ]
        for pattern, repl in random.sample(operations, len(operations)):
            new, changed = self._replace_one_regex(text, pattern, repl)
            if changed:
                return new, "spelling_capitalization"
        return text, None

    def _orthography_abbreviation_case_error(self, text: str) -> Tuple[str, str | None]:
        new, changed = self._replace_one_regex(text, r'\bВУЗ\b', "вуз")
        return (new, "spelling_abbreviation_case") if changed else (text, None)

    def _orthography_borrowed_word_error(self, text: str) -> Tuple[str, str | None]:
        operations = [
            (r'\bпоролон\b', lambda m: self._match_case(m.group(0), "паралон")),
            (r'\bкардинально\b', lambda m: self._match_case(m.group(0), "координально")),
        ]
        for pattern, repl in random.sample(operations, len(operations)):
            new, changed = self._replace_one_regex(text, pattern, repl, flags=re.IGNORECASE)
            if changed:
                return new, "spelling_borrowed_word"
        return text, None

    def _apply_orthography_rule_errors(self, text: str, op_count: int) -> Tuple[str, List[str]]:
        if op_count <= 0:
            return text, []

        operations = [
            self._orthography_tsya_error,
            self._orthography_suffix_pronunciation_error,
            self._orthography_prefix_error,
            self._orthography_n_nn_error,
            self._orthography_soft_hard_sign_error,
            self._orthography_ne_ni_error,
            self._orthography_sibilant_vowel_error,
            self._orthography_ts_i_y_error,
            self._orthography_hyphen_error,
            self._orthography_dictionary_word_error,
            self._orthography_compound_joining_error,
            self._orthography_capitalization_error,
            self._orthography_abbreviation_case_error,
            self._orthography_borrowed_word_error,
        ]
        return self._apply_labeled_operations(text, op_count, operations)

    def _quote_style_error(self, text: str) -> Tuple[str, str | None]:
        pattern = r'«([^»]{2,80})»'
        new, changed = self._replace_one_regex(text, pattern, lambda m: f'"{m.group(1)}"')
        return (new, "punct_quote_style") if changed else (text, None)

    def _remove_quotes_error(self, text: str) -> Tuple[str, str | None]:
        pattern = r'«([^»]{2,80})»'
        new, changed = self._replace_one_regex(text, pattern, lambda m: m.group(1))
        return (new, "punct_remove_quotes") if changed else (text, None)

    def _direct_speech_dash_missing_error(self, text: str) -> Tuple[str, str | None]:
        new, changed = self._replace_one_regex(text, r':\s+—\s+', ': ')
        return (new, "punct_direct_speech_dash_missing") if changed else (text, None)

    def _direct_speech_inner_punct_error(self, text: str) -> Tuple[str, str | None]:
        new, changed = self._replace_one_regex(text, r'([А-Яа-яЁё]+)!»', lambda m: f'{m.group(1)}.»')
        return (new, "punct_direct_speech_inner_punct") if changed else (text, None)

    def _dash_missing_error(self, text: str) -> Tuple[str, str | None]:
        subjects = (
            'Москва', 'Россия', 'Пушкин', 'Толстой', 'Достоевский', 'Знание',
            'Учеба', 'Работа', 'Цель', 'Главное', 'Итог', 'Проект', 'Это'
        )
        pattern = r'\b(' + '|'.join(re.escape(s) for s in subjects) + r')\s+—\s+([А-Яа-яЁё][А-Яа-яЁё-]*(?:\s+[а-яё-]+){0,3})'
        new, changed = self._replace_one_regex(text, pattern, lambda m: f"{m.group(1)} {m.group(2)}")
        return (new, "punct_dash_missing") if changed else (text, None)

    def _dash_extra_error(self, text: str) -> Tuple[str, str | None]:
        pattern = r'\b(Он|Она|Это)\s+([а-яё]{4,})\b'
        new, changed = self._replace_one_regex(text, pattern, lambda m: f'{m.group(1)} — {m.group(2)}')
        return (new, "punct_dash_extra") if changed else (text, None)

    def _bsp_dash_missing_error(self, text: str) -> Tuple[str, str | None]:
        pattern = r'\b([а-яё]{3,})\s+—\s+([а-яё]{3,})\b'
        new, changed = self._replace_one_regex(text, pattern, lambda m: f"{m.group(1)} {m.group(2)}")
        return (new, "punct_bsp_dash_missing") if changed else (text, None)

    def _bracket_missing_close_error(self, text: str) -> Tuple[str, str | None]:
        new, changed = self._replace_one_regex(text, r'\(([^()]{2,40})\)', lambda m: f'({m.group(1)}')
        return (new, "punct_bracket_missing_close") if changed else (text, None)

    def _bracket_extra_error(self, text: str) -> Tuple[str, str | None]:
        pattern = r'\b([А-Яа-яЁё]{4,})\b'
        new, changed = self._replace_one_regex(text, pattern, lambda m: f'({m.group(1)})')
        return (new, "punct_bracket_extra") if changed else (text, None)

    def _bracket_pair_missing_error(self, text: str) -> Tuple[str, str | None]:
        new, changed = self._replace_one_regex(
            text,
            r'\s+\(([^()]{2,60})\)',
            lambda m: f" {m.group(1)}",
        )
        return (new, "punct_bracket_pair_missing") if changed else (text, None)

    def _ellipsis_extra_error(self, text: str) -> Tuple[str, str | None]:
        new, changed = self._replace_one_regex(text, r'\.\.\.', '....')
        return (new, "punct_ellipsis_extra") if changed else (text, None)

    def _ellipsis_missing_error(self, text: str) -> Tuple[str, str | None]:
        new, changed = self._replace_one_regex(text, r'\.\.\.', '')
        return (new, "punct_ellipsis_missing") if changed else (text, None)

    def _list_missing_colon_error(self, text: str) -> Tuple[str, str | None]:
        new, changed = self._replace_one_regex(text, r':(?=\s+[А-Яа-яЁё]+[,;])', '')
        return (new, "punct_list_missing_colon") if changed else (text, None)

    def _list_item_punctuation_error(self, text: str) -> Tuple[str, str | None]:
        new, changed = self._replace_one_regex(text, r';(?=\s+[А-Яа-яЁё]+)', ',')
        return (new, "punct_list_item_punctuation") if changed else (text, None)

    def _quote_punct_order_error(self, text: str) -> Tuple[str, str | None]:
        new, changed = self._replace_one_regex(text, r'»,', ',»')
        return (new, "punct_quote_punct_order") if changed else (text, None)

    def _apply_structural_punctuation_errors(self, text: str, op_count: int) -> Tuple[str, List[str]]:
        if op_count <= 0:
            return text, []

        operations = [
            self._remove_quotes_error,
            self._direct_speech_dash_missing_error,
            self._direct_speech_inner_punct_error,
            self._dash_missing_error,
            self._dash_extra_error,
            self._bsp_dash_missing_error,
            self._bracket_pair_missing_error,
            self._quote_punct_order_error,
        ]
        return self._apply_labeled_operations(text, op_count, operations)

    def _apply_labeled_operations(self, text: str, op_count: int, operations) -> Tuple[str, List[str]]:
        error_types = []
        current = text
        for _ in range(op_count):
            changed = False
            for op in random.sample(operations, len(operations)):
                new, label = op(current)
                if label and new != current:
                    current = new
                    error_types.append(label)
                    changed = True
                    break
            if not changed:
                break
        return current, error_types

    def _apply_strict_labeled_operations(self, text: str, op_count: int, applier) -> Tuple[str, List[str]]:
        """Apply an operation group, accepting only strict-scope labels."""
        if op_count <= 0:
            return text, []
        for _ in range(12):
            new, labels = applier(text, op_count)
            if labels and new != text and error_types_are_strict(labels):
                return new, labels
        return text, []

    def remove_punctuation(self, text: str) -> str:
        """Удалить ОДИН случайный знак препинания."""
        new, _ = self._remove_one_punctuation(text)
        return new


    def wrong_punctuation(self, text: str) -> str:
        """Заменить ОДИН случайный знак препинания на другой."""
        new, _ = self._wrong_punctuation(text)
        return new
    
    @staticmethod
    def _match_case(source: str, replacement: str) -> str:
        if source.isupper():
            return replacement.upper()
        if source[:1].isupper():
            return replacement[:1].upper() + replacement[1:]
        return replacement
    
    def introduce_double_char(self, word: str) -> str:
        """Удвоение случайного символа."""
        if len(word) > 2:
            editable_end = max(1, len(word) - (3 if len(word) >= 6 else 2))
            pos = random.randint(0, editable_end - 1)
            word = word[:pos] + word[pos] + word[pos:]
        return word
    
    def introduce_extra_char(self, word: str) -> str:
        """Вставка случайного лишнего символа."""
        chars = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
        if len(word) > 2:
            editable_end = max(1, len(word) - (3 if len(word) >= 6 else 2))
            pos = random.randint(0, editable_end)
            extra = random.choice(chars)
            word = word[:pos] + extra + word[pos:]
        return word

    def introduce_errors(
        self,
        text: str,
        error_rate: float = 1.0,
        k_range: Tuple[int, int] | None = None,
        k_weights: List[float] | None = None,
        max_errors_override: int | None = None,
        profile: str | None = None,
        return_error_types: bool = False,
) -> str | Tuple[str, List[str]]:
        """Внести ошибки в текст.
        k_range=(k_min,k_max) позволяет сделать curriculum: контролировать, сколько слов испортить.
        """
        original_text = text
        if not text or not text.strip():
            return (text, []) if return_error_types else text

        cfg = self.error_profiles.get(profile, {}) if profile else {}
        strict_scope = bool(self.error_config.get('strict_scope', False))

        word_k_range = cfg.get("word_k_range", k_range)
        word_k_weights = cfg.get("word_k_weights", k_weights)
        word_error_rate = cfg.get("word_error_rate", error_rate)
        max_word_errors = cfg.get("max_word_errors", max_errors_override)
        if strict_scope:
            word_k_range = (0, 0)
            word_error_rate = 0.0
            max_word_errors = 0

        text_with_errors, error_types = self._apply_word_errors(
            text,
            error_rate=word_error_rate,
            k_range=word_k_range,
            k_weights=word_k_weights,
            max_errors_override=max_word_errors,
        )

        punct_count = self._sample_op_count(cfg.get("punct_ops", (0, 0)))
        if not profile and self.error_config.get('punct_remove', False) and random.random() < 0.20:
            punct_count = max(punct_count, 1)
        if self.error_config.get('punct_wrong', False) and punct_count > 0:
            if strict_scope:
                candidate_text, punct_types = self._apply_strict_labeled_operations(
                    text_with_errors,
                    punct_count,
                    self._apply_punctuation_errors,
                )
            else:
                candidate_text, punct_types = self._apply_punctuation_errors(text_with_errors, punct_count)
            if punct_types:
                text_with_errors = candidate_text
                error_types.extend(punct_types)

        orthography_count = self._sample_op_count(cfg.get("orthography_ops", (0, 0)))
        if orthography_count > 0:
            if strict_scope:
                candidate_text, orthography_types = self._apply_strict_labeled_operations(
                    text_with_errors,
                    orthography_count,
                    self._apply_orthography_rule_errors,
                )
            else:
                candidate_text, orthography_types = self._apply_orthography_rule_errors(text_with_errors, orthography_count)
            if orthography_types:
                text_with_errors = candidate_text
                error_types.extend(orthography_types)

        structural_punct_count = self._sample_op_count(cfg.get("structural_punct_ops", (0, 0)))
        if structural_punct_count > 0:
            if strict_scope:
                candidate_text, structural_types = self._apply_strict_labeled_operations(
                    text_with_errors,
                    structural_punct_count,
                    self._apply_structural_punctuation_errors,
                )
            else:
                candidate_text, structural_types = self._apply_structural_punctuation_errors(
                    text_with_errors,
                    structural_punct_count,
                )
            if structural_types:
                text_with_errors = candidate_text
                error_types.extend(structural_types)

        if strict_scope and error_types and not error_types_are_strict(error_types):
            text_with_errors = original_text
            error_types = []

        ensure_error = bool(self.error_config.get('ensure_error', True))
        if ensure_error and text_with_errors == original_text:
            # Для punctuation_only сначала пробуем сделать именно пунктуационную ошибку.
            if profile == "punctuation_only":
                if strict_scope:
                    text_with_errors, punct_types = self._apply_strict_labeled_operations(
                        text_with_errors,
                        1,
                        self._apply_punctuation_errors,
                    )
                else:
                    text_with_errors, punct_types = self._apply_punctuation_errors(text_with_errors, 1)
                error_types.extend(punct_types)
            elif profile == "orthography_rules":
                if strict_scope:
                    text_with_errors, orthography_types = self._apply_strict_labeled_operations(
                        text_with_errors,
                        1,
                        self._apply_orthography_rule_errors,
                    )
                else:
                    text_with_errors, orthography_types = self._apply_orthography_rule_errors(text_with_errors, 1)
                error_types.extend(orthography_types)
            elif profile == "punctuation_structural":
                if strict_scope:
                    text_with_errors, structural_types = self._apply_strict_labeled_operations(
                        text_with_errors,
                        1,
                        self._apply_structural_punctuation_errors,
                    )
                else:
                    text_with_errors, structural_types = self._apply_structural_punctuation_errors(text_with_errors, 1)
                error_types.extend(structural_types)
            elif strict_scope:
                text_with_errors, orthography_types = self._apply_strict_labeled_operations(
                    text_with_errors,
                    1,
                    self._apply_orthography_rule_errors,
                )
                error_types.extend(orthography_types)

            if text_with_errors == original_text and not strict_scope:
                text_with_errors, word_types = self._apply_word_errors(
                    text_with_errors,
                    error_rate=0.25,
                    k_range=(1, 1),
                    max_errors_override=1,
                )
                error_types.extend(word_types)

        if strict_scope and error_types and not error_types_are_strict(error_types):
            text_with_errors = original_text
            error_types = []

        if strict_scope and text_with_errors == original_text and not error_types:
            error_types = [ERROR_SCOPE_CLEAN]

        if not error_types and text_with_errors != original_text:
            error_types.append("unknown")

        return (text_with_errors, error_types) if return_error_types else text_with_errors

    def generate_dataset_curriculum(
        self,
        texts: List[str],
        samples_per_text: int = 3,
        clean_ratio: float = 0.45,
        curriculum=None
    ) -> pd.DataFrame:
        """
        Curriculum-генерация: мешаем профили ошибок.
        curriculum: список словарей вида:
          {"name":"punctuation_only", "profile":"punctuation_only", "p":0.18}
        Старый формат с k_range/error_rate также поддерживается для совместимости.
        """
        if curriculum is None:
            curriculum = [
                {"name": "orthography_rules", "profile": "orthography_rules", "p": 0.32},
                {"name": "punctuation_only", "profile": "punctuation_only", "p": 0.34},
                {"name": "punctuation_structural", "profile": "punctuation_structural", "p": 0.24},
                {"name": "mixed", "profile": "mixed", "p": 0.08},
                {"name": "hard", "profile": "hard", "p": 0.02},
            ]

        probs = [c["p"] for c in curriculum]

        valid_texts = [text for text in texts if text and len(text) >= 10]
        total_slots = len(valid_texts) * max(1, int(samples_per_text))
        clean_slots = self._clean_slot_indices(total_slots, clean_ratio)

        data = []
        slot_idx = 0
        print("Генерация датасета (CURRICULUM)...")
        for text in tqdm(valid_texts):
            for _ in range(samples_per_text):
                # часть примеров оставляем чистыми (чтобы модель не “чинила воздух”)
                if slot_idx in clean_slots:
                    data.append({
                        "error_text": text,
                        "correct_text": text,
                        "difficulty": "clean",
                        "error_types": "clean",
                        "source_kind": "synthetic",
                        "source_dataset": "synthetic",
                        "source_domain": "",
                    })
                    slot_idx += 1
                    continue

                cfg = random.choices(curriculum, weights=probs, k=1)[0]
                profile = cfg.get("profile")
                if profile:
                    err, error_types = self.introduce_errors(
                        text,
                        profile=profile,
                        return_error_types=True,
                    )
                else:
                    err, error_types = self.introduce_errors(
                        text,
                        error_rate=cfg.get("error_rate", 0.25),
                        k_range=cfg.get("k_range"),
                        k_weights=cfg.get("k_weights"),
                        max_errors_override=cfg.get("max_errors"),
                        return_error_types=True,
                    )

                data.append({
                    "error_text": err,
                    "correct_text": text,
                    "difficulty": cfg.get("name", profile or "unknown"),
                    "error_types": "|".join(error_types) if error_types else "unknown",
                    "source_kind": "synthetic",
                    "source_dataset": "synthetic",
                    "source_domain": "",
                })
                slot_idx += 1

        df = pd.DataFrame(data).reset_index(drop=True)
        print(f"Создано {len(df)} примеров (curriculum)")
        return df

    def curated_v10_examples(self, split: str = "train") -> pd.DataFrame:
        """Small deterministic seed set for rare V10 error families."""
        rows = [
            ("Мы решили действовать по новому.", "Мы решили действовать по-новому.", "spelling_compound_joining"),
            ("В виду дождя матч перенесли.", "Ввиду дождя матч перенесли.", "spelling_compound_joining"),
            ("Железно-дорожный вокзал закрыли.", "Железнодорожный вокзал закрыли.", "spelling_compound_joining"),
            ("Мини футбол стал популярным.", "Мини-футбол стал популярным.", "spelling_compound_joining"),
            ("москва стала центром конференции.", "Москва стала центром конференции.", "spelling_capitalization"),
            ("президент Российской Федерации подписал указ.", "Президент Российской Федерации подписал указ.", "spelling_capitalization"),
            ("Этот вуз открыл лабораторию.", "Этот ВУЗ открыл лабораторию.", "spelling_abbreviation_case"),
            ("Мягкий паралон лежал на полу.", "Мягкий поролон лежал на полу.", "spelling_borrowed_word"),
            ("Нужно координально изменить подход.", "Нужно кардинально изменить подход.", "spelling_borrowed_word"),
            ("Я знаю что он придет.", "Я знаю, что он придет.", "punct_remove_comma_before_clause"),
            ("Конечно мы придем.", "Конечно, мы придем.", "punct_remove_intro_comma"),
            ("На столе лежали книги тетради и ручки.", "На столе лежали книги, тетради и ручки.", "punct_remove_homogeneous_comma"),
            ("Он пришел, и сел.", "Он пришел и сел.", "punct_extra_comma_before_single_i"),
            ("Я понял поезд ушел.", "Я понял: поезд ушел.", "punct_bsp_colon_missing"),
            ("Солнце взошло город проснулся.", "Солнце взошло — город проснулся.", "punct_bsp_dash_missing"),
            ("Он сказал: Привет.", "Он сказал: «Привет».", "punct_remove_quotes"),
            ("Он сказал: Привет!", "Он сказал: — Привет!", "punct_direct_speech_dash_missing"),
            ("Он сказал: «Привет.»", "Он сказал: «Привет!»", "punct_direct_speech_inner_punct"),
            ("Москва столица России.", "Москва — столица России.", "punct_dash_missing"),
            ("Он — студент университета.", "Он студент университета.", "punct_dash_extra"),
            ("Он пришел вчера.", "Он пришел (вчера).", "punct_bracket_pair_missing"),
            ("Он прочитал «Войну и мир,» затем ушел.", "Он прочитал «Войну и мир», затем ушел.", "punct_quote_punct_order"),
        ]
        return pd.DataFrame(
            [
                {
                    "error_text": error_text,
                    "correct_text": correct_text,
                    "difficulty": "curated_v10",
                    "error_types": error_type,
                    "source_kind": "synthetic",
                    "source_dataset": "curated_v10",
                    "source_domain": "",
                    "split": split,
                }
                for error_text, correct_text, error_type in rows
            ]
        )


    def generate_dataset(self, texts: List[str], samples_per_text: int = 3,
                     clean_ratio: float = 0.45, error_rate: float = 0.25,
                     profile_weights: Dict[str, float] | None = None) -> pd.DataFrame:
        valid_texts = [text for text in texts if text and len(text) >= 10]
        total_slots = len(valid_texts) * max(1, int(samples_per_text))
        clean_slots = self._clean_slot_indices(total_slots, clean_ratio)
        data = []
        slot_idx = 0
    
        print("Генерация датасета с ошибками...")
        for text in tqdm(valid_texts):
            for _ in range(samples_per_text):
                # часть примеров оставляем чистыми
                if slot_idx in clean_slots:
                    error_text = text
                    difficulty = "clean"
                    error_types = ["clean"]
                else:
                    profile = self._choose_profile(profile_weights)
                    error_text, error_types = self.introduce_errors(
                        text,
                        error_rate=error_rate,
                        profile=profile,
                        return_error_types=True,
                    )
                    difficulty = profile
    
                data.append({
                    'error_text': error_text,
                    'correct_text': text,
                    'difficulty': difficulty,
                    'error_types': "|".join(error_types) if error_types else "unknown",
                    'source_kind': 'synthetic',
                    'source_dataset': 'synthetic',
                    'source_domain': '',
                })
                slot_idx += 1
    
        df = pd.DataFrame(data).reset_index(drop=True)
        print(f"Создано {len(df)} обучающих примеров")
        return df

    
    def save_dataset(self, df: pd.DataFrame, output_path: str):
        """
        Сохранение датасета.
        
        Args:
            df: DataFrame с данными
            output_path: Путь для сохранения
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Датасет сохранен в {output_path}")


def main():
    """Основная функция для подготовки данных."""
    print("=" * 60)
    print("  ПОДГОТОВКА ДАТАСЕТА ДЛЯ HYBRID-МОДЕЛИ")
    print("=" * 60)
    
    # Создаем директории
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Инициализируем генератор
    generator = DatasetGenerator()
    
    # Загружаем или генерируем тексты
    raw_texts_path = 'data/raw/texts.txt'
    texts = generator.load_texts_from_file(raw_texts_path)
    
    print(f"\nЗагружено текстов: {len(texts)}")
    print(f"Пример текста: {texts[0][:100]}...")
    
    # Сплитим ЧИСТЫЕ тексты до аугментации (иначе модель 'видит' одни и те же предложения в train/val/test)
    train_texts, val_texts, test_texts = split_texts(texts, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)

    print(f"\nСплиты (по чистым предложениям):")
    print(f"  • Train texts: {len(train_texts)}")
    print(f"  • Val texts:   {len(val_texts)}")
    print(f"  • Test texts:  {len(test_texts)}")

    # Генерируем датасеты по сплитам
    SAMPLES_PER_TEXT = 3
    CURRICULUM = [
        {"name": "orthography_rules", "profile": "orthography_rules", "p": 0.32},
        {"name": "punctuation_only", "profile": "punctuation_only", "p": 0.34},
        {"name": "punctuation_structural", "profile": "punctuation_structural", "p": 0.24},
        {"name": "mixed", "profile": "mixed", "p": 0.08},
        {"name": "hard", "profile": "hard", "p": 0.02},
    ]

    df_train = generator.generate_dataset_curriculum(
        train_texts,
        samples_per_text=SAMPLES_PER_TEXT,
        clean_ratio=0.45,
        curriculum=CURRICULUM
    )
    df_train["split"] = "train"
    df_train = pd.concat([df_train, generator.curated_v10_examples("train")], ignore_index=True)


    df_val = generator.generate_dataset_curriculum(
        val_texts,
        samples_per_text=1,
        clean_ratio=0.35,
        curriculum=CURRICULUM,
    )
    df_val["split"] = "val"
    df_val = pd.concat([df_val, generator.curated_v10_examples("val")], ignore_index=True)

    df_test = generator.generate_dataset_curriculum(
        test_texts,
        samples_per_text=1,
        clean_ratio=0.35,
        curriculum=CURRICULUM,
    )
    df_test["split"] = "test"
    df_test = pd.concat([df_test, generator.curated_v10_examples("test")], ignore_index=True)

    df = pd.concat([df_train, df_val, df_test], ignore_index=True)

    # Показываем примеры
    print("\n" + "=" * 60)
    print("  ПРИМЕРЫ СГЕНЕРИРОВАННЫХ ДАННЫХ (TRAIN)")
    print("=" * 60)
    for i in range(min(3, len(df_train))):
        print(f"\nС ошибками: {df_train.iloc[i]['error_text']}")
        print(f"Правильный:  {df_train.iloc[i]['correct_text']}")

    # Сохраняем общий датасет (с колонкой split) и отдельные файлы
    output_path = 'data/processed/dataset.csv'
    generator.save_dataset(df, output_path)

    generator.save_dataset(df_train, 'data/processed/train.csv')
    generator.save_dataset(df_val, 'data/processed/val.csv')
    generator.save_dataset(df_test, 'data/processed/test.csv')

    # Статистика
    print("\n" + "=" * 60)
    print("  СТАТИСТИКА ДАТАСЕТА")
    print("=" * 60)
    print(f"Всего примеров: {len(df)}")
    print(f"Train/Val/Test: {len(df_train)}/{len(df_val)}/{len(df_test)}")
    print(f"Средняя длина текста с ошибками: {df['error_text'].str.len().mean():.1f} символов")
    print(f"Средняя длина правильного текста: {df['correct_text'].str.len().mean():.1f} символов")

    print("\nПодготовка данных завершена!")
    print(f"Данные сохранены в: {output_path}")



if __name__ == "__main__":
    main()
