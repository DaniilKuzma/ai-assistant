"""
Модуль для выделения различий между исходным и исправленным текстом.

Используется для подсветки ошибок в GUI.
"""

import difflib
from typing import List, Tuple, Dict


class TextHighlighter:
    """Класс для анализа и подсветки различий между текстами."""
    
    @staticmethod
    def find_word_differences(original: str, corrected: str) -> Tuple[List[Tuple[int, int, str]], List[Tuple[int, int, str]]]:
        """
        Находит различия между двумя текстами на уровне слов и символов.
        
        Args:
            original: Исходный текст (с ошибками)
            corrected: Исправленный текст
            
        Returns:
            Tuple из двух списков:
            - Позиции для подсветки в исходном тексте [(start, end, type), ...]
            - Позиции для подсветки в исправленном тексте [(start, end, type), ...]
        """
        # Разбиваем на слова с сохранением пробелов
        original_words = original.split()
        corrected_words = corrected.split()
        
        original_highlights = []
        corrected_highlights = []
        
        # Используем SequenceMatcher для поиска различий на уровне слов
        matcher = difflib.SequenceMatcher(None, original_words, corrected_words)
        
        original_pos = 0  # Позиция в исходном тексте
        corrected_pos = 0  # Позиция в исправленном тексте
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            # Обновляем позицию для пропущенных равных слов
            if tag == 'equal':
                # Пропускаем равные слова
                for idx in range(i1, i2):
                    original_pos += len(original_words[idx])
                    if idx < len(original_words) - 1:
                        original_pos += 1  # Пробел
                        
                for idx in range(j1, j2):
                    corrected_pos += len(corrected_words[idx])
                    if idx < len(corrected_words) - 1:
                        corrected_pos += 1  # Пробел
            
            elif tag == 'replace':
                # Замена слов - анализируем посимвольно
                orig_chunk = ' '.join(original_words[i1:i2])
                corr_chunk = ' '.join(corrected_words[j1:j2])
                
                # Находим различия внутри слов
                char_highlights_orig, char_highlights_corr = TextHighlighter._find_char_differences(
                    orig_chunk, corr_chunk, original_pos, corrected_pos
                )
                
                original_highlights.extend(char_highlights_orig)
                corrected_highlights.extend(char_highlights_corr)
                
                original_pos += len(orig_chunk)
                corrected_pos += len(corr_chunk)
                
                # Учитываем пробелы после
                if i2 < len(original_words):
                    original_pos += 1
                if j2 < len(corrected_words):
                    corrected_pos += 1
            
            elif tag == 'delete':
                # Удаленные слова (есть в оригинале, нет в исправленном)
                for idx in range(i1, i2):
                    word = original_words[idx]
                    original_highlights.append((original_pos, original_pos + len(word), 'delete'))
                    original_pos += len(word)
                    if idx < len(original_words) - 1:
                        original_pos += 1
            
            elif tag == 'insert':
                # Вставленные слова (нет в оригинале, есть в исправленном)
                for idx in range(j1, j2):
                    word = corrected_words[idx]
                    corrected_highlights.append((corrected_pos, corrected_pos + len(word), 'insert'))
                    corrected_pos += len(word)
                    if idx < len(corrected_words) - 1:
                        corrected_pos += 1
        
        return original_highlights, corrected_highlights
    
    @staticmethod
    def _find_char_differences(
        orig: str, 
        corr: str, 
        orig_offset: int, 
        corr_offset: int
    ) -> Tuple[List[Tuple[int, int, str]], List[Tuple[int, int, str]]]:
        """
        Находит различия на уровне символов между двумя строками.
        
        Args:
            orig: Исходная строка
            corr: Исправленная строка
            orig_offset: Смещение позиции в исходном тексте
            corr_offset: Смещение позиции в исправленном тексте
            
        Returns:
            Tuple из двух списков позиций для подсветки
        """
        orig_highlights = []
        corr_highlights = []
        
        # Используем SequenceMatcher для посимвольного сравнения
        matcher = difflib.SequenceMatcher(None, orig, corr)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                # Замена символов
                if i2 - i1 > 0:
                    orig_highlights.append((orig_offset + i1, orig_offset + i2, 'replace'))
                if j2 - j1 > 0:
                    corr_highlights.append((corr_offset + j1, corr_offset + j2, 'replace'))
            
            elif tag == 'delete':
                # Удаленные символы (подсветить в оригинале)
                orig_highlights.append((orig_offset + i1, orig_offset + i2, 'delete'))
            
            elif tag == 'insert':
                # Вставленные символы (подсветить в исправленном)
                corr_highlights.append((corr_offset + j1, corr_offset + j2, 'insert'))
        
        return orig_highlights, corr_highlights
    
    @staticmethod
    def merge_overlapping_highlights(highlights: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
        """
        Объединяет перекрывающиеся или соседние выделения.
        
        Args:
            highlights: Список выделений (start, end, type)
            
        Returns:
            Объединенный список выделений
        """
        if not highlights:
            return []
        
        # Сортируем по начальной позиции
        sorted_highlights = sorted(highlights, key=lambda x: x[0])
        merged = [sorted_highlights[0]]
        
        for current in sorted_highlights[1:]:
            last = merged[-1]
            
            # Если текущее выделение пересекается или соседствует с последним
            if current[0] <= last[1] + 1 and current[2] == last[2]:
                # Объединяем
                merged[-1] = (last[0], max(last[1], current[1]), last[2])
            else:
                # Добавляем как новое
                merged.append(current)
        
        return merged
    
    @staticmethod
    def find_differences_optimized(original: str, corrected: str) -> Dict:
        """
        Оптимизированный метод для нахождения различий.
        Использует комбинированный подход: сначала по словам, затем по символам.
        
        Args:
            original: Исходный текст
            corrected: Исправленный текст
            
        Returns:
            Словарь с информацией о различиях:
            {
                'original_highlights': [(start, end, type), ...],
                'corrected_highlights': [(start, end, type), ...],
                'stats': {'errors': N, 'corrections': M}
            }
        """
        orig_highlights, corr_highlights = TextHighlighter.find_word_differences(
            original, corrected
        )
        
        # Объединяем соседние выделения для более красивого отображения
        orig_highlights = TextHighlighter.merge_overlapping_highlights(orig_highlights)
        corr_highlights = TextHighlighter.merge_overlapping_highlights(corr_highlights)
        
        return {
            'original_highlights': orig_highlights,
            'corrected_highlights': corr_highlights,
            'stats': {
                'errors': len(orig_highlights),
                'corrections': len(corr_highlights)
            }
        }


def test_highlighter():
    """Тестовая функция для проверки работы highlighter."""
    highlighter = TextHighlighter()
    
    test_cases = [
        ("машиное обучене", "машинное обучение"),  # replace
        ("нейроые сети", "нейронные сети"),  # delete
        ("глубоке обучение", "глубокое обучение"),  # swap
        ("даннных анализ", "данных анализ"),  # double
        ("модлеь обучение", "модель обучение"),  # extra
    ]
    
    print("=" * 70)
    print("ТЕСТ TEXT HIGHLIGHTER")
    print("=" * 70)
    
    for orig, corr in test_cases:
        print(f"\nИсходный:     {orig}")
        print(f"Исправленный: {corr}")
        
        result = highlighter.find_differences_optimized(orig, corr)
        
        print(f"Ошибки в исходном: {result['original_highlights']}")
        print(f"Исправления:       {result['corrected_highlights']}")
        print(f"Статистика: {result['stats']}")


if __name__ == "__main__":
    test_highlighter()
