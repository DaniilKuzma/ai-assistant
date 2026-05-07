"""
GUI приложение для интеллектуального ассистента проверки орфографии.

Используется tkinter для создания графического интерфейса.
"""

import sys
import os
from pathlib import Path
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Добавляем путь к модулям src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox, filedialog
    from tkinter import font as tkfont
except ModuleNotFoundError as exc:
    if exc.name != "tkinter":
        raise
    print(
        "Tkinter is not installed in this WSL Python environment.\n"
        "Install the Ubuntu system package and run the app again:\n\n"
        "  sudo apt update\n"
        "  sudo apt install python3.10-tk\n\n"
        "Then launch:\n"
        "  PYTHONPATH=src .venv/bin/python app/main.py",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from docx import Document  # python-docx
except ImportError:
    Document = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
except ImportError:
    canvas = None
    A4 = None
    mm = None
    SimpleDocTemplate = None
    Paragraph = None

from hybrid_corrector import HybridCorrector
from text_highlighter import TextHighlighter


class SpellingAssistantGUI:
    """Класс графического интерфейса приложения."""
    
    def __init__(self, root):
        """
        Инициализация GUI.
        
        Args:
            root: Корневое окно tkinter
        """
        self.root = root
        self.root.title("Интеллектуальный ассистент проверки орфографии")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Настройка стиля
        self.setup_style()
        
        # Переменные
        self.corrector = None
        self.loading = False
        self.highlighter = TextHighlighter()
        self.current_original_text = ""
        self.current_corrected_text = ""
        self.highlight_enabled = tk.BooleanVar(value=False)
        
        # Создание интерфейса
        self.create_widgets()
        
        # Загрузка модели
        self.load_model()
    
    def setup_style(self):
        """Настройка стилей приложения."""
        self.root.configure(bg='#f0f0f0')
        
        # Шрифты
        self.title_font = tkfont.Font(family="Arial", size=14, weight="bold")
        self.text_font = tkfont.Font(family="Consolas", size=11)
        self.button_font = tkfont.Font(family="Arial", size=10, weight="bold")
        
        # Цвета
        self.primary_color = "#4CAF50"
        self.secondary_color = "#2196F3"
        self.error_color = "#f44336"
        self.bg_color = "#ffffff"
    
    def create_widgets(self):
        """Создание виджетов интерфейса."""
        # Заголовок
        header_frame = tk.Frame(self.root, bg=self.primary_color, height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="Интеллектуальный ассистент проверки орфографии",
            font=("Arial", 18, "bold"),
            bg=self.primary_color,
            fg="white"
        )
        title_label.pack(pady=20)
        
        # Основной контейнер
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Левая панель - ввод текста
        left_frame = tk.LabelFrame(
            main_frame,
            text="Текст с ошибками",
            font=self.title_font,
            bg=self.bg_color,
            relief=tk.GROOVE,
            borderwidth=2
        )
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.input_text = scrolledtext.ScrolledText(
            left_frame,
            wrap=tk.WORD,
            font=self.text_font,
            height=20,
            borderwidth=0,
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.input_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.input_text.insert(1.0, "Введите текст с возможными ошибками здесь...")
        self.input_text.bind("<FocusIn>", self.clear_placeholder)
        
        # Правая панель - исправленный текст
        right_frame = tk.LabelFrame(
            main_frame,
            text="Исправленный текст",
            font=self.title_font,
            bg=self.bg_color,
            relief=tk.GROOVE,
            borderwidth=2
        )
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.output_text = scrolledtext.ScrolledText(
            right_frame,
            wrap=tk.WORD,
            font=self.text_font,
            height=20,
            borderwidth=0,
            relief=tk.FLAT,
            padx=10,
            pady=10,
            state=tk.DISABLED
        )
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Панель управления
        control_frame = tk.Frame(self.root, bg='#f0f0f0')
        control_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Кнопки
        button_frame = tk.Frame(control_frame, bg='#f0f0f0')
        button_frame.pack(pady=10)
        
        self.check_button = tk.Button(
            button_frame,
            text="Проверить текст",
            command=self.check_text,
            font=self.button_font,
            bg=self.primary_color,
            fg="white",
            padx=20,
            pady=10,
            relief=tk.FLAT,
            cursor="hand2"
        )
        self.check_button.pack(side=tk.LEFT, padx=5)
        
        clear_button = tk.Button(
            button_frame,
            text="Очистить",
            command=self.clear_text,
            font=self.button_font,
            bg=self.error_color,
            fg="white",
            padx=20,
            pady=10,
            relief=tk.FLAT,
            cursor="hand2"
        )
        clear_button.pack(side=tk.LEFT, padx=5)
        
        copy_button = tk.Button(
            button_frame,
            text="Копировать результат",
            command=self.copy_result,
            font=self.button_font,
            bg=self.secondary_color,
            fg="white",
            padx=20,
            pady=10,
            relief=tk.FLAT,
            cursor="hand2"
        )
        copy_button.pack(side=tk.LEFT, padx=5)
        
        # Кнопки для работы с файлами
        file_button = tk.Button(
            button_frame,
            text="Загрузить файл",
            command=self.load_file,
            font=self.button_font,
            bg="#9C27B0",
            fg="white",
            padx=20,
            pady=10,
            relief=tk.FLAT,
            cursor="hand2"
        )
        file_button.pack(side=tk.LEFT, padx=5)
        
        save_button = tk.Button(
            button_frame,
            text="Сохранить результат",
            command=self.save_result,
            font=self.button_font,
            bg="#FF9800",
            fg="white",
            padx=20,
            pady=10,
            relief=tk.FLAT,
            cursor="hand2"
        )
        save_button.pack(side=tk.LEFT, padx=5)
        
        # Чекбокс для подсветки ошибок
        self.highlight_checkbox = tk.Checkbutton(
            button_frame,
            text="Подсветить ошибки",
            variable=self.highlight_enabled,
            command=self.toggle_highlight,
            font=("Arial", 10),
            bg='#f0f0f0',
            activebackground='#f0f0f0',
            cursor="hand2"
        )
        self.highlight_checkbox.pack(side=tk.LEFT, padx=15)
        
        # Статус-бар
        self.status_frame = tk.Frame(control_frame, bg='#f0f0f0')
        self.status_frame.pack(fill=tk.X)
        
        self.status_label = tk.Label(
            self.status_frame,
            text="Готов к работе",
            font=("Arial", 10),
            bg='#f0f0f0',
            fg='#555'
        )
        self.status_label.pack(side=tk.LEFT)
        
        self.confidence_label = tk.Label(
            self.status_frame,
            text="",
            font=("Arial", 10),
            bg='#f0f0f0',
            fg='#555'
        )
        self.confidence_label.pack(side=tk.RIGHT)
        
        # Настраиваем теги для подсветки
        self.setup_highlight_tags()
    
    def load_model(self):
        self.status_label.config(text="Загрузка модели...", fg='orange')
        self.check_button.config(state=tk.DISABLED)
        self.root.update()

        try:
            root_dir = Path(__file__).resolve().parent.parent
            models_dir = root_dir / "models"

            self.corrector = HybridCorrector(
                model_path=str(models_dir / "hybrid_corrector.keras"),
                preprocessor_path=str(models_dir / "hybrid_preprocessor.pkl"),
                candidate_generator_path=str(models_dir / "candidate_generator.pkl"),
                strictness="strict",
                context_device="auto",
                candidate_top_k=16,
            )

            if self.corrector.is_trained:
                self.status_label.config(text="Hybrid-модель загружена.", fg='green')
            else:
                self.status_label.config(
                    text="Hybrid-модель еще не обучена; включен словарный режим.",
                    fg='orange'
                )
            self.check_button.config(state=tk.NORMAL)

        except Exception as e:
            self.status_label.config(text=f"Ошибка загрузки: {str(e)}", fg='red')
            messagebox.showerror(
                "Ошибка",
                f"Не удалось загрузить модель:\n\n{str(e)}\n\n"
                f"Убедитесь, что модель обучена и сохранена в папке 'models/'."
            )
    
    def clear_placeholder(self, event):
        """Очистка placeholder текста."""
        if self.input_text.get(1.0, tk.END).strip() == "Введите текст с возможными ошибками здесь...":
            self.input_text.delete(1.0, tk.END)
    
    def check_text(self):
        """Проверка и исправление текста."""
        if self.corrector is None:
            messagebox.showwarning("Предупреждение", "Модель не загружена!")
            return
        
        # Получаем текст
        input_text = self.input_text.get(1.0, tk.END).strip()
        
        if not input_text or input_text == "Введите текст с возможными ошибками здесь...":
            messagebox.showwarning("Предупреждение", "Введите текст для проверки!")
            return
        
        # Обновляем статус
        self.status_label.config(text="Проверка текста...", fg='orange')
        self.check_button.config(state=tk.DISABLED)
        self.root.update()
        
        try:
            lines = [line.rstrip("\r") for line in input_text.split("\n")]
            corrected_lines = [""] * len(lines)
            non_empty_indices = [index for index, line in enumerate(lines) if line.strip()]
            non_empty_lines = [lines[index] for index in non_empty_indices]
            confidence = 1.0

            if non_empty_lines:
                if hasattr(self.corrector, "correct_many"):
                    results = self.corrector.correct_many(non_empty_lines, batch_size=512)
                    confidences = []
                    for index, result in zip(non_empty_indices, results):
                        corrected_lines[index] = result.corrected
                        confidences.append(float(result.confidence))
                    if confidences:
                        confidence = sum(confidences) / len(confidences)
                else:
                    confidences = []
                    for index in non_empty_indices:
                        corrected_lines[index] = self.corrector.correct(lines[index])
                        confidences.append(float(self.corrector.get_confidence(lines[index])))
                    if confidences:
                        confidence = sum(confidences) / len(confidences)

            corrected_text = "\n".join(corrected_lines)
            
            # Сохраняем тексты для подсветки
            self.current_original_text = input_text
            self.current_corrected_text = corrected_text
            
            # Отображаем результат
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(1.0, corrected_text)
            self.output_text.config(state=tk.DISABLED)
            
            # Если подсветка включена, применяем её
            if self.highlight_enabled.get():
                self.apply_highlighting()
            
            # Обновляем статус
            self.status_label.config(text="Проверка завершена!", fg='green')
            #self.confidence_label.config(text=f"Уверенность: {confidence:.1%}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка:\n{str(e)}")
            self.status_label.config(text="Ошибка при проверке", fg='red')
        
        finally:
            self.check_button.config(state=tk.NORMAL)
    
    def clear_text(self):
        """Очистка всех полей."""
        self.input_text.delete(1.0, tk.END)
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state=tk.DISABLED)
        self.status_label.config(text="Готов к работе", fg='#555')
        self.confidence_label.config(text="")
        
        # Очищаем сохраненные тексты
        self.current_original_text = ""
        self.current_corrected_text = ""
        self.highlight_enabled.set(False)
    
    def copy_result(self):
        """Копирование результата в буфер обмена."""
        result_text = self.output_text.get(1.0, tk.END).strip()
        if result_text:
            self.root.clipboard_clear()
            self.root.clipboard_append(result_text)
            messagebox.showinfo("Успех", "Результат скопирован в буфер обмена!")
        else:
            messagebox.showwarning("Предупреждение", "Нет результата для копирования!")
    
    def toggle_highlight(self):
        """Переключение подсветки ошибок."""
        if self.highlight_enabled.get():
            # Включаем подсветку
            if self.current_original_text and self.current_corrected_text:
                self.apply_highlighting()
            else:
                messagebox.showinfo(
                    "Информация",
                    "Сначала проверьте текст, чтобы увидеть подсветку ошибок."
                )
                self.highlight_enabled.set(False)
        else:
            # Выключаем подсветку
            self.remove_highlighting()
    
    def apply_highlighting(self):
        """Применение подсветки различий между текстами."""
        if not self.current_original_text or not self.current_corrected_text:
            return
        
        # Настраиваем теги для подсветки
        self.setup_highlight_tags()
        
        try:
            # Находим различия
            result = self.highlighter.find_differences_optimized(
                self.current_original_text,
                self.current_corrected_text
            )
            
            # Применяем подсветку к исходному тексту (красный)
            self.input_text.tag_remove('error', '1.0', tk.END)
            for start, end, _ in result['original_highlights']:
                start_idx = f"1.0+{start}c"
                end_idx = f"1.0+{end}c"
                self.input_text.tag_add('error', start_idx, end_idx)
            
            # Применяем подсветку к исправленному тексту (зеленый)
            self.output_text.config(state=tk.NORMAL)
            self.output_text.tag_remove('correction', '1.0', tk.END)
            for start, end, _ in result['corrected_highlights']:
                start_idx = f"1.0+{start}c"
                end_idx = f"1.0+{end}c"
                self.output_text.tag_add('correction', start_idx, end_idx)
            self.output_text.config(state=tk.DISABLED)
            
            # Обновляем статус
            errors_count = result['stats']['errors']
            corrections_count = result['stats']['corrections']
            self.status_label.config(
                text=f"Найдено различий: {errors_count} в исходном, {corrections_count} в исправленном",
                fg='blue'
            )
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось применить подсветку:\n{str(e)}")
    
    def remove_highlighting(self):
        """Удаление подсветки."""
        # Удаляем все теги подсветки
        self.input_text.tag_remove('error', '1.0', tk.END)
        
        self.output_text.config(state=tk.NORMAL)
        self.output_text.tag_remove('correction', '1.0', tk.END)
        self.output_text.config(state=tk.DISABLED)
        
        # Возвращаем статус
        self.status_label.config(text="Проверка завершена!", fg='green')
    
    def setup_highlight_tags(self):
        """Настройка тегов для подсветки текста."""
        # Тег для ошибок (красный фон)
        self.input_text.tag_configure(
            'error',
            background='#ffcccc',  # Светло-красный фон
            foreground='#cc0000'   # Темно-красный текст
        )
        
        # Тег для исправлений (зеленый фон)
        self.output_text.tag_configure(
            'correction',
            background='#ccffcc',  # Светло-зеленый фон
            foreground='#009900'   # Темно-зеленый текст
        )
    
    def _clean_pdf_text(self, text: str) -> str:
        """
        Умная очистка текста из PDF: объединяет разорванные строки, сохраняя абзацы.
        
        Args:
            text: Исходный текст из PDF
            
        Returns:
            Очищенный текст с правильными параграфами
        """
        import re
        
        lines = text.split('\n')
        cleaned_lines = []
        current_paragraph = []
        
        def is_paragraph_start(line: str) -> bool:
            """Определяет, начинается ли новый абзац."""
            # Нумерация (1), 2), 3) или 1., 2., 3.
            if re.match(r'^\d+[).]\s', line):
                return True
            # Маркеры списка (-, •, *, etc.)
            if re.match(r'^[\-•*]\s', line):
                return True
            # Короткие заголовки (все слова с большой буквы)
            words = line.split()
            if len(words) <= 4 and all(w[0].isupper() for w in words if w and w[0].isalpha()):
                return True
            return False
        
        def should_merge_lines(prev_line: str, current_line: str) -> bool:
            """Определяет, нужно ли объединить строки."""
            # НЕ объединяем, если текущая строка - начало нового абзаца (нумерация, маркер)
            if is_paragraph_start(current_line):
                return False
            
            # ПРИОРИТЕТ: НЕ объединяем, если предыдущая строка заканчивается точкой/вопросом/восклицанием
            # и текущая начинается с большой буквы (новое предложение)
            if (prev_line.rstrip().endswith(('.', '!', '?')) and 
                current_line and current_line[0].isupper()):
                return False
            
            # Объединяем, если предыдущая строка ОЧЕНЬ короткая (< 20 символов)
            # Это вероятно разрыв слова внутри предложения
            if len(prev_line) < 20:
                return True
            
            # Объединяем, если предыдущая заканчивается двоеточием (продолжение списка)
            if prev_line.rstrip().endswith(':'):
                return True
            
            # Объединяем, если предыдущая строка короткая (< 60 символов)
            # и не заканчивается точкой (вероятно разрыв)
            if len(prev_line) < 60 and not prev_line.rstrip().endswith(('.', '!', '?')):
                return True
            
            # Объединяем, если предыдущая строка не заканчивается знаком препинания
            if not prev_line.rstrip().endswith(('.', '!', '?', ':', ';')):
                return True
            
            return False
        
        for line in lines:
            stripped = line.strip()
            
            # Пустая строка - сохраняем только если есть накопленный параграф
            if not stripped:
                if current_paragraph:
                    cleaned_lines.append(' '.join(current_paragraph))
                    current_paragraph = []
                continue
            
            # Если это первая строка параграфа
            if not current_paragraph:
                current_paragraph.append(stripped)
                continue
            
            # Проверяем, объединять ли с предыдущей строкой
            last_line = current_paragraph[-1]
            
            if should_merge_lines(last_line, stripped):
                current_paragraph.append(stripped)
            else:
                # Завершаем текущий параграф, начинаем новый
                cleaned_lines.append(' '.join(current_paragraph))
                current_paragraph = [stripped]
        
        # Добавляем последний параграф
        if current_paragraph:
            cleaned_lines.append(' '.join(current_paragraph))
        
        # Объединяем строки БЕЗ пустых строк между ними
        result = '\n'.join(cleaned_lines)
        
        return result.strip()
    
    def load_file(self):
        """Загрузка текста из файла (txt, docx, pdf)."""
        filename = filedialog.askopenfilename(
            title="Выберите файл",
            filetypes=[
                ("Все поддерживаемые", "*.txt;*.docx;*.pdf"),
                ("Text files", "*.txt"),
                ("Word files", "*.docx"),
                ("PDF files", "*.pdf"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            try:
                content = ""
                file_ext = os.path.splitext(filename)[1].lower()
                
                if file_ext == ".txt":
                    # Загрузка TXT
                    with open(filename, 'r', encoding='utf-8') as f:
                        content = f.read()
                
                elif file_ext == ".docx":
                    # Загрузка DOCX
                    if Document is None:
                        messagebox.showerror(
                            "Ошибка",
                            "Библиотека python-docx не установлена.\n"
                            "Установите её командой: pip install python-docx"
                        )
                        return
                    
                    doc = Document(filename)
                    paragraphs = [para.text for para in doc.paragraphs]
                    content = "\n".join(paragraphs)
                
                elif file_ext == ".pdf":
                    # Загрузка PDF
                    if fitz is None:
                        messagebox.showerror(
                            "Ошибка",
                            "Библиотека PyMuPDF не установлена.\n"
                            "Установите её командой: pip install PyMuPDF"
                        )
                        return
                    
                    pdf_document = fitz.open(filename)
                    text_parts = []
                    for page_num in range(pdf_document.page_count):
                        page = pdf_document[page_num]
                        page_text = page.get_text()
                        text_parts.append(page_text)
                    pdf_document.close()
                    
                    # Объединяем текст всех страниц
                    raw_content = "\n".join(text_parts)
                    
                    # Умная очистка: объединяем разорванные строки
                    content = self._clean_pdf_text(raw_content)
                
                else:
                    messagebox.showwarning(
                        "Предупреждение",
                        f"Формат файла {file_ext} не поддерживается.\n"
                        "Поддерживаемые форматы: .txt, .docx, .pdf"
                    )
                    return
                
                # Вставляем содержимое в текстовое поле
                self.input_text.delete(1.0, tk.END)
                self.input_text.insert(1.0, content)
                self.status_label.config(
                    text=f"Загружен: {os.path.basename(filename)}", 
                    fg='green'
                )
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить файл:\n{str(e)}")
    
    def save_result(self):
        """Сохранение результата в файл (txt, docx, pdf)."""
        result_text = self.output_text.get(1.0, tk.END).strip()
        
        if not result_text:
            messagebox.showwarning("Предупреждение", "Нет результата для сохранения!")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Сохранить результат",
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt"),
                ("Word files", "*.docx"),
                ("PDF files", "*.pdf"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            try:
                file_ext = os.path.splitext(filename)[1].lower()
                
                if file_ext == ".txt":
                    # Сохранение TXT
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(result_text)
                
                elif file_ext == ".docx":
                    # Сохранение DOCX
                    if Document is None:
                        messagebox.showerror(
                            "Ошибка",
                            "Библиотека python-docx не установлена.\n"
                            "Установите её командой: pip install python-docx"
                        )
                        return
                    
                    doc = Document()
                    # Разбиваем текст на параграфы
                    paragraphs = result_text.split('\n')
                    for para_text in paragraphs:
                        doc.add_paragraph(para_text)
                    doc.save(filename)
                
                elif file_ext == ".pdf":
                    # Сохранение PDF
                    if SimpleDocTemplate is None:
                        messagebox.showerror(
                            "Ошибка",
                            "Библиотека reportlab не установлена.\n"
                            "Установите её командой: pip install reportlab"
                        )
                        return
                    
                    self._save_as_pdf(filename, result_text)
                
                else:
                    # Если расширение не указано или неизвестно, сохраняем как TXT
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(result_text)
                
                messagebox.showinfo("Успех", f"Результат сохранен:\n{filename}")
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{str(e)}")
    
    def _save_as_pdf(self, filename, text):
        """Вспомогательный метод для сохранения текста в PDF с поддержкой русского языка."""
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.units import cm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        
        # Регистрируем русский шрифт (DejaVu Sans)
        # Этот шрифт обычно есть в системе или устанавливается с reportlab
        try:
            # Попытка загрузить системный шрифт
            font_paths = [
                # Windows
                r"C:\Windows\Fonts\DejaVuSans.ttf",
                r"C:\Windows\Fonts\Arial.ttf",
                # Для других систем можно добавить пути
            ]
            
            font_registered = False
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        pdfmetrics.registerFont(TTFont('CustomFont', font_path))
                        font_registered = True
                        break
                    except:
                        continue
            
            if not font_registered:
                # Используем встроенный шрифт Helvetica (ограниченная поддержка кириллицы)
                font_name = 'Helvetica'
            else:
                font_name = 'CustomFont'
                
        except Exception as e:
            # В случае ошибки используем стандартный шрифт
            font_name = 'Helvetica'
        
        # Создаем PDF документ
        doc = SimpleDocTemplate(
            filename,
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )
        
        # Стили
        styles = getSampleStyleSheet()
        style = ParagraphStyle(
            'CustomStyle',
            parent=styles['Normal'],
            fontName=font_name,
            fontSize=12,
            leading=16,
            alignment=TA_LEFT
        )
        
        # Создаем содержимое
        story = []
        
        # Разбиваем текст на параграфы
        paragraphs = text.split('\n')
        for para_text in paragraphs:
            if para_text.strip():
                # Экранируем специальные символы для reportlab
                para_text = para_text.replace('&', '&amp;')
                para_text = para_text.replace('<', '&lt;')
                para_text = para_text.replace('>', '&gt;')
                
                p = Paragraph(para_text, style)
                story.append(p)
            else:
                # Добавляем пустую строку
                story.append(Spacer(1, 0.3*cm))
        
        # Генерируем PDF
        doc.build(story)


def main():
    """Запуск приложения."""
    root = tk.Tk()
    app = SpellingAssistantGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

