from pathlib import Path
import inspect
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.app import streamlit_app
from src.runtime.corrector import Corrector
from src.runtime.explanations import explanation_for
from src.runtime.neural_backend import DirectNeuralBackend
from src.schema.edits import CorrectionResult, RuntimeEdit
from src.schema.labels import RULE_LABELS


ROOT = Path(__file__).resolve().parents[1]


class FakeRuntimeCorrector:
    received_config = None
    from_config_calls = 0
    next_neural_backend = None

    def __init__(self, neural_backend=None):
        self.neural_backend = neural_backend

    @classmethod
    def from_config(cls, config):
        cls.received_config = config
        cls.from_config_calls += 1
        return cls(cls.next_neural_backend)


def _reset_fakes():
    FakeRuntimeCorrector.received_config = None
    FakeRuntimeCorrector.from_config_calls = 0
    FakeRuntimeCorrector.next_neural_backend = None


def test_streamlit_corrector_returns_deterministic_fallback_without_model_artifacts(tmp_path: Path, monkeypatch):
    _reset_fakes()
    config = {"paths": {"adapter_output_dir": str(tmp_path / "missing_adapter"), "heads_output_dir": str(tmp_path / "missing_heads")}}

    monkeypatch.setattr(streamlit_app, "load_config", lambda _path: config, raising=False)
    monkeypatch.setattr(streamlit_app, "Corrector", FakeRuntimeCorrector)

    result = streamlit_app.build_streamlit_corrector(tmp_path / "config.yaml")

    assert isinstance(result.corrector, FakeRuntimeCorrector)
    assert result.kind == "deterministic_fallback"
    assert result.error is None
    assert FakeRuntimeCorrector.from_config_calls == 1
    assert FakeRuntimeCorrector.received_config == config


def test_streamlit_corrector_returns_direct_neural_when_runtime_backend_loaded(tmp_path: Path, monkeypatch):
    _reset_fakes()
    backend = object()
    FakeRuntimeCorrector.next_neural_backend = backend
    config = {"runtime": {"neural_token_edits": True, "neural_punctuation": True}}

    monkeypatch.setattr(streamlit_app, "load_config", lambda _path: config, raising=False)
    monkeypatch.setattr(streamlit_app, "Corrector", FakeRuntimeCorrector)

    result = streamlit_app.build_streamlit_corrector(tmp_path / "config.yaml")

    assert isinstance(result.corrector, FakeRuntimeCorrector)
    assert result.corrector.neural_backend is backend
    assert result.kind == "direct_neural"
    assert result.error is None
    assert FakeRuntimeCorrector.from_config_calls == 1
    assert FakeRuntimeCorrector.received_config == config


def test_streamlit_app_does_not_expose_legacy_trained_model_corrector():
    assert not hasattr(streamlit_app, "TrainedModelCorrector")


def test_streamlit_app_import_does_not_require_streamlit_package():
    script = """
import builtins
import importlib

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "streamlit":
        raise ModuleNotFoundError("streamlit intentionally blocked")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
module = importlib.import_module("src.app.streamlit_app")
assert hasattr(module, "build_streamlit_corrector")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_corrector_from_config_accepts_config_path(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "runtime:\n  neural_token_edits: false\n  neural_punctuation: false\n",
        encoding="utf-8",
    )

    corrector = Corrector.from_config(config_path)

    assert isinstance(corrector, Corrector)
    assert corrector.neural_backend is None


def test_direct_neural_backend_refuses_missing_architecture_marker(tmp_path: Path):
    heads_dir = tmp_path / "heads"
    heads_dir.mkdir()
    (heads_dir / "heads.pt").write_bytes(b"not a torch artifact")
    config = {
        "model": {"lora": {"enabled": False}},
        "paths": {
            "heads_output_dir": str(heads_dir),
            "adapter_output_dir": str(tmp_path / "adapters"),
        },
    }

    with pytest.raises(RuntimeError, match="architecture marker"):
        DirectNeuralBackend.from_config(config)


def test_direct_neural_backend_refuses_debug_marker_by_default(tmp_path: Path):
    heads_dir = tmp_path / "heads"
    heads_dir.mkdir()
    (heads_dir / "heads.pt").write_bytes(b"not a torch artifact")
    (heads_dir / "architecture.json").write_text(
        json.dumps({"architecture": "direct_edit_tagger_v1", "debug_model": True}),
        encoding="utf-8",
    )
    config = {
        "model": {"lora": {"enabled": False}},
        "paths": {
            "heads_output_dir": str(heads_dir),
            "adapter_output_dir": str(tmp_path / "adapters"),
        },
    }

    with pytest.raises(RuntimeError, match="debug_model"):
        DirectNeuralBackend.from_config(config)


def test_streamlit_api_exposes_no_memory_or_user_response_helpers():
    corrector_params = inspect.signature(streamlit_app.build_streamlit_corrector).parameters
    cached_params = inspect.signature(streamlit_app._cached_streamlit_corrector).parameters
    main_source = inspect.getsource(streamlit_app.main)

    assert "memory_enabled" not in corrector_params
    assert "doc_id" not in corrector_params
    assert "memory_enabled" not in cached_params
    assert "ID документа" not in main_source
    assert "doc_id" not in main_source
    obsolete_helper = "build_" + "feed" + "back_rows"
    assert not hasattr(streamlit_app, obsolete_helper)


def test_build_edit_rows_returns_compact_edit_rows():
    result = CorrectionResult(
        source_text="Она пришла сдесь утром.",
        corrected_text="Она пришла здесь утром.",
        edits=[
            RuntimeEdit(
                start=11,
                end=17,
                source="сдесь",
                replacement="здесь",
                edit_type="spelling_replace",
                rule_id="spelling_zdes",
                confidence=0.92,
                explanation="Исправлено написание слова.",
            )
        ],
    )

    assert streamlit_app.build_edit_rows(result.edits) == [
        {
            "Было": "сдесь",
            "Стало": "здесь",
            "Правило": "Исправлено написание слова.",
        }
    ]


def test_build_edit_rows_uses_rule_explanation_when_edit_explanation_is_empty():
    result = CorrectionResult(
        source_text="Кто то пришёл.",
        corrected_text="Кто-то пришёл.",
        edits=[
            RuntimeEdit(
                start=0,
                end=6,
                source="Кто то",
                replacement="Кто-то",
                edit_type="hyphen",
                rule_id="hyphen_particles",
                confidence=0.91,
                explanation="",
            )
        ],
    )

    assert streamlit_app.build_edit_rows(result.edits) == [
        {
            "Было": "Кто то",
            "Стало": "Кто-то",
            "Правило": explanation_for("hyphen_particles"),
        }
    ]


def test_build_edit_rows_uses_specific_fallback_when_rule_is_unknown():
    result = CorrectionResult(
        source_text="лекарствяный",
        corrected_text="лекарственный",
        edits=[
            RuntimeEdit(
                start=0,
                end=12,
                source="лекарствяный",
                replacement="лекарственный",
                edit_type="spelling",
                rule_id="",
                confidence=0.91,
                explanation="",
            )
        ],
    )

    assert streamlit_app.build_edit_rows(result.edits) == [
        {
            "Было": "лекарствяный",
            "Стало": "лекарственный",
            "Правило": "Слово исправлено по нормативному написанию.",
        }
    ]


def test_build_edit_rows_labels_empty_source_and_replacement_cells():
    result = CorrectionResult(
        source_text="Он пришёл",
        corrected_text="Он пришёл.",
        edits=[
            RuntimeEdit(
                start=8,
                end=8,
                source="",
                replacement=".",
                edit_type="punctuation",
                rule_id="final_punctuation",
                confidence=1.0,
                explanation="",
            ),
            RuntimeEdit(
                start=2,
                end=3,
                source=",",
                replacement="",
                edit_type="punctuation",
                rule_id="punct_dash_syntax",
                confidence=0.9,
                explanation="",
            ),
        ],
    )

    assert streamlit_app.build_edit_rows(result.edits) == [
        {
            "Было": "Отсутствует",
            "Стало": ".",
            "Правило": explanation_for("final_punctuation"),
        },
        {
            "Было": ",",
            "Стало": "Удалить",
            "Правило": explanation_for("punct_dash_syntax"),
        },
    ]


def test_current_user_facing_rule_labels_have_explanations():
    missing = [
        rule_id
        for rule_id in RULE_LABELS
        if rule_id not in {"none", "clean_identity"} and not explanation_for(rule_id)
    ]

    assert missing == []
    assert explanation_for("spacing_normalization")


def test_build_edit_summary_counts_spelling_and_punctuation_edits():
    edits = [
        RuntimeEdit(0, 6, "Кто то", "Кто-то", "hyphen", "hyphen_particles"),
        RuntimeEdit(8, 8, "", ".", "punctuation", "final_punctuation"),
        RuntimeEdit(9, 19, "непроверил", "не проверил", "split_join", "ne_verb"),
    ]

    assert streamlit_app.build_edit_summary(edits) == {
        "total": 3,
        "orthography": 2,
        "punctuation": 1,
    }


def test_render_edit_panel_uses_expandable_user_facing_cards(monkeypatch):
    calls = []

    class FakeExpander:
        def __enter__(self):
            calls.append(("enter_expander",))
            return None

        def __exit__(self, exc_type, exc, tb):
            calls.append(("exit_expander",))
            return False

    fake_st = SimpleNamespace(
        markdown=lambda text, **kwargs: calls.append(("markdown", text, kwargs)),
        info=lambda text: calls.append(("info", text)),
        expander=lambda label, **kwargs: calls.append(("expander", label, kwargs)) or FakeExpander(),
    )
    edit = RuntimeEdit(
        start=0,
        end=6,
        source="Кто то",
        replacement="Кто-то",
        edit_type="hyphen",
        rule_id="hyphen_particles",
        explanation="",
    )

    monkeypatch.setattr(streamlit_app, "st", fake_st)
    streamlit_app.render_edit_panel([edit])

    assert ("expander", "Кто то → Кто-то", {"expanded": True}) in calls
    rendered_markdown = "\n".join(str(call[1]) for call in calls if call[0] == "markdown")
    assert "Исправлений: 1" in rendered_markdown
    assert "Орфография: 1" in rendered_markdown
    assert "Пунктуация: 0" in rendered_markdown
    assert "**Было:** Кто то" in rendered_markdown
    assert "**Стало:** Кто-то" in rendered_markdown
    assert f"**Правило:** {explanation_for('hyphen_particles')}" in rendered_markdown


def test_render_edit_panel_shows_empty_state(monkeypatch):
    calls = []
    fake_st = SimpleNamespace(
        markdown=lambda text, **kwargs: calls.append(("markdown", text, kwargs)),
        info=lambda text: calls.append(("info", text)),
        expander=lambda *args, **kwargs: calls.append(("expander", args, kwargs)),
    )

    monkeypatch.setattr(streamlit_app, "st", fake_st)
    streamlit_app.render_edit_panel([])

    assert ("info", "Правок нет") in calls
    assert not any(call[0] == "expander" for call in calls)


def test_text_tab_uses_inline_tooltips_instead_of_right_edit_panel():
    main_source = inspect.getsource(streamlit_app.main)

    assert 'st.tabs(["Текст", "DOCX-документ"])' in main_source
    assert "render_edit_panel(_latest_text_edits())" not in main_source
    assert "render_edit_panel(edits)" in main_source


def test_incremental_checkboxes_use_repeat_check_wording():
    main_source = inspect.getsource(streamlit_app.main)

    assert "Повторно проверять только изменённые фрагменты" in main_source
    assert "Повторно проверять только изменённые абзацы" in main_source
    assert "Проверять только изменённые фрагменты" not in main_source
    assert "Проверять только изменённые абзацы" not in main_source
    assert main_source.index("Повторно проверять только изменённые абзацы") < main_source.index("if uploaded is not None:")


def test_text_incremental_checkbox_is_rendered_after_model_loading_starts():
    main_source = inspect.getsource(streamlit_app.main)

    load_index = main_source.index("load_result = cached_streamlit_corrector")
    checkbox_index = main_source.index('use_incremental = st.checkbox("Повторно проверять только изменённые фрагменты"')

    assert load_index < checkbox_index


def test_text_incremental_cache_key_is_internal_and_stable():
    assert streamlit_app.text_incremental_cache_key() == "default_text"


def test_build_incremental_summary_returns_segment_counts():
    result = SimpleNamespace(checked_segments=3, reused_segments=2, changed_segments=1)

    assert streamlit_app.build_incremental_summary(result) == {
        "checked": 3,
        "reused": 2,
        "changed": 1,
    }


def test_text_gui_full_check_corrects_sentence_segments_without_cache():
    class SentenceOnlyCorrector:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def correct(self, text: str) -> CorrectionResult:
            self.calls.append(text)
            replacements = {
                "Вернувшись домой брат сразу поставил чайник.": "Вернувшись домой, брат сразу поставил чайник.",
                "Мне нужно учится готовить аккуратно.": "Мне нужно учиться готовить аккуратно.",
            }
            return CorrectionResult(text, replacements.get(text, text), [])

    corrector = SentenceOnlyCorrector()
    source = "Вернувшись домой брат сразу поставил чайник. Мне нужно учится готовить аккуратно."

    result = streamlit_app.correct_text_for_gui(
        corrector,
        source,
        use_incremental=False,
        previous_cache={"stale": object()},
    )

    assert corrector.calls == [
        "Вернувшись домой брат сразу поставил чайник.",
        "Мне нужно учится готовить аккуратно.",
    ]
    assert result.corrected_text == "Вернувшись домой, брат сразу поставил чайник. Мне нужно учиться готовить аккуратно."
    assert result.checked_segments == 2
    assert result.reused_segments == 0


def test_render_latest_text_correction_shows_compact_incremental_status(monkeypatch):
    calls = []
    result = CorrectionResult(
        source_text="Я незнаю.",
        corrected_text="Я не знаю.",
        edits=[],
    )
    fake_st = SimpleNamespace(
        session_state={
            "latest_text_correction": {
                "source": "Я незнаю.",
                "result": result,
                "incremental_summary": {"checked": 2, "reused": 1, "changed": 1},
            }
        },
        text_area=lambda *args, **kwargs: calls.append(("text_area", args, kwargs)),
        caption=lambda text: calls.append(("caption", text)),
        info=lambda text: calls.append(("info", text)),
        markdown=lambda text, **kwargs: calls.append(("markdown", text, kwargs)),
    )

    monkeypatch.setattr(streamlit_app, "st", fake_st)
    streamlit_app._render_latest_text_correction()

    assert (
        "caption",
        "Проверены только изменённые фрагменты: заново проверено 2, без повторной проверки 1, новых или изменённых 1.",
    ) in calls
    assert not any(call[0] == "info" for call in calls)


def test_render_latest_text_correction_shows_edit_summary_before_highlight(monkeypatch):
    calls = []
    result = CorrectionResult(
        source_text="Я незнаю что делать",
        corrected_text="Я не знаю, что делать",
        edits=[
            RuntimeEdit(2, 8, "незнаю", "не знаю", "split_join", "ne_verb"),
            RuntimeEdit(8, 8, "", ",", "punctuation", "comma_subordinate"),
        ],
    )
    fake_st = SimpleNamespace(
        session_state={"latest_text_correction": {"source": "Я незнаю что делать", "result": result}},
        text_area=lambda *args, **kwargs: calls.append(("text_area", args, kwargs)),
        caption=lambda text: calls.append(("caption", text)),
        markdown=lambda text, **kwargs: calls.append(("markdown", text, kwargs)),
    )

    monkeypatch.setattr(streamlit_app, "st", fake_st)
    streamlit_app._render_latest_text_correction()

    markdown_calls = [str(call[1]) for call in calls if call[0] == "markdown"]
    rendered_markdown = "\n".join(markdown_calls)
    assert "Исправлений: 2" in rendered_markdown
    assert "Орфография: 1" in rendered_markdown
    assert "Пунктуация: 1" in rendered_markdown
    assert markdown_calls.index("Подсветка исправлений") > next(
        index for index, text in enumerate(markdown_calls) if "Исправлений: 2" in text
    )


def test_render_latest_text_correction_passes_edits_to_highlight_tooltips(monkeypatch):
    calls = []
    edit = RuntimeEdit(
        start=2,
        end=8,
        source="незнаю",
        replacement="не знаю",
        edit_type="split_join",
        rule_id="ne_verb",
        explanation="Частица «не» с глаголами пишется раздельно.",
    )
    result = CorrectionResult(
        source_text="Я незнаю.",
        corrected_text="Я не знаю.",
        edits=[edit],
    )

    fake_st = SimpleNamespace(
        session_state={"latest_text_correction": {"source": "Я незнаю.", "result": result}},
        text_area=lambda *args, **kwargs: calls.append(("text_area", args, kwargs)),
        caption=lambda text: calls.append(("caption", text)),
        markdown=lambda text, **kwargs: calls.append(("markdown", text, kwargs)),
    )

    monkeypatch.setattr(streamlit_app, "st", fake_st)
    streamlit_app._render_latest_text_correction()

    rendered_markdown = "\n".join(str(call[1]) for call in calls if call[0] == "markdown")
    assert "data-tooltip=" in rendered_markdown
    assert "Частица «не» с глаголами пишется раздельно." in rendered_markdown


def test_docx_cache_key_uses_upload_name_without_exposing_document_id():
    assert streamlit_app.docx_cache_key("file-a.docx") == "file-a.docx"
    assert streamlit_app.docx_cache_key("file-a.docx") != streamlit_app.docx_cache_key("file-b.docx")
    assert streamlit_app.docx_cache_key("  ") == "uploaded_docx"


def test_docx_correction_requires_explicit_button():
    main_source = inspect.getsource(streamlit_app.main)

    assert "if docx_correction_requested(uploaded.name):" in main_source


def test_docx_correction_requested_uses_primary_button(monkeypatch):
    calls = []

    def fake_button(label, **kwargs):
        calls.append((label, kwargs))
        return False

    monkeypatch.setattr(streamlit_app, "st", SimpleNamespace(button=fake_button))

    assert streamlit_app.docx_correction_requested("draft.docx") is False
    assert calls == [
        (
            "Исправить DOCX",
            {"type": "primary", "key": "correct_docx_draft.docx"},
        )
    ]


def test_build_docx_incremental_summary_returns_paragraph_counts_and_total_edits():
    result = SimpleNamespace(checked_paragraphs=4, reused_paragraphs=2, edits=[object(), object(), object()])

    assert streamlit_app.build_docx_incremental_summary(result) == {
        "checked": 4,
        "reused": 2,
        "total_edits": 3,
    }


def test_render_docx_incremental_status_is_hidden_for_full_check(monkeypatch):
    calls = []
    fake_st = SimpleNamespace(
        caption=lambda text: calls.append(("caption", text)),
        info=lambda text: calls.append(("info", text)),
    )

    monkeypatch.setattr(streamlit_app, "st", fake_st)
    streamlit_app.render_docx_incremental_status(
        {"checked": 4, "reused": 0, "total_edits": 2},
        enabled=False,
    )

    assert calls == []


def test_render_docx_incremental_status_uses_small_caption_when_enabled(monkeypatch):
    calls = []
    fake_st = SimpleNamespace(
        caption=lambda text: calls.append(("caption", text)),
        info=lambda text: calls.append(("info", text)),
    )

    monkeypatch.setattr(streamlit_app, "st", fake_st)
    streamlit_app.render_docx_incremental_status(
        {"checked": 4, "reused": 2, "total_edits": 3},
        enabled=True,
    )

    assert calls == [
        (
            "caption",
            "Повторно проверены только изменённые абзацы: заново проверено 4, без повторной проверки 2, исправлений 3.",
        )
    ]


def test_render_highlighted_diff_marks_insertions_and_deletions():
    html = streamlit_app.render_highlighted_diff("Я незнаю что делать", "Я не знаю, что делать.")

    assert "diff-insert" in html
    assert "diff-delete" in html
    assert "не знаю" in html
    assert "," in html


def test_render_highlighted_diff_adds_rule_tooltip_to_corrected_fragment():
    edit = RuntimeEdit(
        start=21,
        end=26,
        source="серце",
        replacement="сердце",
        edit_type="spelling",
        rule_id="",
        explanation="Слово исправлено по словарю.",
    )

    html = streamlit_app.render_highlighted_diff(
        "Врач сказал пациенту серце в порядке.",
        "Врач сказал пациенту сердце в порядке.",
        [edit],
    )

    assert 'class="diff-insert diff-tooltip"' in html
    assert "data-tooltip=" in html
    assert "Было:" not in html
    assert "Стало:" not in html
    assert "Слово исправлено по словарю." in html


def test_render_highlighted_diff_escapes_rule_tooltip_content():
    edit = RuntimeEdit(
        start=0,
        end=3,
        source="bad",
        replacement="good",
        edit_type="spelling",
        rule_id="",
        explanation='Опасное "<script>alert(1)</script>".',
    )

    html = streamlit_app.render_highlighted_diff("bad", "good", [edit])

    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "&quot;&lt;script&gt;alert(1)&lt;/script&gt;&quot;" in html


def test_render_highlighted_diff_splits_adjacent_case_fixes_by_word():
    html = streamlit_app.render_highlighted_diff(
        "рач сказал пациенту что серце в порядке но нужно повторить анализ через неделю. "
        "На столе лежали ключи кошелёк и старые наушники. "
        "Вернувшись домой брат сразу поставил чайник и включил музыку. "
        "Во дворе ребята играли по детски шумно но соседи не жаловались. "
        "летом мария иванова поехала в казань к подруге. "
        "В течении часа сервер отвечал стабильно хотя нагрузка на сервис постепенно увеличивалась после рассылки. "
        "В магазине кассир сказал что акаунт покупателя не открылся а скидка не применилась. "
        "Кое-кто из соседей уже поставил велосипеды в подъезд. "
        "Мне нужно учится готовить аккуратно и не спешить. "
        "Редактор проверил лекарствяный справочник и заметил ошибку. "
        "У дома стоял вырасший за лето куст сирени.",
        "рач сказал пациенту, что сердце в порядке, но нужно повторить анализ через неделю. "
        "На столе лежали ключи, кошелёк и старые наушники. "
        "Вернувшись домой, брат сразу поставил чайник и включил музыку. "
        "Во дворе ребята играли по-детски шумно, но соседи не жаловались. "
        "Летом Мария Иванова поехала в Казань к подруге. "
        "В течение часа сервер отвечал стабильно, хотя нагрузка на сервис постепенно увеличивалась после рассылки. "
        "В магазине кассир сказал, что аккаунт покупателя не открылся, а скидка не применилась. "
        "Кое-кто из соседей уже поставил велосипеды в подъезд. "
        "Мне нужно учиться готовить аккуратно и не спешить. "
        "Редактор проверил лекарственный справочник и заметил ошибку. "
        "У дома стоял выросший за лето куст сирени.",
    )

    assert '<span class="diff-delete">летом мария иванова</span>' not in html
    assert '<span class="diff-delete">летом</span><span class="diff-insert">Летом</span>' in html
    assert '<span class="diff-delete">мария</span><span class="diff-insert">Мария</span>' in html
    assert '<span class="diff-delete">иванова</span><span class="diff-insert">Иванова</span>' in html


def test_render_highlighted_diff_shows_hyphen_insertion_without_fake_deleted_hyphen():
    html = streamlit_app.render_highlighted_diff("по детски и Кое кто", "по-детски и Кое-кто")

    assert '<span class="diff-insert">-</span>' in html
    assert '<span class="diff-delete"> </span>' not in html
    assert '<span class="diff-delete">-</span>' not in html


def test_render_highlighted_diff_adds_tooltip_to_hyphen_insertions():
    edit = RuntimeEdit(
        start=0,
        end=9,
        source="по детски",
        replacement="по-детски",
        edit_type="hyphen",
        rule_id="hyphen_adverbs",
        explanation="Наречие пишется через дефис.",
    )

    html = streamlit_app.render_highlighted_diff("по детски", "по-детски", [edit])

    assert '<span class="diff-insert diff-tooltip"' in html
    assert "Было:" not in html
    assert "Стало:" not in html
    assert "Наречие пишется через дефис." in html


def test_render_highlighted_diff_adds_tooltip_to_inner_replacement_word():
    edit = RuntimeEdit(
        start=0,
        end=9,
        source="В течении",
        replacement="В течение",
        edit_type="split_join",
        rule_id="compound_prepositions",
        explanation="Производный предлог «в течение» пишется с буквой е на конце.",
    )

    html = streamlit_app.render_highlighted_diff(
        "В течении часа сервер отвечал.",
        "В течение часа сервер отвечал.",
        [edit],
    )

    assert '<span class="diff-insert diff-tooltip"' in html
    assert ">течение</span>" in html
    assert "Производный предлог «в течение» пишется с буквой е на конце." in html
    assert "Было:" not in html
    assert "Стало:" not in html


def test_render_highlighted_diff_escapes_user_text():
    html = streamlit_app.render_highlighted_diff("<script>незнаю</script>", "<script>не знаю</script>.")

    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "diff-insert" in html
