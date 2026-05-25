from __future__ import annotations

import copy
from dataclasses import dataclass
import difflib
import html
from pathlib import Path
import re
import tempfile
from typing import Any

from src.config.load_config import load_config
from src.docx.docx_corrector import correct_docx, correct_docx_incremental
from src.docx.docx_reader import read_paragraphs
from src.inference.incremental_corrector import IncrementalCorrector
from src.runtime.corrector import Corrector
from src.schema.edits import CorrectionResult, RuntimeEdit


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
st: Any | None = None


@dataclass(frozen=True)
class StreamlitCorrectorLoadResult:
    corrector: Any
    kind: str
    error: str | None = None


def build_streamlit_corrector(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    memory_enabled: bool | None = None,
    doc_id: str | None = None,
) -> StreamlitCorrectorLoadResult:
    config_path = Path(config_path)
    config = _config_with_streamlit_overrides(load_config(config_path), memory_enabled=memory_enabled, doc_id=doc_id)
    corrector = Corrector.from_config(config)
    kind = "direct_neural" if getattr(corrector, "neural_backend", None) is not None else "deterministic_fallback"
    return StreamlitCorrectorLoadResult(corrector, kind)


def _cached_streamlit_corrector(
    config_path: str = str(DEFAULT_CONFIG_PATH),
    memory_enabled: bool | None = False,
    doc_id: str = "default",
) -> StreamlitCorrectorLoadResult:
    return build_streamlit_corrector(config_path, memory_enabled=memory_enabled, doc_id=doc_id)


def main() -> None:
    global st
    st = _import_streamlit()
    cached_streamlit_corrector = st.cache_resource(show_spinner="Загрузка модели корректора...")(
        _cached_streamlit_corrector
    )
    st.set_page_config(page_title="Система исправления ошибок русского языка", layout="wide")
    st.title("Система исправления ошибок русского языка")

    tab_text, tab_docx = st.tabs(["Text", "DOCX"])

    with tab_text:
        use_memory = st.checkbox(
            "Использовать контекстную память решений",
            value=False,
            disabled=True,
            help="Память решений будет адаптирована под RuntimeEdit позже.",
        )
        doc_id = _normalize_doc_id(st.text_input("ID документа", value="default"))
        use_incremental = st.checkbox("Проверять только изменённые фрагменты", value=False)
        load_result = cached_streamlit_corrector(
            str(DEFAULT_CONFIG_PATH),
            memory_enabled=use_memory,
            doc_id=doc_id,
        )
        corrector = load_result.corrector
        source = st.text_area("Исходный текст", height=180)
        if st.button("Исправить текст", type="primary") and source.strip():
            cache_key = get_incremental_cache_key(doc_id)
            previous_cache_by_doc_id = st.session_state.setdefault("previous_segment_cache_by_doc_id", {})
            last_source_text_by_doc_id = st.session_state.setdefault("last_source_text_by_doc_id", {})
            last_corrected_text_by_doc_id = st.session_state.setdefault("last_corrected_text_by_doc_id", {})
            incremental_summary = None
            if use_incremental:
                previous_cache = previous_cache_by_doc_id.get(cache_key)
                result = IncrementalCorrector(corrector).correct_incremental(source, previous_cache)
                previous_cache_by_doc_id[cache_key] = result.segment_cache
                incremental_summary = build_incremental_summary(result)
            else:
                result = corrector.correct(source)
            last_source_text_by_doc_id[cache_key] = source
            last_corrected_text_by_doc_id[cache_key] = result.corrected_text
            latest_text_correction = {
                "source": source,
                "result": result,
                "doc_id": doc_id,
            }
            if incremental_summary is not None:
                latest_text_correction["incremental_summary"] = incremental_summary
            st.session_state["latest_text_correction"] = latest_text_correction
        _render_latest_text_correction()

    with tab_docx:
        uploaded = st.file_uploader("Word-документ", type=["docx"])
        if uploaded is not None:
            docx_doc_id = _normalize_doc_id(
                st.text_input("ID документа", value=uploaded.name, key=f"docx_doc_id_{uploaded.name}")
            )
            use_docx_incremental = st.checkbox("Проверять только изменённые абзацы", value=False)
            docx_load_result = cached_streamlit_corrector(
                str(DEFAULT_CONFIG_PATH),
                memory_enabled=None,
                doc_id=docx_doc_id,
            )
            corrector = docx_load_result.corrector
            with tempfile.TemporaryDirectory() as tmp:
                input_path = Path(tmp) / "input.docx"
                output_path = Path(tmp) / "corrected.docx"
                input_path.write_bytes(uploaded.getvalue())
                if use_docx_incremental:
                    cache_key = docx_cache_key(docx_doc_id)
                    paragraph_cache_by_doc_id = st.session_state.setdefault("docx_paragraph_cache_by_doc_id", {})
                    result = correct_docx_incremental(
                        input_path,
                        output_path,
                        corrector,
                        previous_cache=paragraph_cache_by_doc_id.get(cache_key),
                    )
                    paragraph_cache_by_doc_id[cache_key] = result.paragraph_cache
                    edits = result.edits
                    docx_summary = build_docx_incremental_summary(result)
                else:
                    paragraphs = read_paragraphs(input_path)
                    edits = correct_docx(input_path, output_path, corrector)
                    docx_summary = {
                        "checked": sum(1 for paragraph in paragraphs if paragraph.strip()),
                        "reused": 0,
                        "total_edits": len(edits),
                    }
                st.info(
                    "\n".join(
                        [
                            f"Проверено абзацев: {docx_summary['checked']}",
                            f"Переиспользовано абзацев: {docx_summary['reused']}",
                            f"Всего исправлений: {docx_summary['total_edits']}",
                        ]
                    )
                )
                st.download_button(
                    "Скачать исправленный DOCX",
                    data=output_path.read_bytes(),
                    file_name="corrected.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
                st.dataframe(build_edit_rows(edits), use_container_width=True)


def build_feedback_rows(result: CorrectionResult) -> list[dict[str, object]]:
    return build_edit_rows(getattr(result, "edits", []))


def build_edit_rows(edits: list[RuntimeEdit]) -> list[dict[str, object]]:
    return [
        {
            "source": str(edit.source),
            "replacement": str(edit.replacement),
            "edit_type": str(edit.edit_type),
            "rule_id": str(edit.rule_id),
            "confidence": float(edit.confidence),
            "explanation": str(edit.explanation),
        }
        for edit in edits
    ]


def get_incremental_cache_key(doc_id: Any) -> str:
    return _normalize_doc_id(doc_id)


def build_incremental_summary(result: Any) -> dict[str, int]:
    return {
        "checked": int(getattr(result, "checked_segments", 0)),
        "reused": int(getattr(result, "reused_segments", 0)),
        "changed": int(getattr(result, "changed_segments", 0)),
    }


def docx_cache_key(doc_id: Any) -> str:
    return _normalize_doc_id(doc_id)


def build_docx_incremental_summary(result: Any) -> dict[str, int]:
    edits = getattr(result, "edits", []) or []
    return {
        "checked": int(getattr(result, "checked_paragraphs", 0)),
        "reused": int(getattr(result, "reused_paragraphs", 0)),
        "total_edits": len(edits),
    }


def _config_with_streamlit_overrides(
    config: dict[str, Any],
    *,
    memory_enabled: bool | None,
    doc_id: str | None,
) -> dict[str, Any]:
    config_copy = copy.deepcopy(config)
    if memory_enabled is None and doc_id is None:
        return config_copy

    memory_config = config_copy.get("correction_memory")
    if not isinstance(memory_config, dict):
        memory_config = {}
        config_copy["correction_memory"] = memory_config
    if memory_enabled is not None:
        memory_config["enabled"] = bool(memory_enabled)
    if doc_id is not None:
        memory_config["doc_id"] = _normalize_doc_id(doc_id)
    return config_copy


def _import_streamlit() -> Any:
    import streamlit

    return streamlit


def _render_latest_text_correction() -> None:
    latest = st.session_state.get("latest_text_correction")
    if not latest:
        return

    source = str(latest["source"])
    result = latest["result"]
    st.text_area("Исправленный текст", value=result.corrected_text, height=180)
    summary = latest.get("incremental_summary")
    if summary:
        st.info(
            "\n".join(
                [
                    f"Проверено фрагментов: {summary['checked']}",
                    f"Переиспользовано без повторной проверки: {summary['reused']}",
                    f"Изменено/добавлено: {summary['changed']}",
                ]
            )
        )
    st.markdown("Подсветка исправлений")
    st.markdown(render_highlighted_diff(source, result.corrected_text), unsafe_allow_html=True)
    st.dataframe(build_feedback_rows(result), use_container_width=True)


def _normalize_doc_id(doc_id: Any) -> str:
    normalized = str(doc_id or "").strip()
    return normalized or "default"


def render_highlighted_diff(source: str, corrected: str) -> str:
    source_tokens = _diff_tokens(source)
    corrected_tokens = _diff_tokens(corrected)
    matcher = difflib.SequenceMatcher(a=source_tokens, b=corrected_tokens)
    chunks: list[str] = []
    for tag, source_start, source_end, corrected_start, corrected_end in matcher.get_opcodes():
        source_fragment = html.escape("".join(source_tokens[source_start:source_end]))
        corrected_fragment = html.escape("".join(corrected_tokens[corrected_start:corrected_end]))
        if tag == "equal":
            chunks.append(corrected_fragment)
        elif tag == "delete":
            chunks.append(f'<span class="diff-delete">{source_fragment}</span>')
        elif tag == "insert":
            chunks.append(f'<span class="diff-insert">{corrected_fragment}</span>')
        elif tag == "replace":
            chunks.append(f'<span class="diff-delete">{source_fragment}</span>')
            chunks.append(f'<span class="diff-insert">{corrected_fragment}</span>')
    return f'{_DIFF_STYLE}<div class="diff-view">{"".join(chunks)}</div>'


def _diff_tokens(text: str) -> list[str]:
    return re.findall(r"\s+|[А-Яа-яЁёA-Za-z0-9]+|[^\w\s]", text, flags=re.UNICODE)


_DIFF_STYLE = """
<style>
.diff-view {
    border: 1px solid #d8dee4;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    background: #ffffff;
    color: #1f2328;
    line-height: 1.75;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}
.diff-insert {
    background: #dafbe1;
    color: #116329;
    border-radius: 4px;
    padding: 0.08rem 0.18rem;
}
.diff-delete {
    background: #ffebe9;
    color: #82071e;
    border-radius: 4px;
    padding: 0.08rem 0.18rem;
    text-decoration: line-through;
}
</style>
"""


if __name__ == "__main__":
    main()
