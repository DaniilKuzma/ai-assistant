from __future__ import annotations

import copy
from dataclasses import dataclass
import difflib
import html
from pathlib import Path
import re
import tempfile
from typing import Any

import streamlit as st

from src.config.load_config import load_config
from src.docx.docx_corrector import correct_docx
from src.inference.corrector import Corrector
from src.inference.model_corrector import TrainedModelCorrector
from src.memory import CorrectionFeedbackService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


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
    if _trained_artifacts_exist(config):
        try:
            return StreamlitCorrectorLoadResult(TrainedModelCorrector.from_config(config), "trained_model")
        except Exception as exc:
            return StreamlitCorrectorLoadResult(Corrector.from_config(config), "rule_fallback", str(exc))
    return StreamlitCorrectorLoadResult(Corrector.from_config(config), "rule_fallback")


@st.cache_resource(show_spinner="Загрузка модели корректора...")
def _cached_streamlit_corrector(
    config_path: str = str(DEFAULT_CONFIG_PATH),
    memory_enabled: bool = False,
    doc_id: str = "default",
) -> StreamlitCorrectorLoadResult:
    return build_streamlit_corrector(config_path, memory_enabled=memory_enabled, doc_id=doc_id)


def main() -> None:
    st.set_page_config(page_title="Система исправления ошибок русского языка", layout="wide")
    st.title("Система исправления ошибок русского языка")

    tab_text, tab_docx = st.tabs(["Text", "DOCX"])

    with tab_text:
        use_memory = st.checkbox("Использовать контекстную память решений", value=False)
        doc_id = _normalize_doc_id(st.text_input("ID документа", value="default"))
        load_result = _cached_streamlit_corrector(
            str(DEFAULT_CONFIG_PATH),
            memory_enabled=use_memory,
            doc_id=doc_id,
        )
        corrector = load_result.corrector
        source = st.text_area("Исходный текст", height=180)
        if st.button("Исправить текст", type="primary") and source.strip():
            result = corrector.correct(source)
            st.session_state["latest_text_correction"] = {
                "source": source,
                "result": result,
                "doc_id": doc_id,
            }
        _render_latest_text_correction(corrector, memory_enabled=use_memory)

    with tab_docx:
        docx_load_result = _cached_streamlit_corrector(str(DEFAULT_CONFIG_PATH), memory_enabled=False, doc_id="default")
        corrector = docx_load_result.corrector
        uploaded = st.file_uploader("Word-документ", type=["docx"])
        if uploaded is not None:
            with tempfile.TemporaryDirectory() as tmp:
                input_path = Path(tmp) / "input.docx"
                output_path = Path(tmp) / "corrected.docx"
                input_path.write_bytes(uploaded.getvalue())
                edits = correct_docx(input_path, output_path, corrector)
                st.download_button(
                    "Скачать исправленный DOCX",
                    data=output_path.read_bytes(),
                    file_name="corrected.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
                st.dataframe(
                    [
                        {
                            "source": edit.source,
                            "replacement": edit.replacement,
                            "type": edit.edit_type,
                            "status": edit.status,
                        }
                        for edit in edits
                    ],
                    use_container_width=True,
                )


def build_feedback_rows(result: Any) -> list[dict[str, str]]:
    return [
        {
            "source": str(edit.source),
            "replacement": str(edit.replacement),
            "type": str(edit.edit_type),
            "status": str(edit.status),
            "reason": str(edit.reason),
        }
        for edit in getattr(result, "edits", [])
    ]


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


def _render_latest_text_correction(corrector: Any, *, memory_enabled: bool) -> None:
    latest = st.session_state.get("latest_text_correction")
    if not latest:
        return

    source = str(latest["source"])
    result = latest["result"]
    doc_id = _normalize_doc_id(latest.get("doc_id"))
    st.text_area("Исправленный текст", value=result.corrected_text, height=180)
    st.markdown("Подсветка исправлений")
    st.markdown(render_highlighted_diff(source, result.corrected_text), unsafe_allow_html=True)
    st.dataframe(build_feedback_rows(result), use_container_width=True)
    _render_memory_feedback_controls(
        source=source,
        result=result,
        corrector=corrector,
        doc_id=doc_id,
        memory_enabled=memory_enabled,
    )
    _render_candidate_decisions(result)


def _render_memory_feedback_controls(
    *,
    source: str,
    result: Any,
    corrector: Any,
    doc_id: str,
    memory_enabled: bool,
) -> None:
    if not memory_enabled:
        return
    memory = getattr(corrector, "correction_memory", None)
    if memory is None:
        return

    editable_edits = [
        (index, edit)
        for index, edit in enumerate(getattr(result, "edits", []))
        if getattr(edit, "status", "") in {"accepted", "proposed"}
    ]
    if not editable_edits:
        return

    service = CorrectionFeedbackService(memory, doc_id=doc_id)
    for index, edit in editable_edits:
        st.caption(f"{edit.source or 'пусто'} -> {edit.replacement or 'пусто'}")
        columns = st.columns(3)
        key_prefix = _feedback_button_key(index, edit, doc_id)
        with columns[0]:
            if st.button("Запомнить: принимать", key=f"{key_prefix}_accept"):
                service.accept_edit(source, edit, metadata={"origin": "streamlit"})
                memory.save()
                st.success("Решение запомнено для этого контекста")
        with columns[1]:
            if st.button("Запомнить: отклонять", key=f"{key_prefix}_reject"):
                service.reject_edit(source, edit, metadata={"origin": "streamlit"})
                memory.save()
                st.success("Решение запомнено для этого контекста")
        with columns[2]:
            if st.button("Игнорировать в этом контексте", key=f"{key_prefix}_ignore"):
                service.ignore_edit(source, edit, metadata={"origin": "streamlit"})
                memory.save()
                st.success("Решение запомнено для этого контекста")


def _render_candidate_decisions(result: Any) -> None:
    decisions = getattr(result, "candidate_decisions", []) or []
    if not decisions:
        return
    with st.expander("Кандидаты модели"):
        st.dataframe(_candidate_decision_rows(decisions), use_container_width=True)


def _candidate_decision_rows(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for decision in decisions:
        candidate = decision.get("candidate")
        rows.append(
            {
                "source": str(decision.get("source", "")),
                "replacement": str(decision.get("replacement", "")),
                "type": str(decision.get("edit_type") or getattr(candidate, "edit_type", "")),
                "selected": bool(decision.get("selected", False)),
                "memory": str(decision.get("memory_decision", "")),
                "status": str(decision.get("validator_status", "")),
                "reason": str(decision.get("validator_reason", "")),
            }
        )
    return rows


def _feedback_button_key(index: int, edit: Any, doc_id: str) -> str:
    doc_key = re.sub(r"\W+", "_", doc_id, flags=re.UNICODE).strip("_") or "default"
    return f"memory_{doc_key}_{index}_{getattr(edit, 'start', -1)}_{getattr(edit, 'end', -1)}"


def _normalize_doc_id(doc_id: Any) -> str:
    normalized = str(doc_id or "").strip()
    return normalized or "default"


def _trained_artifacts_exist(config: dict[str, Any]) -> bool:
    paths = config.get("paths", {})
    adapter_dir = _resolve_project_path(paths.get("adapter_output_dir", "models/current/adapters"))
    heads_path = _resolve_project_path(paths.get("heads_output_dir", "models/current/heads")) / "heads.pt"
    return adapter_dir.exists() and heads_path.exists()


def _resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


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
