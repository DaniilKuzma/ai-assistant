from __future__ import annotations

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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class StreamlitCorrectorLoadResult:
    corrector: Any
    kind: str
    error: str | None = None


def build_streamlit_corrector(config_path: str | Path = DEFAULT_CONFIG_PATH) -> StreamlitCorrectorLoadResult:
    config_path = Path(config_path)
    config = load_config(config_path)
    if _trained_artifacts_exist(config):
        try:
            return StreamlitCorrectorLoadResult(TrainedModelCorrector.from_config(config), "trained_model")
        except Exception as exc:
            return StreamlitCorrectorLoadResult(Corrector(), "rule_fallback", str(exc))
    return StreamlitCorrectorLoadResult(Corrector(), "rule_fallback")


@st.cache_resource(show_spinner="Загрузка модели корректора...")
def _cached_streamlit_corrector(config_path: str = str(DEFAULT_CONFIG_PATH)) -> StreamlitCorrectorLoadResult:
    return build_streamlit_corrector(config_path)


def main() -> None:
    st.set_page_config(page_title="Система исправления ошибок русского языка", layout="wide")
    st.title("Система исправления ошибок русского языка")

    load_result = _cached_streamlit_corrector()
    corrector = load_result.corrector

    tab_text, tab_docx = st.tabs(["Text", "DOCX"])

    with tab_text:
        source = st.text_area("Исходный текст", height=180)
        if st.button("Исправить текст", type="primary") and source.strip():
            result = corrector.correct(source)
            st.text_area("Исправленный текст", value=result.corrected_text, height=180)
            st.markdown("Подсветка исправлений")
            st.markdown(render_highlighted_diff(source, result.corrected_text), unsafe_allow_html=True)
            st.dataframe(
                [
                    {
                        "source": edit.source,
                        "replacement": edit.replacement,
                        "type": edit.edit_type,
                        "status": edit.status,
                        "reason": edit.reason,
                    }
                    for edit in result.edits
                ],
                use_container_width=True,
            )

    with tab_docx:
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


def _trained_artifacts_exist(config: dict[str, Any]) -> bool:
    paths = config.get("paths", {})
    adapter_dir = _resolve_project_path(paths.get("adapter_output_dir", "models/adapters/latest"))
    heads_path = _resolve_project_path(paths.get("heads_output_dir", "models/heads/latest")) / "heads.pt"
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
