from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    st.set_page_config(page_title="Russian Edit Corrector", layout="wide")
    st.title("Russian Edit Corrector")

    load_result = _cached_streamlit_corrector()
    corrector = load_result.corrector
    if load_result.kind == "trained_model":
        st.caption("Активна обученная edit-based модель.")
    else:
        st.warning("Обученная модель не загружена, используется rule fallback.")
        if load_result.error:
            st.caption(load_result.error)

    tab_text, tab_docx = st.tabs(["Text", "DOCX"])

    with tab_text:
        source = st.text_area("Исходный текст", height=180)
        if st.button("Исправить текст", type="primary") and source.strip():
            result = corrector.correct(source)
            st.text_area("Исправленный текст", value=result.corrected_text, height=180)
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


if __name__ == "__main__":
    main()
