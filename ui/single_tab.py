"""Tab 1 — Single RAG Experimenter: tune one pipeline, build it, ask questions."""

from __future__ import annotations

import streamlit as st

from rag_lab.chunking import STRATEGIES, STRATEGY_HELP, size_unit
from rag_lab.errors import RAGLabError
from rag_lab.pipeline import RAGConfig, answer_question, build_index

from . import components as ui
from . import state
from .sidebar import SidebarState

EXAMPLE_QUESTIONS = [
    "What are the power classes defined for Bluetooth LE?",
    "Explain the adaptive frequency hopping (AFH) mechanism in detail.",
    "What is the range of the LE connection interval?",
    "Which channels are used as primary advertising channels in LE?",
    "What association models are available for LE Secure Connections pairing?",
]


def _parameters() -> tuple[str, int, int, int]:
    st.subheader("1 · Chunking & retrieval parameters")
    c1, c2, c3, c4 = st.columns([1.4, 1, 1, 0.8])
    strategy = c1.selectbox("Chunking strategy", STRATEGIES, key="s_strategy")
    c1.caption(STRATEGY_HELP[strategy])
    unit = size_unit(strategy)
    size = c2.slider(f"Chunk size ({unit})", 100, 4000, 1000, 50, key="s_size", help="Maximum length of each chunk.")
    overlap = c3.slider(
        f"Chunk overlap ({unit})", 0, 500, 100, 10, key="s_overlap",
        help="Text repeated between consecutive chunks so a fact split at a boundary still appears whole in one chunk.",
    )
    k = c4.slider("Top-K (k)", 1, 10, 4, key="s_k", help="How many chunks are retrieved and passed to the LLM. Changing k does not require a rebuild.")
    if overlap >= size:
        st.warning(f"Overlap ({overlap}) must be smaller than chunk size ({size}).")
    elif overlap > size * 0.5:
        st.info("Overlap above 50% of the chunk size mostly duplicates text: more vectors, little benefit.")
    return strategy, size, overlap, k


def render_single_tab(sb: SidebarState) -> None:
    st.markdown(
        "Build **one** RAG pipeline, look inside it, and ask questions. "
        "Change one parameter at a time and rebuild to see its effect on chunks, retrieval and answers."
    )
    strategy, size, overlap, k = _parameters()

    # ------------------------------------------------------------------ build
    st.subheader("2 · Build the index")
    build = st.button("🔨 Build RAG Index", type="primary", disabled=not sb.ready or overlap >= size)
    if build:
        state.close_single_index()  # free the previous index before allocating a new one
        try:
            parsed = state.get_parsed(sb.doc, sb.page_start, sb.page_end, sb.strip_boilerplate)
            config = RAGConfig("Single", strategy, size, overlap, k, sb.embedding, sb.vector_store)
            with st.spinner(f"Chunking, embedding with {sb.embedding.label} and indexing into {sb.vector_store}…"):
                bar = st.progress(0.0, text="Starting…")
                index = build_index(parsed, config, progress=lambda f, m: bar.progress(min(1.0, f), text=m))
                bar.empty()
            st.session_state[state.K_SINGLE_INDEX] = index
            st.session_state[state.K_SINGLE_SOURCE] = state.source_key(sb)
            if parsed.removed_boilerplate:
                st.toast(f"Removed {len(parsed.removed_boilerplate)} repeated header/footer line pattern(s).")
        except Exception as exc:
            ui.show_error(exc)

    index = st.session_state.get(state.K_SINGLE_INDEX)
    if index is None:
        st.info("Configure the parameters above and click **Build RAG Index**.")
        return

    cfg = index.config
    if (cfg.strategy, cfg.chunk_size, cfg.chunk_overlap) != (strategy, size, overlap):
        st.warning(f"Parameters changed since the last build ({cfg.summary()}). Click **Build RAG Index** to apply them.", icon="🔁")
    if st.session_state.get(state.K_SINGLE_SOURCE) != state.source_key(sb):
        st.warning("Document, page range, embedding model or vector store changed in the sidebar — rebuild to use them.", icon="🔁")

    ui.build_metrics(index)
    with st.expander("🔬 Inspect chunks (length distribution & sample previews)"):
        ui.chunk_length_histogram(index)
        ui.chunk_previews(index, key="single")

    # ------------------------------------------------------------------ query
    st.subheader("3 · Ask a question")
    example = st.selectbox("Example questions", EXAMPLE_QUESTIONS, key="s_example")
    question = st.text_area("Your question", value=example, height=80, key=f"s_question_{example}")
    if st.button("💬 Submit Question", type="primary", disabled=not question.strip()):
        with st.spinner(f"Retrieving top-{k} chunks and generating with {sb.llm.label}…"):
            st.session_state[state.K_SINGLE_RESULT] = answer_question(index, question.strip(), sb.llm, sb.temperature, k=k)

    result = st.session_state.get(state.K_SINGLE_RESULT)
    if result is None:
        return
    st.markdown(f"#### Answer  \n<span class='cfg-caption'>{result.config.summary()} · {result.llm_label}</span>", unsafe_allow_html=True)
    if result.error:
        st.error(result.error, icon="🚫")
    else:
        with st.container(border=True):
            st.markdown(result.answer)
    ui.query_metrics(result)

    st.markdown(f"#### Retrieved context ({len(result.chunks)} chunks, {result.context_chars:,} chars)")
    ui.retrieved_chunks(result.chunks, index.store.distance_label, expanded_first=True)
    if result.prompt_preview:
        with st.expander("📝 Exact prompt sent to the LLM"):
            st.code(result.prompt_preview, language=None, wrap_lines=True)

