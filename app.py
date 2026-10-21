"""RAG Lab — an interactive Streamlit app for learning Retrieval-Augmented Generation
on the Bluetooth Core Specification.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging

import streamlit as st

from ui import components
from ui.benchmark_tab import render_benchmark_tab
from ui.sidebar import render_memory_section, render_sidebar
from ui.single_tab import render_single_tab

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("pypdf").setLevel(logging.ERROR)  # the BT spec triggers harmless decryption-padding warnings

st.set_page_config(page_title="RAG Lab · Bluetooth Spec", page_icon="📡", layout="wide")
st.markdown(components.CSS, unsafe_allow_html=True)

PIPELINE_EXPLAINER = """
**Retrieval-Augmented Generation** answers questions using a document the LLM was never trained on:

1. **Parse** the PDF into text (here with `pypdf`), page by page.
2. **Chunk** the text into overlapping pieces. *Chunk size* trades precision (small) against context (large);
   *overlap* keeps facts that straddle a boundary intact.
3. **Embed** every chunk into a vector that captures its meaning and store the vectors in a **vector database**
   (FAISS or ChromaDB, in memory).
4. **Retrieve** the *k* chunks whose vectors are closest (cosine similarity) to the question's vector.
5. **Generate**: the LLM receives the question plus those *k* chunks and must answer only from them, citing [n].

Every stage is timed and every intermediate result (chunks, scores, the exact prompt) is visible below.
"""


def main() -> None:
    sb = render_sidebar()

    st.title("📡 RAG Lab — Bluetooth Specification")
    st.caption("Experiment with chunking, embeddings and retrieval, and see how they change LLM answers, accuracy and latency.")
    with st.expander("🎓 How RAG works (read me first)"):
        st.markdown(PIPELINE_EXPLAINER)

    if not sb.ready:
        st.error("No usable PDF. Upload one in the sidebar (or fix the error shown there).")
        render_memory_section()
        st.stop()

    tab_single, tab_bench = st.tabs(["🧪 Single RAG Experimenter", "📊 Multi-RAG Benchmark"])
    with tab_single:
        render_single_tab(sb)
    with tab_bench:
        render_benchmark_tab(sb)
    render_memory_section()


if __name__ == "__main__":
    main()
