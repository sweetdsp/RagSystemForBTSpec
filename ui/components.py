"""Reusable Streamlit widgets shared by both tabs."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from rag_lab.errors import RAGLabError
from rag_lab.pipeline import QueryResult, RAGIndex
from rag_lab.vector_store import ScoredChunk

# Single-series bars use one accent colour (reference palette, slot 1).
BAR_COLOR = "#2a78d6"


def show_error(exc: Exception) -> None:
    """Friendly message for expected errors, full traceback for unexpected ones."""
    if isinstance(exc, RAGLabError):
        st.error(f"**{exc.message}**" + (f"\n\n💡 {exc.hint}" if exc.hint else ""), icon="🚫")
    else:
        st.error(f"Unexpected error: {type(exc).__name__}: {exc}", icon="🐞")
        with st.expander("Traceback"):
            st.exception(exc)


def page_label(meta: dict) -> str:
    a, b = meta.get("page_start"), meta.get("page_end")
    return f"p. {a}" if a == b else f"pp. {a}–{b}"


def build_metrics(index: RAGIndex) -> None:
    """The four headline numbers requested after an index build, plus details."""
    s = index.stats
    unit = index.config.unit
    c = st.columns(4)
    c[0].metric("Total pages", f"{s.total_pages:,}", border=True)
    c[1].metric("Chunks created", f"{s.total_chunks:,}", border=True)
    c[2].metric("Avg chunk size", f"{s.avg_chunk_chars:,.0f} chars", help=f"≈ {s.avg_chunk_tokens:,.0f} tokens. Target was {index.config.chunk_size} {unit}.", border=True)
    c[3].metric("Indexing time", f"{s.total_seconds:.2f} s", help=f"Chunking {s.chunking_seconds:.2f}s + embedding/indexing {s.indexing_seconds:.2f}s", border=True)
    st.caption(
        f"Chunk length min/median/max: {s.min_chunk_chars:,} / {s.median_chunk_chars:,.0f} / {s.max_chunk_chars:,} chars · "
        f"{s.total_embedded_tokens:,} tokens embedded · {index.config.embedding.label} · {index.config.vector_store}"
    )


def chunk_length_histogram(index: RAGIndex, height: int = 180) -> None:
    """Distribution of chunk lengths: shows how faithfully a splitter hits its target."""
    df = pd.DataFrame({"chars": index.chunk_lengths})
    chart = (
        alt.Chart(df)
        .mark_bar(color=BAR_COLOR, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("chars:Q", bin=alt.Bin(maxbins=40), title="Chunk length (characters)"),
            y=alt.Y("count():Q", title="Chunks"),
            tooltip=[alt.Tooltip("chars:Q", bin=alt.Bin(maxbins=40), title="Length bin"), alt.Tooltip("count():Q", title="Chunks")],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, width="stretch")


def chunk_previews(index: RAGIndex, key: str) -> None:
    """Browse an evenly-spaced sample of chunks with their exact length & metadata."""
    samples = index.sample_chunks
    if not samples:
        return
    i = st.slider("Sample chunk", 1, len(samples), 1, key=f"{key}_chunk_slider", help=f"{len(samples)} chunks sampled evenly across the document")
    doc = samples[i - 1]
    m = doc.metadata
    st.markdown(f"**Chunk #{m['chunk_id']}** · {page_label(m)} · **{m['char_len']:,} chars** · {m['token_len']:,} tokens · offset {m['start_index']:,}")
    st.code(doc.page_content, language=None, wrap_lines=True)
    with st.popover("Raw metadata"):
        st.json(m)


def retrieved_chunks(chunks: list[ScoredChunk], distance_label: str = "", expanded_first: bool = False) -> None:
    """Collapsible list of retrieved chunks with similarity, distance and source."""
    for rank, c in enumerate(chunks, start=1):
        m = c.document.metadata
        title = f"[{rank}] cos sim {c.similarity:.3f} · dist {c.distance:.3f} · {page_label(m)} · {len(c.document.page_content):,} chars"
        with st.expander(title, expanded=expanded_first and rank == 1):
            st.caption(f"Chunk #{m.get('chunk_id')} · distance metric: {distance_label or 'see vector store'}")
            st.markdown(f"<div class='chunk-box'>{_escape(c.document.page_content)}</div>", unsafe_allow_html=True)


def query_metrics(r: QueryResult) -> None:
    c = st.columns(4)
    c[0].metric("Retrieval", f"{r.retrieval_seconds * 1000:,.0f} ms", help="Embed the question + vector search", border=True)
    c[1].metric("Generation", f"{r.generation_seconds:.2f} s", help=r.llm_label, border=True)
    c[2].metric("Total latency", f"{r.total_seconds:.2f} s", border=True)
    tokens = f"{r.prompt_tokens:,} → {r.completion_tokens:,}" if r.prompt_tokens else f"~{r.context_chars // 4:,} ctx"
    c[3].metric("Tokens (in → out)", tokens, help="Reported by the provider; otherwise ≈ context chars / 4", border=True)


def single_bar_chart(df: pd.DataFrame, field: str, title: str, fmt: str = ",.2f", height: int = 170) -> None:
    """One measure across the 5 configurations (single series → no legend needed)."""
    chart = (
        alt.Chart(df, title=alt.TitleParams(title, anchor="start", fontSize=13))
        .mark_bar(color=BAR_COLOR, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("Config:N", sort=None, title=None, axis=alt.Axis(labelAngle=0, labelLimit=110)),
            y=alt.Y(f"{field}:Q", title=None, axis=alt.Axis(format=fmt.lstrip(","), grid=True, gridOpacity=0.3)),
            tooltip=["Config:N", alt.Tooltip(f"{field}:Q", format=fmt, title=title)],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, width="stretch")


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


CSS = """
<style>
.chunk-box {
  white-space: pre-wrap; font-size: 0.85rem; line-height: 1.45;
  padding: 0.6rem 0.8rem; border-radius: 0.4rem;
  background: rgba(127,127,127,0.08); border-left: 3px solid #2a78d6;
  max-height: 22rem; overflow-y: auto;
}
.cfg-caption { font-size: 0.8rem; opacity: 0.8; }
</style>
"""
