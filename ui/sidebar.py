"""Sidebar: document selection and the model settings shared by both tabs.

Embedding model, vector store and LLM are global on purpose: in the benchmark
tab only the chunking parameters vary, which keeps the comparison a fair,
controlled experiment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import streamlit as st

from rag_lab import settings
from rag_lab.errors import RAGLabError
from rag_lab.settings import ModelSpec

from . import state
from .components import show_error

DEFAULT_PAGE_LIMIT = int(os.getenv("DEFAULT_PAGE_LIMIT", "400"))


def _human_size(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{n / 1e3:.0f} KB"


@dataclass
class SidebarState:
    doc: state.DocumentSource | None
    page_start: int
    page_end: int
    strip_boilerplate: bool
    embedding: ModelSpec
    vector_store: str
    llm: ModelSpec
    temperature: float

    @property
    def ready(self) -> bool:
        return self.doc is not None


def _provider_model_picker(kind: str, catalog: dict[str, list[str]], default: ModelSpec, disabled: dict[str, str]) -> ModelSpec:
    providers = list(catalog)
    provider = st.selectbox(
        f"{kind} provider",
        providers,
        index=providers.index(default.provider),
        key=f"{kind}_provider",
        format_func=lambda p: f"{p} — {disabled[p]}" if p in disabled else p,
    )
    if provider in disabled:
        st.caption(f"⚠️ {disabled[provider]}")
    options = catalog[provider]
    model = st.selectbox(f"{kind} model", options, key=f"{kind}_model_{provider}")
    if provider != settings.EMB_HASHING and provider != settings.LLM_EXTRACTIVE:
        custom = st.text_input(f"…or custom {kind.lower()} model name", key=f"{kind}_custom_{provider}", placeholder="optional")
        model = custom.strip() or model
    return ModelSpec(provider, model)


def render_sidebar() -> SidebarState:
    sb = st.sidebar
    sb.title("⚙️ RAG Settings")

    # ---------------------------------------------------------------- Document
    sb.header("📄 Document")
    uploaded = sb.file_uploader("Upload a PDF", type=["pdf"], help="Defaults to the Bluetooth Core Specification if present in data/, else the bundled sample primer.")
    doc = None
    try:
        doc = state.resolve_document(uploaded)
    except RAGLabError as exc:
        with sb:
            show_error(exc)

    page_start, page_end, strip = 1, 1, True
    if doc:
        origin = {"upload": "Uploaded file", "default": "Bluetooth spec (DEFAULT_PDF_PATH)", "sample": "Bundled sample primer (fallback)"}[doc.origin]
        sb.caption(f"**{doc.name}** · {doc.total_pages:,} pages · {_human_size(doc.size_bytes)}  \n{origin}")
        if doc.origin == "sample":
            sb.info("Using the small bundled primer. Put the official spec at `data/bluetooth_core_spec.pdf` or upload it for the full experience.", icon="ℹ️")
        c1, c2 = sb.columns(2)
        page_start = c1.number_input("First page", 1, doc.total_pages, 1, key=f"p0_{doc.sha256}")
        page_end = c2.number_input("Last page", 1, doc.total_pages, min(doc.total_pages, DEFAULT_PAGE_LIMIT), key=f"p1_{doc.sha256}")
        if page_end < page_start:
            sb.warning("Last page is before first page; using the first page only.")
            page_end = page_start
        if doc.total_pages > DEFAULT_PAGE_LIMIT:
            sb.caption(f"Large document: only pages {page_start}–{page_end} are indexed. Wider ranges take longer and use more RAM (and cost more with OpenAI embeddings).")
        strip = sb.checkbox("Remove repeated headers/footers", value=True, help="Drops lines that repeat at the top/bottom of most pages (e.g. 'Bluetooth SIG Proprietary | Vol 6, Part B page 2860').")

    # ---------------------------------------------------------------- Models
    sb.header("🧠 Models")
    key_ok = settings.openai_api_key() is not None
    sb.markdown(f"OpenAI key: {'✅ loaded from environment' if key_ok else '❌ not set (add OPENAI_API_KEY to .env)'}")

    emb_disabled = {}
    if not key_ok:
        emb_disabled[settings.EMB_OPENAI] = "needs OPENAI_API_KEY"
    if not settings.package_available("sentence_transformers"):
        emb_disabled[settings.EMB_HF] = "pip install sentence-transformers langchain-huggingface"
    llm_disabled = {settings.LLM_OPENAI: "needs OPENAI_API_KEY"} if not key_ok else {}

    with sb:
        embedding = _provider_model_picker("Embedding", settings.EMBEDDING_MODELS, settings.default_embedding_spec(), emb_disabled)
        vector_store = st.selectbox("Vector store", settings.VECTOR_STORES, help="Both run fully in memory and use cosine similarity, so scores are comparable.")
        st.divider()
        llm = _provider_model_picker("LLM", settings.LLM_MODELS, settings.default_llm_spec(), llm_disabled)
        temperature = st.slider(
            "Temperature",
            0.0,
            1.0,
            0.0,
            0.05,
            disabled=llm.provider == settings.LLM_EXTRACTIVE,
            help="0 = deterministic and factual (best for spec Q&A); higher = more varied wording and more risk of invention.",
        )

    return SidebarState(doc, int(page_start), int(page_end), strip, embedding, vector_store, llm, temperature)


def render_memory_section() -> None:
    """Rendered last (after the tabs) so the live-index count reflects this run's builds."""
    sb = st.sidebar
    sb.header("🧹 Memory")
    sb.caption(f"Live vector indexes: **{state.live_index_count()}**")
    if sb.button("Free all indexes & caches", width="stretch", help="Deletes every index, the parsed PDF and cached embedding models."):
        state.free_everything()
        st.rerun()
