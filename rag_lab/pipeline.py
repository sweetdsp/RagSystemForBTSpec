"""The RAG pipeline: build an index from a parsed PDF, then retrieve + generate.

    ParsedPDF ──split──► chunks ──embed──► vector index ──search(k)──► context ──LLM──► answer

This module is UI-agnostic (no Streamlit imports) so it can be unit-tested and
reused from notebooks or a FastAPI back-end.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field, replace
from typing import Callable

from langchain_core.documents import Document

from . import settings
from .chunking import size_unit, split_document
from .embeddings import get_embeddings, provider_hint
from .errors import IndexBuildError, RAGLabError
from .llm import Generation, generate_answer
from .pdf_loader import ParsedPDF
from .settings import ModelSpec
from .vector_store import ScoredChunk, VectorIndex, build_vector_index

ProgressFn = Callable[[float, str], None]

SAMPLE_CHUNKS_KEPT = 40


@dataclass(frozen=True)
class RAGConfig:
    """Everything that defines one RAG system. Frozen = hashable & comparable."""

    name: str
    strategy: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    embedding: ModelSpec
    vector_store: str

    @property
    def unit(self) -> str:
        return size_unit(self.strategy)

    def summary(self) -> str:
        return f"{self.strategy.replace('TextSplitter', '')} · size {self.chunk_size} {self.unit} · overlap {self.chunk_overlap} · k={self.top_k}"

    def with_k(self, k: int) -> "RAGConfig":
        return replace(self, top_k=k)


@dataclass
class BuildStats:
    total_pages: int
    total_chunks: int
    avg_chunk_chars: float
    median_chunk_chars: float
    min_chunk_chars: int
    max_chunk_chars: int
    avg_chunk_tokens: float
    total_embedded_tokens: int
    chunking_seconds: float
    indexing_seconds: float
    total_seconds: float


@dataclass
class RAGIndex:
    """A built, queryable RAG system."""

    config: RAGConfig
    doc_fingerprint: str
    store: VectorIndex
    stats: BuildStats
    chunk_lengths: list[int]  # every chunk's char length (for the size histogram)
    sample_chunks: list[Document] = field(default_factory=list)

    def close(self) -> None:
        self.store.close()


@dataclass
class QueryResult:
    config: RAGConfig
    question: str
    chunks: list[ScoredChunk]
    answer: str
    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float
    llm_label: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    prompt_preview: str = ""
    error: str | None = None

    @property
    def context_chars(self) -> int:
        return sum(len(c.document.page_content) for c in self.chunks)

    @property
    def mean_similarity(self) -> float:
        return statistics.fmean(c.similarity for c in self.chunks) if self.chunks else 0.0

    @property
    def pages(self) -> list[int]:
        pages: set[int] = set()
        for c in self.chunks:
            m = c.document.metadata
            pages.update(range(int(m.get("page_start", 0)), int(m.get("page_end", 0)) + 1))
        return sorted(pages)


def _evenly_spaced(items: list, n: int) -> list:
    if len(items) <= n:
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def build_index(parsed: ParsedPDF, config: RAGConfig, progress: ProgressFn | None = None) -> RAGIndex:
    """Chunk, embed and index ``parsed`` according to ``config``.

    ``progress(fraction, message)`` is called from the current thread; when used
    from worker threads the callback must be thread-safe (e.g. write to a dict).
    """
    report = progress or (lambda f, m: None)
    t0 = time.perf_counter()

    report(0.0, "Chunking text…")
    chunks = split_document(parsed, config.strategy, config.chunk_size, config.chunk_overlap, max_chunks=settings.MAX_CHUNKS_PER_INDEX)
    t_chunk = time.perf_counter() - t0
    report(0.05, f"Created {len(chunks):,} chunks; loading embedding model…")

    embeddings = get_embeddings(config.embedding)
    t1 = time.perf_counter()
    try:
        store = build_vector_index(
            config.vector_store,
            embeddings,
            chunks,
            progress=lambda f, m: report(0.05 + 0.95 * f, m),
        )
    except RAGLabError:
        raise
    except Exception as exc:
        raise IndexBuildError(f"Embedding/indexing failed: {type(exc).__name__}: {exc}", hint=provider_hint(config.embedding)) from exc
    t_index = time.perf_counter() - t1

    lengths = [c.metadata["char_len"] for c in chunks]
    tokens = [c.metadata["token_len"] for c in chunks]
    stats = BuildStats(
        total_pages=parsed.pages_parsed,
        total_chunks=len(chunks),
        avg_chunk_chars=statistics.fmean(lengths),
        median_chunk_chars=statistics.median(lengths),
        min_chunk_chars=min(lengths),
        max_chunk_chars=max(lengths),
        avg_chunk_tokens=statistics.fmean(tokens),
        total_embedded_tokens=sum(tokens),
        chunking_seconds=t_chunk,
        indexing_seconds=t_index,
        total_seconds=time.perf_counter() - t0,
    )
    report(1.0, "Done")
    return RAGIndex(
        config=config,
        doc_fingerprint=parsed.fingerprint,
        store=store,
        stats=stats,
        chunk_lengths=lengths,
        sample_chunks=_evenly_spaced(chunks, SAMPLE_CHUNKS_KEPT),
    )


def retrieve(index: RAGIndex, question: str, k: int | None = None) -> tuple[list[ScoredChunk], float]:
    """Embed the question and fetch the top-k chunks. Returns ``(chunks, seconds)``."""
    t0 = time.perf_counter()
    try:
        vector = get_embeddings(index.config.embedding).embed_query(question)
        hits = index.store.search(vector, k or index.config.top_k)
    except RAGLabError:
        raise
    except Exception as exc:
        raise IndexBuildError(f"Retrieval failed: {type(exc).__name__}: {exc}", hint=provider_hint(index.config.embedding)) from exc
    return hits, time.perf_counter() - t0


def answer_question(index: RAGIndex, question: str, llm: ModelSpec, temperature: float, k: int | None = None) -> QueryResult:
    """Full RAG round-trip. Errors are captured into ``QueryResult.error`` (never raised)
    so that one failing configuration does not break a 5-way benchmark."""
    t0 = time.perf_counter()
    config = index.config if k is None else index.config.with_k(k)
    chunks: list[ScoredChunk] = []
    retrieval_s = generation_s = 0.0
    try:
        chunks, retrieval_s = retrieve(index, question, config.top_k)
        gen: Generation = generate_answer(question, chunks, llm, temperature)
        generation_s = gen.seconds
        return QueryResult(
            config=config,
            question=question,
            chunks=chunks,
            answer=gen.text,
            retrieval_seconds=retrieval_s,
            generation_seconds=generation_s,
            total_seconds=time.perf_counter() - t0,
            llm_label=gen.model_label,
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
            prompt_preview=gen.prompt_preview,
        )
    except RAGLabError as exc:
        error = exc.message + (f" — {exc.hint}" if exc.hint else "")
    except Exception as exc:  # defensive: surface anything unexpected in the card
        error = f"{type(exc).__name__}: {exc}"
    return QueryResult(
        config=config,
        question=question,
        chunks=chunks,
        answer="",
        retrieval_seconds=retrieval_s,
        generation_seconds=generation_s,
        total_seconds=time.perf_counter() - t0,
        llm_label=llm.label,
        error=error,
    )
