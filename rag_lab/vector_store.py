"""In-memory vector stores (FAISS or ChromaDB) behind one tiny interface.

Both backends are configured for **cosine similarity** so their scores are
directly comparable in the UI:

* FAISS stores L2-normalised vectors in a flat L2 index. For unit vectors the
  squared L2 distance is ``d = 2 - 2·cos``, hence ``cos = 1 - d/2``.
* Chroma uses its ``cosine`` HNSW space, which returns ``d = 1 - cos``.

Vectors are added in batches so the caller can display embedding progress.
"""

from __future__ import annotations

import uuid
import warnings
from dataclasses import dataclass
from typing import Callable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from . import settings
from .errors import IndexBuildError, MissingDependencyError

ProgressFn = Callable[[float, str], None]


@dataclass
class ScoredChunk:
    """A retrieved chunk with its raw distance and the derived cosine similarity."""

    document: Document
    distance: float
    similarity: float


class VectorIndex:
    """Common interface; see :class:`FaissIndex` and :class:`ChromaIndex`."""

    backend: str = ""
    distance_label: str = ""

    def search(self, query_vector: list[float], k: int) -> list[ScoredChunk]:
        raise NotImplementedError

    def close(self) -> None:
        """Release memory held by the index."""


class FaissIndex(VectorIndex):
    backend = settings.STORE_FAISS
    distance_label = "squared L2 (unit vectors)"

    def __init__(self, embeddings: Embeddings):
        try:
            import faiss  # noqa: F401  (import check only)

            # langchain-community is in maintenance mode but its FAISS wrapper is still the
            # reference integration; the version is pinned in requirements.txt.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                from langchain_community.vectorstores import FAISS
        except ImportError as exc:
            raise MissingDependencyError("FAISS needs 'faiss-cpu' and 'langchain-community'.", hint="pip install faiss-cpu langchain-community") from exc
        self._cls = FAISS
        self._embeddings = embeddings
        self._store = None

    def add(self, docs: list[Document]) -> None:
        texts = [d.page_content for d in docs]
        metas = [d.metadata for d in docs]
        if self._store is None:
            self._store = self._cls.from_texts(texts, self._embeddings, metadatas=metas, normalize_L2=True)
        else:
            self._store.add_texts(texts, metadatas=metas)

    def search(self, query_vector: list[float], k: int) -> list[ScoredChunk]:
        hits = self._store.similarity_search_with_score_by_vector(query_vector, k=k)
        return [ScoredChunk(doc, float(d), 1.0 - float(d) / 2.0) for doc, d in hits]

    def close(self) -> None:
        self._store = None


class ChromaIndex(VectorIndex):
    backend = settings.STORE_CHROMA
    distance_label = "cosine distance"

    def __init__(self, embeddings: Embeddings):
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            from langchain_chroma import Chroma
        except ImportError as exc:
            raise MissingDependencyError("ChromaDB needs 'chromadb' and 'langchain-chroma'.", hint="pip install chromadb langchain-chroma") from exc
        # An ephemeral (RAM-only) client; each index gets its own uniquely-named collection.
        self._client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True))
        self._name = f"rag_{uuid.uuid4().hex[:12]}"
        self._store = Chroma(
            client=self._client,
            collection_name=self._name,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def add(self, docs: list[Document]) -> None:
        ids = [f"{self._name}-{d.metadata.get('chunk_id', uuid.uuid4().hex)}" for d in docs]
        self._store.add_texts([d.page_content for d in docs], metadatas=[d.metadata for d in docs], ids=ids)

    def search(self, query_vector: list[float], k: int) -> list[ScoredChunk]:
        hits = self._store.similarity_search_by_vector_with_relevance_scores(query_vector, k=k)
        return [ScoredChunk(doc, float(d), 1.0 - float(d)) for doc, d in hits]

    def close(self) -> None:
        try:
            self._client.delete_collection(self._name)
        except Exception:
            pass  # already gone; nothing else to free


def build_vector_index(
    backend: str,
    embeddings: Embeddings,
    chunks: list[Document],
    batch_size: int = settings.EMBED_BATCH_SIZE,
    progress: ProgressFn | None = None,
) -> VectorIndex:
    """Embed ``chunks`` batch by batch and load them into the chosen backend."""
    index: VectorIndex
    if backend == settings.STORE_FAISS:
        index = FaissIndex(embeddings)
    elif backend == settings.STORE_CHROMA:
        index = ChromaIndex(embeddings)
    else:
        raise IndexBuildError(f"Unknown vector store: {backend!r}")

    total = len(chunks)
    try:
        for start in range(0, total, batch_size):
            batch = chunks[start : start + batch_size]
            index.add(batch)
            if progress:
                done = min(start + batch_size, total)
                progress(done / total, f"Embedded {done:,}/{total:,} chunks")
    except Exception:
        index.close()  # do not leak a half-built collection
        raise
    return index
