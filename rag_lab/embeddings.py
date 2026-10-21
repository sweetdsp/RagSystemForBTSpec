"""Embedding model factory.

Supported backends:

* **OpenAI** – ``text-embedding-3-*`` via ``langchain-openai`` (needs ``OPENAI_API_KEY``).
* **HuggingFace** – local ``sentence-transformers`` models such as ``all-MiniLM-L6-v2``.
* **Ollama** – local models served by ``ollama serve`` (e.g. ``nomic-embed-text``).
* **Offline hashing** – a tiny, dependency-free lexical embedding (the
  "hashing trick"). It needs no download or API key, so the app always works
  in a classroom without internet. It only captures word overlap, not meaning,
  which itself is a useful lesson when compared with a neural model.

Model objects are cached per ``ModelSpec`` because loading a sentence-transformer
takes seconds and hundreds of MB; all indexes built with the same spec share one
instance (they are safe to use from several threads for inference).
"""

from __future__ import annotations

import math
import re
import threading
import zlib

from langchain_core.embeddings import Embeddings

from . import settings
from .errors import MissingAPIKeyError, MissingDependencyError, RAGLabError
from .settings import ModelSpec

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


class HashingEmbeddings(Embeddings):
    """Bag of word uni+bi-grams hashed into a fixed-size, L2-normalised vector."""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        words = _TOKEN_RE.findall(text.lower())
        features = words + [f"{a} {b}" for a, b in zip(words, words[1:])]
        for feat in features:
            h = zlib.crc32(feat.encode())
            # The sign bit spreads collisions so they cancel out on average.
            vec[h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        # Sub-linear term frequency dampens very repetitive chunks.
        vec = [math.copysign(math.log1p(abs(v)), v) for v in vec]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


_cache: dict[ModelSpec, Embeddings] = {}
_cache_lock = threading.Lock()


def get_embeddings(spec: ModelSpec) -> Embeddings:
    """Return a (cached) LangChain ``Embeddings`` object for ``spec``."""
    with _cache_lock:  # avoid loading the same HF model twice from parallel builds
        if spec not in _cache:
            _cache[spec] = _create(spec)
        return _cache[spec]


def clear_cache() -> None:
    """Drop cached models (frees the RAM held by local sentence-transformers)."""
    with _cache_lock:
        _cache.clear()


def _create(spec: ModelSpec) -> Embeddings:
    if spec.provider == settings.EMB_HASHING:
        return HashingEmbeddings()

    if spec.provider == settings.EMB_OPENAI:
        key = settings.openai_api_key()
        if not key:
            raise MissingAPIKeyError(
                "OpenAI embeddings selected but OPENAI_API_KEY is not set.",
                hint="Add it to .env, paste it in the sidebar, or pick a HuggingFace / offline embedding model.",
            )
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise MissingDependencyError("langchain-openai is not installed.", hint="pip install langchain-openai") from exc
        return OpenAIEmbeddings(model=spec.model, api_key=key, max_retries=3)

    if spec.provider == settings.EMB_HF:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise MissingDependencyError(
                "HuggingFace embeddings need 'langchain-huggingface' and 'sentence-transformers'.",
                hint="pip install langchain-huggingface sentence-transformers",
            ) from exc
        try:
            return HuggingFaceEmbeddings(
                model_name=spec.model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
            )
        except Exception as exc:  # download failure, unknown model name, ...
            raise RAGLabError(
                f"Could not load HuggingFace model '{spec.model}'.",
                hint=f"The first run downloads the model from huggingface.co; check your connection. Details: {exc}",
            ) from exc

    if spec.provider == settings.EMB_OLLAMA:
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError as exc:
            raise MissingDependencyError("langchain-ollama is not installed.", hint="pip install langchain-ollama") from exc
        return OllamaEmbeddings(model=spec.model, base_url=settings.OLLAMA_BASE_URL)

    raise RAGLabError(f"Unknown embedding provider: {spec.provider!r}")


def provider_hint(spec: ModelSpec) -> str | None:
    """Troubleshooting hint shown when an embedding call fails at runtime."""
    if spec.provider == settings.EMB_OLLAMA:
        return f"Is Ollama running at {settings.OLLAMA_BASE_URL}? Try: ollama pull {spec.model} && ollama serve"
    if spec.provider == settings.EMB_OPENAI:
        return "Check that your OpenAI key is valid and has quota (HTTP 401/429 errors)."
    return None
