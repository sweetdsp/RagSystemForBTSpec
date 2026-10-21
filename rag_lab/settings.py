"""Central configuration: environment loading, model catalogs and safety limits.

Everything that a learner (or an instructor deploying the app) might want to
tune without touching code is read from environment variables here, so the
rest of the code base can import plain Python constants.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load a `.env` file from the project root (if present) before reading any
# variables. Real environment variables always win over the file.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env_int(name: str, default: int) -> int:
    """Read an integer env var, falling back to ``default`` on bad input."""
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"

#: The full Bluetooth Core Specification (download it yourself, see data/README.md).
DEFAULT_PDF_PATH = Path(os.getenv("DEFAULT_PDF_PATH", DATA_DIR / "bluetooth_core_spec.pdf"))

#: Small, bundled, self-written primer used when the real spec is not present.
SAMPLE_PDF_PATH = DATA_DIR / "sample_bluetooth_primer.pdf"

# ---------------------------------------------------------------------------
# Safety / memory limits
# ---------------------------------------------------------------------------
#: Refuse uploads larger than this (Streamlit also has its own server limit).
MAX_UPLOAD_MB = _env_int("MAX_UPLOAD_MB", 300)

#: Hard cap on chunks per index. Tiny chunks over a 3,000-page spec can easily
#: produce 100k+ vectors, which is slow, memory hungry and (with OpenAI) costly.
MAX_CHUNKS_PER_INDEX = _env_int("MAX_CHUNKS_PER_INDEX", 60_000)

#: Number of chunks embedded per batch (also the progress-bar granularity).
EMBED_BATCH_SIZE = _env_int("EMBED_BATCH_SIZE", 128)

#: Upper bound on worker threads used for the 5-way benchmark.
MAX_PARALLEL_WORKERS = _env_int("MAX_PARALLEL_WORKERS", 5)

# ---------------------------------------------------------------------------
# Provider settings
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_REQUEST_TIMEOUT = _env_int("LLM_REQUEST_TIMEOUT", 120)


def openai_api_key() -> str | None:
    """Return the OpenAI key (re-read each call so the UI can set it at runtime)."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return key or None


def package_available(module: str) -> bool:
    """True if ``module`` can be imported (used to grey-out optional backends)."""
    return importlib.util.find_spec(module) is not None


# ---------------------------------------------------------------------------
# Catalogs shown in the UI
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    """A (provider, model) pair. Frozen so it can be used as a cache key."""

    provider: str
    model: str

    @property
    def label(self) -> str:
        return f"{self.provider} · {self.model}"


# Embedding providers -------------------------------------------------------
EMB_OPENAI = "OpenAI"
EMB_HF = "HuggingFace (local)"
EMB_OLLAMA = "Ollama (local)"
EMB_HASHING = "Offline hashing (no model)"

EMBEDDING_MODELS: dict[str, list[str]] = {
    EMB_OPENAI: ["text-embedding-3-small", "text-embedding-3-large"],
    EMB_HF: ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"],
    EMB_OLLAMA: ["nomic-embed-text", "mxbai-embed-large"],
    EMB_HASHING: ["hashing-1024"],
}

# LLM providers ---------------------------------------------------------------
LLM_OPENAI = "OpenAI"
LLM_OLLAMA = "Ollama (local)"
LLM_EXTRACTIVE = "Offline extractive (no LLM)"

LLM_MODELS: dict[str, list[str]] = {
    LLM_OPENAI: [m.strip() for m in os.getenv("OPENAI_CHAT_MODELS", "gpt-4o-mini,gpt-4.1-mini,gpt-4o").split(",") if m.strip()],
    LLM_OLLAMA: [m.strip() for m in os.getenv("OLLAMA_CHAT_MODELS", "llama3.2,qwen2.5,mistral").split(",") if m.strip()],
    LLM_EXTRACTIVE: ["top-sentences"],
}

# Vector stores -----------------------------------------------------------------
STORE_FAISS = "FAISS (in-memory)"
STORE_CHROMA = "ChromaDB (in-memory)"
VECTOR_STORES = [STORE_FAISS, STORE_CHROMA]


def default_embedding_spec() -> ModelSpec:
    """Pick the best embedding backend that works out of the box."""
    if openai_api_key():
        return ModelSpec(EMB_OPENAI, EMBEDDING_MODELS[EMB_OPENAI][0])
    if package_available("sentence_transformers"):
        return ModelSpec(EMB_HF, EMBEDDING_MODELS[EMB_HF][0])
    return ModelSpec(EMB_HASHING, EMBEDDING_MODELS[EMB_HASHING][0])


def default_llm_spec() -> ModelSpec:
    """Pick OpenAI when a key is configured, otherwise the offline answerer."""
    if openai_api_key():
        return ModelSpec(LLM_OPENAI, LLM_MODELS[LLM_OPENAI][0])
    return ModelSpec(LLM_EXTRACTIVE, LLM_MODELS[LLM_EXTRACTIVE][0])
