"""Domain-specific exceptions.

Every error raised on purpose by ``rag_lab`` derives from :class:`RAGLabError`
and carries a human-friendly message (and optional hint) that the UI can show
directly to a learner instead of a raw stack trace.
"""


class RAGLabError(Exception):
    """Base class for all expected, user-facing errors."""

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.message} ({self.hint})" if self.hint else self.message


class PDFLoadError(RAGLabError):
    """The PDF is missing, corrupt, encrypted or contains no extractable text."""


class MissingAPIKeyError(RAGLabError):
    """A cloud provider was selected but its API key is not configured."""


class MissingDependencyError(RAGLabError):
    """An optional Python package needed by the selected backend is not installed."""


class ChunkingError(RAGLabError):
    """Invalid chunking parameters or a splitter failure."""


class IndexBuildError(RAGLabError):
    """Embedding or vector-store indexing failed."""


class GenerationError(RAGLabError):
    """The LLM call failed (network, auth, rate limit, model not found...)."""
