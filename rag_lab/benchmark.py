"""Multi-RAG benchmark: presets plus parallel build and query helpers.

The helpers return ``concurrent.futures`` objects instead of blocking, so the
UI thread can poll them and redraw per-index progress bars while the workers
run (Streamlit widgets must only be touched from the main script thread).
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from . import settings
from .chunking import RECURSIVE, TOKEN
from .pdf_loader import ParsedPDF
from .pipeline import QueryResult, RAGConfig, RAGIndex, answer_question, build_index
from .settings import ModelSpec


@dataclass
class Preset:
    """An editable benchmark slot (embedding & store are shared, set in the sidebar)."""

    name: str
    strategy: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    rationale: str

    def to_config(self, embedding: ModelSpec, vector_store: str) -> RAGConfig:
        return RAGConfig(self.name, self.strategy, self.chunk_size, self.chunk_overlap, self.top_k, embedding, vector_store)


# Note how chunk_size × k (the "context budget") stays in a similar range for
# configs 1-4: the experiment isolates *granularity*, not the amount of text.
DEFAULT_PRESETS: list[Preset] = [
    Preset("Micro Chunks", RECURSIVE, 250, 25, 6, "Very precise matches, but each chunk may be a sentence fragment that lacks surrounding context."),
    Preset("Standard Small", RECURSIVE, 500, 50, 4, "Roughly a paragraph per chunk — a popular starting point for Q&A over technical docs."),
    Preset("Medium / Baseline", RECURSIVE, 1000, 100, 3, "The LangChain-tutorial default. Good balance of context and precision."),
    Preset("Large Context", RECURSIVE, 2000, 200, 2, "Whole sections per chunk: rich context but diluted embeddings and more irrelevant text."),
    Preset("Token Focused", TOKEN, 256, 16, 4, "Sizes measured in model tokens with minimal overlap; cuts may land mid-sentence."),
]


class ProgressBoard:
    """Thread-safe ``{slot: (fraction, message)}`` written by workers, read by the UI."""

    def __init__(self, n: int):
        self._lock = threading.Lock()
        self._state = {i: (0.0, "Queued") for i in range(n)}

    def callback(self, slot: int):
        def _update(fraction: float, message: str) -> None:
            with self._lock:
                self._state[slot] = (max(0.0, min(1.0, fraction)), message)

        return _update

    def snapshot(self) -> dict[int, tuple[float, str]]:
        with self._lock:
            return dict(self._state)


def submit_builds(
    executor: ThreadPoolExecutor,
    parsed: ParsedPDF,
    configs: list[RAGConfig],
    board: ProgressBoard,
) -> dict[int, Future]:
    """Schedule one ``build_index`` per config. Returns ``{slot: future}``."""
    return {i: executor.submit(build_index, parsed, cfg, board.callback(i)) for i, cfg in enumerate(configs)}


def submit_queries(
    executor: ThreadPoolExecutor,
    indexes: dict[int, RAGIndex],
    question: str,
    llm: ModelSpec,
    temperature: float,
) -> dict[int, Future]:
    """Schedule the same question against every index. Returns ``{slot: future}``."""
    return {slot: executor.submit(answer_question, idx, question, llm, temperature) for slot, idx in indexes.items()}


def make_executor(parallel: bool, n_jobs: int) -> ThreadPoolExecutor:
    workers = max(1, min(settings.MAX_PARALLEL_WORKERS, n_jobs)) if parallel else 1
    return ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rag-bench")


def run_queries_blocking(indexes: dict[int, RAGIndex], question: str, llm: ModelSpec, temperature: float, parallel: bool = True) -> dict[int, QueryResult]:
    """Convenience wrapper for scripts/tests: run all queries and wait."""
    with make_executor(parallel, len(indexes)) as ex:
        futures = submit_queries(ex, indexes, question, llm, temperature)
        return {slot: f.result() for slot, f in futures.items()}
