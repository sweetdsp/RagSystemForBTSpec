"""Chunking strategies (LangChain text splitters).

Pages are concatenated into one long string *before* splitting so that large
chunks can span page boundaries, exactly like they would in a production RAG
pipeline. Each chunk's character offset is then mapped back to the page(s) it
came from, so retrieved context can always be cited by page number.

Units matter and are a common source of confusion for learners:

* ``RecursiveCharacterTextSplitter`` and ``CharacterTextSplitter`` measure
  ``chunk_size`` / ``chunk_overlap`` in **characters**.
* ``TokenTextSplitter`` measures them in **tokens** (tiktoken ``cl100k_base``),
  where one token is roughly 4 characters of English text.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)

from .errors import ChunkingError
from .pdf_loader import ParsedPDF

RECURSIVE = "RecursiveCharacterTextSplitter"
CHARACTER = "CharacterTextSplitter"
TOKEN = "TokenTextSplitter"
STRATEGIES = [RECURSIVE, CHARACTER, TOKEN]

STRATEGY_HELP = {
    RECURSIVE: "Tries paragraph → line → word → character boundaries in order, so chunks stay "
    "close to the target size while respecting natural text structure. The usual default.",
    CHARACTER: "Splits on a single separator (a newline here) and greedily merges pieces. "
    "Simple and predictable, but a long paragraph without the separator yields an oversized chunk.",
    TOKEN: "Splits on model tokens (tiktoken cl100k_base). Sizes match what the embedding/LLM "
    "actually 'sees', but cuts can land mid-sentence.",
}

TIKTOKEN_ENCODING = "cl100k_base"
PAGE_JOINER = "\n\n"


def size_unit(strategy: str) -> str:
    return "tokens" if strategy == TOKEN else "chars"


@lru_cache(maxsize=1)
def _load_encoding():
    """Load tiktoken once. Returns ``(encoding, None)`` or ``(None, (message, hint))``;
    the failure is cached too, so an offline machine does not retry per chunk."""
    try:
        import tiktoken

        return tiktoken.get_encoding(TIKTOKEN_ENCODING), None
    except ImportError:
        return None, ("TokenTextSplitter needs the 'tiktoken' package.", "pip install tiktoken")
    except Exception as exc:  # usually: no network on first use (it downloads a ~1 MB BPE file)
        return None, (
            "Could not load the tiktoken encoding (it is downloaded on first use).",
            f"Check your internet connection or pre-populate TIKTOKEN_CACHE_DIR. Details: {exc}",
        )


def _encoding():
    encoding, error = _load_encoding()
    if error:
        raise ChunkingError(*error)
    return encoding


def token_counting_is_exact() -> bool:
    return _load_encoding()[0] is not None


def count_tokens(text: str) -> int:
    """Exact token count when tiktoken is usable, else the ~4 chars/token rule of thumb."""
    encoding, _ = _load_encoding()
    if encoding is None:
        return max(1, len(text) // 4)
    return len(encoding.encode(text, disallowed_special=()))


def make_splitter(strategy: str, chunk_size: int, chunk_overlap: int) -> TextSplitter:
    """Validate parameters and build the requested LangChain splitter."""
    if chunk_size <= 0:
        raise ChunkingError("Chunk size must be positive.")
    if chunk_overlap < 0:
        raise ChunkingError("Chunk overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ChunkingError(
            f"Chunk overlap ({chunk_overlap}) must be smaller than chunk size ({chunk_size}).",
            hint="Overlap is the part of each chunk repeated from the previous one; it cannot cover the whole chunk.",
        )

    common = dict(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
    if strategy == RECURSIVE:
        return RecursiveCharacterTextSplitter(**common)
    if strategy == CHARACTER:
        return CharacterTextSplitter(separator="\n", **common)
    if strategy == TOKEN:
        _encoding()  # fail fast with a friendly error if tiktoken is unusable
        return TokenTextSplitter(encoding_name=TIKTOKEN_ENCODING, **common)
    raise ChunkingError(f"Unknown chunking strategy: {strategy!r}")


@dataclass
class _JoinedText:
    text: str
    page_starts: list[int]  # char offset where each page begins
    page_numbers: list[int]

    def page_at(self, offset: int) -> int:
        idx = bisect.bisect_right(self.page_starts, max(0, offset)) - 1
        return self.page_numbers[max(0, idx)]


def _join_pages(parsed: ParsedPDF) -> _JoinedText:
    parts, starts, numbers, cursor = [], [], [], 0
    for page in parsed.pages:
        starts.append(cursor)
        numbers.append(page.page_number)
        parts.append(page.text)
        cursor += len(page.text) + len(PAGE_JOINER)
    return _JoinedText(PAGE_JOINER.join(parts), starts, numbers)


def split_document(
    parsed: ParsedPDF,
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    max_chunks: int | None = None,
) -> list[Document]:
    """Split a parsed PDF into LangChain ``Document`` chunks with rich metadata.

    Metadata per chunk: ``chunk_id``, ``source``, ``page_start``, ``page_end``,
    ``start_index`` (char offset in the joined text), ``char_len`` and
    ``token_len``.
    """
    splitter = make_splitter(strategy, chunk_size, chunk_overlap)
    joined = _join_pages(parsed)
    try:
        raw_chunks = splitter.create_documents([joined.text])
    except Exception as exc:
        raise ChunkingError(f"The {strategy} failed: {exc}") from exc

    if max_chunks and len(raw_chunks) > max_chunks:
        raise ChunkingError(
            f"{len(raw_chunks):,} chunks would be created, above the safety limit of {max_chunks:,}.",
            hint="Use a larger chunk size, a narrower page range, or raise MAX_CHUNKS_PER_INDEX in .env.",
        )

    chunks: list[Document] = []
    last_start = 0
    for i, doc in enumerate(raw_chunks):
        text = doc.page_content.strip()
        if not text:
            continue
        start = doc.metadata.get("start_index", -1)
        if start is None or start < 0:  # token decoding can defeat exact substring search
            start = last_start
        last_start = start
        chunks.append(
            Document(
                page_content=text,
                metadata={
                    "chunk_id": i,
                    "source": parsed.name,
                    "page_start": joined.page_at(start),
                    "page_end": joined.page_at(start + len(text) - 1),
                    "start_index": start,
                    "char_len": len(text),
                    "token_len": count_tokens(text),
                    "strategy": strategy,
                },
            )
        )
    if not chunks:
        raise ChunkingError("The splitter produced no chunks; the document text may be empty.")
    return chunks
