"""Offline tests: they use the bundled primer, hashing embeddings and the
extractive answerer, so they need no API key, model download or network."""

from __future__ import annotations

import pytest

from rag_lab import settings
from rag_lab.benchmark import DEFAULT_PRESETS, run_queries_blocking
from rag_lab.chunking import CHARACTER, RECURSIVE, TOKEN, split_document, token_counting_is_exact
from rag_lab.embeddings import get_embeddings
from rag_lab.errors import ChunkingError, MissingAPIKeyError, PDFLoadError
from rag_lab.insights import analyze, token_f1
from rag_lab.pdf_loader import PageText, load_pdf, read_pdf_bytes, strip_running_headers
from rag_lab.pipeline import RAGConfig, answer_question, build_index
from rag_lab.settings import ModelSpec

HASHING = ModelSpec(settings.EMB_HASHING, "hashing-1024")
EXTRACTIVE = ModelSpec(settings.LLM_EXTRACTIVE, "top-sentences")


@pytest.fixture(scope="module")
def parsed():
    return load_pdf(read_pdf_bytes(settings.SAMPLE_PDF_PATH), "sample.pdf")


def test_pdf_parsing_and_header_removal(parsed):
    assert parsed.pages_parsed == parsed.total_pages_in_file >= 5
    assert parsed.removed_boilerplate, "the primer's running header/footer should be detected"
    header = "educational sample, not the official specification"
    assert all(header not in p.text.lower() for p in parsed.pages)


def test_page_range(parsed):
    data = read_pdf_bytes(settings.SAMPLE_PDF_PATH)
    sub = load_pdf(data, "sample.pdf", page_start=2, page_end=3)
    assert [p.page_number for p in sub.pages] == [2, 3]
    with pytest.raises(PDFLoadError):
        load_pdf(data, "sample.pdf", page_start=5, page_end=2)


@pytest.mark.parametrize("payload", [b"", b"hello world", b"%PDF-1.7\n garbage without xref"])
def test_corrupt_pdfs_raise_friendly_error(payload):
    with pytest.raises(PDFLoadError):
        load_pdf(payload, "bad.pdf")


def test_strip_running_headers_handles_page_numbers():
    topics = ["radio", "baseband", "link layer", "security", "gatt", "l2cap"]
    pages = [PageText(i, f"Spec v5 | Vol 6 page {i}\nBody about {t}\nFooter") for i, t in enumerate(topics, 1)]
    removed = strip_running_headers(pages)
    assert "spec v# | vol # page #" in removed
    assert [p.text for p in pages] == [f"Body about {t}" for t in topics]


@pytest.mark.parametrize("strategy", [RECURSIVE, CHARACTER])
def test_character_chunks_respect_size_and_pages(parsed, strategy):
    chunks = split_document(parsed, strategy, 300, 30)
    assert len(chunks) > 10
    assert max(c.metadata["char_len"] for c in chunks) <= 300
    for c in chunks:
        assert 1 <= c.metadata["page_start"] <= c.metadata["page_end"] <= parsed.total_pages_in_file


def test_large_chunks_can_span_pages(parsed):
    chunks = split_document(parsed, RECURSIVE, 3000, 0)
    assert any(c.metadata["page_end"] > c.metadata["page_start"] for c in chunks)


def test_invalid_chunk_params(parsed):
    with pytest.raises(ChunkingError):
        split_document(parsed, RECURSIVE, 100, 100)
    with pytest.raises(ChunkingError):
        split_document(parsed, RECURSIVE, 100, 10, max_chunks=3)


@pytest.mark.skipif(not token_counting_is_exact(), reason="tiktoken encoding not downloadable (offline)")
def test_token_splitter(parsed):
    chunks = split_document(parsed, TOKEN, 128, 8)
    assert max(c.metadata["token_len"] for c in chunks) <= 128


def test_missing_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        get_embeddings(ModelSpec(settings.EMB_OPENAI, "text-embedding-3-small-test"))


@pytest.mark.parametrize("store", settings.VECTOR_STORES)
def test_build_and_answer(parsed, store):
    config = RAGConfig("t", RECURSIVE, 500, 50, 3, HASHING, store)
    index = build_index(parsed, config)
    try:
        assert index.stats.total_chunks == len(index.chunk_lengths)
        result = answer_question(index, "What is the maximum output power of LE Power Class 1.5?", EXTRACTIVE, 0.0)
        assert result.error is None
        assert len(result.chunks) == 3
        sims = [c.similarity for c in result.chunks]
        assert sims == sorted(sims, reverse=True) and -1.0 <= sims[-1] <= sims[0] <= 1.0001
        assert "10 mW" in result.answer
        # Changing k at query time needs no rebuild.
        assert len(answer_question(index, "AFH", EXTRACTIVE, 0.0, k=5).chunks) == 5
    finally:
        index.close()


def test_llm_errors_are_captured_not_raised(parsed, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    index = build_index(parsed, RAGConfig("t", RECURSIVE, 800, 80, 2, HASHING, settings.STORE_FAISS))
    result = answer_question(index, "anything", ModelSpec(settings.LLM_OPENAI, "gpt-4o-mini"), 0.0)
    assert result.error and "OPENAI_API_KEY" in result.error
    index.close()


def test_benchmark_and_insights(parsed):
    presets = [p for p in DEFAULT_PRESETS if p.strategy != TOKEN]
    indexes = {i: build_index(parsed, p.to_config(HASHING, settings.STORE_FAISS)) for i, p in enumerate(presets)}
    try:
        results = run_queries_blocking(indexes, "Explain the adaptive frequency hopping (AFH) mechanism.", EXTRACTIVE, 0.0)
        assert set(results) == set(indexes) and all(r.error is None for r in results.values())
        insights = analyze(results, HASHING, EXTRACTIVE, reference="AFH hops only over used channels; at least 20 channels.")
        assert insights.best_slot in indexes
        assert all(m.reference_f1 is not None and m.groundedness > 0.95 for m in insights.metrics)
        assert insights.observations
    finally:
        for idx in indexes.values():
            idx.close()


def test_token_f1():
    assert token_f1("power class 1 is 100 mW", "power class 1 100 mW") == pytest.approx(1.0)
    assert token_f1("apples", "bluetooth") == 0.0


def test_token_splitter_with_offline_byte_encoding(parsed, monkeypatch):
    """Exercise the TokenTextSplitter path without downloading cl100k_base."""
    import tiktoken

    from rag_lab import chunking

    byte_encoding = tiktoken.Encoding(
        name="bytes-test", pat_str=r"\S+|\s+", mergeable_ranks={bytes([i]): i for i in range(256)}, special_tokens={}
    )
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: byte_encoding)
    chunking._load_encoding.cache_clear()
    try:
        chunks = split_document(parsed, TOKEN, 200, 10)  # 1 token = 1 byte with this encoding
        assert len(chunks) > 20
        assert max(c.metadata["token_len"] for c in chunks) <= 200
        assert all(c.metadata["start_index"] >= 0 for c in chunks)
    finally:
        chunking._load_encoding.cache_clear()
