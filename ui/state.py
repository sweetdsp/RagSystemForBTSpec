"""Session-state helpers: document resolution, parse caching and memory management.

Streamlit re-runs the whole script on every interaction, so anything expensive
(hashing a 100 MB PDF, parsing 3,000 pages, vector indexes) lives in
``st.session_state`` and is rebuilt only when its inputs actually change.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from rag_lab import disk_cache, embeddings, settings
from rag_lab.errors import PDFLoadError
from rag_lab.pdf_loader import OutlineSection, ParsedPDF, count_pages, load_pdf, outline_of, read_pdf_bytes, sha256_bytes

# Session-state keys (one place, to avoid typos)
K_DOC_META = "doc_meta"
K_PARSED = "parsed_pdf"
K_SINGLE_INDEX = "single_index"
K_SINGLE_RESULT = "single_result"
K_SINGLE_SOURCE = "single_source"
K_BENCH_SOURCE = "bench_source"
K_BENCH_INDEXES = "bench_indexes"
K_BENCH_BUILD_INFO = "bench_build_info"
K_BENCH_RESULTS = "bench_results"
K_BENCH_INSIGHTS = "bench_insights"


@dataclass
class DocumentSource:
    """The PDF currently selected in the sidebar (bytes are loaded lazily)."""

    name: str
    origin: str  # "upload" | "default" | "sample"
    sha256: str
    size_bytes: int
    total_pages: int
    _upload: object | None = None
    _path: Path | None = None

    def read_bytes(self) -> bytes:
        if self._upload is not None:
            return self._upload.getvalue()
        return read_pdf_bytes(self._path)


def resolve_document(uploaded) -> DocumentSource:
    """Uploaded file → configured Bluetooth spec → bundled sample primer."""
    if uploaded is not None:
        size = uploaded.size
        if size > settings.MAX_UPLOAD_MB * 1024 * 1024:
            raise PDFLoadError(f"File is {size / 1e6:.0f} MB; the limit is {settings.MAX_UPLOAD_MB} MB.", hint="Raise MAX_UPLOAD_MB in .env (and server.maxUploadSize in .streamlit/config.toml).")
        cache_key = ("upload", uploaded.file_id)
        origin, path, upload, name = "upload", None, uploaded, uploaded.name
    else:
        path = settings.DEFAULT_PDF_PATH if settings.DEFAULT_PDF_PATH.is_file() else settings.SAMPLE_PDF_PATH
        origin = "default" if path == settings.DEFAULT_PDF_PATH else "sample"
        stat = path.stat()
        size, upload, name = stat.st_size, None, path.name
        cache_key = ("path", str(path), stat.st_mtime_ns)

    meta = st.session_state.get(K_DOC_META)
    if not meta or meta["key"] != cache_key:
        data = upload.getvalue() if upload is not None else read_pdf_bytes(path)
        meta = {"key": cache_key, "sha": sha256_bytes(data), "pages": count_pages(data)}
        del data
        st.session_state[K_DOC_META] = meta
    return DocumentSource(name, origin, meta["sha"], size, meta["pages"], upload, path)


def source_key(sb) -> tuple:
    """Everything outside the chunking parameters that an index depends on.

    Stored next to each index so the UI can warn when the sidebar has changed
    since the build (``sb`` is a ``SidebarState``).
    """
    sha = sb.doc.sha256 if sb.doc else None
    return (sha, sb.page_start, sb.page_end, sb.strip_boilerplate, sb.embedding, sb.vector_store)


@st.cache_resource(max_entries=3, show_spinner=False)
def page_text_cache(sha256: str) -> dict[int, str]:
    """Process-wide ``{page: text}`` cache per document (shared by all sessions).

    Extracting the 3,141-page Bluetooth spec takes about a minute; afterwards any page
    range of it can be re-parsed in well under a second. Seeded from ``.cache/`` on disk.
    """
    return disk_cache.load_pages(sha256)


@st.cache_data(max_entries=3, show_spinner="Reading the PDF's bookmarks…")
def document_outline(sha256: str, _doc: DocumentSource) -> list[OutlineSection]:
    """Bookmarks per document (slow in pypdf for big outlines, so cached by hash)."""
    cached = disk_cache.load_outline(sha256)
    if cached is not None:
        return cached
    try:
        outline = outline_of(_doc.read_bytes())
    except Exception:
        return []  # bookmarks are a convenience; never block the app on them
    disk_cache.save_outline(sha256, outline)
    return outline


def get_parsed(doc: DocumentSource, page_start: int, page_end: int, strip_boilerplate: bool) -> ParsedPDF:
    """Parse (or reuse) the selected page range. Only one parse is kept in memory."""
    key = (doc.sha256, page_start, page_end, strip_boilerplate)
    cached = st.session_state.get(K_PARSED)
    if cached and cached[0] == key:
        return cached[1]

    st.session_state.pop(K_PARSED, None)  # free the previous parse before making a new one
    page_cache = page_text_cache(doc.sha256)
    cached_before = len(page_cache)
    bar = st.progress(0.0, text=f"Parsing {doc.name}…")
    parsed = load_pdf(
        doc.read_bytes(),
        doc.name,
        page_start,
        page_end,
        progress=lambda f, m: bar.progress(min(1.0, f), text=m),
        strip_boilerplate=strip_boilerplate,
        page_cache=page_cache,
        outline=document_outline(doc.sha256, doc),
    )
    bar.empty()
    if len(page_cache) > cached_before:
        disk_cache.save_pages(doc.sha256, page_cache)
    st.session_state[K_PARSED] = (key, parsed)
    return parsed


def close_single_index() -> None:
    idx = st.session_state.pop(K_SINGLE_INDEX, None)
    if idx is not None:
        idx.close()
    st.session_state.pop(K_SINGLE_RESULT, None)
    st.session_state.pop(K_SINGLE_SOURCE, None)


def close_bench_indexes() -> None:
    for idx in (st.session_state.pop(K_BENCH_INDEXES, None) or {}).values():
        idx.close()
    for key in (K_BENCH_BUILD_INFO, K_BENCH_RESULTS, K_BENCH_INSIGHTS, K_BENCH_SOURCE):
        st.session_state.pop(key, None)


def free_everything() -> None:
    """Drop every index, the parsed PDF and cached embedding models, then GC."""
    close_single_index()
    close_bench_indexes()
    st.session_state.pop(K_PARSED, None)
    embeddings.clear_cache()
    gc.collect()


def live_index_count() -> int:
    return int(K_SINGLE_INDEX in st.session_state) + len(st.session_state.get(K_BENCH_INDEXES) or {})
