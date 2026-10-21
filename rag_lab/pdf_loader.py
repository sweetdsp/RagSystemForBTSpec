"""PDF parsing with pypdf.

The Bluetooth Core Specification is ~3,500 pages, so this module:

* lets the caller restrict parsing to a page range (huge memory/time saver),
* skips individual pages that fail to extract instead of aborting the whole run,
* raises :class:`PDFLoadError` with actionable hints for corrupt, encrypted or
  scanned (image-only) documents.
"""

from __future__ import annotations

import hashlib
import io
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .errors import PDFLoadError

ProgressFn = Callable[[float, str], None]

_MULTISPACE = re.compile(r"[ \t ]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_DIGITS = re.compile(r"\d+")

#: How many lines at the top/bottom of each page are inspected for running headers/footers.
EDGE_LINES = 3


@dataclass
class PageText:
    """Cleaned text of a single PDF page (1-based ``page_number``)."""

    page_number: int
    text: str


@dataclass
class ParsedPDF:
    """Result of parsing a PDF; shared read-only by every RAG index."""

    name: str
    sha256: str
    file_size_bytes: int
    total_pages_in_file: int
    page_range: tuple[int, int]  # inclusive, 1-based
    pages: list[PageText] = field(default_factory=list)
    skipped_pages: list[int] = field(default_factory=list)
    removed_boilerplate: list[str] = field(default_factory=list)
    parse_seconds: float = 0.0

    @property
    def pages_parsed(self) -> int:
        return len(self.pages)

    @property
    def total_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)

    @property
    def fingerprint(self) -> str:
        """Identifies document, page range and cleaning; used to detect stale indexes."""
        clean = "c" if self.removed_boilerplate else "r"
        return f"{self.sha256[:12]}:{self.page_range[0]}-{self.page_range[1]}:{clean}"


def clean_text(text: str) -> str:
    """Light normalisation that keeps paragraph structure for the splitters."""
    text = text.replace("\x00", "")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)  # re-join words hyphenated across lines
    text = _MULTISPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def _edge_key(line: str) -> str:
    """Normalise a line so 'Vol 6, Part A page 2860' and '... page 2861' compare equal."""
    return _DIGITS.sub("#", line.strip().lower())


def strip_running_headers(pages: list[PageText], min_share: float = 0.5) -> list[str]:
    """Remove header/footer lines repeated on most pages (in place).

    Specs repeat banners such as "BLUETOOTH CORE SPECIFICATION Version 5.4 | Vol 6,
    Part B page 2860" on every page. Left in, they end up in nearly every chunk,
    make unrelated chunks look similar to each other and waste context tokens.
    Returns the normalised patterns that were removed (shown in the UI).
    """
    if len(pages) < 3:
        return []
    counts: Counter[str] = Counter()
    for page in pages:
        lines = [ln for ln in page.text.splitlines() if ln.strip()]
        counts.update({_edge_key(ln) for ln in lines[:EDGE_LINES] + lines[-EDGE_LINES:]})
    threshold = max(3, int(len(pages) * min_share))
    boilerplate = {key for key, n in counts.items() if n >= threshold and key}
    if not boilerplate:
        return []
    for page in pages:
        lines = page.text.splitlines()
        nonblank = [i for i, ln in enumerate(lines) if ln.strip()]
        edges = set(nonblank[:EDGE_LINES] + nonblank[-EDGE_LINES:])
        page.text = "\n".join(ln for i, ln in enumerate(lines) if not (i in edges and _edge_key(ln) in boilerplate)).strip()
    return sorted(boilerplate)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_pdf_bytes(path: str | Path) -> bytes:
    """Read a PDF from disk with a friendly error if it does not exist."""
    path = Path(path)
    if not path.is_file():
        raise PDFLoadError(
            f"PDF not found: {path}",
            hint="Upload a PDF in the sidebar or place the Bluetooth spec at the configured DEFAULT_PDF_PATH.",
        )
    return path.read_bytes()


def count_pages(data: bytes) -> int:
    """Cheap page count (no text extraction) used to size the page-range inputs."""
    return len(_open_reader(data).pages)


def _open_reader(data: bytes) -> PdfReader:
    if not data:
        raise PDFLoadError("The uploaded file is empty.")
    if not data.lstrip()[:5].startswith(b"%PDF"):
        raise PDFLoadError("This file does not look like a PDF (missing %PDF header).")
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            # Many specs are "encrypted" with an empty user password (copy protection only).
            try:
                if not reader.decrypt(""):
                    raise PDFLoadError(
                        "The PDF is password-protected.",
                        hint="Remove the password (e.g. with qpdf --decrypt) and upload it again.",
                    )
            except PDFLoadError:
                raise
            except Exception as exc:  # cryptography backend missing, unsupported algorithm, ...
                raise PDFLoadError(
                    "The PDF is encrypted and could not be decrypted.",
                    hint=f"Install 'cryptography' or decrypt the file first. Details: {exc}",
                ) from exc
        _ = len(reader.pages)  # forces the page tree to be read -> surfaces corruption early
        return reader
    except PDFLoadError:
        raise
    except (PdfReadError, ValueError, KeyError, OSError) as exc:
        raise PDFLoadError(
            "The PDF appears to be corrupt or unreadable.",
            hint=f"Try re-downloading it or re-saving it with a PDF tool. Details: {exc}",
        ) from exc


def load_pdf(
    data: bytes,
    name: str,
    page_start: int | None = None,
    page_end: int | None = None,
    progress: ProgressFn | None = None,
    strip_boilerplate: bool = True,
) -> ParsedPDF:
    """Extract and clean text from ``data``.

    Args:
        data: Raw PDF bytes.
        name: Display name (file name).
        page_start / page_end: Optional inclusive 1-based page range.
        progress: Optional ``callback(fraction, message)`` for UI progress bars.
        strip_boilerplate: Remove running headers/footers repeated on most pages.
    """
    t0 = time.perf_counter()
    reader = _open_reader(data)
    n_pages = len(reader.pages)
    if n_pages == 0:
        raise PDFLoadError("The PDF has no pages.")

    start = max(1, page_start or 1)
    end = min(n_pages, page_end or n_pages)
    if start > end:
        raise PDFLoadError(f"Invalid page range {start}-{end} for a {n_pages}-page document.")

    parsed = ParsedPDF(
        name=name,
        sha256=sha256_bytes(data),
        file_size_bytes=len(data),
        total_pages_in_file=n_pages,
        page_range=(start, end),
    )

    span = end - start + 1
    for i, page_no in enumerate(range(start, end + 1)):
        try:
            raw = reader.pages[page_no - 1].extract_text() or ""
        except Exception:  # a single malformed page must not kill a 3,000-page parse
            parsed.skipped_pages.append(page_no)
            continue
        text = clean_text(raw)
        if text:
            parsed.pages.append(PageText(page_number=page_no, text=text))
        if progress and (i % 25 == 0 or i == span - 1):
            progress((i + 1) / span, f"Parsing page {page_no}/{end}")

    if strip_boilerplate:
        parsed.removed_boilerplate = strip_running_headers(parsed.pages)
        parsed.pages = [p for p in parsed.pages if p.text]

    if not parsed.pages:
        raise PDFLoadError(
            "No extractable text was found in the selected pages.",
            hint="The PDF may be scanned images only; run OCR (e.g. ocrmypdf) first, or choose another page range.",
        )

    parsed.parse_seconds = time.perf_counter() - t0
    return parsed
