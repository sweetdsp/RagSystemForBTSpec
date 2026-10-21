"""PDF parsing with pypdf.

The Bluetooth Core Specification is ~3,100 pages, so this module:

* lets the caller restrict parsing to a page range (huge memory/time saver) and
  reuse already-extracted pages through a ``page_cache``,
* reads the PDF bookmarks (outline) so every page knows its section, e.g.
  "Vol 6: Low Energy Controller › Part A: Physical Layer Specification",
* removes running headers/footers and control characters,
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

#: Bump when text extraction/cleaning changes, so on-disk page caches are rebuilt.
PARSER_VERSION = 2

_MULTISPACE = re.compile(r"[ \t ]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_DIGITS = re.compile(r"\d+")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")  # e.g. figure labels in symbol fonts

#: Lines at the top/bottom of each page inspected for running headers/footers.
#: (The Bluetooth spec stacks four header lines: section title + version date,
#: "Bluetooth SIG Proprietary", "... | Vol 6, Part B page 2799" and the Part title.)
TOP_LINES = 4
BOTTOM_LINES = 3

#: A top line is a running header if it repeats on >= LOCAL_HITS of the
#: 2*LOCAL_WINDOW+1 surrounding pages (headers change per chapter, so a single
#: document-wide frequency threshold misses most of them).
LOCAL_WINDOW = 5
LOCAL_HITS = 6

#: Outline entries covering more than this share of the document are wrappers
#: ("Core Specification v5.4") and are left out of section labels.
WRAPPER_SHARE = 0.8
SECTION_SEPARATOR = " › "


@dataclass
class PageText:
    """Cleaned text of a single PDF page (1-based ``page_number``)."""

    page_number: int
    text: str
    section: str = ""  # bookmark breadcrumb, "" when the PDF has no outline


@dataclass(frozen=True)
class OutlineSection:
    """A bookmark with its inclusive page span (1-based)."""

    title: str
    depth: int  # 0 = top level of the kept outline
    start: int
    end: int

    @property
    def label(self) -> str:
        return f"{'↳ ' * self.depth}{self.title} (pp. {self.start}–{self.end})"


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
    outline: list[OutlineSection] = field(default_factory=list)
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


def _is_glyph_garbage(line: str) -> bool:
    """True for lines extracted from figures drawn with symbol fonts.

    Such text comes out as runs of unrelated accented/Greek letters (e.g.
    "/^ͺ^ǇŶĐͺĞůĂǇĨŽƌ"): meaningless to an embedding model and pure noise in a chunk.
    """
    letters = [ch for ch in line if ch.isalpha()]
    exotic = sum(1 for ch in letters if ord(ch) > 0x24F or 0x100 <= ord(ch) <= 0x24F)
    return exotic >= 4 and exotic >= 0.4 * len(letters)


def clean_text(text: str) -> str:
    """Light normalisation that keeps paragraph structure for the splitters."""
    text = _CONTROL_CHARS.sub("", text)
    text = "\n".join(line for line in text.splitlines() if not _is_glyph_garbage(line))
    text = _HYPHEN_BREAK.sub(r"\1\2", text)  # re-join words hyphenated across lines
    text = _MULTISPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def _edge_key(line: str) -> str:
    """Normalise a line so 'Vol 6, Part A page 2860' and '... page 2861' compare equal."""
    return _DIGITS.sub("#", line.strip().lower())


def _edge_keys(text: str) -> tuple[list[str], list[str]]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return [_edge_key(ln) for ln in lines[:TOP_LINES]], [_edge_key(ln) for ln in lines[-BOTTOM_LINES:]]


def _tail(key: str, n_words: int = 3) -> str:
    words = key.split()
    return " ".join(words[-n_words:]) if len(words) > n_words else ""


def strip_running_headers(pages: list[PageText], min_share: float = 0.5) -> list[str]:
    """Remove running header/footer lines (in place).

    Specs print banners such as "BLUETOOTH CORE SPECIFICATION Version 5.4 | Vol 6,
    Part B page 2860" on every page. Left in, they end up in nearly every chunk,
    make unrelated chunks look alike and waste context tokens. A line (with
    digits ignored) is removed when it sits at the top/bottom edge of

    * at least ``min_share`` of all pages (document-wide banners/footers), or,
      for top lines only, of most neighbouring pages (chapter-specific headers),
      or ends with the same three words as the top lines of most pages.

    Returns the removed patterns, most frequent first (shown in the UI).
    """
    if len(pages) < 3:
        return []
    keys = [_edge_keys(p.text) for p in pages]
    threshold = max(3, int(len(pages) * min_share))
    counts = Counter(k for top, bottom in keys for k in set(top + bottom) if k)
    global_bp = {k for k, n in counts.items() if n >= threshold}
    # Headers that merge a varying title with a fixed tail ("<section> Version Date: #-#-#"):
    # a tail shared by the top lines of most pages marks those lines as headers too.
    tails = Counter(t for top, _ in keys for t in {_tail(k) for k in top} if t)
    header_tails = {t for t, n in tails.items() if n >= threshold}

    removed: Counter[str] = Counter()
    for i, page in enumerate(pages):
        lo, hi = max(0, i - LOCAL_WINDOW), min(len(pages), i + LOCAL_WINDOW + 1)
        local = Counter(k for top, _ in keys[lo:hi] for k in set(top))
        top_bp = global_bp | {k for k in keys[i][0] if k and (local[k] >= LOCAL_HITS or _tail(k) in header_tails)}

        lines = page.text.splitlines()
        nonblank = [j for j, ln in enumerate(lines) if ln.strip()]
        top_idx, bottom_idx = set(nonblank[:TOP_LINES]), set(nonblank[-BOTTOM_LINES:])
        kept = []
        for j, ln in enumerate(lines):
            key = _edge_key(ln)
            if (j in top_idx and key in top_bp) or (j in bottom_idx and key in global_bp):
                removed[key] += 1
            else:
                kept.append(ln)
        page.text = "\n".join(kept).strip()
    return [k for k, _ in removed.most_common()]


def read_outline(reader: PdfReader, max_depth: int = 5, max_level: int = 1) -> list[OutlineSection]:
    """Flatten the PDF bookmarks into sections with page spans.

    Wrapper entries spanning almost the whole file are dropped and the first
    ``max_level + 1`` remaining levels are kept (for the Bluetooth spec: Vol and
    Part, ~60 entries). Returns ``[]`` for PDFs without bookmarks.

    Note: pypdf materialises the whole outline first, which takes ~10 s for the
    Bluetooth spec, so callers should cache the result per document.
    """
    n_pages = len(reader.pages)
    raw: list[tuple[int, str, int]] = []

    def walk(items, depth: int) -> None:
        for item in items:
            if isinstance(item, list):
                if depth < max_depth:
                    walk(item, depth + 1)
                continue
            try:
                raw.append((depth, str(item.title).strip(), reader.get_destination_page_number(item) + 1))
            except Exception:
                continue  # dangling bookmark

    try:
        walk(reader.outline, 0)
    except Exception:
        return []
    if not raw:
        return []

    # Each entry ends where the next entry at the same or a shallower depth begins.
    spans = []
    for i, (depth, title, start) in enumerate(raw):
        end = n_pages
        for depth2, _, start2 in raw[i + 1 :]:
            if depth2 <= depth:
                end = max(start, start2 - 1)
                break
        spans.append((depth, title, start, end))

    # Depth counted in non-wrapper ancestors, so "Core Specification v5.4 › Vol 6 › Part A"
    # becomes Vol (0) › Part (1).
    sections: list[OutlineSection] = []
    stack: list[tuple[int, bool]] = []  # (raw depth, is_wrapper) of the current ancestors
    for depth, title, start, end in spans:
        while stack and stack[-1][0] >= depth:
            stack.pop()
        wrapper = (end - start + 1) > WRAPPER_SHARE * n_pages
        level = sum(1 for _, w in stack if not w)
        stack.append((depth, wrapper))
        if not wrapper and level <= max_level:
            sections.append(OutlineSection(title, level, start, end))
    return sections


def section_for_page(outline: list[OutlineSection], page_number: int, max_depth: int = 1) -> str:
    """Breadcrumb of the outline entries containing ``page_number`` (e.g. Vol › Part)."""
    crumbs: dict[int, str] = {}
    for sec in outline:
        if sec.depth <= max_depth and sec.start <= page_number <= sec.end:
            crumbs[sec.depth] = sec.title
    return SECTION_SEPARATOR.join(crumbs[d] for d in sorted(crumbs))


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


def outline_of(data: bytes) -> list[OutlineSection]:
    """Bookmarks of a PDF given as bytes (cache the result; see :func:`read_outline`)."""
    return read_outline(_open_reader(data))


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
    page_cache: dict[int, str] | None = None,
    outline: list[OutlineSection] | None = None,
) -> ParsedPDF:
    """Extract and clean text from ``data``.

    Args:
        data: Raw PDF bytes.
        name: Display name (file name).
        page_start / page_end: Optional inclusive 1-based page range.
        progress: Optional ``callback(fraction, message)`` for UI progress bars.
        strip_boilerplate: Remove running headers/footers.
        page_cache: Optional ``{page_number: cleaned text}`` dict for this document.
            Pages found in it are not re-extracted; newly extracted pages are added.
            Extraction is the slow part (~20 ms/page), so a cache makes changing the
            page range or rebuilding nearly instant.
        outline: Pre-computed :func:`outline_of` result; read from the PDF if None.
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
        outline=read_outline(reader) if outline is None else outline,
    )

    span = end - start + 1
    for i, page_no in enumerate(range(start, end + 1)):
        if page_cache is not None and page_no in page_cache:
            text = page_cache[page_no]
        else:
            try:
                text = clean_text(reader.pages[page_no - 1].extract_text() or "")
            except Exception:  # a single malformed page must not kill a 3,000-page parse
                parsed.skipped_pages.append(page_no)
                continue
            if page_cache is not None:
                page_cache[page_no] = text
        if text:
            section = section_for_page(parsed.outline, page_no)
            parsed.pages.append(PageText(page_number=page_no, text=text, section=section))
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
