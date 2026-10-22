"""Tiny on-disk cache for extracted page text and bookmarks, keyed by PDF hash.

Extracting the 3,141-page Bluetooth spec takes about a minute and reading its
bookmarks ~10 s. Persisting both (≈2 MB gzip) makes every later app start, in
any session, skip that work. Delete the ``.cache/`` folder to reset it.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
from pathlib import Path

from .pdf_loader import PARSER_VERSION, OutlineSection
from .settings import PROJECT_ROOT

log = logging.getLogger(__name__)

CACHE_DIR = Path(os.getenv("RAG_CACHE_DIR", PROJECT_ROOT / ".cache"))


def _path(sha256: str, kind: str) -> Path:
    return CACHE_DIR / f"{sha256[:24]}.v{PARSER_VERSION}.{kind}.json.gz"


def _load(path: Path):
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None
    except Exception as exc:  # corrupt cache file: ignore it, it will be rewritten
        log.warning("Ignoring unreadable cache file %s: %s", path, exc)
        return None


def _save(path: Path, payload) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh)
        tmp.replace(path)  # atomic: readers never see a half-written file
    except OSError as exc:  # read-only filesystem etc.: caching is optional
        log.warning("Could not write cache file %s: %s", path, exc)


def load_pages(sha256: str) -> dict[int, str]:
    data = _load(_path(sha256, "pages")) or {}
    return {int(k): v for k, v in data.items()}


def save_pages(sha256: str, pages: dict[int, str]) -> None:
    _save(_path(sha256, "pages"), {str(k): v for k, v in pages.items()})


def load_outline(sha256: str) -> list[OutlineSection] | None:
    data = _load(_path(sha256, "outline"))
    return None if data is None else [OutlineSection(**row) for row in data]


def save_outline(sha256: str, outline: list[OutlineSection]) -> None:
    _save(_path(sha256, "outline"), [vars(s) for s in outline])
