"""Small lexical helpers shared by the extractive answerer and the insight metrics."""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_\-\.]*[a-z0-9]|[a-z0-9]")

STOPWORDS = frozenset(
    """a about above after again against all also am an and any are as at be because been before being
    below between both but by can could did do does doing down during each either explain few for from
    further had has have having he her here hers how i if in into is it its itself just me more most my
    no nor not of off on once only or other our out over own same she should so some such than that the
    their them then there these they this those through to too under until up very was we were what when
    where which while who whom why will with would you your describe detail details defined define
    mechanism shall may used use using""".split()
)


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def content_words(text: str) -> set[str]:
    """Lower-cased words minus stopwords and 1-2 letter fragments (keeps 'le', 'ms', numbers)."""
    return {w for w in words(text) if w not in STOPWORDS and (len(w) > 2 or w.isdigit() or w in {"le", "ms", "db"})}
