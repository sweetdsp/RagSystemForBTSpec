"""Answer generation: prompt construction and LLM providers.

Three providers are supported:

* **OpenAI** chat models via ``langchain-openai``.
* **Ollama** local chat models via ``langchain-ollama``.
* **Offline extractive** – no LLM at all: it returns the retrieved sentences
  that best overlap the question. It lets the app run fully offline and shows
  learners what "retrieval only" looks like compared with real generation.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from . import settings
from .errors import GenerationError, MissingAPIKeyError, MissingDependencyError
from .settings import ModelSpec
from .text_utils import content_words
from .vector_store import ScoredChunk

SYSTEM_PROMPT = (
    "You are an expert assistant for the Bluetooth Core Specification. "
    "Answer the user's question using ONLY the numbered context passages provided. "
    "Cite the passages you rely on with bracketed numbers like [1] or [2][3]. "
    "If the context does not contain enough information, say so explicitly rather than guessing. "
    "Be precise with numbers, units, field names and section references; when useful, name the "
    "Volume/Part shown in a passage's header."
)


@dataclass
class Generation:
    text: str
    seconds: float
    model_label: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    prompt_preview: str = ""
    extra: dict = field(default_factory=dict)


def format_context(chunks: list[ScoredChunk]) -> str:
    """Render retrieved chunks as numbered, page-cited passages for the prompt."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        m = c.document.metadata
        pages = f"p.{m.get('page_start')}" if m.get("page_start") == m.get("page_end") else f"pp.{m.get('page_start')}-{m.get('page_end')}"
        source = f"{m['section']}, {pages}" if m.get("section") else pages
        blocks.append(f"[{i}] ({source})\n{c.document.page_content}")
    return "\n\n---\n\n".join(blocks)


def build_user_prompt(question: str, chunks: list[ScoredChunk]) -> str:
    return f"Context passages:\n\n{format_context(chunks)}\n\n---\n\nQuestion: {question}\n\nAnswer:"


def _chat_model(spec: ModelSpec, temperature: float):
    if spec.provider == settings.LLM_OPENAI:
        key = settings.openai_api_key()
        if not key:
            raise MissingAPIKeyError(
                "OpenAI LLM selected but OPENAI_API_KEY is not set.",
                hint="Add it to .env, paste it in the sidebar, or choose Ollama / offline extractive mode.",
            )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise MissingDependencyError("langchain-openai is not installed.", hint="pip install langchain-openai") from exc
        return ChatOpenAI(model=spec.model, temperature=temperature, api_key=key, timeout=settings.LLM_REQUEST_TIMEOUT, max_retries=2)

    if spec.provider == settings.LLM_OLLAMA:
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise MissingDependencyError("langchain-ollama is not installed.", hint="pip install langchain-ollama") from exc
        return ChatOllama(model=spec.model, temperature=temperature, base_url=settings.OLLAMA_BASE_URL)

    raise GenerationError(f"Unknown LLM provider: {spec.provider!r}")


def chat(spec: ModelSpec, system: str, user: str, temperature: float) -> tuple[str, dict]:
    """Single chat completion. Returns ``(text, usage_dict)``; raises GenerationError."""
    model = _chat_model(spec, temperature)
    try:
        reply = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    except Exception as exc:
        hint = None
        if spec.provider == settings.LLM_OLLAMA:
            hint = f"Is Ollama running at {settings.OLLAMA_BASE_URL} with '{spec.model}' pulled? (ollama pull {spec.model})"
        elif spec.provider == settings.LLM_OPENAI:
            hint = "Check your API key, model name, quota and network connection."
        raise GenerationError(f"{spec.label} call failed: {type(exc).__name__}: {exc}", hint=hint) from exc
    usage = getattr(reply, "usage_metadata", None) or {}
    return str(reply.content).strip(), dict(usage)


def generate_answer(question: str, chunks: list[ScoredChunk], spec: ModelSpec, temperature: float) -> Generation:
    """Produce the final answer for ``question`` grounded in ``chunks``."""
    user_prompt = build_user_prompt(question, chunks)
    t0 = time.perf_counter()

    if spec.provider == settings.LLM_EXTRACTIVE:
        text = extractive_answer(question, chunks)
        return Generation(text, time.perf_counter() - t0, spec.label, prompt_preview=user_prompt)

    text, usage = chat(spec, SYSTEM_PROMPT, user_prompt, temperature)
    return Generation(
        text=text,
        seconds=time.perf_counter() - t0,
        model_label=spec.label,
        prompt_tokens=usage.get("input_tokens"),
        completion_tokens=usage.get("output_tokens"),
        prompt_preview=user_prompt,
    )


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def extractive_answer(question: str, chunks: list[ScoredChunk], max_sentences: int = 4) -> str:
    """Pick the retrieved sentences sharing the most content words with the question."""
    q_words = content_words(question)
    scored: list[tuple[float, int, int, str]] = []
    for ci, chunk in enumerate(chunks, start=1):
        for si, sent in enumerate(_SENTENCE_SPLIT.split(chunk.document.page_content)):
            sent = " ".join(sent.split())
            if len(sent) < 25:
                continue
            overlap = len(q_words & content_words(sent))
            if overlap:
                # Prefer overlap, then higher-ranked chunks, then earlier sentences.
                scored.append((overlap + 0.1 * chunk.similarity, -ci, -si, sent))
    if not scored:
        return "_Offline extractive mode: none of the retrieved passages share keywords with the question._"
    best = sorted(scored, reverse=True)[:max_sentences]
    best.sort(key=lambda s: (-s[1], -s[2]))  # restore reading order
    lines = [f"- {sent} [{-ci}]" for _, ci, _, sent in best]
    return "_Offline extractive mode (no LLM called) — most relevant retrieved sentences:_\n\n" + "\n".join(lines)
