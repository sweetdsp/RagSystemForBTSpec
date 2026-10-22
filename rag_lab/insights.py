"""Educational analysis of a 5-way benchmark.

Without a labelled test set we cannot measure true accuracy, so this module
computes transparent *proxy* metrics and turns them into plain-language
observations. Every metric is explained in the UI so learners understand
what it can and cannot tell them:

* **Question coverage** – share of the question's keywords present in the
  retrieved context. Low → retrieval probably missed relevant text.
* **Groundedness** – share of the answer's keywords that also appear in the
  retrieved context. Low → the LLM added information from elsewhere (its
  parametric memory, i.e. a possible hallucination) or paraphrased heavily.
* **Consensus** – mean cosine similarity of an answer to the other answers.
  Outliers are worth reading carefully: they are either uniquely right or wrong.
* **Reference match** (optional) – token-F1 against a reference answer you type.
* **LLM judge** (optional) – the selected LLM grades each answer 1-5.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, field

from . import settings
from .embeddings import get_embeddings
from .errors import RAGLabError
from .llm import chat, format_context
from .pipeline import QueryResult
from .settings import ModelSpec
from .text_utils import content_words

_CITATION_RE = re.compile(r"\[\d+\]")
_ITALIC_NOTE_RE = re.compile(r"^_.*_$", re.M)  # e.g. the extractive answerer's intro line


@dataclass
class SlotMetrics:
    slot: int
    name: str
    chunk_size: int
    unit: str
    top_k: int
    context_chars: int
    mean_similarity: float
    top_similarity: float
    question_coverage: float
    groundedness: float
    answer_words: int
    consensus: float | None = None
    reference_f1: float | None = None
    judge_score: float | None = None
    judge_comment: str = ""
    total_seconds: float = 0.0
    retrieval_seconds: float = 0.0
    generation_seconds: float = 0.0
    error: str | None = None

    @property
    def approx_chars(self) -> int:
        """Chunk size in characters (token sizes converted with the ~4 chars/token rule)."""
        return self.chunk_size * 4 if self.unit == "tokens" else self.chunk_size


@dataclass
class BenchmarkInsights:
    metrics: list[SlotMetrics]
    best_slot: int | None
    best_basis: str
    observations: list[str] = field(default_factory=list)


def _ratio(part: set, whole: set) -> float:
    return len(part & whole) / len(whole) if whole else 0.0


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style F1 over content words (order-insensitive)."""
    p, r = content_words(prediction), content_words(reference)
    common = len(p & r)
    if not common:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _consensus(answers: dict[int, str], embedding: ModelSpec) -> dict[int, float]:
    """Mean similarity of each answer to the others (embedding-based, lexical fallback)."""
    slots = list(answers)
    if len(slots) < 2:
        return {}
    try:
        vectors = dict(zip(slots, get_embeddings(embedding).embed_documents([answers[s] for s in slots])))
        sim = lambda a, b: _cosine(vectors[a], vectors[b])  # noqa: E731
    except Exception:
        bags = {s: content_words(answers[s]) for s in slots}
        sim = lambda a, b: len(bags[a] & bags[b]) / (len(bags[a] | bags[b]) or 1)  # noqa: E731
    return {s: statistics.fmean(sim(s, o) for o in slots if o != s) for s in slots}


JUDGE_SYSTEM = (
    "You are a strict grader of answers about the Bluetooth Core Specification. "
    "Grade the ANSWER to the QUESTION using the CONTEXT it was generated from"
    "{reference_clause}. Return ONLY a JSON object: "
    '{{"correctness": 1-5, "completeness": 1-5, "faithfulness": 1-5, "comment": "<one short sentence>"}}. '
    "faithfulness = every claim is supported by the context."
)


def judge_answer(result: QueryResult, llm: ModelSpec, reference: str | None) -> tuple[float, str]:
    """Ask the LLM to grade one answer. Returns ``(mean score 1-5, comment)``."""
    system = JUDGE_SYSTEM.format(reference_clause=" and the REFERENCE answer" if reference else "")
    user = f"QUESTION:\n{result.question}\n\nCONTEXT:\n{format_context(result.chunks)}\n\n"
    if reference:
        user += f"REFERENCE:\n{reference}\n\n"
    user += f"ANSWER:\n{result.answer}"
    text, _ = chat(llm, system, user, temperature=0.0)
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise RAGLabError("Judge returned no JSON.", hint=text[:200])
    data = json.loads(match.group(0))
    scores = [float(data[k]) for k in ("correctness", "completeness", "faithfulness") if k in data]
    if not scores:
        raise RAGLabError("Judge JSON had no scores.", hint=text[:200])
    return statistics.fmean(scores), str(data.get("comment", ""))


def analyze(
    results: dict[int, QueryResult],
    embedding: ModelSpec,
    llm: ModelSpec,
    reference: str | None = None,
    use_judge: bool = False,
) -> BenchmarkInsights:
    """Compute per-slot metrics, pick a "best" config and write observations."""
    reference = (reference or "").strip() or None
    q_words = content_words(next(iter(results.values())).question) if results else set()

    metrics: list[SlotMetrics] = []
    for slot, r in sorted(results.items()):
        ctx_words = set().union(*(content_words(c.document.page_content) for c in r.chunks)) if r.chunks else set()
        ans_words = content_words(_CITATION_RE.sub(" ", _ITALIC_NOTE_RE.sub(" ", r.answer)))
        metrics.append(
            SlotMetrics(
                slot=slot,
                name=r.config.name,
                chunk_size=r.config.chunk_size,
                unit=r.config.unit,
                top_k=r.config.top_k,
                context_chars=r.context_chars,
                mean_similarity=r.mean_similarity,
                top_similarity=max((c.similarity for c in r.chunks), default=0.0),
                question_coverage=_ratio(ctx_words, q_words),
                groundedness=_ratio(ctx_words, ans_words),
                answer_words=len(r.answer.split()),
                total_seconds=r.total_seconds,
                retrieval_seconds=r.retrieval_seconds,
                generation_seconds=r.generation_seconds,
                error=r.error,
            )
        )

    ok = [m for m in metrics if not m.error]
    by_slot = {m.slot: m for m in metrics}
    for slot, value in _consensus({m.slot: results[m.slot].answer for m in ok}, embedding).items():
        by_slot[slot].consensus = value
    if reference:
        for m in ok:
            m.reference_f1 = token_f1(results[m.slot].answer, reference)
    judge_errors: list[str] = []
    if use_judge and llm.provider != settings.LLM_EXTRACTIVE:
        for m in ok:
            try:
                m.judge_score, m.judge_comment = judge_answer(results[m.slot], llm, reference)
            except Exception as exc:  # judging is best-effort
                judge_errors.append(f"{m.name}: {exc}")

    best, basis = _pick_best(ok, reference is not None)
    observations = _observations(ok, metrics, results, best, basis, llm)
    if judge_errors:
        observations.append("⚠️ LLM judge failed for: " + "; ".join(judge_errors))
    return BenchmarkInsights(metrics=metrics, best_slot=best, best_basis=basis, observations=observations)


def composite_score(m: SlotMetrics) -> float:
    """Reference-free heuristic: grounded, on-topic answers that agree with the others."""
    return 0.4 * m.groundedness + 0.3 * m.question_coverage + 0.3 * (m.consensus or 0.0)


def _pick_best(ok: list[SlotMetrics], has_reference: bool) -> tuple[int | None, str]:
    if not ok:
        return None, "no successful runs"
    if has_reference:
        return max(ok, key=lambda m: m.reference_f1 or 0.0).slot, "highest token-F1 against your reference answer"
    judged = [m for m in ok if m.judge_score is not None]
    if judged:
        return max(judged, key=lambda m: m.judge_score).slot, "highest LLM-judge score"
    return max(ok, key=composite_score).slot, "highest heuristic score (0.4·groundedness + 0.3·coverage + 0.3·consensus)"


def _pct(x: float) -> str:
    return f"{x:.0%}"


def _observations(ok, metrics, results, best, basis, llm) -> list[str]:
    obs: list[str] = []
    for m in metrics:
        if m.error:
            obs.append(f"❌ **{m.name}** failed: {m.error}")
    if not ok:
        return obs

    by_slot = {m.slot: m for m in ok}
    if best is not None:
        b = by_slot[best]
        obs.append(
            f"🏆 **{b.name}** ({b.chunk_size} {b.unit}, k={b.top_k}) looks strongest by {basis}. "
            "Read its answer next to the others before trusting the number: proxies can be fooled."
        )

    smallest = min(ok, key=lambda m: m.approx_chars)
    largest = max(ok, key=lambda m: m.approx_chars)
    if smallest.slot != largest.slot:
        if smallest.question_coverage + 0.1 < largest.question_coverage:
            obs.append(
                f"🔍 **Smaller chunks lost context:** {smallest.name} covered {_pct(smallest.question_coverage)} of the question's "
                f"keywords vs {_pct(largest.question_coverage)} for {largest.name}. Tiny chunks match precisely but often "
                "separate a definition from the details around it."
            )
        elif largest.question_coverage + 0.1 < smallest.question_coverage:
            obs.append(
                f"🔍 **Small chunks retrieved better here:** {smallest.name} covered {_pct(smallest.question_coverage)} of the "
                f"question's keywords vs {_pct(largest.question_coverage)} for {largest.name}; with only k={largest.top_k} large "
                "chunks, one off-topic hit wastes a big share of the context."
            )
        if largest.mean_similarity + 0.03 < smallest.mean_similarity:
            obs.append(
                f"🌫️ **Large chunks dilute the embedding:** mean similarity {largest.mean_similarity:.2f} ({largest.name}) vs "
                f"{smallest.mean_similarity:.2f} ({smallest.name}). One vector has to summarise many topics, so the match to a "
                "specific question is weaker, and more unrelated text (noise) reaches the LLM."
            )
        ctx_ratio = largest.context_chars / max(1, smallest.context_chars)
        obs.append(
            f"📏 **Context sent to the LLM** ranged from {smallest.context_chars:,} chars ({smallest.name}) to "
            f"{largest.context_chars:,} chars ({largest.name}), ×{ctx_ratio:.1f}. More context costs more tokens and latency "
            "and can bury the key sentence ('lost in the middle')."
        )

    if llm.provider == settings.LLM_EXTRACTIVE:
        obs.append("ℹ️ Offline extractive mode copies sentences from the context, so groundedness is ~100% by construction. Switch to a real LLM to study hallucination.")
    else:
        weak = [m for m in ok if m.groundedness < 0.6]
        if weak:
            names = ", ".join(f"{m.name} ({_pct(m.groundedness)})" for m in weak)
            obs.append(
                f"⚠️ **Possible unsupported content:** {names} — many answer keywords do not appear in the retrieved context. "
                "The model may be answering from memory (hallucination risk) or paraphrasing; check the cited passages."
            )
        else:
            obs.append("✅ All answers are well grounded: most of their keywords appear in the retrieved context.")

    with_consensus = [m for m in ok if m.consensus is not None]
    if len(with_consensus) >= 3:
        outlier = min(with_consensus, key=lambda m: m.consensus)
        avg = statistics.fmean(m.consensus for m in with_consensus)
        if outlier.consensus < avg - 0.08:
            obs.append(f"🧭 **{outlier.name}** disagrees most with the others (consensus {outlier.consensus:.2f} vs avg {avg:.2f}). It is either uniquely right or uniquely wrong — inspect its context.")

    page_sets = {m.slot: set(results[m.slot].pages) for m in ok}
    common = set.intersection(*page_sets.values()) if page_sets else set()
    if common:
        obs.append(f"📄 Every configuration retrieved text from page(s) {', '.join(map(str, sorted(common)[:10]))} — strong evidence that the answer lives there.")
    else:
        obs.append("📄 No single page was retrieved by every configuration: chunking changed *what* was found, not just how much.")

    fastest = min(ok, key=lambda m: m.total_seconds)
    slowest = max(ok, key=lambda m: m.total_seconds)
    obs.append(
        f"⏱️ Fastest: **{fastest.name}** ({fastest.total_seconds:.2f}s); slowest: **{slowest.name}** ({slowest.total_seconds:.2f}s). "
        "Retrieval is usually milliseconds — generation time grows with the amount of context and the answer length."
    )
    return obs
