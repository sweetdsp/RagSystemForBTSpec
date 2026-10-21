"""Tab 2 — Multi-RAG Benchmark: build 5 configurations and compare their answers."""

from __future__ import annotations

import time
from concurrent.futures import wait

import pandas as pd
import streamlit as st

from rag_lab.benchmark import DEFAULT_PRESETS, Preset, ProgressBoard, make_executor, submit_builds, submit_queries
from rag_lab.chunking import STRATEGIES, TOKEN
from rag_lab.insights import analyze, composite_score
from rag_lab.pipeline import RAGConfig
from rag_lab.settings import LLM_EXTRACTIVE

from . import components as ui
from . import state
from .sidebar import SidebarState

K_PRESETS_DF = "bench_presets_df"
K_EDITOR_VERSION = "bench_editor_version"

BENCH_QUESTIONS = [
    "Explain the adaptive frequency hopping (AFH) mechanism in detail.",
    "What are the power classes defined for Bluetooth LE?",
    "How does LE Channel Selection Algorithm #2 differ from #1?",
    "What is the minimum number of used channels in an AFH channel map?",
]


def _presets_frame(presets: list[Preset]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Name": p.name, "Strategy": p.strategy, "Chunk size": p.chunk_size, "Overlap": p.chunk_overlap, "k": p.top_k, "Why": p.rationale} for p in presets]
    )


def _config_editor() -> list[Preset] | None:
    """Editable table of the 5 presets. Returns validated presets or None."""
    if K_PRESETS_DF not in st.session_state:
        st.session_state[K_PRESETS_DF] = _presets_frame(DEFAULT_PRESETS)
        st.session_state[K_EDITOR_VERSION] = 0

    c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
    c1.subheader("1 · The five configurations")
    if c2.button("↺ Reset presets", width="stretch"):
        st.session_state[K_PRESETS_DF] = _presets_frame(DEFAULT_PRESETS)
        st.session_state[K_EDITOR_VERSION] += 1  # new widget key = discard the user's edits
        st.rerun()

    st.caption(
        "Edit any cell. Sizes are in **characters** for the character splitters and **tokens** for TokenTextSplitter. "
        "Note how `chunk size × k` (the context budget) stays roughly constant for configs 1–4, so the experiment "
        "isolates *granularity* rather than the amount of text the LLM receives."
    )
    edited = st.data_editor(
        st.session_state[K_PRESETS_DF],
        key=f"bench_editor_{st.session_state[K_EDITOR_VERSION]}",
        hide_index=True,
        num_rows="fixed",
        width="stretch",
        column_config={
            "Name": st.column_config.TextColumn(width="medium", required=True, max_chars=30),
            "Strategy": st.column_config.SelectboxColumn(options=STRATEGIES, required=True, width="medium"),
            "Chunk size": st.column_config.NumberColumn(min_value=50, max_value=8000, step=50, required=True),
            "Overlap": st.column_config.NumberColumn(min_value=0, max_value=2000, step=5, required=True),
            "k": st.column_config.NumberColumn(min_value=1, max_value=10, step=1, required=True),
            "Why": st.column_config.TextColumn("What it demonstrates", width="large", disabled=True),
        },
    )

    presets, problems = [], []
    for i, row in edited.iterrows():
        name = str(row["Name"] or f"Config {i + 1}").strip()
        size, overlap, k = int(row["Chunk size"]), int(row["Overlap"]), int(row["k"])
        if overlap >= size:
            problems.append(f"**{name}**: overlap ({overlap}) must be smaller than chunk size ({size}).")
        presets.append(Preset(name, row["Strategy"], size, overlap, k, str(row["Why"])))
    names = [p.name for p in presets]
    if len(set(names)) != len(names):
        problems.append("Configuration names must be unique.")
    for p in problems:
        st.error(p)
    return None if problems else presets


def _build_all(sb: SidebarState, presets: list[Preset], parallel: bool) -> None:
    state.close_bench_indexes()
    parsed = state.get_parsed(sb.doc, sb.page_start, sb.page_end, sb.strip_boilerplate)
    configs: list[RAGConfig] = [p.to_config(sb.embedding, sb.vector_store) for p in presets]

    board = ProgressBoard(len(configs))
    bars = []
    with st.status(f"Building {len(configs)} RAG systems {'in parallel' if parallel else 'sequentially'}…", expanded=True) as status:
        for cfg in configs:
            bars.append(st.progress(0.0, text=f"{cfg.name}: queued"))
        log = st.empty()
        log_lines: list[str] = []
        t0 = time.perf_counter()

        with make_executor(parallel, len(configs)) as executor:
            futures = submit_builds(executor, parsed, configs, board)
            pending = set(futures.values())
            while pending:
                _, pending = wait(pending, timeout=0.25)
                for slot, (frac, msg) in board.snapshot().items():
                    bars[slot].progress(frac, text=f"{configs[slot].name}: {msg}")

        indexes, info = {}, {}
        for slot, fut in futures.items():
            name = configs[slot].name
            try:
                idx = fut.result()
                indexes[slot] = idx
                info[slot] = {"ok": True, "seconds": idx.stats.total_seconds}
                bars[slot].progress(1.0, text=f"✅ {name}: {idx.stats.total_chunks:,} chunks in {idx.stats.total_seconds:.2f}s")
                log_lines.append(f"✅ `{name}` — {idx.stats.total_chunks:,} chunks, avg {idx.stats.avg_chunk_chars:,.0f} chars, {idx.stats.total_seconds:.2f}s")
            except Exception as exc:
                msg = getattr(exc, "message", None) or f"{type(exc).__name__}: {exc}"
                hint = getattr(exc, "hint", None)
                info[slot] = {"ok": False, "error": msg + (f" — {hint}" if hint else "")}
                bars[slot].progress(0.0, text=f"❌ {name}: failed")
                log_lines.append(f"❌ `{name}` — {msg}")
        log.markdown("\n\n".join(log_lines))
        wall = time.perf_counter() - t0
        n_ok = len(indexes)
        status.update(
            label=f"Built {n_ok}/{len(configs)} RAG systems in {wall:.1f}s wall-clock",
            state="complete" if n_ok == len(configs) else ("error" if n_ok == 0 else "complete"),
            expanded=n_ok < len(configs),
        )

    st.session_state[state.K_BENCH_INDEXES] = indexes
    st.session_state[state.K_BENCH_BUILD_INFO] = {"configs": configs, "info": info, "wall": wall}
    st.session_state[state.K_BENCH_SOURCE] = state.source_key(sb)


def _build_summary() -> None:
    build = st.session_state.get(state.K_BENCH_BUILD_INFO)
    indexes = st.session_state.get(state.K_BENCH_INDEXES) or {}
    if not build:
        return
    rows = []
    for slot, cfg in enumerate(build["configs"]):
        idx = indexes.get(slot)
        rows.append(
            {
                "Config": cfg.name,
                "Strategy": cfg.strategy.replace("TextSplitter", ""),
                "Size": f"{cfg.chunk_size} {cfg.unit}",
                "Overlap": cfg.chunk_overlap,
                "k": cfg.top_k,
                "Chunks": idx.stats.total_chunks if idx else None,
                "Avg chars": round(idx.stats.avg_chunk_chars) if idx else None,
                "Build (s)": round(idx.stats.total_seconds, 2) if idx else None,
                "Status": "✅ ready" if idx else f"❌ {build['info'][slot].get('error', '')}",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with st.expander("🔬 Compare chunk length distributions & samples"):
        tabs = st.tabs([cfg.name for cfg in build["configs"]])
        for slot, tab in enumerate(tabs):
            with tab:
                if slot in indexes:
                    ui.chunk_length_histogram(indexes[slot], height=150)
                    ui.chunk_previews(indexes[slot], key=f"bench_{slot}")
                else:
                    st.error(build["info"][slot].get("error", "Build failed"))


def _answer_card(slot: int, result, metrics, is_best: bool, distance_label: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{'🏆 ' if is_best else ''}{slot + 1}. {result.config.name}**")
        st.markdown(
            f"<span class='cfg-caption'>{result.config.strategy.replace('TextSplitter', '')} · size {result.config.chunk_size} {result.config.unit} · "
            f"overlap {result.config.chunk_overlap} · k={result.config.top_k}</span>",
            unsafe_allow_html=True,
        )
        if result.error:
            st.error(result.error, icon="🚫")
            return
        c1, c2 = st.columns(2)
        c1.metric("Latency", f"{result.total_seconds:.2f}s", help=f"retrieval {result.retrieval_seconds * 1000:.0f} ms + generation {result.generation_seconds:.2f}s")
        c2.metric("Context", f"{result.context_chars / 1000:.1f}k", help=f"{result.context_chars:,} characters of retrieved context sent to the LLM")
        st.markdown(result.answer)
        if metrics is not None:
            st.caption(
                f"coverage {metrics.question_coverage:.0%} · grounded {metrics.groundedness:.0%} · mean sim {metrics.mean_similarity:.2f}"
                + (f" · judge {metrics.judge_score:.1f}/5" if metrics.judge_score is not None else "")
                + (f" · ref-F1 {metrics.reference_f1:.2f}" if metrics.reference_f1 is not None else "")
            )
            if metrics.judge_comment:
                st.caption(f"🧑‍⚖️ {metrics.judge_comment}")
        with st.expander(f"Retrieved snippets ({len(result.chunks)})"):
            ui.retrieved_chunks(result.chunks, distance_label)


def _insights_section(insights, results) -> None:
    st.subheader("4 · Educational insights")
    st.caption(
        "These are transparent *proxy* metrics, not ground truth. Add a reference answer or enable the LLM judge for a "
        "stronger accuracy signal, and always read the answers yourself."
    )
    for line in insights.observations:
        st.markdown(f"- {line}")

    ok = [m for m in insights.metrics if not m.error]
    if not ok:
        return
    df = pd.DataFrame(
        [
            {
                "Config": m.name,
                "Chunk size": f"{m.chunk_size} {m.unit}",
                "k": m.top_k,
                "Context chars": m.context_chars,
                "Mean similarity": m.mean_similarity,
                "Question coverage": m.question_coverage,
                "Groundedness": m.groundedness,
                "Consensus": m.consensus,
                "Reference F1": m.reference_f1,
                "Judge (1-5)": m.judge_score,
                "Heuristic score": composite_score(m),
                "Latency (s)": m.total_seconds,
                "Answer words": m.answer_words,
            }
            for m in ok
        ]
    )
    df = df.dropna(axis=1, how="all")
    pct = st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=1.0)
    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        column_config={
            "Question coverage": pct,
            "Groundedness": pct,
            "Mean similarity": st.column_config.NumberColumn(format="%.3f"),
            "Consensus": st.column_config.NumberColumn(format="%.3f"),
            "Reference F1": st.column_config.NumberColumn(format="%.2f"),
            "Judge (1-5)": st.column_config.NumberColumn(format="%.1f"),
            "Heuristic score": st.column_config.NumberColumn(format="%.3f"),
            "Latency (s)": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    c1, c2 = st.columns(2)
    with c1:
        ui.single_bar_chart(df, "Latency (s)", "Total latency (s)")
        ui.single_bar_chart(df, "Mean similarity", "Mean cosine similarity of retrieved chunks", fmt=".3f")
    with c2:
        ui.single_bar_chart(df, "Context chars", "Context sent to the LLM (characters)", fmt=",.0f")
        ui.single_bar_chart(df, "Groundedness", "Groundedness (answer keywords found in context)", fmt=".0%")

    with st.expander("📚 How to read these metrics & questions for discussion"):
        st.markdown(
            """
- **Mean similarity** — how close the retrieved chunks are to the question in embedding space. Small chunks
  usually score higher because each vector represents one focused idea.
- **Question coverage** — share of the question's keywords present in the retrieved text. Low coverage means
  retrieval likely missed something, no matter how good the LLM is.
- **Groundedness** — share of the answer's keywords that appear in the retrieved text. Low values flag content
  that came from the model's memory (possible hallucination) or heavy paraphrasing.
- **Consensus** — similarity of an answer to the other four. A low outlier deserves a careful read.
- **Reference F1 / LLM judge** — optional accuracy signals; the judge is itself an LLM and can be wrong.

**Discuss:**
1. Which configuration gave the most *precise* answer (exact numbers, field names)? Which the most *complete*?
2. Did the micro-chunk answer miss context that sat in a neighbouring sentence or table row?
3. Did the large-chunk answer include unrelated details (noise) or contradict the spec?
4. Re-run with temperature 0.8. Which configuration's answer changed the most, and why?
5. Change only `k` for the micro-chunk config (e.g. 6 → 12). Does more context fix the lost-context problem?
"""
        )


def render_benchmark_tab(sb: SidebarState) -> None:
    st.markdown(
        "Prepare **five RAG systems** that differ only in chunking and top-k, ask them the same question, and "
        f"compare. Embedding (**{sb.embedding.label}**), vector store (**{sb.vector_store}**) and LLM "
        f"(**{sb.llm.label}**) come from the sidebar, so the comparison is a controlled experiment."
    )
    presets = _config_editor()

    st.subheader("2 · Build")
    c1, c2 = st.columns([2, 3], vertical_alignment="center")
    parallel = c2.toggle("Build & query in parallel (threads)", value=True, help="Parallel builds overlap network calls (OpenAI/Ollama). With local CPU models the gain is smaller.")
    has_token_config = bool(presets) and any(p.strategy == TOKEN for p in presets)
    if c1.button("🚀 Prepare 5 Parallel RAG Systems", type="primary", disabled=not sb.ready or presets is None, width="stretch"):
        try:
            _build_all(sb, presets, parallel)
        except Exception as exc:
            ui.show_error(exc)
    if has_token_config:
        st.caption(f"ℹ️ {TOKEN} sizes are in tokens: 256 tokens ≈ {256 * 4:,} characters of English text.")

    indexes = st.session_state.get(state.K_BENCH_INDEXES) or {}
    _build_summary()
    if not indexes:
        st.info("Click **Prepare 5 Parallel RAG Systems** to build the indexes.")
        return
    if st.session_state.get(state.K_BENCH_SOURCE) != state.source_key(sb):
        st.warning("Document, page range, embedding model or vector store changed in the sidebar — rebuild to use them.", icon="🔁")
    built = st.session_state[state.K_BENCH_BUILD_INFO]["configs"]
    if presets and [p.to_config(sb.embedding, sb.vector_store) for p in presets] != built:
        st.warning("The configuration table changed since the last build. Rebuild to apply the edits.", icon="🔁")

    # ------------------------------------------------------------------ query
    st.subheader("3 · Ask all systems the same question")
    example = st.selectbox("Example questions", BENCH_QUESTIONS, key="b_example")
    question = st.text_area("Benchmark question", value=example, height=80, key=f"b_question_{example}")
    with st.expander("Optional: accuracy signals"):
        reference = st.text_area("Reference answer (ground truth)", key="b_reference", height=100, placeholder="Paste the correct answer from the spec to score each config by token-F1…")
        judge = st.checkbox(
            "Ask the LLM to judge each answer (1–5)", value=False, disabled=sb.llm.provider == LLM_EXTRACTIVE,
            help="Makes 5 extra LLM calls. Unavailable in offline extractive mode.",
        )

    if st.button(f"⚖️ Benchmark All {len(indexes)} RAGs", type="primary", disabled=not question.strip()):
        with st.spinner(f"Querying {len(indexes)} systems with {sb.llm.label}…"):
            with make_executor(parallel, len(indexes)) as executor:
                futures = submit_queries(executor, indexes, question.strip(), sb.llm, sb.temperature)
                results = {slot: f.result() for slot, f in futures.items()}
        with st.spinner("Analysing answers…"):
            try:
                insights = analyze(results, sb.embedding, sb.llm, reference=reference, use_judge=judge)
            except Exception as exc:  # insights are a bonus; never lose the answers
                ui.show_error(exc)
                insights = None
        st.session_state[state.K_BENCH_RESULTS] = results
        st.session_state[state.K_BENCH_INSIGHTS] = insights

    results = st.session_state.get(state.K_BENCH_RESULTS)
    if not results:
        return
    insights = st.session_state.get(state.K_BENCH_INSIGHTS)
    metrics_by_slot = {m.slot: m for m in insights.metrics} if insights else {}
    best = insights.best_slot if insights else None

    layout = st.segmented_control("Layout", ["Side by side", "Grid (2 per row)", "Stacked"], default="Side by side", key="b_layout")
    per_row = {"Side by side": len(results), "Grid (2 per row)": 2, "Stacked": 1}.get(layout or "Side by side", len(results))
    slots = sorted(results)
    for row_start in range(0, len(slots), per_row):
        cols = st.columns(per_row)
        for col, slot in zip(cols, slots[row_start : row_start + per_row]):
            with col:
                _answer_card(slot, results[slot], metrics_by_slot.get(slot), slot == best, indexes[slot].store.distance_label if slot in indexes else "")

    if insights:
        _insights_section(insights, results)
