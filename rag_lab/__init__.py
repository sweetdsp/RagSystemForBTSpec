"""rag_lab — a small, UI-agnostic RAG toolkit used by the Streamlit teaching app.

Modules:
    settings      environment variables, model catalogs, safety limits
    pdf_loader    PDF → cleaned page texts (pypdf)
    chunking      LangChain text splitters + page mapping
    embeddings    OpenAI / HuggingFace / Ollama / offline hashing embeddings
    vector_store  in-memory FAISS or ChromaDB with comparable cosine scores
    llm           prompt + OpenAI / Ollama / offline extractive generation
    pipeline      build_index() and answer_question()
    benchmark     5-config presets and parallel execution helpers
    insights      proxy metrics and plain-language observations
"""
