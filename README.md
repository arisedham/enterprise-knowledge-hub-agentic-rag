# Enterprise Knowledge Hub — Agentic RAG (Company FAQ / Policy Assistant)

A demo Agentic RAG system that answers employee queries by retrieving from multiple internal sources (PDFs, Confluence exports, runbooks) and external references. The system demonstrates "agency" — an LLM-driven orchestration of Router → Retriever → Verifier → Synthesizer agents — producing grounded answers with provenance and inline citations.

## Features
- Multi-source retrieval: PDFs, DOCX, Confluence HTML, structured tables.
- Agentic orchestration:
  - Router Agent chooses which knowledge sources to query.
  - Retriever Agents perform hybrid retrieval (sparse + dense).
  - Refinement Agent rephrases queries if coverage is low.
  - Synthesizer Agent composes answers with inline citations.
  - Verifier Agent computes confidence and detects contradictions.
- Streamlit demo UI showing retrieval path and highlighted sources.
- Test suite for ingestion, retrieval, and end-to-end QA.

## Architecture
See `docs/ARCHITECTURE.md` for diagrams and component breakdown:
- Ingestion → Chunking → Embedding → Vector DB
- Router → Retriever(s) → Aggregator → Verifier → Synthesizer
- Streamlit frontend → Backend API

## Tech stack 
- LLM: OpenAI 
- Embeddings: OpenAI / local embedding model
- Vector DB: Chroma (demo) / Faiss / Milvus (production)
- Backend: FastAPI
- Frontend: Streamlit
- Orchestration: Python (LangChain or custom agent flow)
- Tests: pytest, tox
- CI: GitHub Actions (run tests, lint, build docker)

## Quickstart (local demo)
> Prereqs: Python 3.10+, pip, Git, (optionally) OpenAI API key

1. Clone
```bash
git clone https://github.com/<your-repo>/enterprise-knowledge-hub.git
cd enterprise-knowledge-hub
