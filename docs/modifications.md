# Modifications Report

## 1. Original Project

(HISTORICAL) The original uploaded notebook implemented the first two-thirds of a Retrieval-Augmented Generation (RAG) pipeline over a single biomedical review article (`biomedicines-12-01543.pdf`). It included PDF text extraction, layout-aware parsing, cleaning, token-based chunking, and embedding generation using local HuggingFace models.

It lacked full generation, proper safety guardrails, citation verification, and a production-grade secure architecture.

## 2. Hardened Cloud Architecture (Current State)

The project has been completely overhauled into a strict, production-ready, cloud-only architecture. 

**Architectural Constraints:**
- **Embeddings:** `nvidia/nemotron-3-embed-1b:free` via OpenRouter Cloud.
- **Reranking:** `nvidia/llama-nemotron-rerank-vl-1b-v2:free` via OpenRouter Cloud.
- **Generation:** `openai/gpt-oss-20b` via Groq Cloud.
- **Safety / Security:** Lakera Guard Cloud API.
- **Grounding Validation:** LLM-as-a-judge using `openai/gpt-oss-20b` (Groq).

Local ML models (SentenceTransformers, HF inference) have been entirely removed to ensure environment stability and adherence to cloud-only mandates.

## 3. Key Bug Fixes & Hardening (Final Pass)

- **Fail-Closed Security**: `secure_generate_answer()` strictly fails closed. If Lakera Guard fails, the reranker fails, or the grounding validation fails, the system returns an abstention (`abstained = True`) rather than proceeding with unverified or potentially unsafe content.
- **No-Citation Grounding**: If the LLM generates a substantive answer without valid citations, it fails the grounding check and the pipeline safely abstains.
- **Thread-Safety**: Added `threading.Lock()` to `_bm25_cache` and `_semantic_cache` to support concurrent requests through the Gradio UI.
- **Citation Ranges**: `extract_citation_numbers()` was upgraded to correctly parse and unpack citation ranges like `[12-15]` into discrete citations.
- **Development Bypass**: Added a Lakera Guard safety bypass (`RAG_DEV_MODE=1` + `SKIP_SAFETY_CHECK=1`) exclusively for local development testing, preserving strict fail-closed behavior for production.
- **Dead Code Removal**: Cleaned up outdated `parse_sections()` and fragile docstore reconstruction logic.
- **Query Logs**: Query logs (`rag_query_log.jsonl`) are strictly `.gitignore`d to prevent PII or usage data from leaking into version control.

## 4. UI & Workflow

- The Gradio UI (`src/app.py`) has been streamlined to rely entirely on `secure_generate_answer()` from `src/pipeline.py`.
- Legacy reranker toggles and generation fallbacks have been removed. 
- All metadata (including chunk tracking, page associations, and references) is persisted to `data/processed/` and verified on load to maintain index integrity.