# Fibromyalgia Guarded Medical RAG Pipeline

A highly secure, cloud-only Retrieval-Augmented Generation (RAG) pipeline built over the biomedical review article: *"Fibromyalgia: A Review of the Pathophysiological Mechanisms and Multidisciplinary Treatment Strategies"*.

## Strict Cloud Model Architecture

This project strictly implements a verified model stack using exclusively Cloud APIs. No local inference, fallback generation models, or open-ended configuration are permitted in this architecture.

- **Embeddings**: `nvidia/nemotron-3-embed-1b:free` via OpenRouter Cloud
- **Retrieval Engine**: Hybrid FAISS Dense Search + BM25 Lexical Search (with deduplication)
- **Reranking**: `nvidia/llama-nemotron-rerank-vl-1b-v2:free` via OpenRouter Cloud
- **Generation**: `openai/gpt-oss-20b` via Groq Cloud
- **Grounding Validation (LLM-as-a-judge)**: `openai/gpt-oss-20b` via Groq Cloud
- **Content Safety**: Lakera Guard Cloud API (Strict Fail-Closed)

## Architecture Features

- **Strict Validation & Safety**: Input validation blocks prompt injection and personal data. Lakera Guard API is enforced; if Lakera fails or times out, the system safely abstains rather than failing open.
- **Robust Grounding**: Generated claims are verified sentence-by-sentence using an LLM-as-a-judge against the retrieved excerpts. A failure of the grounding service correctly results in an "uncertain" state (fail-closed).
- **Intelligent Caching**: Semantic cache (stable chunk-id hashed) only stores fully grounded, safe, and successful answers.
- **Index Protection**: FAISS index stores metadata detailing the model and provider. Mismatches automatically rebuild the index.

## Privacy & Data Flow
**WARNING: DO NOT SUBMIT SENSITIVE PATIENT DATA (PHI/PII).**
All queries and retrieved context are sent to external third-party cloud providers (Groq, OpenRouter, Lakera). This system is designed for public medical literature RAG and does NOT have a local-inference privacy boundary. Always sanitize inputs before querying the application.

## Project Structure

```text
project/
├── src/
│   ├── pipeline.py
│   └── app.py
├── data/
├── docs/
├── notebooks/
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Setup & Installation

1. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy the `.env.example` to `.env` and fill in your API keys (see below).

## Environment Variables

Set these in your `.env` file or export them directly in your environment:

```bash
# Generation & Grounding Provider (Groq)
GROQ_API_KEY="gsk_..."

# Embeddings & Reranking Provider (OpenRouter)
OPENROUTER_API_KEY="sk-or-..."

# Safety & Guardrails (Lakera Guard)
LAKERA_GUARD_API_KEY="lakera-..."

# Local Development Bypass (DO NOT USE IN PRODUCTION)
# RAG_DEV_MODE=1
# SKIP_SAFETY_CHECK=1
```
*Note: The pipeline relies heavily on these keys. If any are missing, the pipeline will properly fail closed.*

### Development Bypass
For local testing or development in environments where Lakera Guard is unavailable, you can bypass the content safety check by setting **both** `RAG_DEV_MODE=1` and `SKIP_SAFETY_CHECK=1`. This will explicitly flag safety as bypassed and print loud warnings to `sys.stderr`. **This bypass must never be enabled in a production environment.**

## Running the Application

Launch the Gradio Web UI:

```bash
python src/app.py
```

The first launch will parse the PDF, run the text chunking, and build the FAISS and BM25 indices (saved to `data/processed/faiss_index`). Subsequent launches will load the cached indices instantly.

Open the displayed local URL (typically `http://127.0.0.1:7860`) in your browser to interact with the system.

## Evaluation

The `pipeline.py` script includes functions like `evaluate_pipeline()`, `evaluate_answers()`, and `run_guardrail_red_team()` to test retrieval accuracy, grounding validation, and adversarial red-team safety blocks. Technical failures (like timeouts) are strictly distinguished from actual answer quality.