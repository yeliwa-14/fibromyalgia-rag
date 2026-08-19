"""Gradio interface for the Fibromyalgia Guarded RAG pipeline.

Run with:
    python src/app.py
or
    python app.py

Features:
- Hybrid Search (FAISS + BM25) with Cross-Encoder Reranking
- Multi-layer Guardrail checks (Input validation, Lakera safety, Grounding, Dosage leak)
- Comprehensive source metrics (Dense, BM25, and Rerank scores)
- Active LLM Provider & Embedding Backend status display
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Ensure src directory is in sys.path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import gradio as gr
from pipeline import (
    EMBEDDING_MODEL,
    get_retriever,
    load_references,
    secure_generate_answer,
)

retriever = None
references = None

def _get_or_init_retriever():
    global retriever, references
    if retriever is None:
        try:
            print("Initializing Fibromyalgia RAG retrieval index & documents...")
            retriever = get_retriever(k=3)
            references = load_references()
            print("Ready.")
        except Exception as e:
            return None, str(e)
    return retriever, None


def answer_question(question: str):
    if not question or not question.strip():
        return "من فضلك اكتب سؤال أولاً.", "", "{}"

    ret, err = _get_or_init_retriever()
    if err or ret is None:
        return f"System configuration error: {err}. Please check your API keys.", "", "{}"

    result = secure_generate_answer(
        question,
        vectorstore=ret.vectorstore,
        top_n=5,
    )

    answer = result.get("answer") or "Could not generate an answer."

    # Format sources table & markdown
    sources = result.get("sources", [])
    if sources:
        formatted_sources = []
        for i, s in enumerate(sources, start=1):
            pages = ", ".join(str(p) for p in (s.get("pages") or []))
            r_score = (
                f"{s['rerank_score']:.4f}"
                if s.get("rerank_score") is not None
                else "N/A"
            )
            d_score = (
                f"{s['dense_score']:.4f}"
                if s.get("dense_score") is not None
                else "N/A"
            )
            b_score = (
                f"{s['bm25_score']:.4f}"
                if s.get("bm25_score") is not None
                else "N/A"
            )
            formatted_sources.append(
                f"### [{i}] Section: {s.get('section')} (p. {pages})\n"
                f"- **Chunk ID**: `{s.get('chunk_id')}` | **File**: `{s.get('source')}`\n"
                f"- **Scores**: Rerank: `{r_score}` | Dense (L2): `{d_score}` | BM25: `{b_score}`"
            )
        sources_md = "\n\n".join(formatted_sources)
    else:
        sources_md = "_No source passages retrieved for this query._"

    guardrails_json = json.dumps(
        result.get("guardrails", {}), indent=2, default=str
    )

    return answer, sources_md, guardrails_json


EXAMPLE_QUESTIONS = [
    "What are the FDA-approved drugs for fibromyalgia?",
    "What is fibromyalgia characterized by?",
    "What non-pharmacological treatments are discussed for fibromyalgia?",
    "What diagnostic criteria are used for fibromyalgia?",
    "What is the proposed underlying mechanism of fibromyalgia pain?",
]

with gr.Blocks(title="Fibromyalgia Guarded RAG") as demo:
    has_llm_key = bool(os.environ.get("GROQ_API_KEY"))
    gen_status = (
        "Groq Cloud API" if has_llm_key else "Generation Unavailable (No API Key)"
    )

    gr.Markdown(
        "# 🩺 Fibromyalgia Research Article Q&A (Guarded RAG)\n"
        "Ask questions grounded strictly in the review article: "
        "*'Fibromyalgia: A Review of the Pathophysiological Mechanisms and Multidisciplinary Treatment Strategies'*\n\n"
        f"**Generation mode:** {gen_status} | "
        f"**Embedding model:** {EMBEDDING_MODEL}"
    )

    with gr.Row():
        with gr.Column(scale=3):
            question_box = gr.Textbox(
                label="Your Question",
                placeholder="e.g. What are the FDA-approved drugs for fibromyalgia?",
                lines=2,
            )
            ask_btn = gr.Button("Ask Question", variant="primary")

    with gr.Row():
        with gr.Column(scale=2):
            answer_box = gr.Textbox(
                label="Generated Grounded Answer", lines=8, interactive=False
            )
        with gr.Column(scale=2):
            sources_box = gr.Markdown(label="Retrieved Evidence & Scores")

    with gr.Accordion("Guardrail Safety & Provenance Report", open=False):
        guardrails_box = gr.Code(label="Guardrails Decision Log (JSON)", language="json")

    gr.Examples(examples=EXAMPLE_QUESTIONS, inputs=question_box)

    ask_btn.click(
        fn=answer_question,
        inputs=[question_box],
        outputs=[answer_box, sources_box, guardrails_box],
    )
    question_box.submit(
        fn=answer_question,
        inputs=[question_box],
        outputs=[answer_box, sources_box, guardrails_box],
    )


if __name__ == "__main__":
    demo.launch()