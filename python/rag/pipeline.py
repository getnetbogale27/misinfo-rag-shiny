"""End-to-end RAG pipeline using FAISS retrieval."""

from __future__ import annotations

from typing import Dict

from rag.generator import generate_answer
from rag.retriever import retrieve_top_chunks


def _confidence_from_verdict(verdict: str) -> float:
    normalized = verdict.strip().lower()
    if normalized in {"true", "false"}:
        return 0.75
    return 0.5


def run_pipeline(claim: str) -> Dict[str, object]:
    top_chunks = retrieve_top_chunks(claim, top_k=3)
    generated = generate_answer(claim, top_chunks)

    verdict = generated.get("verdict", "Uncertain")
    explanation = generated.get("explanation", "No explanation available.")

    return {
        "verdict": verdict,
        "confidence": _confidence_from_verdict(str(verdict)),
        "explanation": explanation,
        "evidence": top_chunks,
    }
