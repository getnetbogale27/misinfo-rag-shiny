"""End-to-end RAG pipeline using FAISS retrieval."""

from __future__ import annotations

from typing import Dict

from rag.generator import generate_answer
from rag.retriever import retrieve_top_chunks


AMHARIC_BLOCK_START = 0x1200
AMHARIC_BLOCK_END = 0x137F


def detect_language(text: str) -> str:
    """Detect claim language as English (en) or Amharic (am).

    Uses a lightweight Unicode-range heuristic for Ethiopic script.
    """

    if not text:
        return "en"

    for ch in text:
        code = ord(ch)
        if AMHARIC_BLOCK_START <= code <= AMHARIC_BLOCK_END:
            return "am"
    return "en"


def _confidence_from_verdict(verdict: str) -> float:
    normalized = verdict.strip().lower()
    if normalized in {"true", "false"}:
        return 0.75
    return 0.5


def run_pipeline(claim: str) -> Dict[str, object]:
    language = detect_language(claim)
    top_chunks = retrieve_top_chunks(claim, top_k=5)
    generated = generate_answer(claim, top_chunks, language=language)

    verdict = generated.get("verdict", "Uncertain")
    explanation = generated.get("explanation", "No explanation available.")

    confidence = generated.get("confidence")
    if confidence is None:
        confidence = _confidence_from_verdict(str(verdict))

    return {
        "verdict": verdict,
        "confidence": float(confidence),
        "explanation": explanation,
        "evidence": top_chunks,
        "language": language,
    }
