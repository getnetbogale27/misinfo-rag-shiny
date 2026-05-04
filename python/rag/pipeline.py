"""End-to-end simple RAG pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from rag.generator import generate_answer
from rag.retriever import retrieve_top_chunks

_KB_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge_base"
_CHUNK_SIZE_WORDS = 300
_CHUNK_OVERLAP_WORDS = 40

_LOADED_CHUNKS: List[str] | None = None


def _load_documents(directory: Path) -> List[str]:
    docs: List[str] = []
    if not directory.exists():
        return docs

    for path in sorted(directory.iterdir()):
        if path.is_dir():
            continue
        if path.suffix.lower() not in {".txt", ".md", ".csv", ".json", ".rds"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                docs.append(text)
        except Exception:
            continue
    return docs


def _chunk_text(text: str, size: int = _CHUNK_SIZE_WORDS, overlap: int = _CHUNK_OVERLAP_WORDS) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        piece = words[start : start + size]
        if not piece:
            continue
        chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


def _load_chunks_once() -> List[str]:
    global _LOADED_CHUNKS
    if _LOADED_CHUNKS is not None:
        return _LOADED_CHUNKS

    documents = _load_documents(_KB_DIR)
    all_chunks: List[str] = []
    for doc in documents:
        all_chunks.extend(_chunk_text(doc))

    _LOADED_CHUNKS = all_chunks
    return _LOADED_CHUNKS


def _confidence_from_verdict(verdict: str) -> float:
    normalized = verdict.strip().lower()
    if normalized == "true" or normalized == "false":
        return 0.75
    return 0.5


def run_pipeline(claim: str) -> Dict[str, object]:
    chunks = _load_chunks_once()
    top_chunks = retrieve_top_chunks(claim, chunks, top_k=3)
    generated = generate_answer(claim, top_chunks)

    verdict = generated.get("verdict", "Uncertain")
    explanation = generated.get("explanation", "No explanation available.")

    return {
        "verdict": verdict,
        "confidence": _confidence_from_verdict(str(verdict)),
        "explanation": explanation,
        "evidence": top_chunks,
    }
