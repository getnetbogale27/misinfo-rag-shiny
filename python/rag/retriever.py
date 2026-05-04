"""FAISS-backed retriever for multilingual semantic search."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback
    np = None

try:
    import faiss
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback
    faiss = None

from rag.embeddings import get_multilingual_embedding

_INDEX_DIR = Path(__file__).resolve().parents[1] / "vectorstore" / "faiss_index"
_INDEX_PATH = _INDEX_DIR / "index.faiss"
_DOCSTORE_PATH = _INDEX_DIR / "docstore.json"


def _missing_index_payload() -> dict[str, str]:
    return {
        "status": "no_index",
        "message": "FAISS index not found. Run build_index.py first.",
    }


@lru_cache(maxsize=1)
def _load_index_and_docstore() -> Tuple[Any, Dict[str, str]]:
    if faiss is None:
        raise ModuleNotFoundError("faiss is not installed")
    if not _INDEX_PATH.exists() or not _DOCSTORE_PATH.exists():
        raise FileNotFoundError(_missing_index_payload()["message"])

    index = faiss.read_index(str(_INDEX_PATH))
    docstore = json.loads(_DOCSTORE_PATH.read_text(encoding="utf-8"))
    return index, docstore


def get_retrieval_status() -> dict[str, str]:
    """Return retrieval subsystem health in a structured format."""

    if faiss is None or np is None:
        return {
            "status": "unavailable",
            "message": "FAISS or numpy dependency missing. Using reasoning-only fallback.",
        }

    if not _INDEX_PATH.exists() or not _DOCSTORE_PATH.exists():
        return _missing_index_payload()

    return {"status": "ready", "message": "FAISS retrieval is available."}


def _embed_query(query: str) -> np.ndarray:
    query_vector = np.array([get_multilingual_embedding(query)], dtype="float32")
    faiss.normalize_L2(query_vector)
    return query_vector


def retrieve_top_chunks(query: str, top_k: int = 5) -> List[str]:
    if not query.strip():
        return []

    status = get_retrieval_status()
    if status["status"] != "ready":
        return [status["message"]]

    try:
        index, docstore = _load_index_and_docstore()
        query_vector = _embed_query(query)

        _, indices = index.search(query_vector, top_k)

        chunks: List[str] = []
        for idx in indices[0].tolist():
            if idx < 0:
                continue
            chunk = docstore.get(str(idx))
            if chunk:
                chunks.append(chunk)
        return chunks
    except Exception:
        return ["Retrieval unavailable: FAISS index or dependency is missing."]
