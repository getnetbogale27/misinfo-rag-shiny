"""FAISS-backed retriever for semantic search."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np

from rag.embeddings import get_embedding

_INDEX_DIR = Path(__file__).resolve().parents[1] / "vectorstore" / "faiss_index"
_INDEX_PATH = _INDEX_DIR / "index.faiss"
_DOCSTORE_PATH = _INDEX_DIR / "docstore.json"


@lru_cache(maxsize=1)
def _load_index_and_docstore() -> Tuple[faiss.Index, Dict[str, str]]:
    if not _INDEX_PATH.exists() or not _DOCSTORE_PATH.exists():
        raise FileNotFoundError("FAISS index or docstore missing. Run vectorstore/build_index.py first.")

    index = faiss.read_index(str(_INDEX_PATH))
    docstore = json.loads(_DOCSTORE_PATH.read_text(encoding="utf-8"))
    return index, docstore


def _embed_query(query: str) -> np.ndarray:
    query_vector = np.array([get_embedding(query)], dtype="float32")
    faiss.normalize_L2(query_vector)
    return query_vector


def retrieve_top_chunks(query: str, top_k: int = 3) -> List[str]:
    if not query.strip():
        return []

    index, docstore = _load_index_and_docstore()
    query_vector = _embed_query(query)

    distances, indices = index.search(query_vector, top_k)
    _ = distances

    chunks: List[str] = []
    for idx in indices[0].tolist():
        if idx < 0:
            continue
        chunk = docstore.get(str(idx))
        if chunk:
            chunks.append(chunk)

    return chunks
