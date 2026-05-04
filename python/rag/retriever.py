"""Retriever for simple embedding + cosine similarity search."""

from __future__ import annotations

from typing import Dict, List

from rag.embeddings import get_embedding

# In-memory embedding cache: chunk text -> embedding vector.
_EMBEDDING_CACHE: Dict[str, List[float]] = {}


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""

    if not vec_a or not vec_b:
        return 0.0

    size = min(len(vec_a), len(vec_b))
    if size == 0:
        return 0.0

    dot = sum(vec_a[i] * vec_b[i] for i in range(size))
    norm_a = sum(vec_a[i] * vec_a[i] for i in range(size)) ** 0.5
    norm_b = sum(vec_b[i] * vec_b[i] for i in range(size)) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def _chunk_embedding(chunk: str) -> List[float]:
    """Get cached embedding for a chunk, computing if needed."""

    if chunk not in _EMBEDDING_CACHE:
        _EMBEDDING_CACHE[chunk] = get_embedding(chunk)
    return _EMBEDDING_CACHE[chunk]


def retrieve_top_chunks(query: str, chunks: List[str], top_k: int = 3) -> List[str]:
    """Retrieve top-k similar chunks for a query."""

    if not chunks:
        return []

    query_embedding = get_embedding(query)
    scored = []

    for chunk in chunks:
        similarity = _cosine_similarity(query_embedding, _chunk_embedding(chunk))
        scored.append((similarity, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]
