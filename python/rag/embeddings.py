"""Embedding utilities for simple local RAG."""

from __future__ import annotations

import hashlib
import os
from typing import List


_FALLBACK_DIM = 256
_OPENAI_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def _fallback_embedding(text: str, dim: int = _FALLBACK_DIM) -> List[float]:
    """Deterministic embedding fallback using hash buckets.

    This keeps the RAG pipeline working even when no external embedding
    service/model is configured.
    """

    vector = [0.0] * dim
    tokens = text.lower().split()
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    norm = sum(x * x for x in vector) ** 0.5
    if norm == 0:
        return vector

    return [x / norm for x in vector]


def get_embedding(text: str) -> List[float]:
    """Return an embedding vector for text.

    Preferred order:
    1) OpenAI embeddings API (if OPENAI_API_KEY is available and openai package exists)
    2) Deterministic local fallback embedding
    """

    if not text:
        return [0.0] * _FALLBACK_DIM

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.embeddings.create(model=_OPENAI_MODEL, input=text)
            return response.data[0].embedding
        except Exception:
            # Fail open to local deterministic embeddings.
            pass

    return _fallback_embedding(text)
