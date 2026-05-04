"""Build and persist FAISS index for knowledge base chunks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import faiss
import numpy as np

from rag.embeddings import get_embedding

KB_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge_base"
INDEX_DIR = Path(__file__).resolve().parent / "faiss_index"
INDEX_PATH = INDEX_DIR / "index.faiss"
DOCSTORE_PATH = INDEX_DIR / "docstore.json"

CHUNK_SIZE_WORDS = 300
CHUNK_OVERLAP_WORDS = 40


def _load_rds_text(path: Path) -> List[str]:
    try:
        import pyreadr  # type: ignore

        result = pyreadr.read_r(str(path))
    except Exception:
        return []

    docs: List[str] = []
    for _, frame in result.items():
        for col in frame.columns:
            series = frame[col].dropna().astype(str)
            docs.extend([value.strip() for value in series if value.strip()])
    return docs


def _load_documents(directory: Path) -> List[str]:
    docs: List[str] = []
    if not directory.exists():
        return docs

    for path in sorted(directory.iterdir()):
        if path.is_dir():
            continue

        suffix = path.suffix.lower()
        if suffix == ".rds":
            docs.extend(_load_rds_text(path))
            continue

        if suffix not in {".txt", ".md", ".csv", ".json"}:
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            continue

        if text:
            docs.append(text)

    return docs


def _chunk_text(text: str, size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> List[str]:
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


def _embed_chunks(chunks: Iterable[str]) -> np.ndarray:
    vectors = [get_embedding(chunk) for chunk in chunks]
    if not vectors:
        return np.empty((0, 0), dtype="float32")

    matrix = np.array(vectors, dtype="float32")
    faiss.normalize_L2(matrix)
    return matrix


def build_index() -> None:
    docs = _load_documents(KB_DIR)
    chunks: List[str] = []
    for doc in docs:
        chunks.extend(_chunk_text(doc))

    embeddings = _embed_chunks(chunks)
    if embeddings.size == 0:
        raise RuntimeError("No chunks/embeddings generated from knowledge base.")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    docstore = {str(i): chunk for i, chunk in enumerate(chunks)}
    DOCSTORE_PATH.write_text(json.dumps(docstore, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    build_index()
