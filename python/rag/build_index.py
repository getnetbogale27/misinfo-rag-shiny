"""Build FAISS index and metadata for RAG retrieval.

Run:
    python rag/build_index.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

try:
    import faiss
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("faiss is required. Install faiss-cpu first.") from exc

import numpy as np

from rag.embeddings import get_multilingual_embedding

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
KB_DIR = DATA_DIR / "knowledge_base"
INDEX_DIR = Path(__file__).resolve().parents[1] / "vectorstore" / "faiss_index"
INDEX_PATH = INDEX_DIR / "index.faiss"
DOCSTORE_PATH = INDEX_DIR / "docstore.json"
META_PATH = INDEX_DIR / "metadata.json"



def _load_documents(directory: Path) -> list[str]:
    docs: list[str] = []
    if not directory.exists():
        return docs

    for path in sorted(directory.iterdir()):
        if path.is_dir() or path.suffix.lower() not in {".txt", ".md", ".json", ".csv"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            docs.append(text)
    return docs


def _chunk_text(text: str, chunk_size: int = 300, overlap: int = 40) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_size]
        if not chunk:
            continue
        chunks.append(" ".join(chunk))
        if start + chunk_size >= len(words):
            break
    return chunks


def _embed_chunks(chunks: Iterable[str]) -> np.ndarray:
    vectors = [get_multilingual_embedding(chunk) for chunk in chunks]
    if not vectors:
        return np.empty((0, 0), dtype="float32")

    matrix = np.array(vectors, dtype="float32")
    faiss.normalize_L2(matrix)
    return matrix


def build_index() -> dict[str, object]:
    documents = _load_documents(KB_DIR)
    chunks: List[str] = []
    for doc in documents:
        chunks.extend(_chunk_text(doc))

    embeddings = _embed_chunks(chunks)
    if embeddings.size == 0:
        raise RuntimeError("No embeddings generated. Ensure knowledge-base files exist in data/knowledge_base.")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    docstore = {str(i): chunk for i, chunk in enumerate(chunks)}
    DOCSTORE_PATH.write_text(json.dumps(docstore, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = {
        "documents": len(documents),
        "chunks": len(chunks),
        "dimension": int(dim),
        "index_path": str(INDEX_PATH),
        "docstore_path": str(DOCSTORE_PATH),
    }
    META_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    info = build_index()
    print(json.dumps({"status": "success", "metadata": info}, indent=2))
