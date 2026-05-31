from __future__ import annotations

import math
import os
import re
from collections import Counter

import numpy as np

from .models import PatientRecord

try:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import faiss
except Exception:  # pragma: no cover - fallback keeps the app runnable without faiss.
    faiss = None


class SimpleVectorStore:
    """FAISS-backed local retrieval with a deterministic hashing embedding fallback."""

    def __init__(self, records: list[PatientRecord], min_relevance: float | None = None):
        self.records = records
        self.documents = [record.searchable_text() for record in records]
        self.vectors = [self._vectorize(document) for document in self.documents]
        self.embedding_dim = 384
        self.min_relevance = min_relevance if min_relevance is not None else float(os.getenv("VECTOR_MIN_RELEVANCE", "0.08"))
        self.embeddings = self._build_embeddings(self.documents)
        self.index = self._build_faiss_index(self.embeddings)

    def _vectorize(self, text: str) -> Counter:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        stop_words = {"the", "and", "or", "a", "an", "to", "of", "for", "in", "with", "is"}
        return Counter(token for token in tokens if token not in stop_words)

    def _cosine(self, left: Counter, right: Counter) -> float:
        common = set(left) & set(right)
        numerator = sum(left[token] * right[token] for token in common)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if not left_norm or not right_norm:
            return 0.0
        return numerator / (left_norm * right_norm)

    def _embed_text(self, text: str) -> np.ndarray:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        vector = np.zeros(self.embedding_dim, dtype="float32")
        for token in tokens:
            vector[hash(token) % self.embedding_dim] += 1.0
        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm
        return vector

    def _build_embeddings(self, documents: list[str]) -> np.ndarray:
        if not documents:
            return np.zeros((0, self.embedding_dim), dtype="float32")
        return np.vstack([self._embed_text(document) for document in documents]).astype("float32")

    def _build_faiss_index(self, embeddings: np.ndarray):
        if faiss is None or embeddings.size == 0:
            return None
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(embeddings)
        return index

    def search(self, query: str, limit: int = 4) -> list[tuple[PatientRecord, float]]:
        if not self.records:
            return []
        if self.index is not None:
            query_embedding = self._embed_text(query).reshape(1, -1).astype("float32")
            scores, indexes = self.index.search(query_embedding, min(limit, len(self.records)))
            return [
                (self.records[int(index)], float(score))
                for score, index in zip(scores[0], indexes[0])
                if index >= 0 and score >= self.min_relevance
            ]
        query_vector = self._vectorize(query)
        ranked = sorted(
            enumerate(self.vectors),
            key=lambda item: self._cosine(query_vector, item[1]),
            reverse=True,
        )
        ranked = [(index, self._cosine(query_vector, vector)) for index, vector in ranked]
        return [(self.records[index], float(score)) for index, score in ranked[:limit] if score >= self.min_relevance]
