import faiss
import numpy as np
from typing import List, Tuple
from src.chunking.splitters import TextChunk
from src.embeddings.embedder import DenseEmbedder


class FAISSVectorStore:
    """High-performance vector index using FAISS with cosine similarity."""

    def __init__(self, embedder: DenseEmbedder):
        self.embedder = embedder
        self.dim = embedder.dim
        # IndexFlatIP gives exact cosine similarity on L2-normalized vectors
        self.index = faiss.IndexFlatIP(self.dim)
        self.chunks: List[TextChunk] = []

    def add_chunks(self, chunks: List[TextChunk]):
        if not chunks:
            return
        self.chunks.extend(chunks)
        texts = [f"{c.section_heading}\n{c.text}" for c in chunks]
        embeddings = self.embedder.embed_texts(texts)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[TextChunk, float]]:
        if self.index.ntotal == 0:
            return []

        query_vec = self.embedder.embed_query(query)
        scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))

        results: List[Tuple[TextChunk, float]] = []
        for rank in range(indices.shape[1]):
            idx = indices[0][rank]
            if idx >= 0 and idx < len(self.chunks):
                # Cosine similarity in range [-1, 1], map to [0, 1]
                raw_score = float(scores[0][rank])
                normalized_score = max(0.0, min(1.0, (raw_score + 1.0) / 2.0))
                results.append((self.chunks[idx], normalized_score))

        return results

    def clear(self):
        self.index.reset()
        self.chunks.clear()

