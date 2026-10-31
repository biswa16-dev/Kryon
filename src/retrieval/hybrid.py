from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel
from src.chunking.splitters import TextChunk
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.vector_store import FAISSVectorStore


class ScoredChunk(BaseModel):
    chunk: TextChunk
    score: float
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    rank: int = 1


class HybridRetriever:
    """Combines BM25 keyword search with FAISS dense vector retrieval via Reciprocal Rank Fusion."""

    def __init__(self, vector_store: FAISSVectorStore, bm25_retriever: BM25Retriever):
        self.vector_store = vector_store
        self.bm25 = bm25_retriever

    def search(
        self,
        query: str,
        top_k: int = 5,
        search_mode: str = "hybrid",
        alpha: float = 0.6,  # 0.6 semantic, 0.4 BM25
        doc_filter: Optional[str] = None,
    ) -> List[ScoredChunk]:
        """
        Executes search under requested mode:
        - 'semantic': Dense vector search only
        - 'bm25': Keyword BM25 search only
        - 'hybrid': Reciprocal Rank Fusion (RRF) + weighted combination
        """
        if search_mode == "semantic":
            raw_results = self.vector_store.search(query, top_k=top_k * 2)
            filtered = [
                ScoredChunk(chunk=c, score=s, semantic_score=s, bm25_score=0.0, rank=idx + 1)
                for idx, (c, s) in enumerate(raw_results)
                if not doc_filter or doc_filter.lower() in c.filename.lower()
            ]
            return filtered[:top_k]

        elif search_mode == "bm25":
            raw_results = self.bm25.search(query, top_k=top_k * 2)
            filtered = [
                ScoredChunk(chunk=c, score=s, semantic_score=0.0, bm25_score=s, rank=idx + 1)
                for idx, (c, s) in enumerate(raw_results)
                if not doc_filter or doc_filter.lower() in c.filename.lower()
            ]
            return filtered[:top_k]

        # Hybrid search with Reciprocal Rank Fusion (RRF, k=60)
        dense_results = self.vector_store.search(query, top_k=20)
        bm25_results = self.bm25.search(query, top_k=20)

        chunk_map: Dict[str, TextChunk] = {}
        dense_ranks: Dict[str, int] = {}
        bm25_ranks: Dict[str, int] = {}
        dense_scores: Dict[str, float] = {}
        bm25_scores: Dict[str, float] = {}

        for rank, (chunk, score) in enumerate(dense_results):
            chunk_map[chunk.chunk_id] = chunk
            dense_ranks[chunk.chunk_id] = rank + 1
            dense_scores[chunk.chunk_id] = score

        for rank, (chunk, score) in enumerate(bm25_results):
            chunk_map[chunk.chunk_id] = chunk
            bm25_ranks[chunk.chunk_id] = rank + 1
            bm25_scores[chunk.chunk_id] = score

        rrf_scores: Dict[str, float] = {}
        k_rrf = 60.0

        for chunk_id in chunk_map.keys():
            r_dense = dense_ranks.get(chunk_id, 100)
            r_bm25 = bm25_ranks.get(chunk_id, 100)
            s_dense = dense_scores.get(chunk_id, 0.0)
            s_bm25 = bm25_scores.get(chunk_id, 0.0)

            # RRF formula with alpha weighting
            score_rrf = (alpha / (k_rrf + r_dense)) + ((1.0 - alpha) / (k_rrf + r_bm25))
            # Blended with normalized confidence
            blended = (score_rrf * 30.0) * 0.4 + (alpha * s_dense + (1.0 - alpha) * s_bm25) * 0.6
            rrf_scores[chunk_id] = blended

        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        scored_chunks: List[ScoredChunk] = []
        rank_counter = 1
        for cid in sorted_ids:
            chunk = chunk_map[cid]
            if doc_filter and doc_filter.lower() not in chunk.filename.lower():
                continue
            scored_chunks.append(
                ScoredChunk(
                    chunk=chunk,
                    score=min(1.0, float(rrf_scores[cid])),
                    semantic_score=float(dense_scores.get(cid, 0.0)),
                    bm25_score=float(bm25_scores.get(cid, 0.0)),
                    rank=rank_counter,
                )
            )
            rank_counter += 1
            if len(scored_chunks) >= top_k:
                break

        return scored_chunks

