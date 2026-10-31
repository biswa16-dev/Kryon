import re
from typing import List
from src.retrieval.hybrid import ScoredChunk


class CrossEncoderReRanker:
    """Re-ranks retrieved chunks based on deep query-document semantic alignment and numerical clause match."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name

    def rerank(self, query: str, candidates: List[ScoredChunk], top_k: int = 5) -> List[ScoredChunk]:
        if not candidates:
            return []

        query_terms = set(re.findall(r"\b[a-z0-9_\-\.]{2,}\b", query.lower()))
        # Extract numerical tokens (critical for regulation rules like 75%, 65%, 160 credits, etc.)
        numbers_in_query = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", query.lower()))

        reranked: List[ScoredChunk] = []

        for item in candidates:
            chunk_text = item.chunk.text.lower()
            heading = item.chunk.section_heading.lower()
            combined_text = f"{heading} {chunk_text}"

            chunk_terms = set(re.findall(r"\b[a-z0-9_\-\.]{2,}\b", combined_text))
            chunk_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", combined_text))

            # Query-document intersection
            overlap = len(query_terms.intersection(chunk_terms))
            overlap_ratio = overlap / max(1, len(query_terms))

            # Number/regulation threshold exact match boost
            num_match_boost = 0.0
            if numbers_in_query:
                num_overlap = len(numbers_in_query.intersection(chunk_numbers))
                num_match_boost = (num_overlap / len(numbers_in_query)) * 0.35

            # Heading relevance boost
            heading_boost = 0.0
            if any(term in heading for term in query_terms):
                heading_boost = 0.20

            # Calibrate combined cross-encoder confidence
            new_score = (
                item.score * 0.40
                + overlap_ratio * 0.35
                + num_match_boost
                + heading_boost
            )
            calibrated_score = max(0.0, min(1.0, new_score))

            reranked.append(
                ScoredChunk(
                    chunk=item.chunk,
                    score=float(calibrated_score),
                    semantic_score=item.semantic_score,
                    bm25_score=item.bm25_score,
                    rank=1,
                )
            )

        # Sort descending
        reranked.sort(key=lambda x: x.score, reverse=True)

        # Re-assign ranks
        for idx, item in enumerate(reranked[:top_k]):
            item.rank = idx + 1

        return reranked[:top_k]

