import math
import re
from typing import List, Dict, Any, Tuple
from collections import Counter
from src.chunking.splitters import TextChunk


STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "did", "do", "does", "doing", "don", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she",
    "should", "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself",
}


def tokenize(text: str) -> List[str]:
    words = re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())
    return [w for w in words if w not in STOP_WORDS]


class BM25Retriever:
    """Okapi BM25 retriever for exact keyword matching, policy section numbers, and code lookups."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[TextChunk] = []
        self.corpus_size: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_lens: List[int] = []
        self.doc_term_freqs: List[Counter] = []
        self.idf: Dict[str, float] = {}

    def fit(self, chunks: List[TextChunk]):
        self.chunks = chunks
        self.corpus_size = len(chunks)
        if self.corpus_size == 0:
            return

        tokenized_corpus = [tokenize(c.text + " " + c.section_heading) for c in chunks]
        self.doc_lens = [len(doc) for doc in tokenized_corpus]
        self.avg_doc_len = sum(self.doc_lens) / max(1, self.corpus_size)
        self.doc_term_freqs = [Counter(doc) for doc in tokenized_corpus]

        # Calculate document frequencies
        df: Dict[str, int] = {}
        for doc in tokenized_corpus:
            unique_terms = set(doc)
            for term in unique_terms:
                df[term] = df.get(term, 0) + 1

        # Calculate IDF
        self.idf = {}
        for term, freq in df.items():
            self.idf[term] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def search(self, query: str, top_k: int = 10) -> List[Tuple[TextChunk, float]]:
        if self.corpus_size == 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return [(self.chunks[i], 0.0) for i in range(min(top_k, len(self.chunks)))]

        scores: List[float] = [0.0] * self.corpus_size

        for token in query_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]

            for i in range(self.corpus_size):
                tf = self.doc_term_freqs[i].get(token, 0)
                if tf > 0:
                    doc_len = self.doc_lens[i]
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1.0, self.avg_doc_len)))
                    scores[i] += idf_val * (numerator / denominator)

        # Normalize scores to 0.0 - 1.0 range
        max_score = max(scores) if scores and max(scores) > 0 else 1.0
        normalized_scores = [s / max_score for s in scores]

        # Sort descending
        ranked_indices = sorted(range(self.corpus_size), key=lambda idx: scores[idx], reverse=True)
        results: List[Tuple[TextChunk, float]] = []
        for idx in ranked_indices[:top_k]:
            results.append((self.chunks[idx], float(normalized_scores[idx])))

        return results

