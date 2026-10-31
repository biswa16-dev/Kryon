import json
import time
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel
from src.ingestion.parser import DocumentParser, ParsedDocument
from src.chunking.splitters import ChunkingEngine, TextChunk
from src.embeddings.embedder import DenseEmbedder
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.hybrid import HybridRetriever
from src.reranking.reranker import CrossEncoderReRanker
from src.generation.grounded_generator import GroundedRAGGenerator


class ExperimentMetrics(BaseModel):
    experiment_id: str
    chunk_size: int
    chunk_overlap: int
    retrieval_mode: str
    use_reranker: bool
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    recall_at_5: float
    mrr: float
    citation_accuracy: float
    refusal_accuracy: float
    faithfulness_score: float
    avg_latency_ms: float
    total_evaluated: int


class RAGEvaluator:
    """Systematic evaluation harness to test chunk sizes, overlaps, and retrieval paradigms."""

    def __init__(self, raw_data_dir: str = "data/raw", questions_file: str = "data/evaluation/questions.json"):
        self.raw_data_dir = Path(raw_data_dir)
        self.questions_file = Path(questions_file)
        self.parsed_docs: List[ParsedDocument] = []
        self._load_documents()

    def _load_documents(self):
        self.parsed_docs = []
        for file_path in self.raw_data_dir.glob("*.md"):
            doc = DocumentParser.parse_file(file_path)
            self.parsed_docs.append(doc)

    def load_questions(self) -> List[Dict[str, Any]]:
        with open(self.questions_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_experiment(
        self,
        experiment_id: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        retrieval_mode: str = "hybrid",
        use_reranker: bool = True,
        confidence_threshold: float = 0.50,
    ) -> ExperimentMetrics:
        # 1. Chunk documents according to experiment parameters
        all_chunks: List[TextChunk] = []
        for doc in self.parsed_docs:
            chunks = ChunkingEngine.chunk_document(
                doc,
                strategy="recursive",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            all_chunks.extend(chunks)

        # 2. Build Index
        embedder = DenseEmbedder()
        vector_store = FAISSVectorStore(embedder)
        vector_store.add_chunks(all_chunks)

        bm25 = BM25Retriever()
        bm25.fit(all_chunks)

        retriever = HybridRetriever(vector_store, bm25)
        reranker = CrossEncoderReRanker() if use_reranker else None
        generator = GroundedRAGGenerator(confidence_threshold=confidence_threshold)

        # 3. Evaluate Questions
        questions = self.load_questions()
        p1_list, p3_list, p5_list, rec5_list, rr_list = [], [], [], [], []
        citation_matches = 0
        refusal_correct = 0
        faithfulness_scores = []
        latencies = []
        out_of_scope_count = 0

        for q in questions:
            t0 = time.time()
            query_str = q["question"]
            is_in_scope = q.get("is_in_scope", True)
            expected_sources = [s.lower() for s in q.get("expected_sources", [])]
            target_section = q.get("target_section", "").lower()

            # Retrieve
            retrieved = retriever.search(query_str, top_k=10, search_mode=retrieval_mode)
            if reranker:
                retrieved = reranker.rerank(query_str, retrieved, top_k=5)
            else:
                retrieved = retrieved[:5]

            # Generate
            gen_res = generator.generate(query_str, retrieved)
            latencies.append((time.time() - t0) * 1000)

            # Evaluate Out-of-Scope Refusal
            if not is_in_scope:
                out_of_scope_count += 1
                if gen_res.is_refusal:
                    refusal_correct += 1
                    faithfulness_scores.append(1.0)
                else:
                    faithfulness_scores.append(0.0)
                continue

            # In-scope Retrieval Accuracy
            # Check how many retrieved chunks match expected document/section
            hits_at_1 = 0
            hits_at_3 = 0
            hits_at_5 = 0
            found_reciprocal_rank = 0.0

            for rank, item in enumerate(retrieved):
                item_doc = item.chunk.filename.lower()
                item_sec = item.chunk.section_heading.lower()
                is_match = (
                    any(exp in item_doc for exp in expected_sources)
                    or (target_section and target_section in item_sec)
                )

                if is_match:
                    if rank == 0:
                        hits_at_1 = 1
                    if rank < 3:
                        hits_at_3 += 1
                    if rank < 5:
                        hits_at_5 += 1
                    if found_reciprocal_rank == 0.0:
                        found_reciprocal_rank = 1.0 / (rank + 1)

            p1_list.append(hits_at_1)
            p3_list.append(hits_at_3 / 3.0)
            p5_list.append(hits_at_5 / 5.0)
            rec5_list.append(1.0 if hits_at_5 > 0 else 0.0)
            rr_list.append(found_reciprocal_rank)

            # Citation accuracy
            if gen_res.citations:
                top_cit_doc = gen_res.citations[0].document.lower()
                if any(exp in top_cit_doc for exp in expected_sources):
                    citation_matches += 1

            # Faithfulness score
            ground_truth = q.get("ground_truth_answer", "").lower()
            key_facts = [w for w in ground_truth.split() if len(w) > 4][:5]
            ans_lower = gen_res.answer.lower()
            overlap = sum(1 for k in key_facts if k in ans_lower)
            faithfulness_scores.append(overlap / max(1, len(key_facts)))

        in_scope_count = len(questions) - out_of_scope_count

        return ExperimentMetrics(
            experiment_id=experiment_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            retrieval_mode=retrieval_mode,
            use_reranker=use_reranker,
            precision_at_1=round(sum(p1_list) / max(1, len(p1_list)), 3),
            precision_at_3=round(sum(p3_list) / max(1, len(p3_list)), 3),
            precision_at_5=round(sum(p5_list) / max(1, len(p5_list)), 3),
            recall_at_5=round(sum(rec5_list) / max(1, len(rec5_list)), 3),
            mrr=round(sum(rr_list) / max(1, len(rr_list)), 3),
            citation_accuracy=round(citation_matches / max(1, in_scope_count), 3),
            refusal_accuracy=round(refusal_correct / max(1, out_of_scope_count), 3),
            faithfulness_score=round(sum(faithfulness_scores) / max(1, len(faithfulness_scores)), 3),
            avg_latency_ms=round(sum(latencies) / max(1, len(latencies)), 2),
            total_evaluated=len(questions),
        )

    def run_all_standard_experiments(self) -> List[ExperimentMetrics]:
        configs = [
            ("EXP-001 (Chunk 256, Overlap 0%, Semantic)", 256, 0, "semantic", False),
            ("EXP-002 (Chunk 256, Overlap 20%, Semantic)", 256, 50, "semantic", False),
            ("EXP-003 (Chunk 512, Overlap 10%, Semantic)", 512, 50, "semantic", False),
            ("EXP-004 (Chunk 512, Overlap 20%, Semantic)", 512, 100, "semantic", False),
            ("EXP-005 (Chunk 1024, Overlap 10%, Semantic)", 1024, 100, "semantic", False),
            ("EXP-006 (Chunk 512, Overlap 20%, BM25 Only)", 512, 100, "bm25", False),
            ("EXP-007 (Chunk 512, Overlap 20%, Hybrid RRF)", 512, 100, "hybrid", False),
            ("EXP-008 (Chunk 512, Overlap 20%, Hybrid + Re-Rank)", 512, 100, "hybrid", True),
        ]
        results: List[ExperimentMetrics] = []
        for exp_id, c_size, c_overlap, r_mode, r_rank in configs:
            m = self.run_experiment(
                experiment_id=exp_id,
                chunk_size=c_size,
                chunk_overlap=c_overlap,
                retrieval_mode=r_mode,
                use_reranker=r_rank,
            )
            results.append(m)
        return results

