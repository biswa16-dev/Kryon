import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from src.retrieval.hybrid import ScoredChunk

load_dotenv()


class Citation(BaseModel):
    document: str
    section: str
    page: int
    quote_snippet: str
    confidence: float


class GenerationResult(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    confidence_score: float
    is_refusal: bool
    retrieved_chunks: List[Dict[str, Any]]
    execution_time_ms: float
    model_used: str


class GroundedRAGGenerator:
    """
    Production RAG Generator powered by fine-tuned flan-t5-base.
    
    Pipeline:
    1. Builds a context prompt from retrieved chunks
    2. Feeds it to the fine-tuned flan-t5-base model for natural answer generation
    3. Applies confidence gating to refuse low-confidence / out-of-scope queries
    4. Attaches inline source citations from retrieved documents
    
    Fully self-hosted, zero external APIs, CPU-deployable.
    """

    def __init__(
        self,
        confidence_threshold: Optional[float] = None,
        model_dir: str = "models/kryon-academic-t5",
        max_context_chars: int = 1200,
        max_new_tokens: int = 300,
    ):
        if confidence_threshold is None:
            try:
                self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))
            except ValueError:
                self.confidence_threshold = 0.45
        else:
            self.confidence_threshold = confidence_threshold

        self.max_context_chars = max_context_chars
        self.max_new_tokens = max_new_tokens
        self.model_dir = Path(model_dir)
        self.enable_local_model = os.getenv("ENABLE_LOCAL_MODEL", "true").lower() in ("true", "1", "yes")
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._device = "cpu"
        self._load_model()

    def _load_model(self):
        """Load the fine-tuned flan-t5-base model from disk when enabled and available."""
        if not self.enable_local_model:
            print("[Generator] Local model loading disabled (ENABLE_LOCAL_MODEL=false) — using extractive fallback")
            return

        if self.model_dir.exists() and (self.model_dir / "model.safetensors").exists():
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

                self._torch = torch
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
                self._model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_dir))
                self._model.to(self._device)
                self._model.eval()
                print(f"[Generator] Loaded fine-tuned model from {self.model_dir}")
            except Exception as e:
                print(f"[Generator] Model load failed: {e}")
        else:
            print(f"[Generator] No model found at {self.model_dir} — will use extractive fallback")

    def _build_context(self, scored_chunks: List[ScoredChunk]) -> str:
        """Build a context string from the top retrieved chunks."""
        context_parts = []
        total_chars = 0

        for sc in scored_chunks:
            chunk = sc.chunk
            section_label = f"[{chunk.section_heading}]" if chunk.section_heading else ""
            text = chunk.text.strip()
            part = f"{section_label} {text}".strip()

            if total_chars + len(part) > self.max_context_chars:
                remaining = self.max_context_chars - total_chars
                if remaining > 100:
                    context_parts.append(part[:remaining] + "...")
                break

            context_parts.append(part)
            total_chars += len(part)

        return "\n\n".join(context_parts)

    def _generate_with_model(self, query: str, context: str) -> str:
        """Generate an answer using the fine-tuned flan-t5-base model."""
        if self._model is None or self._tokenizer is None or self._torch is None:
            return ""

        prompt = (
            f"Answer the following academic question based on the given context.\n\n"
            f"Context: {context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        inputs = self._tokenizer(
            prompt,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(self._device)

        with self._torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.2,
                temperature=0.7,
                do_sample=False,
            )

        answer = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()

    def _extractive_fallback(self, query: str, scored_chunks: List[ScoredChunk]) -> str:
        """Fallback: extract and format answer directly from top chunk when model is unavailable."""
        if not scored_chunks:
            return ""
        top = scored_chunks[0].chunk
        doc_name = top.doc_title or top.filename
        section = top.section_heading

        lines = [line.strip() for line in top.text.split("\n") if line.strip()]
        clean_points = []
        for line in lines:
            cl = line.lstrip("-*#•0123456789. ")
            if len(cl) > 20 and not cl.startswith("#"):
                clean_points.append(cl)

        points_text = "\n".join([f"- {p}" for p in clean_points[:5]]) if clean_points else f"- {top.text[:400]}"

        return (
            f"Based on **{doc_name}** under **§{section}**:\n\n"
            f"{points_text}\n\n"
            f"> 📄 **Source**: `{top.filename}` (§{section}, Page {top.page_number})"
        )

    def compute_confidence(self, query: str, scored_chunks: List[ScoredChunk]) -> float:
        """Multi-signal confidence scoring combining retrieval score and keyword coverage."""
        if not scored_chunks:
            return 0.0

        top_chunk = scored_chunks[0]
        top_score = top_chunk.score

        query_words = set(re.findall(r"\b[a-z0-9]{2,}\b", query.lower()))
        stop_words = {"what", "when", "where", "which", "how", "many", "does", "rules",
                       "policy", "the", "and", "is", "are", "can", "will", "about",
                       "tell", "me", "please", "bro", "hey", "this", "that", "for"}
        filtered_query = [w for w in query_words if w not in stop_words]

        if not filtered_query:
            return top_score

        combined_text = ""
        for sc in scored_chunks[:3]:
            combined_text += " " + sc.chunk.text.lower() + " " + sc.chunk.section_heading.lower()

        matched_words = [w for w in filtered_query if w in combined_text]
        coverage = len(matched_words) / len(filtered_query)

        combined_confidence = (top_score * 0.50) + (coverage * 0.50)
        return float(max(0.0, min(1.0, combined_confidence)))

    def _build_citations(self, scored_chunks: List[ScoredChunk], top_n: int = 3) -> List[Citation]:
        """Build citation list from scored chunks."""
        return [
            Citation(
                document=sc.chunk.filename,
                section=sc.chunk.section_heading,
                page=sc.chunk.page_number,
                quote_snippet=sc.chunk.text.strip().replace("\n", " ")[:140] + "...",
                confidence=round(sc.score, 3),
            )
            for sc in scored_chunks[:top_n]
        ]

    def _build_retrieved_chunks_info(self, scored_chunks: List[ScoredChunk]) -> List[Dict[str, Any]]:
        """Format retrieved chunk metadata for the API response."""
        return [
            {
                "chunk_id": c.chunk.chunk_id,
                "title": c.chunk.doc_title,
                "filename": c.chunk.filename,
                "section": c.chunk.section_heading,
                "page": c.chunk.page_number,
                "score": round(c.score, 3),
                "semantic_score": round(c.semantic_score, 3),
                "bm25_score": round(c.bm25_score, 3),
                "text_preview": c.chunk.text[:220] + "...",
            }
            for c in scored_chunks
        ]

    def _append_source_footer(self, answer: str, scored_chunks: List[ScoredChunk]) -> str:
        """Append a source citation footer to the generated answer."""
        if not scored_chunks:
            return answer

        top = scored_chunks[0].chunk
        footer = f"\n\n> 📄 **Source**: `{top.filename}` (§{top.section_heading}, Page {top.page_number})"

        # Don't duplicate if answer already contains a source footer
        if "📄" in answer or "Source" in answer[-100:]:
            return answer

        return answer + footer

    def generate(self, query: str, scored_chunks: List[ScoredChunk]) -> GenerationResult:
        """Main generation pipeline: model inference -> confidence gating -> citation attachment."""
        t0 = time.time()
        confidence = self.compute_confidence(query, scored_chunks)

        # 1. Confidence Gate — refuse low-confidence / out-of-scope queries
        if confidence < self.confidence_threshold or not scored_chunks:
            exec_time = (time.time() - t0) * 1000
            refusal_msg = (
                f"I'm sorry, but I couldn't find sufficient information in the official university "
                f"academic regulations to answer your question about: *\"{query}\"*.\n\n"
                f"My knowledge is limited to academic regulations, examination policies, grading systems, "
                f"attendance rules, and course syllabi. If your question is about something else "
                f"(like hostel rules, campus facilities, or events), please check with the relevant "
                f"university department.\n\n"
                f"**Confidence**: {confidence:.1%}"
            )
            return GenerationResult(
                query=query,
                answer=refusal_msg,
                citations=[],
                confidence_score=confidence,
                is_refusal=True,
                retrieved_chunks=self._build_retrieved_chunks_info(scored_chunks[:3]),
                execution_time_ms=round(exec_time, 2),
                model_used="ConfidenceGate (Refusal)",
            )

        # 2. Build context from retrieved chunks
        context = self._build_context(scored_chunks)

        # 3. Generate answer with the fine-tuned model
        answer = self._generate_with_model(query, context)

        # 4. Fallback to extractive if model produced empty or very short output
        if len(answer.strip()) < 20:
            answer = self._extractive_fallback(query, scored_chunks)
            model_label = "Extractive Fallback"
        else:
            # Append source citation footer
            answer = self._append_source_footer(answer, scored_chunks)
            model_label = "kryon-academic-t5 (flan-t5-base + LoRA)"

        # 5. Build citations
        citations = self._build_citations(scored_chunks)

        exec_time = (time.time() - t0) * 1000

        return GenerationResult(
            query=query,
            answer=answer,
            citations=citations,
            confidence_score=confidence,
            is_refusal=False,
            retrieved_chunks=self._build_retrieved_chunks_info(scored_chunks),
            execution_time_ms=round(exec_time, 2),
            model_used=model_label,
        )
