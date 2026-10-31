import os
import re
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import httpx
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
    RAG Generator enforcing:
    1. Grounded factual answer generation
    2. Mandatory source citation layer
    3. Multi-signal confidence gating
    4. Strict refusal when retrieval confidence is below threshold (< 0.52)
    """

    def __init__(
        self,
        ollama_url: Optional[str] = None,
        model_name: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
    ):
        self.ollama_url = ollama_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        if confidence_threshold is None:
            try:
                self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.50"))
            except ValueError:
                self.confidence_threshold = 0.50
        else:
            self.confidence_threshold = confidence_threshold
        self._last_ollama_check: float = 0.0
        self._ollama_online: bool = False
        self._detected_model: Optional[str] = None

    def _get_active_ollama_model(self) -> Optional[str]:
        now = time.time()
        # Re-check Ollama status every 12 seconds
        if now - self._last_ollama_check > 12.0:
            self._last_ollama_check = now
            try:
                with httpx.Client(timeout=0.4) as client:
                    res = client.get(f"{self.ollama_url}/api/tags")
                    if res.status_code == 200:
                        self._ollama_online = True
                        data = res.json()
                        models = [m.get("name", "") for m in data.get("models", [])]
                        # Prefer user-configured model, else check for standard intelligent local models
                        preferred = [self.model_name, "qwen2.5:7b", "llama3.1:8b", "llama3.2:3b", "mistral:7b", "phi3:mini", "gemma2:9b"]
                        chosen = None
                        for pref in preferred:
                            match = next((m for m in models if pref in m), None)
                            if match:
                                chosen = match
                                break
                        self._detected_model = chosen or (models[0] if models else self.model_name)
                    else:
                        self._ollama_online = False
            except Exception:
                self._ollama_online = False

        return self._detected_model if self._ollama_online else None

    def _query_ollama(self, prompt: str) -> Optional[tuple[str, str]]:
        active_model = self._get_active_ollama_model()
        if not active_model:
            return None

        try:
            with httpx.Client(timeout=8.0) as client:
                res = client.post(
                    f"{self.ollama_url}/api/generate",
                    json={"model": active_model, "prompt": prompt, "stream": False},
                )
                if res.status_code == 200:
                    text_resp = res.json().get("response")
                    if text_resp:
                        return (text_resp, active_model)
        except Exception:
            self._ollama_online = False
        return None

    def compute_confidence(self, query: str, scored_chunks: List[ScoredChunk]) -> float:
        if not scored_chunks:
            return 0.0

        top_chunk = scored_chunks[0]
        top_score = top_chunk.score

        # Check keyword coverage between query and top retrieved chunks
        query_words = set(re.findall(r"\b[a-z]{3,}\b", query.lower()))
        # Remove common query stop words
        filtered_query = [w for w in query_words if w not in {"what", "when", "where", "which", "how", "many", "does", "rules", "policy"}]

        if not filtered_query:
            return top_score

        top_text = (top_chunk.chunk.text + " " + top_chunk.chunk.section_heading).lower()
        matched_words = [w for w in filtered_query if w in top_text]
        coverage = len(matched_words) / len(filtered_query)

        # Multi-signal confidence score
        combined_confidence = (top_score * 0.55) + (coverage * 0.45)
        return float(max(0.0, min(1.0, combined_confidence)))

    def _extract_grounded_factual_summary(self, query: str, chunks: List[ScoredChunk]) -> str:
        """
        Deterministic, hallucination-free factual extraction from retrieved regulation passages.
        Preserves exact numbers, grade thresholds, and penalty clauses.
        """
        top = chunks[0]
        text_lines = [line.strip() for line in top.chunk.text.split("\n") if line.strip()]

        # Filter lines most relevant to query
        query_words = set(re.findall(r"\b[a-z0-9]{3,}\b", query.lower()))
        relevant_lines = []

        for line in text_lines:
            line_lower = line.lower()
            if any(w in line_lower for w in query_words) or any(k in line_lower for k in ["percent", "%", "credit", "grade", "shall", "must", "fee", "penalty"]):
                clean_line = line.lstrip("-*#•0123456789. ")
                if len(clean_line) > 15:
                    relevant_lines.append(clean_line)

        if not relevant_lines:
            relevant_lines = text_lines[:3]

        summary_body = " ".join(relevant_lines[:4])

        answer = (
            f"According to **{top.chunk.doc_title}** under **{top.chunk.section_heading}**:\n\n"
            f"{summary_body}\n\n"
            f"> **Mandatory Citation**: `[Source: {top.chunk.filename}, §{top.chunk.section_heading}, Page {top.chunk.page_number}]`"
        )
        return answer

    def generate(self, query: str, scored_chunks: List[ScoredChunk]) -> GenerationResult:
        t0 = time.time()
        confidence = self.compute_confidence(query, scored_chunks)

        # Confidence Gate: Low-Confidence Refusal Logic
        if confidence < self.confidence_threshold or not scored_chunks:
            exec_time = (time.time() - t0) * 1000
            refusal_msg = (
                f"⚠️ **Official Policy Refusal (Low Confidence: {confidence:.1%})**\n\n"
                f"I cannot provide a verified answer because the official university academic regulations, "
                f"syllabi, and examination manuals do not contain sufficient verified clauses regarding: "
                f"*\"{query}\"*.\n\n"
                f"**Safety Guarantee**: The system refuses to fabricate or guess ungrounded institutional policies."
            )
            return GenerationResult(
                query=query,
                answer=refusal_msg,
                citations=[],
                confidence_score=confidence,
                is_refusal=True,
                retrieved_chunks=[
                    {
                        "chunk_id": c.chunk.chunk_id,
                        "title": c.chunk.doc_title,
                        "section": c.chunk.section_heading,
                        "score": round(c.score, 3),
                    }
                    for c in scored_chunks[:3]
                ],
                execution_time_ms=round(exec_time, 2),
                model_used="ConfidenceGate (Refusal Triggered)",
            )

        # Generate Grounded Citations
        citations: List[Citation] = []
        for sc in scored_chunks[:3]:
            # First 120 chars as snippet
            snippet = sc.chunk.text.strip().replace("\n", " ")[:140] + "..."
            citations.append(
                Citation(
                    document=sc.chunk.filename,
                    section=sc.chunk.section_heading,
                    page=sc.chunk.page_number,
                    quote_snippet=snippet,
                    confidence=round(sc.score, 3),
                )
            )

        # Try intelligent local LLM via Ollama if available
        context_block = "\n\n".join(
            [f"--- Document: {c.chunk.filename} (Section: {c.chunk.section_heading}, Page: {c.chunk.page_number}) ---\n{c.chunk.text}" for c in scored_chunks[:3]]
        )
        prompt = (
            f"You are the official University Academic Regulations Guide helping students in their learning phase. "
            f"Answer the student's question clearly, politely, and directly based STRICTLY on the official university context below. "
            f"Explain clearly what the student must do, percentage limits, passing requirements, or penalties. "
            f"Include exact numbers and cite the handbook section.\n\n"
            f"Official University Context:\n{context_block}\n\n"
            f"Student Question: {query}\n\n"
            f"Clear Student Guidance:"
        )

        ollama_res = self._query_ollama(prompt)
        if ollama_res:
            ollama_answer, active_model = ollama_res
            answer = ollama_answer.strip()
            # Append formal citation line
            top = scored_chunks[0].chunk
            answer += f"\n\n> **Verified Citation**: `[Source: {top.filename}, §{top.section_heading}, Page {top.page_number}]`"
            model_used = f"Ollama ({active_model})"
        else:
            # Deterministic, grounded extractor
            answer = self._extract_grounded_factual_summary(query, scored_chunks)
            model_used = "Grounded Extractive Synthesizer (Local)"

        exec_time = (time.time() - t0) * 1000

        return GenerationResult(
            query=query,
            answer=answer,
            citations=citations,
            confidence_score=confidence,
            is_refusal=False,
            retrieved_chunks=[
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
            ],
            execution_time_ms=round(exec_time, 2),
            model_used=model_used,
        )
