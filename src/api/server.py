import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from src.ingestion.parser import DocumentParser, ParsedDocument
from src.chunking.splitters import ChunkingEngine, TextChunk
from src.embeddings.embedder import DenseEmbedder
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.hybrid import HybridRetriever, ScoredChunk
from src.reranking.reranker import CrossEncoderReRanker
from src.generation.grounded_generator import GroundedRAGGenerator, GenerationResult
from src.evaluation.evaluator import RAGEvaluator, ExperimentMetrics


# Pydantic Schemas
class QueryRequest(BaseModel):
    query: str
    search_mode: str = os.getenv("SEARCH_MODE", "hybrid")  # "hybrid", "semantic", "bm25"
    top_k: int = int(os.getenv("TOP_K", "5"))
    use_reranker: bool = os.getenv("USE_RERANKER", "true").lower() in ("true", "1", "yes")
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.50"))
    doc_filter: Optional[str] = None


class RAGSystem:
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir or os.getenv("DATA_DIR", "data/raw"))
        self.parsed_docs: List[ParsedDocument] = []
        self.chunks: List[TextChunk] = []
        self._last_index_time: float = 0.0
        self.embedder = DenseEmbedder()
        self.vector_store = FAISSVectorStore(self.embedder)
        self.bm25 = BM25Retriever()
        self.retriever = HybridRetriever(self.vector_store, self.bm25)
        self.reranker = CrossEncoderReRanker()
        self.generator = GroundedRAGGenerator()
        self.evaluator = RAGEvaluator(raw_data_dir=str(self.data_dir))
        self.cached_experiments: List[Dict[str, Any]] = []
        self.load_and_index()

    def load_and_index(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "512"))
        chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "50"))
        self.parsed_docs = []
        self.chunks = []
        self.vector_store.clear()

        supported_exts = {
            ".md", ".markdown", ".txt", ".pdf", ".docx", ".doc",
            ".xlsx", ".xls", ".csv", ".tsv", ".json",
            ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"
        }

        # Sort files with most recently modified/uploaded first
        files = [p for p in self.data_dir.iterdir() if p.is_file() and p.suffix.lower() in supported_exts]
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        for path in files:
            try:
                doc = DocumentParser.parse_file(path)
                self.parsed_docs.append(doc)
                doc_chunks = ChunkingEngine.chunk_document(
                    doc, strategy="recursive", chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                self.chunks.extend(doc_chunks)
            except Exception as e:
                print(f"Error parsing document {path.name}: {e}")

        self.vector_store.add_chunks(self.chunks)
        self.bm25.fit(self.chunks)
        self._last_index_time = time.time()

    def sync_if_needed(self) -> bool:
        """Auto-rescan data_dir if any file was added, removed, or modified."""
        supported_exts = {
            ".md", ".markdown", ".txt", ".pdf", ".docx", ".doc",
            ".xlsx", ".xls", ".csv", ".tsv", ".json",
            ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"
        }
        current_files = [p for p in self.data_dir.iterdir() if p.is_file() and p.suffix.lower() in supported_exts]
        indexed_names = {d.filename for d in self.parsed_docs}

        if len(current_files) != len(self.parsed_docs) or any(p.name not in indexed_names for p in current_files):
            self.load_and_index()
            return True

        if any(p.stat().st_mtime > getattr(self, "_last_index_time", 0.0) for p in current_files):
            self.load_and_index()
            return True

        return False

    def query(self, req: QueryRequest) -> GenerationResult:
        # 1. Retrieve candidates
        candidates = self.retriever.search(
            query=req.query,
            top_k=max(req.top_k * 2, 8),
            search_mode=req.search_mode,
            doc_filter=req.doc_filter,
        )

        # 2. Re-rank
        if req.use_reranker:
            final_chunks = self.reranker.rerank(req.query, candidates, top_k=req.top_k)
        else:
            final_chunks = candidates[:req.top_k]

        # 3. Generate grounded answer
        self.generator.confidence_threshold = req.confidence_threshold
        result = self.generator.generate(req.query, final_chunks)
        return result


app = FastAPI(
    title="RAG Academic Regulations & Syllabus QA System",
    description="Production-grade RAG pipeline grounded in official academic regulations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system instance
rag_system = RAGSystem()


@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "documents_indexed": len(rag_system.parsed_docs),
        "chunks_indexed": len(rag_system.chunks),
        "vector_index_size": rag_system.vector_store.index.ntotal,
        "bm25_corpus_size": rag_system.bm25.corpus_size,
    }


@app.post("/api/v1/query")
def execute_query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    result = rag_system.query(req)
    return result.model_dump()


@app.get("/api/v1/documents")
def list_documents():
    rag_system.sync_if_needed()
    docs_summary = []
    for doc in rag_system.parsed_docs:
        doc_chunks = [c for c in rag_system.chunks if c.doc_id == doc.doc_id]
        docs_summary.append(
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "filename": doc.filename,
                "sections_count": len(doc.sections),
                "chunks_count": len(doc_chunks),
                "sha256": doc.sha256[:12],
                "char_count": len(doc.full_text),
                "sections": [{"heading": s.heading, "page": s.page_estimate} for s in doc.sections[:10]],
            }
        )
    return {"total_documents": len(docs_summary), "documents": docs_summary}


@app.post("/api/v1/documents/refresh")
def refresh_documents_index():
    rag_system.load_and_index()
    return {
        "status": "success",
        "message": f"Successfully re-scanned repository. Indexed {len(rag_system.parsed_docs)} documents and {len(rag_system.chunks)} chunks.",
        "documents_count": len(rag_system.parsed_docs),
        "chunks_count": len(rag_system.chunks),
    }


@app.post("/api/v1/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    target_path = rag_system.data_dir / file.filename
    content = await file.read()
    with open(target_path, "wb") as f:
        f.write(content)
    # Touch mtime to current timestamp so it sorts to the top
    try:
        now = time.time()
        os.utime(target_path, (now, now))
    except Exception:
        pass
    # Re-index
    rag_system.load_and_index()
    return {"message": f"Successfully ingested and indexed {file.filename}"}


@app.delete("/api/v1/documents/{filename}")
def delete_document(filename: str):
    clean_name = Path(filename).name
    target_path = (rag_system.data_dir / clean_name).resolve()

    # Safety check against path traversal
    data_dir_resolved = rag_system.data_dir.resolve()
    if not str(target_path).startswith(str(data_dir_resolved)):
        raise HTTPException(status_code=400, detail="Invalid filename parameter.")

    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail=f"Document '{clean_name}' not found.")

    try:
        target_path.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete '{clean_name}': {str(e)}")

    # Re-index to update vector index and BM25 index
    rag_system.load_and_index()
    return {"message": f"Successfully removed '{clean_name}' and updated search index."}


@app.get("/api/v1/experiments")
def get_experiments():
    if not rag_system.cached_experiments:
        # Run benchmarks once and cache
        results = rag_system.evaluator.run_all_standard_experiments()
        rag_system.cached_experiments = [r.model_dump() for r in results]
    return {"experiments": rag_system.cached_experiments}


@app.post("/api/v1/experiments/run")
def re_run_experiments():
    results = rag_system.evaluator.run_all_standard_experiments()
    rag_system.cached_experiments = [r.model_dump() for r in results]
    return {"message": "Evaluation suite executed successfully", "experiments": rag_system.cached_experiments}


# Auth & Password Reset Schemas & Endpoints
class SendOTPRequest(BaseModel):
    email: str

class VerifyOTPRequest(BaseModel):
    email: str
    otp: str
    new_password: str

otp_store: Dict[str, Dict[str, Any]] = {}

@app.post("/api/v1/auth/send-otp")
def send_otp(req: SendOTPRequest):
    email = req.email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")
    import random
    code = f"{random.randint(100000, 999999)}"
    otp_store[email] = {"code": code, "expires_at": time.time() + 600}
    print(f"\n[KRYON AUTH] Generated OTP for {email}: {code} (valid for 10 mins)\n")
    return {
        "status": "success",
        "message": f"Verification OTP sent to {email}",
        "dev_code": code
    }

@app.post("/api/v1/auth/verify-otp")
def verify_otp(req: VerifyOTPRequest):
    email = req.email.strip().lower()
    record = otp_store.get(email)
    if not record:
        raise HTTPException(status_code=400, detail="No OTP requested for this email. Please request a new code.")
    if time.time() > record["expires_at"]:
        del otp_store[email]
        raise HTTPException(status_code=400, detail="OTP has expired. Please request a new code.")
    if req.otp.strip() != record["code"]:
        raise HTTPException(status_code=400, detail="Invalid OTP code. Please check and try again.")
    if len(req.new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters long.")
    del otp_store[email]
    return {
        "status": "success",
        "message": "Password reset successful! You can now log in with your new password."
    }


# Mount static assets and serve frontend
frontend_dir = Path("frontend")
if frontend_dir.exists():
    static_dir = frontend_dir / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    def serve_index():
        index_file = frontend_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "Frontend index.html ready."}

    @app.get("/signup.html")
    @app.get("/signup")
    @app.get("/login.html")
    @app.get("/login")
    def serve_signup():
        signup_file = frontend_dir / "signup.html"
        if signup_file.exists():
            return FileResponse(signup_file)
        raise HTTPException(status_code=404, detail="Auth page not found")

    @app.get("/favicon.ico")
    def serve_favicon():
        fav = frontend_dir / "static" / "favicon.ico"
        if fav.exists():
            return FileResponse(fav)
        return JSONResponse(status_code=204, content=None)

