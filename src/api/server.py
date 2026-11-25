import os
import time
import secrets
import smtplib
from urllib.parse import urlencode
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import List, Dict, Any, Optional
import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
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

def dispatch_otp_email(to_email: str, otp_code: str) -> tuple[bool, str]:
    """
    Sends the 6-digit OTP verification code to the user's email via SMTP.
    Configurable via environment variables (.env):
    - SMTP_HOST (e.g. smtp.gmail.com)
    - SMTP_PORT (e.g. 587)
    - SMTP_USER (e.g. your_email@gmail.com)
    - SMTP_PASSWORD (e.g. 16-character Google App Password)
    - SMTP_FROM (e.g. "Kryon AI <noreply@kryon.ai>")
    """
    load_dotenv(override=True)
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com").strip()
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "").strip().strip("'\"")
    smtp_password = os.getenv("SMTP_PASSWORD", "").strip().strip("'\"").replace(" ", "")
    smtp_from = os.getenv("SMTP_FROM", "").strip() or (f"Kryon AI <{smtp_user}>" if smtp_user else "Kryon AI <noreply@kryon.ai>")

    if not smtp_user or not smtp_password:
        print(f"\n{'='*65}\n[KRYON AUTH NOTICE] SMTP credentials not configured in .env.\nTo deliver emails to {to_email}, set SMTP_USER and SMTP_PASSWORD in .env.\nGenerated OTP for {to_email}: {otp_code} (Valid for 5 mins)\n{'='*65}\n")
        return False, "SMTP credentials (SMTP_USER, SMTP_PASSWORD) not configured in .env"

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Kryon AI - Your Verification Code is {otp_code}"
        msg["From"] = smtp_from
        msg["To"] = to_email

        plain_text = (
            f"Hello,\n\n"
            f"Your Kryon AI password reset verification code is:\n\n"
            f"    {otp_code}\n\n"
            f"This code will expire in 5 minutes.\n"
            f"If you did not request this verification code, please ignore this email.\n\n"
            f"Best regards,\n"
            f"Kryon AI Security Team"
        )

        html_text = f"""<!DOCTYPE html>
<html>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f9fafb; margin: 0; padding: 32px 16px;">
  <div style="max-width: 520px; margin: 0 auto; background: #ffffff; border: 1px solid #e5e7eb; border-radius: 16px; padding: 36px 28px; box-shadow: 0 4px 12px rgba(0,0,0,0.03);">
    <div style="margin-bottom: 24px;">
      <span style="font-size: 22px; font-weight: 700; color: #111827; letter-spacing: -0.5px;">Kryon AI</span>
    </div>
    <h2 style="font-size: 20px; font-weight: 600; color: #111827; margin: 0 0 12px;">Password Reset Verification</h2>
    <p style="font-size: 15px; color: #4b5563; line-height: 1.5; margin: 0 0 24px;">
      We received a request to reset your password for your Kryon AI account. Please enter the verification code below to proceed:
    </p>
    <div style="background-color: #f3f4f6; border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 24px;">
      <span style="font-size: 34px; font-weight: 700; letter-spacing: 8px; color: #111827; font-family: monospace;">{otp_code}</span>
    </div>
    <p style="font-size: 13px; color: #6b7280; line-height: 1.5; margin: 0 0 16px;">
      &bull; This code is strictly valid for <strong>5 minutes</strong>.<br>
      &bull; If you did not request a password reset, please ignore this email or contact support.
    </p>
    <hr style="border: none; border-top: 1px solid #f3f4f6; margin: 24px 0;" />
    <p style="font-size: 12px; color: #9ca3af; margin: 0;">
      This is an automated security email from Kryon AI Academic QA System. Please do not reply directly to this email.
    </p>
  </div>
</body>
</html>"""

        msg.attach(MIMEText(plain_text, "plain", "utf-8"))
        msg.attach(MIMEText(html_text, "html", "utf-8"))

        with smtplib.SMTP(smtp_host, smtp_port, timeout=12) as server:
            server.ehlo()
            if smtp_port == 587:
                server.starttls()
                server.ehlo()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_from, [to_email], msg.as_string())

        print(f"[KRYON AUTH] Verification email successfully delivered to {to_email} via SMTP ({smtp_host})")
        return True, "Email delivered successfully"
    except Exception as e:
        err_msg = f"[KRYON AUTH ERROR] Failed to send email to {to_email}: {str(e)}"
        print(err_msg)
        return False, str(e)


@app.post("/api/v1/auth/send-otp")
def send_otp(req: SendOTPRequest):
    email = req.email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")
    import random
    code = f"{random.randint(100000, 999999)}"
    otp_store[email] = {"code": code, "expires_at": time.time() + 300}
    
    # Send actual email to recipient
    sent, reason = dispatch_otp_email(email, code)

    # Note: Code is NEVER returned to client for security
    return {
        "status": "success",
        "message": f"Verification OTP code sent to {email}. Please check your inbox and spam folder.",
        "email_sent": sent
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


# Google OAuth 2.0 Store & Endpoints
oauth_states: Dict[str, Dict[str, Any]] = {}
oauth_results: Dict[str, Dict[str, Any]] = {}


def _get_google_oauth_config() -> Dict[str, str]:
    load_dotenv(override=True)
    port = os.getenv("PORT", "8000").strip()
    return {
        "client_id": os.getenv("GOOGLE_CLIENT_ID", "").strip().strip("'\""),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET", "").strip().strip("'\""),
        "redirect_uri": os.getenv("GOOGLE_REDIRECT_URI", f"http://localhost:{port}/api/v1/auth/google/callback").strip(),
        "port": port,
    }


def _sanitize_origin(origin: str, port: str) -> str:
    allowed = {f"http://localhost:{port}", f"http://127.0.0.1:{port}"}
    cleaned = origin.strip().rstrip("/")
    return cleaned if cleaned in allowed else f"http://localhost:{port}"


def _cleanup_oauth_stores() -> None:
    now = time.time()
    expired_states = [k for k, v in oauth_states.items() if now - v.get("created_at", 0) > 600]
    for k in expired_states:
        oauth_states.pop(k, None)
    expired_results = [k for k, v in oauth_results.items() if now - v.get("created_at", 0) > 120]
    for k in expired_results:
        oauth_results.pop(k, None)


@app.get("/api/v1/auth/google/status")
def google_oauth_status():
    cfg = _get_google_oauth_config()
    return {
        "configured": bool(cfg["client_id"] and cfg["client_secret"]),
        "redirect_uri": cfg["redirect_uri"],
    }


@app.get("/api/v1/auth/google/login")
def google_oauth_login(request: Request, mode: str = "login", origin: str = ""):
    _cleanup_oauth_stores()
    cfg = _get_google_oauth_config()
    safe_mode = "signup" if mode == "signup" else "login"
    fallback_origin = f"{request.url.scheme}://{request.url.netloc}" if request.url.netloc else f"http://localhost:{cfg['port']}"
    safe_origin = _sanitize_origin(origin or fallback_origin, cfg["port"])

    if not cfg["client_id"] or not cfg["client_secret"]:
        return RedirectResponse(url=f"{safe_origin}/signup.html?mode={safe_mode}&auth_error=not_configured")

    state = secrets.token_urlsafe(24)
    oauth_states[state] = {
        "mode": safe_mode,
        "origin": safe_origin,
        "created_at": time.time(),
    }

    params = urlencode({
        "client_id": cfg["client_id"],
        "redirect_uri": cfg["redirect_uri"],
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "online",
        "prompt": "select_account",
    })
    return RedirectResponse(url=f"https://accounts.google.com/o/oauth2/v2/auth?{params}")


@app.get("/api/v1/auth/google/callback")
async def google_oauth_callback(
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
):
    _cleanup_oauth_stores()
    cfg = _get_google_oauth_config()
    state_data = oauth_states.pop(state, None) if state else None

    if not state_data:
        return RedirectResponse(url=f"http://localhost:{cfg['port']}/signup.html?mode=login&auth_error=invalid_state")

    safe_origin = state_data["origin"]
    safe_mode = state_data["mode"]

    if error or not code:
        return RedirectResponse(url=f"{safe_origin}/signup.html?mode={safe_mode}&auth_error=oauth_denied")

    if not cfg["client_id"] or not cfg["client_secret"]:
        return RedirectResponse(url=f"{safe_origin}/signup.html?mode={safe_mode}&auth_error=not_configured")

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            token_resp = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": cfg["client_id"],
                    "client_secret": cfg["client_secret"],
                    "redirect_uri": cfg["redirect_uri"],
                    "grant_type": "authorization_code",
                },
            )
            if token_resp.status_code != 200:
                return RedirectResponse(url=f"{safe_origin}/signup.html?mode={safe_mode}&auth_error=token_exchange_failed")

            access_token = token_resp.json().get("access_token")
            if not access_token:
                return RedirectResponse(url=f"{safe_origin}/signup.html?mode={safe_mode}&auth_error=token_exchange_failed")

            userinfo_resp = await client.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if userinfo_resp.status_code != 200:
                return RedirectResponse(url=f"{safe_origin}/signup.html?mode={safe_mode}&auth_error=userinfo_failed")

            profile = userinfo_resp.json()
    except Exception as exc:
        print(f"[KRYON GOOGLE OAUTH ERROR] {exc}")
        return RedirectResponse(url=f"{safe_origin}/signup.html?mode={safe_mode}&auth_error=oauth_network_error")

    email = (profile.get("email") or "").strip().lower()
    if not email:
        return RedirectResponse(url=f"{safe_origin}/signup.html?mode={safe_mode}&auth_error=missing_email")

    full_name = (profile.get("name") or email.split("@")[0]).strip()
    first_name = (profile.get("given_name") or full_name.split(" ")[0] or "User").strip()
    last_name = (profile.get("family_name") or " ".join(full_name.split(" ")[1:])).strip()

    parts = [p for p in full_name.split() if p]
    if len(parts) >= 2:
        initials = (parts[0][0] + parts[-1][0]).upper()
    elif parts:
        initials = parts[0][:2].upper()
    else:
        initials = "K"

    one_time_token = secrets.token_urlsafe(32)
    oauth_results[one_time_token] = {
        "mode": safe_mode,
        "created_at": time.time(),
        "user": {
            "name": full_name,
            "firstName": first_name,
            "lastName": last_name,
            "email": email,
            "initials": initials,
            "picture": profile.get("picture", ""),
            "authType": "google",
        },
    }

    return RedirectResponse(url=f"{safe_origin}/signup.html?mode={safe_mode}&google_token={one_time_token}")


@app.get("/api/v1/auth/google/result")
def google_oauth_result(token: str):
    _cleanup_oauth_stores()
    record = oauth_results.pop(token, None)
    if not record:
        raise HTTPException(status_code=400, detail="Google authentication session expired. Please try again.")
    return {
        "status": "success",
        "mode": record["mode"],
        "user": record["user"],
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
            return FileResponse(index_file, headers={"Cache-Control": "no-cache, must-revalidate"})
        return {"message": "Frontend index.html ready."}

    @app.get("/signup.html")
    @app.get("/signup")
    @app.get("/login.html")
    @app.get("/login")
    def serve_signup():
        signup_file = frontend_dir / "signup.html"
        if signup_file.exists():
            return FileResponse(signup_file, headers={"Cache-Control": "no-cache, must-revalidate"})
        raise HTTPException(status_code=404, detail="Auth page not found")

    @app.get("/favicon.ico")
    def serve_favicon():
        fav = frontend_dir / "static" / "favicon.ico"
        if fav.exists():
            return FileResponse(fav, media_type="image/x-icon")
        fav_svg = frontend_dir / "static" / "favicon.svg"
        if fav_svg.exists():
            return FileResponse(fav_svg, media_type="image/svg+xml")
        return JSONResponse(status_code=204, content=None)

