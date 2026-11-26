# ==============================================================================
# Kryon - Multi-Format RAG Academic Regulations Assistant
# Production Docker Container (CPU-optimized, clone-and-run ready)
# ==============================================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files, buffer stdout, and configure defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    ENVIRONMENT=production \
    DATA_DIR=data/raw \
    CHUNK_SIZE=512 \
    CHUNK_OVERLAP=50 \
    SEARCH_MODE=hybrid \
    TOP_K=5 \
    USE_RERANKER=true \
    CONFIDENCE_THRESHOLD=0.50 \
    GOOGLE_REDIRECT_URI=http://localhost:8000/api/v1/auth/google/callback

WORKDIR /app

# Install essential OS libraries for FAISS (OpenMP), Pillow, SSL/OAuth, and healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    libgomp1 \
    libjpeg62-turbo \
    zlib1g \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

# Install CPU-only PyTorch first to keep the image lean, then remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Ensure persistent runtime directories exist even on fresh clones
RUN mkdir -p /app/data/raw /app/models

# Copy application source code, updated frontend, and data
COPY . .

# Build-time smoke check for core ML, vector search, and API packages
RUN python -c "import fastapi, uvicorn, faiss, torch, transformers, peft, safetensors, httpx, PIL, dotenv; print('All core packages verified!')"

# Expose FastAPI application port
EXPOSE 8000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Launch the server
CMD ["python", "run_server.py"]
