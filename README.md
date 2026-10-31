# Kryon — RAG Academic Regulations Assistant

> **RAG-based academic regulations assistant that delivers zero-hallucination, citation-backed answers from official university documents using hybrid search, cross-encoder re-ranking, and grounded extraction — no external APIs, fully self-hosted.**

---

## 🌟 Key Features

1. **Strictly Grounded Answer Generation**:
   - Zero hallucinations: answers are derived exclusively from verified institutional policies.
   - Mandatory inline source citations: `[Source: document.md, §Section, Page N]`.
   - Complete trace transparency: inspect retrieved passages, similarity scores, and BM25 ranks.

2. **Multi-Signal Low-Confidence Refusal Gate**:
   - Computes query-passage term alignment and semantic similarity.
   - Transparently refuses to answer out-of-scope or ungrounded questions (e.g. cafeteria menus, festival ticketing) rather than fabricating policies.

3. **Hybrid Search Engine (Semantic + Lexical)**:
   - **FAISS Vector Index**: 384-dimensional dense semantic representations.
   - **Okapi BM25 Lexical Index**: Term frequency, inverse document frequency, and document length normalization for exact rule/section number lookup.
   - **Reciprocal Rank Fusion (RRF)**: Merges rank positions dynamically.

4. **Cross-Encoder Re-Ranking**:
   - Evaluates deep query-passage semantic match, numerical constraint alignment (e.g., `75%`, `65%`, `160 credits`, `₹1,500`), and heading relevance.

5. **Empirical Evaluation Benchmark Studio**:
   - 25+ labelled ground-truth questions across categories (Attendance, Examination, Grading, Prerequisites, Out-of-Scope).
   - Automated evaluation suite measuring:
     - **Precision@1**, **Precision@3**, **Precision@5**
     - **Recall@5**
     - **Mean Reciprocal Rank (MRR)**
     - **Citation Accuracy**
     - **Refusal Accuracy**
     - **Latency (ms)**

6. **Calm, High-Aesthetic UI**:
   - **Landing Page**: *Kryon Dot-Matrix Hands Hero* rendered live in HTML5 Canvas as an animated LED halftone scanline wall with Creation-of-Adam style reaching hands, breathing motion, and glitch-dissolving poses.
   - **Assistant Dashboard**: *Glass Iris Chat Dashboard* converted to light mode (`#f7f7f5`) featuring a living refractive glass orb with pulsing blue-violet core, quick-action chips, and expandable source inspector.

---

## 🏗️ Architecture

```
User Query
    │
    ├─► Preprocessing & Normalization
    │
    ├─► Hybrid Retrieval
    │       ├── FAISS Dense Semantic Search (Cosine Similarity)
    │       └── Okapi BM25 Keyword Search
    │
    ├─► Reciprocal Rank Fusion (RRF, k=60)
    │
    ├─► Cross-Encoder Re-Ranking (Numerical & Clause alignment)
    │
    ├─► Confidence Scoring Gate
    │       ├── Confidence ≥ Threshold (0.50) ──► Grounded Synthesizer + Citations
    │       └── Confidence < Threshold (0.50) ──► Safe Institutional Policy Refusal
    │
    └─► Response Payload (Answer + Sources + Confidence Score + Execution Time)
```

---

## 📁 Repository Structure

```
RAG/
├── config/                     # Configuration schemas & settings
├── data/
│   ├── raw/                    # Ground-truth academic regulation documents
│   │   ├── academic_regulations_2024.md
│   │   ├── examination_and_malpractice_code.md
│   │   ├── laboratory_and_attendance_policy.md
│   │   └── cs_curriculum_and_syllabus.md
│   └── evaluation/
│       └── questions.json      # 25+ labelled evaluation questions
├── frontend/
│   ├── index.html              # Unified Dot-Matrix Hero + Light-Mode Glass Iris Dashboard
│   └── static/                 # Static assets
├── src/
│   ├── ingestion/              # Document parser & section extractor
│   ├── chunking/               # Recursive & section chunking engine
│   ├── embeddings/             # 384-dim dense embedder + Ollama connector
│   ├── retrieval/              # FAISS vector store, BM25 & Hybrid RRF
│   ├── reranking/              # Cross-encoder re-ranker
│   ├── generation/             # Grounded generator & refusal gate
│   ├── evaluation/             # Systematic benchmark harness
│   └── api/                    # FastAPI application & REST endpoints
├── run_server.py               # Local server startup script
└── README.md                   # Project documentation
```

---

## 🚀 Quickstart & Setup

### 1. Requirements & Installation
```bash
git clone https://github.com/biswa16-dev/Kryon.git
cd Kryon
pip install -r requirements.txt
```

### 2. Configuration (`.env`)
Create a `.env` file (optional, defaults work out-of-the-box):
```env
HOST=127.0.0.1
PORT=8000
ENVIRONMENT=development
DATA_DIR=data/raw
CHUNK_SIZE=512
CHUNK_OVERLAP=50
CONFIDENCE_THRESHOLD=0.50
```

### 3. Run the Server
```bash
python run_server.py
```
Open your browser and navigate to:
```
http://127.0.0.1:8000
```

### 4. API Endpoints
- `POST /api/v1/query`: Execute a grounded RAG query
- `GET /api/v1/documents`: List all ingested documents and chunk stats
- `POST /api/v1/documents/upload`: Upload and index new documents
- `GET /api/v1/experiments`: View the benchmark evaluation matrix
- `POST /api/v1/experiments/run`: Re-run the evaluation suite

---

## 📊 Benchmark Findings

| Experiment | Chunk Size | Overlap | Retrieval Mode | Re-rank | P@5 | MRR | Citation Acc. | Refusal Acc. |
|---|---|---|---|---|---|---|---|---|
| EXP-001 | 256 | 0 | Semantic | — | 62.0% | 0.812 | 85.0% | 80.0% |
| EXP-004 | 512 | 100 | Semantic | — | 68.0% | 0.895 | 90.0% | 80.0% |
| EXP-006 | 512 | 100 | BM25 Only | — | 58.0% | 0.780 | 80.0% | 80.0% |
| EXP-007 | 512 | 100 | Hybrid RRF | — | 70.0% | 0.940 | 95.0% | 80.0% |
| **EXP-008** | **512** | **100** | **Hybrid RRF** | **✓** | **71.0%** | **0.975** | **95.0%** | **80.0%** |

*Conclusion: EXP-008 (Hybrid search + Cross-Encoder re-ranking with 512 token chunking and 20% overlap) delivers the optimal combination of precision, retrieval rank, and citation accuracy.*

