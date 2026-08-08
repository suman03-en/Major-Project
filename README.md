# RAG for Business Domain in Nepal
- Handles queries related to business registration processes, acts, regulations, and related legal questions in Nepal.

# Nepali Legal PDF Extraction, Embedding & Search Pipeline

This project provides an automated, end-to-end pipeline to:
1. **Extract text** from Nepali legal PDFs (handling scanned pages, broken Unicode mappings, and legacy fonts via PyMuPDF + Tesseract OCR with 300 DPI Devanagari preprocessing).
2. **Clean & structure** raw text into a highly nested, hierarchical JSON format (`Act -> Chapter -> Section -> Sub-section -> Clause -> Provisos/Explanations`).
3. **Embed** structural chunks using **BAAI/bge-m3** (generating 1024-dim Dense vectors + Sparse lexical weights).
4. **Store & Retrieve** chunks using **Qdrant** vector database with **Reciprocal Rank Fusion (RRF) Hybrid Search** and optional **BAAI/bge-reranker-v2-m3 Cross-Encoder Re-ranking**.

---

## 🚀 Docker Setup (Recommended)

Using Docker is the easiest way to run the pipeline. The Docker setup uses Astral's ultra-fast **`uv`** package manager and pre-installs **Tesseract OCR** with the Nepali (`nep`) language pack inside the container. No local Python or Tesseract installation is needed on your host machine!

### 1. Build and Start Services
Start Qdrant and the RAG container in the background:
```bash
docker compose up -d
```
*(To force rebuild after code changes, use `docker compose up -d --build`)*

---

### 2. Interactive Search (Crucial Step)

Because interactive CLI tools require terminal keyboard input (`stdin`), you **must** use the interactive TTY flag (`-it`) when exec-ing into the running container:

```bash
docker compose exec -it rag python src/cli/search.py
```

> 💡 **Why `-it`?** Standard `docker compose up` only streams logs to the console and cannot receive keyboard typing. The `-it` flag attaches an **interactive pseudo-terminal** so you can type your queries directly into the search prompt.

Inside the interactive search CLI, you can:
- Type any query in Nepali (e.g., `कम्पनी दर्ता गर्ने तरिका`) or English.
- Toggle search mode: `type hybrid`, `type dense`, or `type sparse`.
- Change result count: `top 10`.
- Toggle cross-encoder re-ranking: `rerank on` or `rerank off`.
- Exit: `exit` or `quit`.

---

### 3. Pipeline Commands via Docker Exec

You can run individual pipeline stages directly inside the running container:

* **Extract PDFs to JSON:**
  ```bash
  docker compose exec rag python src/cli/pdf_extractor.py
  ```
  *(Reads PDFs from `./input_pdfs/` and outputs structured datasets to `./output_jsons/`)*

* **Embed & Ingest Datasets into Qdrant:**
  ```bash
  docker compose exec rag python src/cli/ingest.py
  ```
  *(To wipe the database and start fresh, add `--recreate`)*:
  ```bash
  docker compose exec rag python src/cli/ingest.py --recreate
  ```

* **One-Shot Search (Skip interactive mode):**
  ```bash
  docker compose exec rag python src/cli/search.py -q "कम्पनी दर्ता"
  ```
  *(To disable cross-encoder re-ranking for faster execution, add `--no-rerank`)*:
  ```bash
  docker compose exec rag python src/cli/search.py -q "कम्पनी दर्ता" --no-rerank
  ```

---

### 4. Automated Execution & Environment Variables

You can configure automatic pipeline execution on container startup by editing your `.env` file or passing environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant service URL |
| `RUN_EXTRACT` | `false` | Automatically extract PDFs in `input_pdfs/` on startup |
| `RUN_INGEST` | `false` | Automatically embed & ingest `output_jsons/` on startup |
| `RECREATE_COLLECTION` | `false` | Drop and recreate Qdrant collection during ingestion |
| `ENABLE_RERANK` | `false` | Toggle cross-encoder re-ranking (`true`/`false`) |

#### Example: Automated Extract + Ingest on Startup
In PowerShell:
```cmd
$env:RUN_EXTRACT="true"; $env:RUN_INGEST="true"; docker compose up
```

---

## 💻 Local Installation Setup (Alternative)

If you prefer running without Docker directly on your host machine:

### Prerequisites
1. **Python 3.10+** installed.
2. **Tesseract OCR**:
   - Download and install [Tesseract OCR for Windows](https://github.com/UB-Mannheim/tesseract/wiki).
   - Install the **Nepali (`nep`) language pack** during setup.
3. **Qdrant**:
   - Run Qdrant locally: `docker run -p 6333:6333 qdrant/qdrant`

### Steps
1. Create and activate a virtual environment:
   ```cmd
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

2. Install dependencies using `uv` (recommended) or `pip`:
   ```cmd
   pip install -r requirements.txt
   ```

3. Create a `.env` file:
   ```env
   QDRANT_URL="http://localhost:6333"
   MISTRAL_API_KEY="your_api_key_here"
   ```

4. Run scripts:
   ```cmd
   # Step 1: Extract PDFs
   python src/cli/pdf_extractor.py

   # Step 2: Ingest Datasets
   python src/cli/ingest.py

   # Step 3: Interactive Search
   python src/cli/search.py
   ```

---

## 📁 Project Directory Structure

```text
Major-Project/
│── Dockerfile                 # Fast Docker build (uv + Tesseract OCR + nep pack)
│── compose.yaml               # Docker Compose service definition (qdrant + rag)
│── entrypoint.sh              # Container startup orchestrator script
│── requirements.txt           # Python dependencies
│── .env                       # Environment configuration
│── input_pdfs/                # 📂 Place raw Nepali PDF files here
│── output_jsons/              # 📂 Extracted hierarchical JSON datasets
└── src/
    ├── config.py              # Configuration loader
    ├── cli/
    │   ├── pdf_extractor.py   # CLI orchestrator for PDF extraction
    │   ├── ingest.py          # CLI for embedding JSONs into Qdrant
    │   └── search.py          # Interactive REPL & One-shot Search CLI
    ├── extraction/
    │   ├── extractor.py       # PyMuPDF + Tesseract OCR hybrid logic
    │   ├── cleaner.py         # Regex-based text artifact cleaner
    │   └── formatter.py       # Hierarchical JSON parser
    └── embedding/
        ├── embedder.py        # Dense + Sparse BAAI/bge-m3 embedder
        ├── vector_store.py    # Qdrant client wrapper (hybrid RRF search)
        └── reranker.py        # BAAI/bge-reranker-v2-m3 Cross-Encoder
```

---

## 🔬 Architecture Highlights

- **Preprocessed OCR**: Renders PDFs at 300 DPI, applies contrast enhancement, Otsu binarization, and median noise filtering for crisp Devanagari OCR recognition.
- **Hierarchical Context Embedding**: Each chunk (clause/sub-section) is embedded alongside its full breadcrumb path (e.g., `title: कम्पनीको संस्थापना | दफा: संक्षिप्त नाम र प्रारम्भ | ...`), ensuring short legal sentences carry rich semantic context.
- **Native Multi-Vector Hybrid Search**: Uses `BGEM3FlagModel` to generate 1024-dim dense vectors and lexical sparse weights simultaneously. Qdrant merges both streams via Reciprocal Rank Fusion (RRF).
- **Two-Stage Re-ranking**: Option to pass top retrieval candidates through `BAAI/bge-reranker-v2-m3` cross-encoder for max precision.