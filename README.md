# RAG for Business Domain in Nepal
- Handles queries related to business registration processes, acts, regulations, and related legal questions in Nepal.

# Nepali Legal PDF Extraction, Embedding & Search Pipeline

This project provides an automated, end-to-end pipeline to:
1. **Extract text** from Nepali legal PDFs (handling scanned pages, broken Unicode mappings, and legacy fonts via PyMuPDF + Tesseract OCR with 300 DPI Devanagari preprocessing).
2. **Clean & structure** raw text into a highly nested, hierarchical JSON format (`Act -> Chapter -> Section -> Sub-section -> Clause -> Provisos/Explanations`).
3. **Embed** structural chunks using **BAAI/bge-m3** (generating 1024-dim Dense vectors + Sparse lexical weights).
4. **Store & Retrieve** chunks using **Qdrant** vector database with **Reciprocal Rank Fusion (RRF) Hybrid Search** and optional **BAAI/bge-reranker-v2-m3 Cross-Encoder Re-ranking**.
5. **Knowledge Graph** construction using **Neo4j** to build procedural pipelines of steps, documents, fees, and offices from the NER extracted entities.

---

## 📖 Official Documentation

For a detailed breakdown of the system architecture, workflows, and a file-by-file explanation of the codebase, please refer to the official docs:

👉 **[Read the Official Documentation](docs/index.md)**

---

## 🚀 Docker Setup (Recommended)

The Docker stack decouples **CPU-only web/RAG serving** (default, fast build) from **GPU-accelerated Surya OCR** (on-demand `--profile gpu`):

### 1. Default CPU Stack (Web App + Qdrant + Neo4j)
Start Qdrant, Neo4j, and the slim CPU-only `rag` container (uses `requirements-dev.txt`, no CUDA downloads or NVIDIA Container Toolkit required):
```bash
docker compose up -d --build
```

### 2. On-Demand GPU Extraction (Surya OCR)
When extracting scanned Nepali PDFs on an NVIDIA GPU, use the `gpu` profile (`Dockerfile.gpu` + `requirements.txt` with `torch==2.6.0+cu121` pinned to PyTorch's CDN to avoid `pypi.nvidia.com` timeouts):
```bash
docker compose --profile gpu run --rm ocr python -m src.cli.pdf_extractor
```
Extracted files are written to `./extracted_jsons/`, which is shared with the `rag` container.

### 3. Interactive Search, Web UI & Pipeline Commands
You **must** use `-it` when exec-ing into the running container for interactive CLI menus:
```bash
# Vector Search:
docker compose exec -it rag python -m src.cli.search

# Graph Query:
docker compose exec -it rag python -m src.cli.query_graph
```

* **Embed & Ingest Datasets into Qdrant:**
  ```bash
  docker compose exec rag python -m src.cli.ingest
  ```
* **NER Extraction:**
  ```bash
  docker compose exec rag python -m src.cli.extract_ner -i extracted_jsons/dataset.json
  ```
* **Ingest NER Graph into Neo4j:**
  ```bash
  docker compose exec rag python -m src.cli.ingest_graph
  ```

---

## 💻 Local Installation Setup (CPU Dev Environment)

For local development, unit tests, `ruff` linting, and running the FastAPI web app without Docker, use **Python 3.12** with `requirements-dev.txt` (CPU-only PyTorch; Python 3.13 is avoided because `transformers==4.44.2` / `tokenizers` lacks prebuilt 3.13 wheels):

1. Create a Python 3.12 virtual environment and install lightweight dev dependencies using [`uv`](https://docs.astral.sh/uv/):
   ```powershell
   uv venv .venv --python 3.12
   uv pip install -r requirements-dev.txt --index-strategy unsafe-best-match
   ```
   *(Note: `--index-strategy unsafe-best-match` lets `uv` resolve packages across both PyPI and PyTorch's CPU wheel index.)*
2. Copy `.env.example` to `.env` and fill in your LLM API keys (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_HELPER_MODEL`).
3. Run tests, linter, or the local web server:
   ```powershell
   .\.venv\Scripts\python.exe -m unittest discover -s tests -v
   .\.venv\Scripts\ruff.exe check src/ web/ tests/
   .\.venv\Scripts\uvicorn.exe web.main:app --reload
   ```