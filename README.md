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

Using Docker is the easiest way to run the pipeline. 

### 1. Build and Start Services
Start Qdrant, Neo4j, and the RAG container in the background:
```bash
docker compose up -d
```

### 2. Interactive Search & Graph Query (Crucial Step)
You **must** use the interactive TTY flag (`-it`) when exec-ing into the running container to allow keyboard input:
```bash
# For Vector Search:
docker compose exec -it rag python -m src.cli.search

# For Graph Query:
docker compose exec -it rag python -m src.cli.query_graph
```

### 3. Pipeline Commands via Docker Exec
You can run individual pipeline stages directly inside the running container:

* **Extract PDFs to JSON:**
  ```bash
  docker compose exec rag python -m src.cli.pdf_extractor
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

## 💻 Local Installation Setup (Alternative)

If you prefer running without Docker directly on your host machine:

### Prerequisites
1. **Python 3.10+** installed.
2. **Tesseract OCR** with the **Nepali (`nep`) language pack**.
3. **Qdrant** running locally (`docker run -p 6333:6333 qdrant/qdrant`).
4. **Neo4j** running locally (e.g., `neo4j:5-community` via docker).

### Steps
1. Create and activate a virtual environment:
   ```cmd
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
2. Install the project dependencies (editable mode):
   ```cmd
   pip install -e .
   ```
3. Create a `.env` file based on `.env.example`:
   ```env
   QDRANT_URL="http://localhost:6333"
   NEO4J_URI="bolt://localhost:7687"
   NEO4J_USER="neo4j"
   NEO4J_PASSWORD="neo4j_pass"
   MISTRAL_API_KEY="your_api_key_here"
   ```
4. Run scripts using module execution:
   ```cmd
   python -m src.cli.pdf_extractor
   python -m src.cli.ingest
   python -m src.cli.ingest_graph
   python -m src.cli.search
   python -m src.cli.query_graph
   ```