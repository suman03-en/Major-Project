# Workflows & Data Flow

This document outlines the four primary workflows of the system. Understanding these pipelines is crucial for grasping how the different modules interact.

---

## 1. PDF Extraction Pipeline
**Goal:** Convert raw, scanned Nepali legal PDFs into highly structured, hierarchical JSON.

**Flow:**
1. **Input**: PDF files placed in `input_pdfs/`.
2. **OCR (`extractor.py`)**: The PDF is rendered into images at 300 DPI. OpenCV applies preprocessing (contrast, Otsu binarization, median blur) to make the Devanagari script clear. Tesseract OCR extracts the raw text.
3. **Cleaning (`cleaner.py`)**: Regex rules remove common OCR artifacts, broken Unicode, and normalize whitespace.
4. **Formatting (`formatter.py`)**: The text is parsed line-by-line using Regex to identify standard legal hierarchy (`परिच्छेद`, `दफा`, `उपदफा`, `खण्ड`, `तर`, `स्पष्टीकरणः`). 
5. **Output**: A nested JSON dataset saved to `extracted_jsons/`.

---

## 2. Ingestion & Embedding Pipeline
**Goal:** Take structured JSON datasets and ingest them into Qdrant for semantic search.

**Flow:**
1. **Input**: JSON files in `extracted_jsons/`.
2. **Context Enrichment (`embedder.py`)**: Before embedding, a chunk (e.g., a sub-section) is prepended with its hierarchy path. 
   - *Example: "Title: Company Act | Chapter: 2 | Section: 3 | Text: The company must be registered."*
3. **Multi-Vector Embedding (`embedder.py`)**: The enriched text is passed through `BAAI/bge-m3`. It generates:
   - A 1024-dimensional Dense vector (for semantic meaning).
   - Lexical sparse weights (for keyword matching).
4. **Qdrant Upsert (`vector_store.py`)**: The vectors and the raw chunk payload are pushed to the Qdrant collection (`nepali_legal_chunks`).

---

## 3. Hybrid Search Pipeline
**Goal:** Retrieve the most relevant legal clauses for a user's query.

**Flow:**
1. **Query Processing (`search.py`)**: User enters a query (Nepali or English).
2. **Query Embedding (`embedder.py`)**: The query is converted into Dense and Sparse vectors.
3. **Qdrant Search (`vector_store.py`)**: Qdrant performs a Prefetch search:
   - Finds top candidates using Dense vectors.
   - Finds top candidates using Sparse vectors.
   - Fuses the results using Reciprocal Rank Fusion (RRF) to get the best combined ranking.
4. **Reranking [Optional] (`reranker.py`)**: The top `N` results are passed through a Cross-Encoder (`bge-reranker-v2-m3`) which scores the (Query, Document) pairs for maximum precision.

---

## 4. Named Entity Recognition (NER) Pipeline
**Goal:** Extract structured business processes (fees, steps, documents, offices) from the legal text.

**Flow:**
1. **Input**: The `extracted_jsons/` dataset.
2. **Pre-Filtering (`ner_filter.py`)**: An algorithm scores every chunk based on keywords (e.g., "दस्तुर", "दर्ता"). Chunks scoring below a threshold are skipped to save LLM compute time and API costs.
3. **LLM Extraction (`ner_extractor.py`)**: High-scoring chunks are sent to an LLM (Ollama or Mistral) with a strict prompt demanding JSON output matching our Pydantic schemas.
4. **Validation (`schemas.py`)**: The returned JSON is validated. If it fails, the system automatically retries with a fallback cleaning mechanism.
5. **Output**: A new dataset saved to `ner_outputs/` containing the structured entities mapped back to their original chunks.
