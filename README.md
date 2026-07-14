# RAG for Business Domain in Nepal
- It handles queries related to the business registration process and related legal questions in Nepal.

# Nepali Legal PDF Extraction, Embedding & Search Pipeline

This project provides an automated, end-to-end pipeline to:
1. Extract text from Nepali legal PDFs (handling broken Unicode mappings and legacy fonts via OCR).
2. Clean and structure the text into a highly nested, hierarchical JSON format.
3. Embed the structural chunks using **BAAI/bge-m3** (Dense + Sparse representations).
4. Store and retrieve them using the **Qdrant** vector database with **Reciprocal Rank Fusion (RRF) Hybrid Search**.

## Prerequisites

1. **Python 3.10+** installed on your system.
2. **Tesseract OCR**:
   - Download and install [Tesseract OCR for Windows](https://github.com/UB-Mannheim/tesseract/wiki).
   - During installation, check the box to install the **Nepali (`nep`) language pack**.
   - By default, the script looks for Tesseract at `C:\Program Files\Tesseract-OCR\tesseract.exe`. Update the path in `src/pipeline/extractor.py` if installed elsewhere.
3. **Qdrant**:
   - Run Qdrant locally (e.g., via Docker: `docker run -p 6333:6333 qdrant/qdrant`) or download the Windows executable.

## Installation Setup

1. Create a virtual environment (recommended):
   ```cmd
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

2. Install the required Python packages (including `FlagEmbedding` for multi-vector BGE-M3):
   ```cmd
   pip install -r requirements.txt
   pip install FlagEmbedding
   ```

3. Create a `.env` file in the root directory and add your Qdrant URL:
   ```env
   QDRANT_URL="http://localhost:6333"
   ```

## Project Directory Structure

```text
Major project/
│── requirements.txt           # Python dependencies
│── README.md                  # This file
│── .env                       # Environment configuration (e.g., Qdrant URL)
│── input_pdfs/                # 📂 Place your raw Nepali PDFs here!
│── output_jsons/              # 📂 Generated JSON datasets appear here
└── src/
    │── config.py              # Configuration and settings loader
    │── pdf_extractor.py       # Main orchestration script for PDF extraction
    │── ingest.py              # Script to embed JSONs and upsert to Qdrant
    │── search.py              # Interactive CLI for testing RAG search
    ├── pipeline/
    │   ├── extractor.py       # PyMuPDF + Tesseract OCR hybrid logic
    │   ├── cleaner.py         # Regex-based text artifact cleaner
    │   └── formatter.py       # Regex-based structural JSON parser
    └── embedding/
        ├── embedder.py        # Dense + Sparse BGE-M3 text embedder
        └── vector_store.py    # Qdrant client wrapper (handles hybrid RRF)
```

## Usage Instructions

### 1. Extract PDFs to JSON
1. Place one or more Nepali legal PDF files (e.g., `company_act.pdf`) into the `input_pdfs/` directory.
2. Run the main extraction script:
   ```cmd
   python src/pdf_extractor.py
   ```
   *This extracts text using OCR, builds the document hierarchy (Chapters, Sections, etc.), and outputs to `output_jsons/`.*

### 2. Embed and Ingest Data
1. Ensure your Qdrant server is running and the URL matches your `.env` configuration.
2. Run the ingestion script to embed the chunks into dense and sparse vectors, and upsert them to Qdrant:
   ```cmd
   python src/ingest.py
   ```
   *(To wipe the database and start fresh, run `python src/ingest.py --recreate`)*

### 3. Search the Database
Use the interactive REPL to test vector retrieval:
```cmd
python src/search.py
```
Inside the REPL, you can dynamically adjust search parameters:
- `top <N>`: Change how many results are returned (e.g., `top 10`).
- `type <dense|sparse|hybrid>`: Toggle the search algorithm. `hybrid` is the default and provides the highest accuracy by combining semantic meaning (dense) with exact keyword matching (sparse) via Reciprocal Rank Fusion.

## Architecture Details

- **Extraction**: Renders PDFs to 300 DPI images and leverages Tesseract's `psm 4` to ensure bulleted clauses like `(क)` remain attached to their definitions instead of splitting into distinct columns.
- **Formatting**: Builds an exact hierarchical schema (Chapter → Section → Clause). Extracts `तर` (Provisos) and `स्पष्टीकरण` (Explanations) directly.
- **Embedding Context**: Small clauses like `"कम्पनीको नाम,"` are embedded along with their full hierarchy titles and provisos in a single string, ensuring short texts have massive semantic meaning in the vector space.
- **Multi-Vector Hybrid Search**: Uses `BGEM3FlagModel` to generate a 1024-dimensional semantic vector and a lexical token weight map simultaneously. Qdrant's `Prefetch` API merges both strategies natively.