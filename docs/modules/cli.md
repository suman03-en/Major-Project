# CLI Module (`src/cli/`)

This module contains the command-line interfaces (CLI) that wire the underlying modules together. They are the entry points for running the pipeline.

---

## 1. `pdf_extractor.py`

**Purpose**: Orchestrates the conversion of raw PDFs in `input_pdfs/` to structured JSON in `extracted_jsons/`.

**Workflow**:
1. Uses Python's `glob` to find all `.pdf` files.
2. Initializes the `PdfExtractor` and `TextCleaner`.
3. Does a first-pass OCR on the first 2 pages to extract metadata (e.g., Act Name, Year).
4. Initializes the `RegexFormatter` with the extracted metadata.
5. Does a second-pass OCR on the entire document, cleaning the text line-by-line.
6. Passes the combined text to the formatter and saves the output JSON.

**Usage**:
```bash
python -m src.cli.pdf_extractor
```

---

## 2. `ingest.py`

**Purpose**: Orchestrates reading the extracted JSONs and pushing them into the Qdrant database.

**Workflow**:
1. Connects to Qdrant. If the `--recreate` flag is passed, it drops the existing `nepali_legal_chunks` collection.
2. Initializes the `LegalChunkEmbedder`.
3. Reads all `_dataset.json` files from `extracted_jsons/`.
4. Passes the chunks to the embedder to get Dense and Sparse vectors.
5. Upserts the vectors and payloads to Qdrant.

**Usage**:
```bash
python -m src.cli.ingest --recreate
```

---

## 3. `search.py`

**Purpose**: An interactive REPL (Read-Eval-Print Loop) terminal interface for querying the vector database.

**Workflow**:
1. Prompts the user for a query.
2. Encodes the query into vectors.
3. Performs a Hybrid Search on Qdrant.
4. Optionally performs a Cross-Encoder Rerank on the top results.
5. Prints the results beautifully to the terminal, including the source Act, the hierarchical path, and the clause text.

**Usage**:
```bash
# Interactive mode
python -m src.cli.search

# One-shot query mode
python -m src.cli.search -q "कम्पनी दर्ता" --no-rerank
```

---

## 4. `extract_ner.py`

**Purpose**: Orchestrates the NER pipeline to extract business entities.

**Workflow**:
1. Reads a specific JSON dataset (passed via `--input`).
2. Runs the `ClauseFilter` to skip irrelevant chunks.
3. Sends the relevant chunks to the `NERExtractor` (Ollama or Mistral).
4. Saves the structured responses to `ner_outputs/`.

**Usage**:
```bash
# Filter only (no LLM calls)
python -m src.cli.extract_ner -i extracted_jsons/dataset.json --filter-only

# Run extraction using Ollama
python -m src.cli.extract_ner -i extracted_jsons/dataset.json --provider ollama
```
