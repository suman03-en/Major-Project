# Nepali Legal RAG Pipeline

Welcome to the documentation for the **Nepali Legal Document RAG Pipeline**. This project processes scanned Nepali legal documents (specifically in the business registration domain), extracts the text, structures it hierarchically, and powers a highly accurate Hybrid Search and Named Entity Recognition (NER) pipeline.

## Overview

The system is designed to handle the complexities of the Nepali language and Devanagari script. It extracts structure from raw PDFs to ensure that legal clauses maintain their context (e.g., Chapter > Section > Sub-section > Clause) when embedded into the vector database.

### Core Capabilities
1. **PDF Extraction & Formatting**: Converts scanned PDFs into hierarchical JSON datasets using OCR.
2. **Context-Aware Embedding**: Embeds short clauses alongside their full breadcrumb path to preserve semantic meaning.
3. **Hybrid Search**: Combines Dense (Semantic) and Sparse (Lexical/BM25) search vectors using Reciprocal Rank Fusion (RRF).
4. **Information Extraction (NER)**: Uses LLMs (Local Ollama or Mistral API) to extract specific entities like process steps, fees, durations, and required documents from the legal text.

## Architecture

![Architecture](https://qdrant.tech/images/hybrid-search.png) <!-- Conceptual placeholder -->

### Tech Stack
- **OCR Engine**: PyMuPDF + Tesseract (with `nep` language pack)
- **Embeddings**: `BAAI/bge-m3` (Dense + Sparse Multi-Vector Model)
- **Vector Database**: Qdrant
- **Cross-Encoder Reranker**: `BAAI/bge-reranker-v2-m3`
- **LLM/NER**: Ollama (`qwen2.5`) / Mistral API
- **Data Validation**: Pydantic
- **Packaging**: `uv`, `pyproject.toml`

## Documentation Structure

If you're new to the codebase, we recommend reading through the docs in this order:

1. [**Workflows**](workflows.md): Understand how data flows through the system from start to finish.
2. **Module Deep Dives**:
   - [Extraction Pipeline](modules/extraction.md): How PDFs become JSON.
   - [Embedding Pipeline](modules/embedding.md): How JSON becomes Vectors in Qdrant.
   - [Knowledge Base & NER](modules/knowledge_base.md): How we extract structured data using LLMs.
   - [CLI Orchestration](modules/cli.md): How the scripts in `src/cli/` wire everything together.
   - [Evaluation Module](modules/evaluation.md): How we measure our retrieval accuracy (MRR, NDCG).
