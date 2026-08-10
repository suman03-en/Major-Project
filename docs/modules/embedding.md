# Embedding Module (`src/embedding/`)

This module handles converting text chunks into mathematical vectors (Embeddings) and interacting with the Qdrant Vector Database.

---

## 1. `embedder.py`

**Purpose**: Wraps the `BAAI/bge-m3` model to generate both Dense semantic vectors and Sparse lexical weights.

**Key Components**:
- **`LegalChunkEmbedder` Class**: Auto-detects if a GPU (`cuda`) is available and loads the `BGEM3FlagModel`.
- **`_build_embedding_text(chunk)`**: Crucial for accuracy. It prepends the chunk's hierarchy (Chapter, Section, Sub-section) and appends its provisos and explanations to the raw text. This ensures that a short sentence like *"Must register within 35 days"* is embedded with the context *"Company Act -> Section 3 -> Must register..."*.
- **`embed_chunks(chunks)`**: Takes the JSON chunks, processes them through the model, and returns a list of dictionaries formatted exactly for Qdrant insertion (with deterministic UUIDs).

**Dependencies**: `FlagEmbedding`, `torch`.

---

## 2. `vector_store.py`

**Purpose**: Manages the connection to Qdrant, handles database initialization, and performs the actual search queries.

**Key Components**:
- **`QdrantVectorStore` Class**: Connects to the database and ensures the `nepali_legal_chunks` collection exists with the correct vector configurations (1024-dim Dense, Sparse Vector support).
- **`hybrid_search()`**: The core retrieval function.
  - Uses Qdrant's `Prefetch` API to run two concurrent searches: one for Dense vectors and one for Sparse vectors.
  - Applies **Reciprocal Rank Fusion (RRF)** to fuse the two lists of results, giving preference to documents that score high in both semantic and keyword matching.

**Dependencies**: `qdrant-client`.

---

## 3. `reranker.py`

**Purpose**: Provides an optional second stage of ranking for maximum precision using a Cross-Encoder model.

**Key Components**:
- **`CrossEncoderReranker` Class**: Loads the `BAAI/bge-reranker-v2-m3` model.
- **`rerank()`**: Takes the top `N` results from the hybrid search and passes them through the cross-encoder alongside the user's query. The cross-encoder looks at the query and document simultaneously to generate a highly accurate relevance score. The results are then re-sorted based on this new score.
