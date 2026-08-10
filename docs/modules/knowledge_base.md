# Knowledge Base & NER Module (`src/knowledge_base/`)

This module is responsible for Named Entity Recognition (NER)—extracting structured business data (like fees, timelines, and required documents) from the raw legal text using Large Language Models (LLMs).

---

## 1. `schemas.py`

**Purpose**: Defines the rigid data structures we expect the LLM to return.

**Key Components**:
- Uses **Pydantic** `BaseModel` to define schemas:
  - `OfficeEntity`: The government office responsible (Name, Level).
  - `ProcessStep`: Action required, documents needed, duration, and price/fees.
  - `ExtractedEntity`: Grouping of Process Steps.
  - `ClauseNEROutput`: The top-level root object holding the extraction results.
- These schemas are passed directly to the LLMs (via structured output tools) to guarantee the JSON matches our database requirements.

**Dependencies**: `pydantic`.

---

## 2. `ner_filter.py`

**Purpose**: Acts as a gatekeeper to prevent wasting expensive LLM API calls on irrelevant legal text.

**Key Components**:
- **`ClauseFilter` Class**: Contains predefined sets of highly specific Nepali keywords related to business processes (e.g., `दर्ता`, `शुल्क`, `निवेदन`, `दिनभित्र`).
- **`score_chunk(chunk)`**: Calculates a relevance score for a text chunk based on keyword matches. If a chunk scores below the defined threshold, it is skipped by the NER pipeline entirely.

---

## 3. `ner_extractor.py`

**Purpose**: The core LLM integration that reads the text and extracts the Pydantic schemas.

**Key Components**:
- **`NERExtractor` Class**: Supports two providers:
  - **Ollama**: For local, free inference (default model: `qwen2.5:3b`).
  - **Mistral API**: For high-quality, cloud-based inference (requires `.env` key).
- **`_build_system_prompt()`**: Contains the complex instruction set guiding the LLM on how to interpret Nepali legal text and map it to our specific JSON structure.
- **Retry Logic**: If the LLM returns invalid JSON (a common issue), the `_extract_mistral` and `_extract_ollama` methods catch the `ValidationError`, attempt to clean the raw string, and retry the extraction automatically.
