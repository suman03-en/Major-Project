# Nepali Legal RAG

A grounded legal and business-registration assistant for Nepal that combines statutory text retrieval with procedural knowledge-graph navigation.

## Language

**Act**:
A statutory enactment, regulation (Niyamawali), or official legal instrument ingested into the knowledge base.
_Avoid_: Document, file, law-text

**Section**:
A numbered statutory division (`दफा`) within an Act, identified by `sec` without requiring a chapter number.
_Avoid_: Article, part

**Provision**:
A single retrievable statutory unit (subsection, clause, proviso, or section summary) stored in the vector index with citations and temporal status.
_Avoid_: Chunk, snippet, record

**Procedure**:
A named multi-step administrative workflow (`Process → Step`) in the knowledge graph linking required documents, offices, fees, and wait times.
_Avoid_: Workflow, guide, flow
