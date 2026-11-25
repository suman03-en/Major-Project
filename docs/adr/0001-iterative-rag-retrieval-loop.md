# Bounded Iterative RAG Retrieval Loop

Status: accepted

Single-shot RAG (`1 query → 1 retrieval → 1 generation`) fails on Nepali statutory and business-registration questions because answers span multiple scattered sections and cross-references (`दफा`), and checking only vector similarity before consulting Neo4j discards valid procedural matches. We chose a bounded iterative retrieval loop (`max_loops = 3`, early exit on Loop 1, hard cap of 12 deduplicated provisions) with `search_clauses`, `fetch_section`, and `search_graph` tools over single-shot RAG or unbounded agent loops, trading 1–2 extra free-tier LLM calls on complex queries for multi-section recall within token limits.
