"""
RAG Pipeline
=============
Orchestrates the full Retrieval-Augmented Generation flow:

  1. Encode query → hybrid search in Qdrant → (optional) cross-encoder rerank
  2. Query Neo4j graph for procedural steps matching retrieved chunks
  3. Assemble context (chunk text + structured graph steps)
  4. Generate grounded answer via Mistral LLM

Usage:
    from src.rag.rag_pipeline import RAGPipeline

    rag = RAGPipeline()
    answer = rag.answer("कम्पनी दर्ता कसरी गर्ने?")

    # Streaming:
    for token in rag.answer_stream("कम्पनी दर्ता कसरी गर्ने?"):
        print(token, end="", flush=True)
"""

import logging
from typing import Any, Dict, Iterator, List, Optional

from src.config import get_settings, setup_logging
from src.embedding.embedder import LegalChunkEmbedder
from src.embedding.vector_store import QdrantVectorStore
from src.embedding.reranker import CrossEncoderReranker
from src.knowledge_base.graph_store import Neo4jGraphStore
from src.llm.llm_client import MistralLLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Over-fetch multiplier when reranking is active
# ---------------------------------------------------------------------------
RERANK_FETCH_MULTIPLIER: int = 4
CONTEXT_OVERFETCH: int = 3       # Over-fetch to compensate for empty/short chunks
MIN_CHUNK_TEXT_LEN: int = 10     # Skip chunks shorter than this

# ---------------------------------------------------------------------------
# RAG System Prompt
# ---------------------------------------------------------------------------
RAG_SYSTEM_PROMPT = """You are a helpful legal assistant for Nepali business registration and administrative procedures.

Your job is to answer the user's question using ONLY the context provided below. The context contains:
1. **Legal Text Chunks** — relevant excerpts from Nepali laws and acts
2. **Procedural Steps** — step-by-step processes extracted from a knowledge graph

═══════════════════════════════════════════
RULES
═══════════════════════════════════════════
1. Answer ONLY from the provided context. Do NOT use outside knowledge.
2. If the context does not contain enough information to answer, clearly state:
   "उपलब्ध सन्दर्भमा यस प्रश्नको पर्याप्त जानकारी छैन।" (The available context does not have sufficient information for this question.)
3. CITATION RULE (CRITICAL):
   - Each context chunk has a "📌 उद्धृत गर्नुहोस्" (Cite As) label.
   - You MUST use ONLY that label when citing sources.
   - NEVER infer section/subsection numbers from inside the text body.
     Text may contain cross-references like "उपदफा (४) बमोजिम" — these
     refer to OTHER sections, not the current chunk's own location.
   - Example: If a chunk's cite label says "दफा ६१, उपदफा (७)",
     always cite it as "दफा ६१ को उपदफा (७)", even if the text mentions "उपदफा (४)".
4. For procedural questions, present steps in a clear numbered format.
5. Include relevant details like:
   - Required documents (आवश्यक कागजातहरू)
   - Government offices to visit (सम्बन्धित कार्यालय)
   - Fees (दस्तुर/शुल्क)
   - Time limits (समय सीमा)
6. Answer in the SAME LANGUAGE as the user's question (Nepali or English).
7. Be concise but thorough. Do not repeat the same information.
8. If multiple processes or acts are relevant, organize your answer clearly by process/act.

═══════════════════════════════════════════
RESPONSE FORMAT
═══════════════════════════════════════════
- Use clear headings and numbered lists for procedures
- Bold important terms, offices, and deadlines
- At the end, add a "स्रोत" (Source) section listing the acts referenced"""


class RAGPipeline:
    """
    Full RAG pipeline combining vector retrieval, graph lookup, and LLM generation.

    Parameters
    ----------
    embedder : LegalChunkEmbedder | None
        Pre-initialised embedder. Created automatically if ``None``.
    vector_store : QdrantVectorStore | None
        Pre-initialised Qdrant store. Created automatically if ``None``.
    graph_store : Neo4jGraphStore | None
        Pre-initialised Neo4j store. Set ``use_graph=False`` to skip.
    llm_client : MistralLLMClient | None
        Pre-initialised LLM client. Created automatically if ``None``.
    reranker : CrossEncoderReranker | None
        Pre-initialised cross-encoder reranker. Disabled by default.
    use_graph : bool
        Whether to query Neo4j for procedural steps (default: True).
    use_reranker : bool
        Whether to apply cross-encoder reranking (default: False).
    top_k : int
        Number of chunks to retrieve (default: 5).
    search_type : str
        Qdrant search type: ``"dense"``, ``"sparse"``, or ``"hybrid"`` (default).
    """

    def __init__(
        self,
        embedder: Optional[LegalChunkEmbedder] = None,
        vector_store: Optional[QdrantVectorStore] = None,
        graph_store: Optional[Neo4jGraphStore] = None,
        llm_client: Optional[MistralLLMClient] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        use_graph: bool = True,
        use_reranker: bool = False,
        top_k: int = 5,
        search_type: str = "hybrid",
    ) -> None:
        settings = get_settings()

        # --- Embedder ---
        if embedder is not None:
            self.embedder = embedder
        else:
            logger.info("Loading BGE-M3 embedder...")
            self.embedder = LegalChunkEmbedder(show_progress=False)

        # --- Vector store ---
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            self.vector_store = QdrantVectorStore(url=settings.QDRANT_URL)

        # --- Reranker ---
        self.use_reranker = use_reranker
        if use_reranker and reranker is not None:
            self.reranker = reranker
        elif use_reranker:
            logger.info("Loading cross-encoder reranker...")
            self.reranker = CrossEncoderReranker()
        else:
            self.reranker = None

        # --- Graph store ---
        self.use_graph = use_graph
        self.graph_store: Optional[Neo4jGraphStore] = None
        if use_graph:
            if graph_store is not None:
                self.graph_store = graph_store
            else:
                try:
                    self.graph_store = Neo4jGraphStore(
                        settings.NEO4J_URI, settings.NEO4J_USER, settings.NEO4J_PASSWORD
                    )
                    logger.info("Connected to Neo4j graph store.")
                except Exception as e:
                    logger.warning(
                        "Could not connect to Neo4j (%s). "
                        "RAG will work without graph context.", e
                    )
                    self.graph_store = None

        # --- LLM ---
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            self.llm_client = MistralLLMClient()

        self.top_k = top_k
        self.search_type = search_type

    # ------------------------------------------------------------------
    # Stage 1: Retrieve relevant chunks from Qdrant
    # ------------------------------------------------------------------

    def retrieve_chunks(
        self,
        query: str,
        top_k: Optional[int] = None,
        act_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Encode the query and retrieve the most relevant chunks from Qdrant.
        Filters empty/stub chunks and optionally applies cross-encoder reranking.
        """
        k = top_k or self.top_k

        embeddings = self.embedder.embed_texts([query])
        query_dense = embeddings["dense_vecs"][0].tolist()
        query_sparse = embeddings["lexical_weights"][0]

        # Over-fetch: always fetch extra to compensate for empty chunks,
        # and even more when reranking is active.
        if self.reranker:
            fetch_k = k * RERANK_FETCH_MULTIPLIER
        else:
            fetch_k = k * CONTEXT_OVERFETCH

        results = self.vector_store.search(
            query_dense=query_dense,
            query_sparse=query_sparse,
            search_type=self.search_type,
            top_k=fetch_k,
            act_filter=act_filter,
        )

        # Filter out empty or stub chunks (section headers with no text)
        results = [
            r for r in results
            if len(r.get("text", "").strip()) >= MIN_CHUNK_TEXT_LEN
        ]

        # Cross-encoder reranking
        if self.reranker and results:
            results = self.reranker.rerank(query=query, candidates=results, top_k=k)
        else:
            # Trim to top_k after filtering
            results = results[:k]

        logger.info("Retrieved %d chunks for query: %s", len(results), query[:80])
        return results

    # ------------------------------------------------------------------
    # Stage 2: Retrieve structured steps from Neo4j graph
    # ------------------------------------------------------------------

    def retrieve_graph_context(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Query Neo4j for procedural steps related to the retrieved chunks.

        Looks up process names from the graph that match act sources found
        in the retrieved chunks, then fetches full step details.
        """
        if not self.graph_store:
            return []

        try:
            # Get all process names from the graph
            all_processes = self.graph_store.get_all_processes()
            if not all_processes:
                return []

            # Find processes that are mentioned in the query or chunk text
            matching_processes: List[str] = []
            search_text = query.lower()
            for chunk in chunks:
                search_text += " " + chunk.get("text", "").lower()

            for process_name in all_processes:
                # Check if any part of the process name appears in the context
                if process_name.lower() in search_text:
                    matching_processes.append(process_name)

            # If no exact matches, try partial keyword matching
            if not matching_processes:
                query_words = set(query.split())
                for process_name in all_processes:
                    process_words = set(process_name.split())
                    # If at least one significant word overlaps (skip very short words)
                    overlap = query_words & process_words
                    significant_overlap = [w for w in overlap if len(w) > 2]
                    if significant_overlap:
                        matching_processes.append(process_name)

            # Fetch full step details for matching processes
            graph_results: List[Dict[str, Any]] = []
            for process_name in matching_processes[:3]:  # Cap at 3 to avoid context overflow
                steps = self.graph_store.get_steps_for_process(process_name)
                docs = self.graph_store.get_documents_for_process(process_name)
                offices = self.graph_store.get_offices_for_process(process_name)

                graph_results.append({
                    "process_name": process_name,
                    "steps": steps,
                    "documents": docs,
                    "offices": offices,
                })

            logger.info(
                "Graph context: %d matching processes found for query.",
                len(graph_results),
            )
            return graph_results

        except Exception as e:
            logger.warning("Graph retrieval failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Stage 3: Assemble the context prompt
    # ------------------------------------------------------------------

    @staticmethod
    def _format_citation(act_source: str, hierarchy: dict) -> str:
        """Build an explicit Nepali citation string from the hierarchy."""
        parts: list[str] = []
        
        act_name = "कम्पनी ऐन, २०६३" if "company_act_2063" in act_source else act_source
        parts.append(act_name)
        
        if "sec" in hierarchy:
            parts.append(f"दफा {hierarchy['sec']}")
        if "sub" in hierarchy:
            parts.append(f"उपदफा ({hierarchy['sub']})")
        if "clause" in hierarchy:
            parts.append(f"खण्ड ({hierarchy['clause']})")
            
        return ", ".join(parts)

    @staticmethod
    def _format_hierarchy(hierarchy: dict) -> str:
        """Build a readable breadcrumb from a hierarchy dict."""
        parts: list[str] = []
        if "ch_title" in hierarchy:
            parts.append(f"परिच्छेद {hierarchy.get('ch', '?')}: {hierarchy['ch_title']}")
        if "sec_title" in hierarchy:
            parts.append(f"दफा {hierarchy.get('sec', '?')}: {hierarchy['sec_title']}")
        if "sub" in hierarchy:
            parts.append(f"उपदफा ({hierarchy['sub']})")
        if "clause" in hierarchy:
            parts.append(f"खण्ड ({hierarchy['clause']})")
        return " → ".join(parts) if parts else "—"

    def _fetch_section_siblings(
        self,
        chunk: Dict[str, Any],
        seen_ids: set,
    ) -> List[Dict[str, Any]]:
        """
        Fetch sibling chunks from Qdrant that share the same act + chapter +
        section as the given chunk. This pulls in ALL clauses/subsections
        under the same section (e.g., all penalty tiers under दफा ४३).

        Uses Qdrant scroll with payload filtering — no embedding needed.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        h = chunk.get("hierarchy", {})
        act = chunk.get("act_source", "")
        ch = h.get("ch")
        sec = h.get("sec")

        if ch is None or sec is None or not act:
            return []

        try:
            scroll_filter = Filter(
                must=[
                    FieldCondition(key="act_source", match=MatchValue(value=act)),
                    FieldCondition(key="hierarchy.ch", match=MatchValue(value=ch)),
                    FieldCondition(key="hierarchy.sec", match=MatchValue(value=sec)),
                ]
            )
            points, _ = self.vector_store.client.scroll(
                collection_name=self.vector_store.collection_name,
                scroll_filter=scroll_filter,
                limit=20,  # Cap at 20 siblings per section
                with_payload=True,
                with_vectors=False,
            )

            siblings = []
            for pt in points:
                payload = pt.payload or {}
                cid = payload.get("chunk_id", "")
                text = payload.get("text", "")
                if cid in seen_ids or len(text.strip()) < MIN_CHUNK_TEXT_LEN:
                    continue
                siblings.append({
                    "chunk_id": cid,
                    "act_source": payload.get("act_source", ""),
                    "type": payload.get("type", ""),
                    "hierarchy": payload.get("hierarchy", {}),
                    "text": text,
                    "stats": payload.get("stats", {}),
                    "provisos": payload.get("provisos", []),
                    "explanations": payload.get("explanations", []),
                })
            return siblings

        except Exception as e:
            logger.warning("Sibling fetch failed: %s", e)
            return []

    def assemble_context(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        graph_context: List[Dict[str, Any]],
    ) -> str:
        """
        Build the full context string that gets sent to the LLM along with
        the user's question.

        For clause-level and subsection-level chunks, sibling chunks from the
        same section are fetched from Qdrant to give the LLM full context
        (e.g., all penalty tiers when one clause is retrieved).
        """
        sections: List[str] = []

        # Collect all sibling chunks to provide parent/sibling context
        expanded_chunks: List[Dict[str, Any]] = []
        seen_ids: set = set()
        expanded_sections: set = set()  # Track (act, ch, sec) already expanded

        for chunk in chunks:
            cid = chunk.get("chunk_id", "")
            if cid not in seen_ids:
                expanded_chunks.append(chunk)
                seen_ids.add(cid)

            # For clause or subsection chunks, fetch siblings from Qdrant
            chunk_type = chunk.get("type", "")
            h = chunk.get("hierarchy", {})
            section_key = (chunk.get("act_source", ""), h.get("ch"), h.get("sec"))

            if chunk_type in ("clause", "subsection") and section_key not in expanded_sections:
                expanded_sections.add(section_key)
                siblings = self._fetch_section_siblings(chunk, seen_ids)
                for sib in siblings:
                    sid = sib.get("chunk_id", "")
                    if sid not in seen_ids:
                        expanded_chunks.append(sib)
                        seen_ids.add(sid)

        # ── Section 1: Legal text chunks ─────────────────────────────
        if expanded_chunks:
            sections.append("═══ कानूनी पाठ सन्दर्भहरू (Legal Text Context) ═══\n")
            for i, chunk in enumerate(expanded_chunks, 1):
                act = chunk.get("act_source", "Unknown")
                h_dict = chunk.get("hierarchy", {})
                hierarchy = self._format_hierarchy(h_dict)
                citation = self._format_citation(act, h_dict)
                text = chunk.get("text", "")

                # Include provisos and explanations if present
                extra_parts = []
                for proviso in chunk.get("provisos", []):
                    extra_parts.append(f"  तर: {proviso}")
                for explanation in chunk.get("explanations", []):
                    extra_parts.append(f"  स्पष्टीकरण: {explanation}")
                extra_text = "\n".join(extra_parts)

                section = (
                    f"[सन्दर्भ {i}]\n"
                    f"  ऐन: {act}\n"
                    f"  स्थान: {hierarchy}\n"
                    f"  📌 उद्धृत गर्नुहोस्: {citation}\n"
                    f"  पाठ: {text}"
                )
                if extra_text:
                    section += f"\n{extra_text}"
                sections.append(section)

        # ── Section 2: Graph-derived procedural steps ────────────────
        if graph_context:
            sections.append("\n═══ प्रक्रियागत चरणहरू (Procedural Steps from Knowledge Graph) ═══\n")
            for proc in graph_context:
                process_name = proc["process_name"]
                steps = proc.get("steps", [])
                docs = proc.get("documents", [])
                offices = proc.get("offices", [])

                sections.append(f"प्रक्रिया: {process_name}")

                if steps:
                    for step in steps:
                        step_text = f"  चरण {step['step_number']}: {step['action']}"
                        if step.get("office"):
                            step_text += f"\n    कार्यालय: {step['office']}"
                        if step.get("documents") and any(step["documents"]):
                            valid_docs = [d for d in step["documents"] if d]
                            if valid_docs:
                                step_text += f"\n    कागजात: {', '.join(valid_docs)}"
                        if step.get("fees") and any(step["fees"]):
                            valid_fees = [str(f) for f in step["fees"] if f]
                            if valid_fees:
                                step_text += f"\n    दस्तुर: {', '.join(valid_fees)}"
                        if step.get("duration"):
                            step_text += f"\n    समय: {step['duration']}"
                        if step.get("prerequisite"):
                            step_text += f"\n    पूर्वशर्त: {step['prerequisite']}"
                        sections.append(step_text)

                if docs:
                    sections.append(
                        f"  सम्पूर्ण आवश्यक कागजातहरू: {', '.join(docs)}"
                    )
                if offices:
                    sections.append(
                        f"  सम्बन्धित कार्यालयहरू: {', '.join(offices)}"
                    )
                sections.append("")  # blank line between processes

        # ── Final assembly ───────────────────────────────────────────
        context_text = "\n".join(sections)
        return f"{context_text}\n\n═══ प्रश्न (Question) ═══\n{query}"

    # ------------------------------------------------------------------
    # Stage 4: Generate answer
    # ------------------------------------------------------------------

    def answer(
        self,
        query: str,
        top_k: Optional[int] = None,
        act_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the full RAG pipeline and return the generated answer with metadata.

        Returns
        -------
        dict
            ``answer``   : str — the generated response
            ``chunks``   : list — retrieved chunk context
            ``graph``    : list — graph-derived procedural steps
            ``context``  : str — the full assembled context sent to the LLM
        """
        # 1. Retrieve chunks
        chunks = self.retrieve_chunks(query, top_k=top_k, act_filter=act_filter)

        # 2. Retrieve graph context
        graph_context = []
        if self.use_graph and self.graph_store:
            graph_context = self.retrieve_graph_context(chunks, query)

        # 3. Assemble context
        context = self.assemble_context(query, chunks, graph_context)

        # 4. Generate answer
        response = self.llm_client.generate(RAG_SYSTEM_PROMPT, context)

        return {
            "answer": response,
            "chunks": chunks,
            "graph": graph_context,
            "context": context,
        }

    def answer_stream(
        self,
        query: str,
        top_k: Optional[int] = None,
        act_filter: Optional[str] = None,
    ) -> Iterator[str]:
        """
        Run the full RAG pipeline and stream the generated answer token-by-token.

        The retrieval stages run synchronously first, then generation is streamed.
        Callers can access the retrieval context via ``last_retrieval_context``
        after initiating the stream.
        """
        # 1. Retrieve chunks
        chunks = self.retrieve_chunks(query, top_k=top_k, act_filter=act_filter)

        # 2. Retrieve graph context
        graph_context = []
        if self.use_graph and self.graph_store:
            graph_context = self.retrieve_graph_context(chunks, query)

        # 3. Assemble context
        context = self.assemble_context(query, chunks, graph_context)

        # Store for post-stream inspection
        self.last_retrieval_context = {
            "chunks": chunks,
            "graph": graph_context,
            "context": context,
        }

        # 4. Stream the LLM response
        yield from self.llm_client.stream(RAG_SYSTEM_PROMPT, context)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close underlying connections (graph store)."""
        if self.graph_store:
            try:
                self.graph_store.close()
            except Exception:
                pass
