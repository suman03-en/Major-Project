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
from collections.abc import Callable, Iterator
from typing import Any

from src.config import get_settings
from src.embedding.embedder import LegalChunkEmbedder
from src.embedding.reranker import CrossEncoderReranker
from src.embedding.vector_store import QdrantVectorStore
from src.knowledge_base.graph_store import Neo4jGraphStore
from src.llm.base import BaseLLMClient
from src.llm.llm_client import create_llm_client
from src.rag.verifier import VerificationLayer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Over-fetch multiplier when reranking is active
# ---------------------------------------------------------------------------
RERANK_FETCH_MULTIPLIER: int = 4
CONTEXT_OVERFETCH: int = 3       # Over-fetch to compensate for empty/short chunks
MIN_CHUNK_TEXT_LEN: int = 10     # Skip chunks shorter than this
RELEVANCE_THRESHOLD: float = 0.08  # Min reranker score when no graph steps match

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
2. SYNTHESIS RULE (IMPORTANT):
   - Legal acts rarely state a complete "how-to" guide in one single paragraph; instead, steps, eligibility, required documents, offices, and fees are spread across multiple sections (दफा/उपदफा) and procedural nodes.
   - Combine and synthesize all relevant details from the provided chunks and graph steps into a clear, actionable answer.
   - If the retrieved context only partially covers the question after retrieval completes, provide a grounded partial answer with exact citations for what is supported, and explicitly note any remaining uncovered gaps or missing details.
   - Only state "उपलब्ध सन्दर्भमा यस प्रश्नको पर्याप्त जानकारी छैन।" if the provided context truly contains no relevant provisions or steps for the question.
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
    llm_client : BaseLLMClient | None
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
        embedder: LegalChunkEmbedder | None = None,
        vector_store: QdrantVectorStore | None = None,
        graph_store: Neo4jGraphStore | None = None,
        llm_client: BaseLLMClient | None = None,
        reranker: CrossEncoderReranker | None = None,
        use_graph: bool = True,
        use_reranker: bool = False,
        top_k: int = 5,
        search_type: str = "hybrid",
        verifier: VerificationLayer | None = None,
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
        self.graph_store: Neo4jGraphStore | None = None
        if use_graph:
            if graph_store is not None:
                self.graph_store = graph_store
            else:
                try:
                    self.graph_store = Neo4jGraphStore(
                        settings.NEO4J_URI, settings.NEO4J_USER, settings.NEO4J_PASSWORD
                    )
                    logger.info("Connected to Neo4j graph store.")
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Could not connect to Neo4j (%s). "
                        "RAG will work without graph context.", e
                    )
                    self.graph_store = None

        # --- LLM ---
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            self.llm_client = create_llm_client()

        self.top_k = top_k
        self.search_type = search_type
        self.verifier = verifier or VerificationLayer()

    # ------------------------------------------------------------------
    # Stage 1: Retrieve relevant chunks from Qdrant
    # ------------------------------------------------------------------

    def retrieve_chunks(
        self,
        query: str,
        top_k: int | None = None,
        act_filter: str | None = None,
    ) -> list[dict[str, Any]]:
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

    def search_clauses(
        self,
        query: str,
        top_k: int = 8,
        act_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search statutory provisions and procedural clauses in Qdrant."""
        return self.retrieve_chunks(query, top_k=top_k, act_filter=act_filter)

    # ------------------------------------------------------------------
    # Stage 2: Retrieve structured steps from Neo4j graph
    # ------------------------------------------------------------------

    def retrieve_graph_context(
        self,
        chunks: list[dict[str, Any]],
        query: str,
    ) -> list[dict[str, Any]]:
        """
        Query Neo4j for procedural steps related to the retrieved chunks.

        Looks up process names from the graph that match act sources found
        in the retrieved chunks, then fetches full step details.
        """
        if not self.graph_store:
            return []

        try:
            graph_results = self.graph_store.find_relevant_processes(
                query=query, chunks=chunks
            )
            logger.info(
                "Graph context: %d matching processes found for query.",
                len(graph_results),
            )
            return graph_results

        except Exception as e:  # noqa: BLE001
            logger.warning("Graph retrieval failed: %s", e)
            return []

    def search_graph(
        self,
        query: str,
        chunks: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Search Neo4j procedural graph for Procedures matching *query* and *chunks*."""
        return self.retrieve_graph_context(chunks or [], query)

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

    def _points_to_chunks(
        self,
        points: list[Any],
        seen_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Convert Qdrant scroll points into normalized provision dicts."""
        chunks: list[dict[str, Any]] = []
        for pt in points:
            payload = pt.payload or {}
            cid = payload.get("chunk_id", "")
            text = payload.get("text", "")
            if (seen_ids is not None and cid in seen_ids) or len(text.strip()) < MIN_CHUNK_TEXT_LEN:
                continue
            chunks.append({
                "chunk_id": cid,
                "act_source": payload.get("act_source", ""),
                "type": payload.get("type", ""),
                "hierarchy": payload.get("hierarchy", {}),
                "text": text,
                "stats": payload.get("stats", {}),
                "provisos": payload.get("provisos", []),
                "explanations": payload.get("explanations", []),
                "status": payload.get("status"),
            })
        return chunks

    def fetch_section(
        self,
        sec_num: int,
        act_source: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch all provisions for a given section number from Qdrant by scrolling
        ``hierarchy.sec == sec_num`` (without requiring ``hierarchy.ch``).
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        must = [FieldCondition(key="hierarchy.sec", match=MatchValue(value=sec_num))]
        if act_source:
            must.append(FieldCondition(key="act_source", match=MatchValue(value=act_source)))

        try:
            points, _ = self.vector_store.client.scroll(
                collection_name=self.vector_store.collection_name,
                scroll_filter=Filter(must=must),
                limit=30,
                with_payload=True,
                with_vectors=False,
            )
            return self._points_to_chunks(points)
        except Exception as e:  # noqa: BLE001
            logger.warning("fetch_section(%s) failed: %s", sec_num, e)
            return []

    def _fetch_section_siblings(
        self,
        chunk: dict[str, Any],
        seen_ids: set,
    ) -> list[dict[str, Any]]:
        """
        Fetch sibling chunks from Qdrant that share the same act + chapter +
        section as the given chunk. This pulls in ALL clauses/subsections
        under the same section (e.g., all penalty tiers under दफा ४३).

        Uses Qdrant scroll with payload filtering — no embedding needed.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

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
            return self._points_to_chunks(points, seen_ids=seen_ids)

        except Exception as e:  # noqa: BLE001
            logger.warning("Sibling fetch failed: %s", e)
            return []

    def assemble_context(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        graph_context: list[dict[str, Any]],
        response_language: str | None = None,
    ) -> str:
        """
        Build the full context string that gets sent to the LLM along with
        the user's question.

        For clause-level and subsection-level chunks, sibling chunks from the
        same section are fetched from Qdrant to give the LLM full context
        (e.g., all penalty tiers when one clause is retrieved).
        """
        sections: list[str] = []

        # Collect all sibling chunks to provide parent/sibling context
        expanded_chunks: list[dict[str, Any]] = []
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
        lang_hint = ""
        if response_language == "en":
            lang_hint = "\n[Instruction: Respond in English.]"
        elif response_language == "roman":
            lang_hint = "\n[Instruction: Respond in clear Romanized Nepali (or English if technical legal terms are clearer).]"
        return f"{context_text}\n\n═══ प्रश्न (Question) ═══\n{query}{lang_hint}"

    # ------------------------------------------------------------------
    # Stage 4: Generate answer
    # ------------------------------------------------------------------

    def answer(
        self,
        query: str,
        top_k: int | None = None,
        act_filter: str | None = None,
        retrieval_query: str | None = None,
        response_language: str | None = None,
        on_step: Callable[[str, str], None] | None = None,
    ) -> dict[str, Any]:
        """
        Run the full RAG pipeline and return the generated answer with metadata.

        Parameters
        ----------
        query:
            Original user question (used in the LLM prompt so the response
            matches the user's language).
        retrieval_query:
            Optional normalized/translated query (e.g. Devanagari Nepali) used
            for Qdrant + Neo4j retrieval.  Defaults to *query*.
        response_language:
            Optional hint (``"ne"``, ``"en"``, ``"roman"``) appended to the
            prompt when *query* was translated.
        on_step:
            Optional ``(stage, message)`` callback for live progress updates.

        Returns
        -------
        dict
            ``answer``   : str — the generated response
            ``chunks``   : list — retrieved chunk context
            ``graph``    : list — graph-derived procedural steps
            ``context``  : str — the full assembled context sent to the LLM
        """
        def _emit(stage: str, msg: str) -> None:
            if on_step:
                on_step(stage, msg)

        search_query = retrieval_query or query

        # 1. Retrieve chunks
        _emit("retrieve_vector", "Searching legal clauses in Qdrant…")
        chunks = self.retrieve_chunks(search_query, top_k=top_k, act_filter=act_filter)
        _emit("retrieve_vector_done", f"Retrieved {len(chunks)} relevant legal chunk(s)")

        # 2. Retrieve graph context before gating so procedural graph matches
        #    are never discarded just because vector chunks scored modestly.
        graph_context = []
        if self.use_graph and self.graph_store:
            _emit("retrieve_graph", "Checking Neo4j knowledge graph for procedures…")
            graph_context = self.retrieve_graph_context(chunks, search_query)
            _emit("retrieve_graph_done", f"Found {len(graph_context)} matching procedure(s)")

        # 2b. Relevance gate — only short-circuit when BOTH vector reranker score
        #     is below threshold AND no graph procedures matched.
        if self.reranker and chunks and not graph_context:
            top_score = chunks[0].get("score", 1.0)
            if top_score < RELEVANCE_THRESHOLD:
                logger.info(
                    "Relevance gate: top score %.3f < %.3f and no graph match — skipping context for query: %s",
                    top_score,
                    RELEVANCE_THRESHOLD,
                    query[:60],
                )
                no_ctx_response = (
                    "उपलब्ध कानूनी सन्दर्भमा यस प्रश्नको पर्याप्त जानकारी फेला परेन।"
                    if response_language == "ne"
                    else "I could not find sufficient relevant information in the available Nepali legal documents for this question."
                )
                return {
                    "answer": no_ctx_response,
                    "chunks": [],
                    "graph": [],
                    "context": "",
                }

        # 3. Exclude inactive provisions from context assembly (still verified below)
        active_chunks = self.verifier.filter_active_provisions(chunks)
        context = self.assemble_context(
            query, active_chunks, graph_context, response_language=response_language
        )

        # 4. Generate answer
        _emit("generate", "Generating grounded legal answer…")
        response = self.llm_client.generate(RAG_SYSTEM_PROMPT, context)

        # 5. Deterministic verification (0 LLM calls)
        _emit("verify", "Verifying citations and temporal status…")
        verification = self.verifier.verify(
            response, chunks=chunks, graph=graph_context
        )

        return {
            "answer": verification.answer,
            "chunks": chunks,
            "graph": graph_context,
            "context": context,
            "verification": verification,
        }

    def answer_stream(
        self,
        query: str,
        top_k: int | None = None,
        act_filter: str | None = None,
        retrieval_query: str | None = None,
        response_language: str | None = None,
    ) -> Iterator[str]:
        """
        Run the full RAG pipeline and stream the generated answer token-by-token.

        The retrieval stages run synchronously first, then generation is streamed.
        Callers can access the retrieval context via ``last_retrieval_context``
        after initiating the stream.
        """
        search_query = retrieval_query or query

        # 1. Retrieve chunks
        chunks = self.retrieve_chunks(search_query, top_k=top_k, act_filter=act_filter)

        # 2. Retrieve graph context
        graph_context = []
        if self.use_graph and self.graph_store:
            graph_context = self.retrieve_graph_context(chunks, search_query)

        # 3. Exclude inactive provisions from context assembly (still verified below)
        active_chunks = self.verifier.filter_active_provisions(chunks)
        context = self.assemble_context(
            query, active_chunks, graph_context, response_language=response_language
        )

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
            except Exception as exc:  # noqa: BLE001
                logger.debug("Graph store close failed: %s", exc)
