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
import math
from collections.abc import Callable, Iterator
from typing import Any

from src.config import get_settings
from src.embedding.embedder import LegalChunkEmbedder
from src.embedding.reranker import CrossEncoderReranker
from src.embedding.vector_store import QdrantVectorStore
from src.knowledge_base.graph_store import Neo4jGraphStore
from src.llm.base import BaseLLMClient
from src.llm.llm_client import (
    create_llm_client,
    handle_rate_limit_response,
)
from src.rag.verifier import VerificationLayer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Over-fetch multiplier when reranking is active
# ---------------------------------------------------------------------------
RERANK_FETCH_MULTIPLIER: int = 4
CONTEXT_OVERFETCH: int = 3       # Over-fetch to compensate for empty/short chunks
MIN_CHUNK_TEXT_LEN: int = 10     # Skip chunks shorter than this
RELEVANCE_THRESHOLD: float = 0.08  # Min reranker score when no graph steps match


def estimate_tokens(text: str) -> int:
    """Estimate LLM token count with script-aware weighting for Devanagari text."""
    if not text:
        return 0
    total = 0.0
    for ch in text:
        code = ord(ch)
        if 0x0900 <= code <= 0x097F:
            total += 0.9
        elif code > 0x007F:
            total += 0.8
        else:
            total += 0.25
    return math.ceil(total)


def _truncate_to_token_budget(text: str, max_tokens: int) -> str:
    """Truncate *text* cleanly so its estimated tokens fit within *max_tokens*."""
    if max_tokens <= 0 or not text:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = text[:mid].rstrip() + "…"
        if estimate_tokens(candidate) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + "…" if lo > 0 else ""


def _infer_response_language(query: str, response_language: str | None) -> str:
    """Resolve explicit *response_language* or infer ``'ne'``/``'en'`` from *query*."""
    if response_language:
        return response_language
    return "ne" if any(0x0900 <= ord(ch) <= 0x097F for ch in query) else "en"


# ---------------------------------------------------------------------------
# RAG System Prompt (condensed for free-tier TPM efficiency)
# ---------------------------------------------------------------------------
RAG_SYSTEM_PROMPT = """You are a grounded legal assistant for Nepali business registration and administrative procedures.
Answer ONLY from the provided Legal Provisions and Procedural Steps; do not use outside knowledge.

RULES:
1. Synthesis: Combine scattered Provisions (दफा/उपदफा) and Procedural Steps into a clear, actionable answer. If context only partially covers the question, answer what is supported with exact citations and note remaining gaps. Only state "उपलब्ध सन्दर्भमा यस प्रश्नको पर्याप्त जानकारी छैन।" when no relevant Provisions or Procedures exist.
2. Inline Citations (CRITICAL): Each Provision has a "📌 उद्धृत गर्नुहोस्" label. Cite ONLY that exact label inline right where the claim appears (e.g., "(कम्पनी ऐन, २०६३, दफा ३(१))")—never infer section/subsection numbers from cross-references inside the text body (e.g., "उपदफा (४) बमोजिम").
3. Procedures & Details: Use numbered steps and include required documents (आवश्यक कागजातहरू), offices (सम्बन्धित कार्यालय), fees (दस्तुर/शुल्क), and time limits (समय सीमा).
4. Language & Format: Reply concisely in the user's question language (Nepali or English), organized by process/act with bold key terms/offices/deadlines. Do NOT append a "स्रोत" or "Sources" list at the end—keep all citations inline."""


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
        max_context_tokens: int | None = None,
        max_siblings_per_section: int | None = None,
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
        self.max_context_tokens = (
            max_context_tokens
            if max_context_tokens is not None
            else int(getattr(settings, "MAX_CONTEXT_TOKENS", 3500))
        )
        self.max_siblings_per_section = (
            max_siblings_per_section
            if max_siblings_per_section is not None
            else int(getattr(settings, "MAX_SIBLINGS_PER_SECTION", 2))
        )

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

    def _format_provision_block(
        self,
        index: int,
        chunk: dict[str, Any],
        include_explanations: bool = True,
        text_override: str | None = None,
    ) -> str:
        """Format a single Provision entry with its canonical citation label."""
        act = chunk.get("act_source", "Unknown")
        h_dict = chunk.get("hierarchy", {})
        hierarchy = self._format_hierarchy(h_dict)
        citation = self._format_citation(act, h_dict)
        text = chunk.get("text", "") if text_override is None else text_override

        extra_parts = [f"  तर: {proviso}" for proviso in chunk.get("provisos", [])]
        if include_explanations:
            for explanation in chunk.get("explanations", []):
                extra_parts.append(f"  स्पष्टीकरण: {explanation}")

        block = (
            f"[सन्दर्भ {index}]\n"
            f"  ऐन: {act}\n"
            f"  स्थान: {hierarchy}\n"
            f"  📌 उद्धृत गर्नुहोस्: {citation}\n"
            f"  पाठ: {text}"
        )
        if extra_parts:
            block += "\n" + "\n".join(extra_parts)
        return block

    def _format_procedure_block(self, proc: dict[str, Any]) -> str:
        """Format a Procedure entry compactly, avoiding duplicate step rollups."""
        lines = [f"प्रक्रिया: {proc['process_name']}"]
        steps = proc.get("steps", [])
        step_docs: set[str] = set()
        step_offices: set[str] = set()

        for step in steps:
            step_text = f"  चरण {step['step_number']}: {step['action']}"
            office = step.get("office")
            if office:
                step_offices.add(str(office))
                step_text += f"\n    कार्यालय: {office}"
            if step.get("documents") and any(step["documents"]):
                valid_docs = [str(d) for d in step["documents"] if d]
                if valid_docs:
                    step_docs.update(valid_docs)
                    step_text += f"\n    कागजात: {', '.join(valid_docs)}"
            if step.get("fees") and any(step["fees"]):
                valid_fees = [str(f) for f in step["fees"] if f]
                if valid_fees:
                    step_text += f"\n    दस्तुर: {', '.join(valid_fees)}"
            if step.get("duration"):
                step_text += f"\n    समय: {step['duration']}"
            if step.get("prerequisite"):
                step_text += f"\n    पूर्वशर्त: {step['prerequisite']}"
            lines.append(step_text)

        missing_docs = [str(d) for d in proc.get("documents", []) if d and str(d) not in step_docs]
        if missing_docs:
            lines.append(f"  सम्पूर्ण आवश्यक कागजातहरू: {', '.join(missing_docs)}")
        missing_offices = [
            str(o) for o in proc.get("offices", []) if o and str(o) not in step_offices
        ]
        if missing_offices:
            lines.append(f"  सम्बन्धित कार्यालयहरू: {', '.join(missing_offices)}")
        return "\n".join(lines)

    def format_citations(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Deduplicate retrieved Provisions and format readable citation metadata for UI pills."""
        formatted: list[dict[str, Any]] = []
        seen: set[tuple[Any, Any, Any, Any]] = set()
        for chunk in chunks:
            act = str(chunk.get("act_source", "") or "Unknown")
            h = chunk.get("hierarchy") or {}
            key = (act, h.get("sec"), h.get("sub"), h.get("clause"))
            if key in seen:
                continue
            seen.add(key)
            excerpt = " ".join(str(chunk.get("text", "")).split())[:280]
            formatted.append(
                {
                    "label": self._format_citation(act, h),
                    "act_source": act,
                    "hierarchy": self._format_hierarchy(h),
                    "excerpt": excerpt,
                    "status": str(chunk.get("status", "active") or "active"),
                }
            )
        return formatted

    def _pack_primary_provisions(
        self,
        provisions: list[dict[str, Any]],
        prov_budget: int,
        header_cost: int,
    ) -> tuple[list[str], list[dict[str, Any]], set[str], int]:
        """Pack primary Provisions in rank order within *prov_budget*."""
        packed: list[str] = []
        primary: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        used = header_cost if provisions else 0

        for provision in provisions:
            pid = provision.get("chunk_id", "")
            if pid and pid in seen_ids:
                continue
            if pid:
                seen_ids.add(pid)

            idx = len(packed) + 1
            block = self._format_provision_block(idx, provision, include_explanations=True)
            cost = estimate_tokens(block)
            if used + cost > prov_budget:
                compact = self._format_provision_block(idx, provision, include_explanations=False)
                compact_cost = estimate_tokens(compact)
                if used + compact_cost <= prov_budget:
                    block, cost = compact, compact_cost
                elif not packed:
                    shell = self._format_provision_block(
                        idx, provision, include_explanations=False, text_override=""
                    )
                    avail = max(0, prov_budget - used - estimate_tokens(shell))
                    trimmed = _truncate_to_token_budget(provision.get("text", ""), avail)
                    block = self._format_provision_block(
                        idx, provision, include_explanations=False, text_override=trimmed
                    )
                    cost = estimate_tokens(block)
                else:
                    break

            packed.append(block)
            primary.append(provision)
            used += cost

        return packed, primary, seen_ids, used

    def _append_section_siblings(
        self,
        primary_provisions: list[dict[str, Any]],
        packed_provisions: list[str],
        seen_ids: set[str],
        prov_budget: int,
        used_prov_tokens: int,
    ) -> int:
        """Append Section siblings only while spare budget remains, capped per Section."""
        if self.max_siblings_per_section <= 0 or not primary_provisions:
            return used_prov_tokens

        expanded_sections: set[tuple[Any, Any, Any]] = set()
        for provision in primary_provisions:
            if prov_budget - used_prov_tokens <= 40:
                break
            prov_type = provision.get("type", "")
            h = provision.get("hierarchy", {})
            section_key = (provision.get("act_source", ""), h.get("ch"), h.get("sec"))
            if prov_type not in ("clause", "subsection") or section_key in expanded_sections:
                continue
            expanded_sections.add(section_key)
            siblings = self._fetch_section_siblings(provision, seen_ids)
            for sib in siblings[: self.max_siblings_per_section]:
                sid = sib.get("chunk_id", "")
                if sid:
                    if sid in seen_ids:
                        continue
                    seen_ids.add(sid)
                idx = len(packed_provisions) + 1
                sib_block = self._format_provision_block(idx, sib, include_explanations=False)
                sib_cost = estimate_tokens(sib_block)
                if used_prov_tokens + sib_cost > prov_budget:
                    break
                packed_provisions.append(sib_block)
                used_prov_tokens += sib_cost

        return used_prov_tokens

    def _pack_procedures(
        self,
        proc_blocks: list[str],
        proc_header: str,
        budget: int,
        used_total: int,
    ) -> list[str]:
        """Append formatted Procedures within the remaining token *budget*."""
        out: list[str] = []
        proc_added = False
        for block in proc_blocks:
            header_cost = estimate_tokens(proc_header) if not proc_added else 0
            block_cost = estimate_tokens(block)
            if used_total + header_cost + block_cost > budget:
                if proc_added:
                    break
                avail = max(0, budget - used_total - header_cost)
                if avail <= 0:
                    break
                trimmed = _truncate_to_token_budget(block, avail)
                if not trimmed:
                    break
                block = trimmed
                block_cost = estimate_tokens(block)
            if not proc_added:
                out.append(proc_header)
                used_total += header_cost
                proc_added = True
            out.append(block)
            out.append("")
            used_total += block_cost
        return out

    def _pack_history(
        self,
        conversation_history: list[dict[str, str]],
        hist_header: str,
        hist_budget: int,
    ) -> tuple[list[str], int]:
        """Pack newest-first conversation turns within *hist_budget*, keeping chronological order."""
        if not conversation_history or hist_budget <= 0:
            return [], 0
        header_cost = estimate_tokens(hist_header)
        if header_cost >= hist_budget:
            return [], 0
        used = header_cost
        kept_reversed: list[str] = []
        for turn in reversed(conversation_history[-4:]):
            role = "User" if turn.get("role") == "user" else "Assistant"
            text = " ".join(str(turn.get("content", "")).split())[:300]
            if not text:
                continue
            line = f"- {role}: {text}"
            cost = estimate_tokens(line)
            if used + cost > hist_budget:
                break
            kept_reversed.append(line)
            used += cost
        if not kept_reversed:
            return [], 0
        return [hist_header, *reversed(kept_reversed)], used

    def assemble_context(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        graph_context: list[dict[str, Any]],
        response_language: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Build a token-budgeted context string for the LLM.

        Packs directly retrieved Provisions and Procedures first within
        ``max_context_tokens``, optionally includes a bounded sliding window of
        recent conversation turns, and appends Section siblings only while spare
        budget remains (capped at ``max_siblings_per_section`` per Section).
        """
        lang_hint = ""
        if response_language == "en":
            lang_hint = "\n[Instruction: Respond in English.]"
        elif response_language == "roman":
            lang_hint = "\n[Instruction: Respond in clear Romanized Nepali (or English if technical legal terms are clearer).]"
        footer = f"\n\n═══ प्रश्न (Question) ═══\n{query}{lang_hint}"
        budget = max(0, self.max_context_tokens - estimate_tokens(footer))

        hist_header = "\n═══ अघिल्लो संवाद (Recent Conversation) ═══"
        hist_lines: list[str] = []
        hist_reserve = 0
        if conversation_history and budget > 0:
            hist_lines, hist_reserve = self._pack_history(
                conversation_history, hist_header, budget // 4
            )

        content_budget = max(0, budget - hist_reserve)
        proc_blocks = [self._format_procedure_block(proc) for proc in graph_context]
        proc_header = "\n═══ प्रक्रियागत चरणहरू (Procedural Steps from Knowledge Graph) ═══\n"
        proc_reserve = 0
        if proc_blocks and content_budget > 0:
            total_proc_tokens = estimate_tokens(proc_header + "\n\n".join(proc_blocks))
            proc_reserve = min(total_proc_tokens, content_budget // 2)

        prov_budget = max(0, content_budget - proc_reserve)
        prov_header = "═══ कानूनी पाठ सन्दर्भहरू (Legal Text Context) ═══\n"
        packed_provisions, primary_provisions, seen_ids, used_prov_tokens = (
            self._pack_primary_provisions(chunks, prov_budget, estimate_tokens(prov_header))
        )
        used_prov_tokens = self._append_section_siblings(
            primary_provisions, packed_provisions, seen_ids, prov_budget, used_prov_tokens
        )

        sections: list[str] = []
        used_total = 0
        if packed_provisions:
            sections.append(prov_header)
            sections.extend(packed_provisions)
            used_total = used_prov_tokens

        if proc_blocks:
            sections.extend(
                self._pack_procedures(proc_blocks, proc_header, content_budget, used_total)
            )

        if hist_lines:
            sections.extend(hist_lines)

        context_text = "\n".join(sections)
        return f"{context_text}{footer}"

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
        rate_msg = handle_rate_limit_response(
            response, _infer_response_language(query, response_language), "RAG generation"
        )
        if rate_msg is not None:
            return {
                "answer": rate_msg,
                "chunks": chunks,
                "graph": graph_context,
                "context": context,
                "verification": None,
            }

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
        it = iter(self.llm_client.stream(RAG_SYSTEM_PROMPT, context))
        first = next(it, None)
        if first is None:
            return
        rate_msg = handle_rate_limit_response(
            first, _infer_response_language(query, response_language), "RAG streaming"
        )
        if rate_msg is not None:
            yield rate_msg
            return
        yield first
        yield from it

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
