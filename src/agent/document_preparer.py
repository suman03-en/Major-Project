"""
Document Preparer
=================
Builds document checklists and LLM-generated draft templates for the user
based on the documents required by the active Neo4j procedure + relevant
legal clauses from Qdrant.

Usage::

    from src.agent.document_preparer import DocumentPreparer

    preparer = DocumentPreparer(rag_pipeline, graph_store, llm_client)
    result = preparer.prepare("मलाई निवेदन फारम चाहिन्छ", session)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.agent.session_store import SessionState
    from src.knowledge_base.graph_store import Neo4jGraphStore
    from src.llm.base import BaseLLMClient
    from src.rag.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

_DOC_PREP_PROMPT = """You are a Nepali legal document assistant.

The user needs help preparing documents for the following legal process.

{process_context}

{legal_context}

Your task:
1. List ALL required documents as a numbered checklist.
2. For each document, describe:
   - What it is (1-2 sentences)
   - Where to obtain it (office, online portal, etc.)
   - Any specific requirements (notarisation, translation, etc.)
3. If a document template can be drafted (e.g., an application letter), provide a skeleton template.
4. Answer in the SAME LANGUAGE as the user's question.

User request: {query}"""


class DocumentPreparer:
    """
    Generates document checklists and draft templates.

    Parameters
    ----------
    rag_pipeline:
        Live RAGPipeline instance (for Qdrant retrieval).
    graph_store:
        Connected Neo4j store (for process document lists).
    llm_client:
        LLM client for generating the response text.
    """

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        graph_store: Neo4jGraphStore | None,
        llm_client: BaseLLMClient,
    ) -> None:
        self._rag = rag_pipeline
        self._graph = graph_store
        self._llm = llm_client

    def prepare(
        self,
        query: str,
        session: SessionState,
        retrieval_query: str | None = None,
    ) -> dict[str, Any]:
        """
        Build a document checklist response for *query* given the session context.

        Returns
        -------
        dict
            ``answer``    — LLM-generated document checklist text
            ``documents`` — list of document names from the graph
            ``chunks``    — RAG chunks used for legal grounding
        """
        search_query = retrieval_query or query

        # 1. Graph documents (from active procedure or query match)
        graph_docs: list[str] = []
        process_context = ""
        if session.has_active_process and self._graph:
            try:
                graph_docs = self._graph.get_documents_for_process(
                    session.active_process  # type: ignore[arg-type]
                )
                offices = self._graph.get_offices_for_process(
                    session.active_process  # type: ignore[arg-type]
                )
                process_context = (
                    f"Process: {session.active_process}\n"
                    f"Required documents: {', '.join(graph_docs) if graph_docs else 'N/A'}\n"
                    f"Offices involved: {', '.join(offices) if offices else 'N/A'}"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Graph doc fetch failed: %s", exc)

        # 2. RAG chunks for legal grounding
        chunks: list[dict[str, Any]] = []
        legal_context = ""
        try:
            chunks = self._rag.retrieve_chunks(search_query, top_k=3)
            if chunks:
                legal_context = "Relevant legal provisions:\n" + "\n".join(
                    f"- {c.get('text', '')[:200]}" for c in chunks
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("RAG chunk retrieval failed: %s", exc)

        # 3. Build LLM prompt
        prompt = _DOC_PREP_PROMPT.format(
            process_context=process_context or "No active procedure selected.",
            legal_context=legal_context or "No specific legal provisions retrieved.",
            query=query,
        )

        # 4. Generate
        answer = self._llm.generate(
            system_prompt="You are a Nepali legal document assistant.",
            user_message=prompt,
        )

        return {
            "answer": answer,
            "documents": graph_docs,
            "chunks": chunks,
        }
