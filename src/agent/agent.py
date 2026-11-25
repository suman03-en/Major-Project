"""
Legal Agent Orchestrator
========================
Top-level agent that routes each user query to the right subsystem based on
:class:`~src.agent.intent_classifier.IntentType`, maintains session state, and
returns a structured :class:`AgentResponse`.

This is the single entry point for ``web/main.py`` and the CLI.

Usage::

    from src.agent import LegalAgent
    from src.agent.session_store import InMemorySessionStore

    store = InMemorySessionStore()
    agent = LegalAgent(
        rag_pipeline=rag,
        graph_store=graph,
        llm_client=llm,
        session_store=store,
    )
    response = agent.respond("कम्पनी दर्ता कसरी गर्ने?", session_id="abc-123")
    print(response.answer)
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.agent.intent_classifier import IntentClassifier, IntentType
from src.agent.procedure_navigator import NavigationResult, ProcedureNavigator
from src.agent.session_store import InMemorySessionStore, SessionState
from src.agent.translator import QueryTranslator, detect_language, is_romanized_nepali

if TYPE_CHECKING:
    from src.agent.session_store import SessionStore
    from src.knowledge_base.graph_store import Neo4jGraphStore
    from src.llm.base import BaseLLMClient
    from src.rag.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canned greeting messages (no LLM or retrieval needed)
# ---------------------------------------------------------------------------

_GREETING_NE = (
    "नमस्ते! 🙏 म नेपाली कानूनी सहायक हुँ।\n\n"
    "म तपाईंलाई यी विषयमा मद्दत गर्न सक्छु:\n"
    "• **कम्पनी दर्ता** जस्ता व्यापारिक प्रक्रियाहरू (चरण-दर-चरण)\n"
    "• **कानूनी धाराहरू** (ऐन, नियमावली, उपदफा)\n"
    "• **आवश्यक कागजातहरू** को सूची र टेम्पलेट\n\n"
    "कुन विषयमा जान्नु हुन्छ?"
)

_GREETING_EN = (
    "Hello! 👋 I'm your Nepali Legal AI assistant.\n\n"
    "I can help you with:\n"
    "• **Business registration procedures** (step-by-step guidance)\n"
    "• **Legal provisions** from Nepali Acts and Regulations\n"
    "• **Document checklists** and draft templates\n\n"
    "What would you like to know?"
)


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class AgentResponse:
    """
    Structured response returned by :meth:`LegalAgent.respond`.

    Attributes
    ----------
    answer:
        Human-readable text reply (Markdown).
    intent:
        Classified intent of the user's query.
    session_id:
        Session identifier (echo-back so the client can store it).
    navigation:
        Populated for ``PROCEDURAL`` / ``FOLLOW_UP`` / ``HYBRID`` intents.
    chunks:
        Retrieved RAG chunks (for ``LEGAL_FACT`` / ``HYBRID`` / ``DOCUMENT_PREP``).
    graph:
        Graph-derived procedural steps (for ``HYBRID``).
    """

    answer: str
    intent: IntentType
    session_id: str
    navigation: NavigationResult | None = None
    chunks: list[dict[str, Any]] = field(default_factory=list)
    graph: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class LegalAgent:
    """
    Orchestrates intent classification, session management, and subsystem routing.

    Parameters
    ----------
    rag_pipeline:
        A fully initialised :class:`~src.rag.rag_pipeline.RAGPipeline`.
    graph_store:
        Connected Neo4j graph store (``None`` disables procedural navigation
        and document graph lookups).
    llm_client:
        LLM client used by the document preparer sub-module.
    session_store:
        Session backend.  Defaults to :class:`~src.agent.session_store.InMemorySessionStore`.
    """

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        graph_store: Neo4jGraphStore | None = None,
        llm_client: BaseLLMClient | None = None,
        session_store: SessionStore | None = None,
        translator: QueryTranslator | None = None,
        translate_backoff_sec: float = 1.0,
    ) -> None:
        self._rag = rag_pipeline
        self._graph = graph_store
        self._llm = llm_client or rag_pipeline.llm_client
        self._store: SessionStore = session_store or InMemorySessionStore()

        self._classifier = IntentClassifier()
        self._translator = translator or QueryTranslator(self._llm)
        self._translate_backoff_sec = translate_backoff_sec
        self._navigator = ProcedureNavigator(graph_store) if graph_store else None

        # Lazy import to avoid circular imports at module load time
        from src.agent.document_preparer import DocumentPreparer

        self._doc_preparer = DocumentPreparer(
            rag_pipeline=rag_pipeline,
            graph_store=graph_store,
            llm_client=self._llm,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def respond(
        self,
        query: str,
        session_id: str = "",
        on_step: Callable[[str, str], None] | None = None,
    ) -> AgentResponse:
        """
        Process *query* and return an :class:`AgentResponse`.

        Parameters
        ----------
        query:
            Raw user input (Nepali or English).
        session_id:
            UUID string from the client.  An empty string creates a new session.
        on_step:
            Optional ``(stage, message)`` callback invoked as the agent moves
            through classification, translation, retrieval, and generation.
        """
        def _emit(stage: str, msg: str) -> None:
            if on_step:
                on_step(stage, msg)

        # 1. Session
        session = self._store.get_or_create(session_id or str(uuid.uuid4()))

        # 2. Fast-path check for greetings and active follow-ups (0 LLM calls)
        _emit("intent", "Analyzing query intent…")
        lang = detect_language(query)
        intent = self._classifier.classify(
            query,
            has_active_session=session.has_active_process,
        )

        retrieval_query = query
        # Only translate Romanized Nepali -> Devanagari; BGE-M3 handles English & Devanagari natively
        if (
            intent not in (IntentType.GREETING, IntentType.FOLLOW_UP)
            and lang != "ne"
            and is_romanized_nepali(query)
        ):
            _emit("translate", "Translating Romanized Nepali query to Nepali…")
            tr = self._translator.translate(query)
            retrieval_query = tr.retrieval_query
            if self._translate_backoff_sec > 0:
                _emit("backoff", "Waiting 1s before next step…")
                time.sleep(self._translate_backoff_sec)
            lang = tr.language
            if tr.was_translated:
                _emit("translate_done", f"Translated ({lang} → ne): {retrieval_query}")
                # Re-classify on translated Devanagari text if original was UNKNOWN
                translated_intent = self._classifier.classify(
                    retrieval_query,
                    has_active_session=session.has_active_process,
                )
                if intent == IntentType.UNKNOWN:
                    intent = translated_intent

        _emit("route", f"Intent: {intent.value} (lang={lang})")
        session.push_intent(intent)

        logger.info(
            "Agent: session=%s  lang=%s  intent=%s  query=%s",
            session.session_id,
            lang,
            intent.value,
            query[:60],
        )

        # 3. Route
        response = self._route(
            query=query,
            retrieval_query=retrieval_query,
            intent=intent,
            session=session,
            lang=lang,
            on_step=on_step,
        )

        # 4. Persist session
        self._store.save(session)
        return response

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route(
        self,
        query: str,
        retrieval_query: str,
        intent: IntentType,
        session: SessionState,
        lang: str,
        on_step: Callable[[str, str], None] | None = None,
    ) -> AgentResponse:
        """Dispatch to the appropriate subsystem based on *intent*."""
        sid = session.session_id
        step_lang = "en" if lang == "en" else "ne"

        if intent == IntentType.GREETING:
            return AgentResponse(
                answer=_GREETING_NE if lang == "ne" else _GREETING_EN,
                intent=intent,
                session_id=sid,
            )

        if intent == IntentType.FOLLOW_UP:
            return self._handle_follow_up(query, session, step_lang, sid, on_step=on_step)

        if intent == IntentType.PROCEDURAL:
            return self._handle_procedural(
                query, retrieval_query, session, step_lang, lang, sid, on_step=on_step
            )

        if intent == IntentType.DOCUMENT_PREP:
            return self._handle_document_prep(
                query, retrieval_query, session, sid, on_step=on_step
            )

        if intent in (IntentType.LEGAL_FACT, IntentType.UNKNOWN):
            return self._handle_legal_fact(
                query, retrieval_query, session, lang, sid, on_step=on_step
            )

        if intent == IntentType.HYBRID:
            return self._handle_hybrid(
                query, retrieval_query, session, step_lang, lang, sid, on_step=on_step
            )

        # Safety fallback
        return self._handle_legal_fact(
            query, retrieval_query, session, lang, sid, on_step=on_step
        )

    # ------------------------------------------------------------------
    # Intent handlers
    # ------------------------------------------------------------------

    def _handle_follow_up(
        self,
        query: str,
        session: SessionState,
        lang: str,
        sid: str,
        on_step: Callable[[str, str], None] | None = None,
    ) -> AgentResponse:
        """Advance the current procedure by one step."""
        if self._navigator is None or not session.has_active_process:
            # No active procedure → fall back to RAG
            return self._handle_legal_fact(query, query, session, lang, sid, on_step=on_step)

        # Check if user said "done" / "सकियो" → reset navigator
        done_kw = {"done", "सकियो", "sakiyo", "complete", "finish", "reset", "cancel"}
        if any(w in query.lower() for w in done_kw):
            self._navigator.reset(session)
            msg = (
                "ठीक छ! प्रक्रिया रद्द गरियो। के थप सहायता चाहिनुहुन्छ?"
                if lang == "ne"
                else "Procedure cancelled. How else can I help you?"
            )
            return AgentResponse(answer=msg, intent=IntentType.FOLLOW_UP, session_id=sid)

        if on_step:
            on_step("navigate", "Advancing to next procedure step in Neo4j…")
        nav_result = self._navigator.navigate(query, session, advance=True)
        if nav_result is None:
            return self._handle_legal_fact(query, query, session, lang, sid, on_step=on_step)

        return AgentResponse(
            answer=ProcedureNavigator.format_step_text(nav_result, lang),
            intent=IntentType.FOLLOW_UP,
            session_id=sid,
            navigation=nav_result,
        )

    def _handle_procedural(
        self,
        query: str,
        retrieval_query: str,
        session: SessionState,
        step_lang: str,
        lang: str,
        sid: str,
        on_step: Callable[[str, str], None] | None = None,
    ) -> AgentResponse:
        """Start or continue a procedure walk-through."""
        if self._navigator is None:
            # No graph → fall back to RAG for procedural text
            return self._handle_legal_fact(
                query, retrieval_query, session, lang, sid, on_step=on_step
            )

        if on_step:
            on_step("navigate", "Looking up matching procedure in Neo4j graph…")
        nav_result = self._navigator.navigate(retrieval_query, session, advance=False)
        if nav_result is None:
            # Graph has no matching process → fall back to RAG
            if on_step:
                on_step("navigate_miss", "No exact graph procedure found — falling back to RAG…")
            return self._handle_legal_fact(
                query, retrieval_query, session, lang, sid, on_step=on_step
            )

        return AgentResponse(
            answer=ProcedureNavigator.format_step_text(nav_result, step_lang),
            intent=IntentType.PROCEDURAL,
            session_id=sid,
            navigation=nav_result,
        )

    def _handle_legal_fact(
        self,
        query: str,
        retrieval_query: str,
        session: SessionState,
        lang: str,
        sid: str,
        on_step: Callable[[str, str], None] | None = None,
    ) -> AgentResponse:
        """Run the full RAG pipeline and return a grounded legal answer."""
        try:
            result = self._rag.answer(
                query,
                retrieval_query=retrieval_query,
                response_language=lang,
                on_step=on_step,
            )
            session.last_chunks = result.get("chunks", [])
            return AgentResponse(
                answer=result["answer"],
                intent=IntentType.LEGAL_FACT,
                session_id=sid,
                chunks=result.get("chunks", []),
                graph=result.get("graph", []),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("RAG pipeline failed: %s", exc)
            return AgentResponse(
                answer="माफ गर्नुहोस्, अहिले जवाफ दिन सकिएन। कृपया फेरि प्रयास गर्नुहोस्।",
                intent=IntentType.LEGAL_FACT,
                session_id=sid,
            )

    def _handle_document_prep(
        self,
        query: str,
        retrieval_query: str,
        session: SessionState,
        sid: str,
        on_step: Callable[[str, str], None] | None = None,
    ) -> AgentResponse:
        """Generate a document checklist / template."""
        if on_step:
            on_step("doc_prep", "Gathering required documents & generating checklist…")
        result = self._doc_preparer.prepare(
            query, session, retrieval_query=retrieval_query
        )
        session.last_chunks = result.get("chunks", [])
        return AgentResponse(
            answer=result["answer"],
            intent=IntentType.DOCUMENT_PREP,
            session_id=sid,
            chunks=result.get("chunks", []),
        )

    def _handle_hybrid(
        self,
        query: str,
        retrieval_query: str,
        session: SessionState,
        step_lang: str,
        lang: str,
        sid: str,
        on_step: Callable[[str, str], None] | None = None,
    ) -> AgentResponse:
        """Combined: procedural navigation + legal RAG answer."""
        nav_result = None
        nav_text = ""
        if self._navigator:
            if on_step:
                on_step("navigate", "Looking up matching procedure in Neo4j graph…")
            nav_result = self._navigator.navigate(retrieval_query, session, advance=False)
            if nav_result:
                nav_text = ProcedureNavigator.format_step_text(nav_result, step_lang)

        # RAG answer (may add context on top of the navigation)
        try:
            rag_result = self._rag.answer(
                query,
                retrieval_query=retrieval_query,
                response_language=lang,
                on_step=on_step,
            )
            session.last_chunks = rag_result.get("chunks", [])
            rag_answer = rag_result["answer"]
        except Exception as exc:  # noqa: BLE001
            logger.error("RAG failed in hybrid mode: %s", exc)
            rag_answer = ""
            rag_result = {"chunks": [], "graph": []}

        # Merge both outputs
        separator = "\n\n---\n\n" if nav_text and rag_answer else ""
        combined = nav_text + separator + rag_answer

        return AgentResponse(
            answer=combined or "माफ गर्नुहोस्, जानकारी फेला परेन।",
            intent=IntentType.HYBRID,
            session_id=sid,
            navigation=nav_result,
            chunks=rag_result.get("chunks", []),
            graph=rag_result.get("graph", []),
        )
