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
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.agent.intent_classifier import IntentClassifier, IntentType
from src.agent.iterative_retriever import IterativeRetriever
from src.agent.procedure_navigator import NavigationResult, ProcedureNavigator
from src.agent.session_store import InMemorySessionStore, SessionState
from src.agent.translator import QueryTranslator, detect_language, is_romanized_nepali
from src.llm.llm_client import (
    FallbackLLMClient,
    StandardLLMClient,
    create_helper_llm_client,
    handle_rate_limit_response,
)
from src.rag.rag_pipeline import RAG_SYSTEM_PROMPT
from src.rag.verifier import VerificationLayer

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

_REWRITE_SYSTEM_PROMPT = (
    "Rewrite the user's follow-up legal question into a self-contained search query "
    "using the recent conversation history. Resolve pronouns (it, that, त्यसको, उक्त) "
    "to the specific Nepali Act, Section, or business procedure discussed. "
    "Return ONLY the rewritten search query (no quotes, no explanation)."
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
    citations: list[dict[str, Any]] = field(default_factory=list)


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
        translate_backoff_sec: float = 0.0,
        iterative_retriever: IterativeRetriever | None = None,
        verifier: VerificationLayer | None = None,
        helper_llm_client: BaseLLMClient | None = None,
    ) -> None:
        self._rag = rag_pipeline
        self._graph = graph_store
        self._llm = llm_client or rag_pipeline.llm_client
        use_env_helper = isinstance(self._llm, (StandardLLMClient, FallbackLLMClient))
        self._helper_llm = (
            helper_llm_client
            if helper_llm_client is not None
            else ((create_helper_llm_client() if use_env_helper else None) or self._llm)
        )
        self._store: SessionStore = (
            session_store if session_store is not None else InMemorySessionStore()
        )

        self._classifier = IntentClassifier()
        self._translator = translator or QueryTranslator(self._helper_llm)
        self._translate_backoff_sec = translate_backoff_sec
        self._navigator = ProcedureNavigator(graph_store) if graph_store else None
        self._retriever = iterative_retriever or IterativeRetriever(
            rag_pipeline, llm_client=self._helper_llm
        )
        self._verifier = verifier or getattr(rag_pipeline, "verifier", None) or VerificationLayer()

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

    def _rewrite_query_with_history(
        self,
        query: str,
        history: list[dict[str, str]],
    ) -> str:
        """Resolve follow-up pronouns into a standalone search query using recent turns."""
        if not history:
            return query
        recent = "\n".join(
            f"{t['role'].capitalize()}: {t['content'][:250]}" for t in history[-4:]
        )
        prompt = f"Recent conversation:\n{recent}\n\nFollow-up question: {query}"
        try:
            rewritten = self._helper_llm.generate(_REWRITE_SYSTEM_PROMPT, prompt).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("History query rewrite failed: %s", exc)
            return query
        if not rewritten or rewritten.lstrip().startswith("[Error:"):
            return query
        return rewritten.strip('"').strip("'").strip()

    def _prepare_turn(
        self,
        query: str,
        session: SessionState,
        regenerate: bool,
        on_step: Callable[[str, str], None] | None,
    ) -> tuple[str, IntentType, str, list[dict[str, str]]]:
        """Classify, optionally translate, and contextualize *query* against *session*."""
        def _emit(stage: str, msg: str) -> None:
            if on_step:
                on_step(stage, msg)

        prior = list(session.messages)
        if regenerate and prior and prior[-1].role == "assistant":
            prior = prior[:-1]
            if prior and prior[-1].role == "user":
                prior = prior[:-1]
        history_dicts = [
            {"role": m.role, "content": m.content} for m in prior[-4:] if m.content
        ]

        _emit("intent", "Analyzing query intent…")
        lang = detect_language(query)
        intent = self._classifier.classify(
            query,
            has_active_session=session.has_active_process,
        )

        retrieval_query = query
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
                translated_intent = self._classifier.classify(
                    retrieval_query,
                    has_active_session=session.has_active_process,
                )
                if intent == IntentType.UNKNOWN:
                    intent = translated_intent

        if history_dicts and intent not in (IntentType.GREETING, IntentType.FOLLOW_UP):
            _emit("contextualize", "Understanding follow-up from conversation history…")
            retrieval_query = self._rewrite_query_with_history(retrieval_query, history_dicts)

        _emit("route", f"Intent: {intent.value} (lang={lang})")
        return retrieval_query, intent, lang, history_dicts

    def respond(
        self,
        query: str,
        session_id: str = "",
        on_step: Callable[[str, str], None] | None = None,
        regenerate: bool = False,
    ) -> AgentResponse:
        """Process *query*, persist conversation turns, and return an :class:`AgentResponse`."""
        collected_steps: list[str] = []

        def _emit(stage: str, msg: str) -> None:
            collected_steps.append(msg)
            if on_step:
                on_step(stage, msg)

        step_cb = _emit if on_step is not None else None
        session = self._store.get_or_create(session_id or str(uuid.uuid4()))
        retrieval_query, intent, lang, history_dicts = self._prepare_turn(
            query, session, regenerate, _emit
        )
        session.push_intent(intent)

        logger.info(
            "Agent: session=%s  lang=%s  intent=%s  query=%s",
            session.session_id,
            lang,
            intent.value,
            query[:60],
        )

        response = self._route(
            query=query,
            retrieval_query=retrieval_query,
            intent=intent,
            session=session,
            lang=lang,
            on_step=step_cb,
            conversation_history=history_dicts,
        )
        response.citations = self._format_citations_safe(response.chunks)
        self._persist_turn(
            session,
            query=query,
            answer=response.answer,
            citations=response.citations,
            steps=collected_steps,
            regenerate=regenerate,
        )
        return response

    def _persist_turn(
        self,
        session: SessionState,
        query: str,
        answer: str,
        citations: list[dict[str, Any]],
        steps: list[str],
        regenerate: bool,
    ) -> None:
        """Append or replace the latest assistant turn and save *session*."""
        if regenerate:
            if not session.messages:
                session.add_message("user", query)
            session.replace_last_assistant_message(answer, citations=citations, steps=steps)
        else:
            session.add_message("user", query)
            session.add_message("assistant", answer, citations=citations, steps=steps)
        self._store.save(session)

    def _format_citations_safe(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format deduplicated UI citations when the pipeline supports it."""
        if not chunks:
            return []
        formatter = getattr(self._rag, "format_citations", None)
        if callable(formatter):
            res = formatter(chunks)
            if isinstance(res, list):
                return res
        return []

    def respond_stream(
        self,
        query: str,
        session_id: str = "",
        regenerate: bool = False,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """Stream ``('step', ...)``, ``('token', ...)``, and ``('final', ...)`` events."""
        step_events: list[tuple[str, dict[str, Any]]] = []
        collected_steps: list[str] = []

        def _emit(stage: str, msg: str) -> None:
            collected_steps.append(msg)
            step_events.append(("step", {"stage": stage, "message": msg}))

        session = self._store.get_or_create(session_id or str(uuid.uuid4()))
        retrieval_query, intent, lang, history_dicts = self._prepare_turn(
            query, session, regenerate, _emit
        )
        session.push_intent(intent)
        for ev in step_events:
            yield ev
        step_events.clear()

        def _persist_and_build_final(
            answer_text: str,
            chunks: list[dict[str, Any]],
            graph: list[dict[str, Any]],
            nav: NavigationResult | None = None,
        ) -> dict[str, Any]:
            citations = self._format_citations_safe(chunks)
            session.last_chunks = chunks
            self._persist_turn(
                session,
                query=query,
                answer=answer_text,
                citations=citations,
                steps=collected_steps,
                regenerate=regenerate,
            )
            nav_dict = None
            if nav is not None:
                nav_dict = {
                    "process_name": nav.process_name,
                    "total_steps": nav.total_steps,
                    "current_step": nav.current_step,
                    "is_complete": nav.is_complete,
                    "progress_pct": nav.progress_pct,
                    "step": nav.step or {},
                }
            return {
                "answer": answer_text,
                "intent": intent.value,
                "session_id": session.session_id,
                "navigation": nav_dict,
                "chunks": chunks,
                "citations": citations,
                "graph_context": graph,
            }

        if intent not in (IntentType.LEGAL_FACT, IntentType.PROCEDURAL, IntentType.UNKNOWN):
            resp = self._route(
                query=query,
                retrieval_query=retrieval_query,
                intent=intent,
                session=session,
                lang=lang,
                on_step=_emit,
                conversation_history=history_dicts,
            )
            for ev in step_events:
                yield ev
            if resp.answer:
                yield ("token", {"text": resp.answer})
            yield ("final", _persist_and_build_final(resp.answer, resp.chunks, resp.graph, resp.navigation))
            return

        retrieval = self._retriever.retrieve(
            query, retrieval_query=retrieval_query, on_step=_emit
        )
        for ev in step_events:
            yield ev
        step_events.clear()

        active_chunks = self._verifier.filter_active_provisions(retrieval.chunks)
        context = self._rag.assemble_context(
            query,
            active_chunks,
            retrieval.graph,
            response_language=lang,
            conversation_history=history_dicts,
        )
        has_gap = not retrieval.sufficient and bool(retrieval.missing.strip())
        if has_gap:
            context = f"{context}\n[Retrieval Gap: {retrieval.missing.strip()}]"

        yield ("step", {"stage": "generate", "message": "Generating grounded legal answer…"})
        collected_steps.append("Generating grounded legal answer…")

        stream_iter = iter(self._llm.stream(RAG_SYSTEM_PROMPT, context))
        first = next(stream_iter, None)
        if first is None:
            final_answer = "माफ गर्नुहोस्, अहिले जवाफ दिन सकिएन।"
            yield ("token", {"text": final_answer})
            yield ("final", _persist_and_build_final(final_answer, retrieval.chunks, retrieval.graph))
            return

        rate_msg = handle_rate_limit_response(first, lang, "streaming answer generation")
        if rate_msg is not None:
            yield ("token", {"text": rate_msg})
            yield ("final", _persist_and_build_final(rate_msg, retrieval.chunks, retrieval.graph))
            return

        parts = [first]
        yield ("token", {"text": first})
        for tok in stream_iter:
            parts.append(tok)
            yield ("token", {"text": tok})

        raw_answer = "".join(parts)
        yield ("step", {"stage": "verify", "message": "Verifying citations and temporal status…"})
        collected_steps.append("Verifying citations and temporal status…")
        verification = self._verifier.verify(
            raw_answer, chunks=retrieval.chunks, graph=retrieval.graph
        )
        final_answer = verification.answer
        if has_gap:
            gap_note = (
                f"\n\nℹ️ नोट: उपलब्ध सन्दर्भमा सबै विवरण फेला परेन ({retrieval.missing.strip()})।"
                if lang == "ne"
                else f"\n\nℹ️ Note: Some details could not be fully retrieved ({retrieval.missing.strip()})."
            )
            final_answer = f"{final_answer.rstrip()}{gap_note}"

        yield ("final", _persist_and_build_final(final_answer, retrieval.chunks, retrieval.graph))

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
        conversation_history: list[dict[str, str]] | None = None,
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
                query,
                retrieval_query,
                session,
                lang,
                sid,
                on_step=on_step,
                conversation_history=conversation_history,
            )

        if intent == IntentType.DOCUMENT_PREP:
            return self._handle_document_prep(
                query, retrieval_query, session, lang, sid, on_step=on_step
            )

        if intent in (IntentType.LEGAL_FACT, IntentType.UNKNOWN):
            return self._handle_legal_fact(
                query,
                retrieval_query,
                session,
                lang,
                sid,
                on_step=on_step,
                conversation_history=conversation_history,
            )

        if intent == IntentType.HYBRID:
            return self._handle_hybrid(
                query,
                retrieval_query,
                session,
                step_lang,
                lang,
                sid,
                on_step=on_step,
                conversation_history=conversation_history,
            )

        # Safety fallback
        return self._handle_legal_fact(
            query,
            retrieval_query,
            session,
            lang,
            sid,
            on_step=on_step,
            conversation_history=conversation_history,
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
        lang: str,
        sid: str,
        on_step: Callable[[str, str], None] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> AgentResponse:
        """Handle a procedural query by combining Neo4j Procedure search and Qdrant Provision search."""
        try:
            result = self._run_iterative_rag(
                query,
                retrieval_query=retrieval_query,
                lang=lang,
                on_step=on_step,
                conversation_history=conversation_history,
            )
            session.last_chunks = result.get("chunks", [])
            return AgentResponse(
                answer=result["answer"],
                intent=IntentType.PROCEDURAL,
                session_id=sid,
                chunks=result.get("chunks", []),
                graph=result.get("graph", []),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Procedural RAG pipeline failed: %s", exc)
            return AgentResponse(
                answer="माफ गर्नुहोस्, अहिले जवाफ दिन सकिएन। कृपया फेरि प्रयास गर्नुहोस्।",
                intent=IntentType.PROCEDURAL,
                session_id=sid,
            )

    def _run_iterative_rag(
        self,
        query: str,
        retrieval_query: str,
        lang: str,
        on_step: Callable[[str, str], None] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Run bounded iterative retrieval and synthesize a grounded answer."""
        retrieval = self._retriever.retrieve(
            query,
            retrieval_query=retrieval_query,
            on_step=on_step,
        )
        active_chunks = self._verifier.filter_active_provisions(retrieval.chunks)
        context = self._rag.assemble_context(
            query,
            active_chunks,
            retrieval.graph,
            response_language=lang,
            conversation_history=conversation_history,
        )
        has_gap = not retrieval.sufficient and bool(retrieval.missing.strip())
        if has_gap:
            context = f"{context}\n[Retrieval Gap: {retrieval.missing.strip()}]"
        if on_step:
            on_step("generate", "Generating grounded legal answer…")
        answer = self._llm.generate(RAG_SYSTEM_PROMPT, context)
        rate_msg = handle_rate_limit_response(answer, lang, "answer generation")
        if rate_msg is not None:
            return {
                "answer": rate_msg,
                "chunks": retrieval.chunks,
                "graph": retrieval.graph,
                "context": context,
                "verification": None,
            }
        if on_step:
            on_step("verify", "Verifying citations and temporal status…")
        verification = self._verifier.verify(
            answer, chunks=retrieval.chunks, graph=retrieval.graph
        )
        final_answer = verification.answer
        if has_gap:
            gap_note = (
                f"\n\nℹ️ नोट: उपलब्ध सन्दर्भमा सबै विवरण फेला परेन ({retrieval.missing.strip()})।"
                if lang == "ne"
                else f"\n\nℹ️ Note: Some details could not be fully retrieved ({retrieval.missing.strip()})."
            )
            final_answer = f"{final_answer.rstrip()}{gap_note}"
        return {
            "answer": final_answer,
            "chunks": retrieval.chunks,
            "graph": retrieval.graph,
            "context": context,
            "verification": verification,
        }

    def _handle_legal_fact(
        self,
        query: str,
        retrieval_query: str,
        session: SessionState,
        lang: str,
        sid: str,
        on_step: Callable[[str, str], None] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> AgentResponse:
        """Run bounded iterative RAG and return a grounded legal answer."""
        try:
            result = self._run_iterative_rag(
                query,
                retrieval_query=retrieval_query,
                lang=lang,
                on_step=on_step,
                conversation_history=conversation_history,
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
        lang: str,
        sid: str,
        on_step: Callable[[str, str], None] | None = None,
    ) -> AgentResponse:
        """Generate a document checklist / template."""
        if on_step:
            on_step("doc_prep", "Gathering required documents & generating checklist…")
        result = self._doc_preparer.prepare(
            query, session, retrieval_query=retrieval_query
        )
        answer = result["answer"]
        rate_msg = handle_rate_limit_response(answer, lang, "document preparation")
        if rate_msg is not None:
            answer = rate_msg
        session.last_chunks = result.get("chunks", [])
        return AgentResponse(
            answer=answer,
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
        conversation_history: list[dict[str, str]] | None = None,
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
            rag_result = self._run_iterative_rag(
                query,
                retrieval_query=retrieval_query,
                lang=lang,
                on_step=on_step,
                conversation_history=conversation_history,
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
