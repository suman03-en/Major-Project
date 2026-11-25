"""
Tests for the agent layer:
  - IntentClassifier
  - SessionStore / SessionState
  - ProcedureNavigator
  - LegalAgent routing (mocked RAG + Graph)
"""

import unittest
from unittest.mock import MagicMock, patch

from src.agent.intent_classifier import IntentClassifier, IntentType
from src.agent.session_store import InMemorySessionStore, SessionState
from src.agent.translator import QueryTranslator, detect_language, is_romanized_nepali

# ---------------------------------------------------------------------------
# 1. IntentClassifier
# ---------------------------------------------------------------------------

class TestIntentClassifier(unittest.TestCase):

    def setUp(self):
        self.clf = IntentClassifier()

    def test_greeting_english(self):
        self.assertEqual(self.clf.classify("hi"), IntentType.GREETING)

    def test_greeting_nepali(self):
        self.assertEqual(self.clf.classify("नमस्ते"), IntentType.GREETING)

    def test_procedural_nepali(self):
        self.assertEqual(
            self.clf.classify("कम्पनी दर्ता कसरी गर्ने?"),
            IntentType.PROCEDURAL,
        )

    def test_procedural_english(self):
        self.assertEqual(
            self.clf.classify("how to register a company"),
            IntentType.PROCEDURAL,
        )

    def test_legal_fact(self):
        self.assertEqual(
            self.clf.classify("दफा ४५ के भन्छ?"),
            IntentType.LEGAL_FACT,
        )

    def test_document_prep(self):
        self.assertEqual(
            self.clf.classify("मलाई निवेदन फारम चाहिन्छ"),
            IntentType.DOCUMENT_PREP,
        )

    def test_follow_up_with_active_session(self):
        self.assertEqual(
            self.clf.classify("next", has_active_session=True),
            IntentType.FOLLOW_UP,
        )

    def test_follow_up_without_session_is_not_follow_up(self):
        # Without active session "next" alone should NOT be FOLLOW_UP
        result = self.clf.classify("next", has_active_session=False)
        self.assertNotEqual(result, IntentType.FOLLOW_UP)

    def test_hybrid_both_keywords(self):
        result = self.clf.classify("कम्पनी दर्ता प्रक्रिया र दफा ४५ के भन्छ?")
        self.assertEqual(result, IntentType.HYBRID)

    def test_unknown_falls_through(self):
        result = self.clf.classify("abcdef xyz 123")
        self.assertEqual(result, IntentType.UNKNOWN)


# ---------------------------------------------------------------------------
# 2. InMemorySessionStore
# ---------------------------------------------------------------------------

class TestSessionStore(unittest.TestCase):

    def test_creates_new_session(self):
        store = InMemorySessionStore()
        s = store.get_or_create("sess-1")
        self.assertIsInstance(s, SessionState)
        self.assertEqual(s.session_id, "sess-1")

    def test_returns_same_session(self):
        store = InMemorySessionStore()
        s1 = store.get_or_create("sess-1")
        s1.active_process = "कम्पनी दर्ता"
        store.save(s1)
        s2 = store.get_or_create("sess-1")
        self.assertEqual(s2.active_process, "कम्पनी दर्ता")

    def test_delete_removes_session(self):
        store = InMemorySessionStore()
        store.get_or_create("sess-del")
        store.delete("sess-del")
        s = store.get_or_create("sess-del")
        self.assertIsNone(s.active_process)  # fresh session

    def test_max_sessions_eviction(self):
        store = InMemorySessionStore(max_sessions=2)
        store.get_or_create("a")
        store.get_or_create("b")
        store.get_or_create("c")  # triggers eviction
        self.assertLessEqual(len(store), 2)

    def test_push_intent_history(self):
        store = InMemorySessionStore()
        s = store.get_or_create("sess-h")
        s.push_intent(IntentType.GREETING)
        s.push_intent(IntentType.PROCEDURAL)
        self.assertEqual(s.intent_history, [IntentType.GREETING, IntentType.PROCEDURAL])

    def test_has_active_process(self):
        s = SessionState(session_id="x")
        self.assertFalse(s.has_active_process)
        s.active_process = "test"
        self.assertTrue(s.has_active_process)


# ---------------------------------------------------------------------------
# 3. LegalAgent routing (fully mocked)
# ---------------------------------------------------------------------------

class TestLegalAgentRouting(unittest.TestCase):

    def _make_agent(self):
        """Build a LegalAgent with mocked RAG pipeline and no graph."""
        from src.agent.agent import LegalAgent
        from src.agent.iterative_retriever import RetrievalResult
        from src.agent.session_store import InMemorySessionStore

        mock_rag = MagicMock()
        mock_rag.assemble_context.return_value = "assembled context"
        mock_rag.retrieve_chunks.return_value = []
        mock_rag.graph_store = None
        mock_rag.llm_client = MagicMock()
        mock_rag.llm_client.generate.return_value = "LLM answer"

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = RetrievalResult(chunks=[], graph=[], loops_used=1)

        store = InMemorySessionStore()
        agent = LegalAgent(
            rag_pipeline=mock_rag,
            graph_store=None,
            session_store=store,
            translate_backoff_sec=0.0,
        )
        agent._retriever = mock_retriever
        return agent, mock_rag, store

    def test_greeting_returns_no_rag_call(self):
        agent, _, _ = self._make_agent()
        resp = agent.respond("hi", session_id="s1")
        self.assertEqual(resp.intent, IntentType.GREETING)
        agent._retriever.retrieve.assert_not_called()

    def test_legal_fact_calls_rag(self):
        agent, _, _ = self._make_agent()
        resp = agent.respond("दफा ४५ के भन्छ?", session_id="s2")
        self.assertEqual(resp.intent, IntentType.LEGAL_FACT)
        agent._retriever.retrieve.assert_called_once()

    def test_session_id_echoed_back(self):
        agent, _, _ = self._make_agent()
        resp = agent.respond("hi", session_id="echo-me")
        self.assertEqual(resp.session_id, "echo-me")

    def test_new_session_id_generated_when_empty(self):
        agent, _, _ = self._make_agent()
        resp = agent.respond("hi", session_id="")
        self.assertNotEqual(resp.session_id, "")

    def test_procedural_without_graph_falls_back_to_rag(self):
        """With no graph store, PROCEDURAL intent falls back to RAG."""
        agent, _, _ = self._make_agent()
        agent.respond("कम्पनी दर्ता कसरी गर्ने?", session_id="s3")
        # No graph → falls through to legal_fact handler
        agent._retriever.retrieve.assert_called_once()

    def test_english_query_skips_translation_and_keeps_original_query(self):
        agent, mock_rag, _ = self._make_agent()
        resp = agent.respond("what is public company in nepal", session_id="s4")
        self.assertEqual(resp.intent, IntentType.LEGAL_FACT)
        # 1 LLM call for answer generation, 0 for translation
        self.assertEqual(mock_rag.llm_client.generate.call_count, 1)
        agent._retriever.retrieve.assert_called_once_with(
            "what is public company in nepal",
            retrieval_query="what is public company in nepal",
            on_step=None,
        )

    def test_romanized_query_translated_for_retrieval(self):
        agent, mock_rag, _ = self._make_agent()
        mock_rag.llm_client.generate.side_effect = [
            '{"language": "roman", "translated_query": "कम्पनी दर्ता कसरी गर्ने?"}',
            "कम्पनी दर्ता जवाफ",
        ]
        agent.respond("company darta kasari garne?", session_id="s5")
        agent._retriever.retrieve.assert_called_once_with(
            "company darta kasari garne?",
            retrieval_query="कम्पनी दर्ता कसरी गर्ने?",
            on_step=None,
        )

    def test_greeting_skips_translation_llm_call(self):
        agent, mock_rag, _ = self._make_agent()
        agent.respond("namaste", session_id="s6")
        mock_rag.llm_client.generate.assert_not_called()
        agent._retriever.retrieve.assert_not_called()

    def test_on_step_emits_intermediate_stages(self):
        agent, mock_rag, _ = self._make_agent()
        mock_rag.llm_client.generate.side_effect = [
            '{"language": "roman", "translated_query": "कम्पनी दर्ता कसरी गर्ने?"}',
            "कम्पनी दर्ता जवाफ",
        ]
        stages = []
        agent.respond(
            "company darta kasari garne?",
            session_id="s7",
            on_step=lambda stage, msg: stages.append((stage, msg)),
        )
        stage_names = [s[0] for s in stages]
        self.assertIn("intent", stage_names)
        self.assertIn("translate", stage_names)
        self.assertIn("translate_done", stage_names)
        self.assertIn("route", stage_names)

    def test_translation_skips_sleep_by_default_and_honors_explicit_backoff(self):
        from src.agent.agent import LegalAgent
        from src.agent.iterative_retriever import RetrievalResult
        from src.agent.session_store import InMemorySessionStore

        mock_rag = MagicMock()
        mock_rag.assemble_context.return_value = "context"
        mock_rag.graph_store = None
        mock_rag.llm_client = MagicMock()
        mock_rag.llm_client.generate.side_effect = [
            '{"language": "roman", "translated_query": "कम्पनी दर्ता कसरी गर्ने?"}',
            "OK",
            '{"language": "roman", "translated_query": "कम्पनी दर्ता कसरी गर्ने?"}',
            "OK",
        ]
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = RetrievalResult(chunks=[], graph=[], loops_used=1)

        agent_default = LegalAgent(
            rag_pipeline=mock_rag,
            graph_store=None,
            session_store=InMemorySessionStore(),
        )
        agent_default._retriever = mock_retriever
        with patch("src.agent.agent.time.sleep") as mock_sleep:
            agent_default.respond("company darta kasari garne?", session_id="s-default")
        mock_sleep.assert_not_called()

        agent_explicit = LegalAgent(
            rag_pipeline=mock_rag,
            graph_store=None,
            session_store=InMemorySessionStore(),
            translate_backoff_sec=1.0,
        )
        agent_explicit._retriever = mock_retriever
        with patch("src.agent.agent.time.sleep") as mock_sleep:
            agent_explicit.respond("company darta kasari garne?", session_id="s-backoff")
        mock_sleep.assert_called_once_with(1.0)

    def test_helper_llm_client_routes_translation_and_completeness_only(self):
        from src.agent.agent import LegalAgent
        from src.agent.session_store import InMemorySessionStore

        primary_llm = MagicMock()
        primary_llm.generate.return_value = "Final synthesized answer."
        helper_llm = MagicMock()
        helper_llm.generate.side_effect = [
            '{"language": "roman", "translated_query": "कम्पनी दर्ता कसरी गर्ने?"}',
            '{"sufficient": true, "missing": "", "next_actions": []}',
        ]

        mock_rag = MagicMock()
        mock_rag.graph_store = None
        mock_rag.llm_client = primary_llm
        mock_rag.verifier = None
        mock_rag.retrieve_chunks.return_value = [
            {"chunk_id": "c-1", "act_source": "company_act", "hierarchy": {"sec": 3}, "text": "पाठ"}
        ]
        mock_rag.assemble_context.return_value = "context"

        agent = LegalAgent(
            rag_pipeline=mock_rag,
            graph_store=None,
            llm_client=primary_llm,
            helper_llm_client=helper_llm,
            session_store=InMemorySessionStore(),
        )
        resp = agent.respond("company darta kasari garne?", session_id="s-helper")
        self.assertEqual(resp.answer, "Final synthesized answer.")
        self.assertEqual(helper_llm.generate.call_count, 2)
        primary_llm.generate.assert_called_once()

    def test_rate_limit_error_returns_clean_localized_message(self):
        from src.agent.agent import LegalAgent
        from src.agent.iterative_retriever import RetrievalResult
        from src.agent.session_store import InMemorySessionStore

        primary_llm = MagicMock()
        primary_llm.generate.return_value = (
            '[Error: LLM generation failed — HTTP 429: {"error":{"message":"Rate limit reached"}}]'
        )
        mock_rag = MagicMock()
        mock_rag.graph_store = None
        mock_rag.llm_client = primary_llm
        mock_rag.assemble_context.return_value = "context"

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = RetrievalResult(chunks=[], graph=[], loops_used=1)

        agent = LegalAgent(
            rag_pipeline=mock_rag,
            graph_store=None,
            llm_client=primary_llm,
            session_store=InMemorySessionStore(),
        )
        agent._retriever = mock_retriever

        resp_en = agent.respond("What does section 3 say?", session_id="s-429-en")
        self.assertNotIn("[Error:", resp_en.answer)
        self.assertNotIn("{", resp_en.answer)
        self.assertIn("rate", resp_en.answer.lower())

        resp_ne = agent.respond("दफा ३ के भन्छ?", session_id="s-429-ne")
        self.assertNotIn("[Error:", resp_ne.answer)
        self.assertNotIn("{", resp_ne.answer)
        self.assertIn("मिनेट", resp_ne.answer)

    def test_multi_turn_continuity_rewrites_followup_and_persists_turns(self):
        from src.agent.agent import LegalAgent
        from src.agent.iterative_retriever import RetrievalResult
        from src.agent.session_store import InMemorySessionStore

        primary_llm = MagicMock()
        primary_llm.generate.side_effect = ["Answer 1 about Section 43.", "Answer 2 about penalties."]
        helper_llm = MagicMock()
        helper_llm.generate.return_value = "कम्पनी ऐन २०६३ दफा ४३ को जरिवाना"

        mock_rag = MagicMock()
        mock_rag.graph_store = None
        mock_rag.llm_client = primary_llm
        mock_rag.verifier = None
        mock_rag.assemble_context.return_value = "context"

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = RetrievalResult(chunks=[], graph=[], loops_used=1)

        store = InMemorySessionStore()
        agent = LegalAgent(
            rag_pipeline=mock_rag,
            graph_store=None,
            llm_client=primary_llm,
            helper_llm_client=helper_llm,
            session_store=store,
        )
        agent._retriever = mock_retriever

        # Turn 1: 0 helper rewrite calls
        agent.respond("दफा ४३ के भन्छ?", session_id="conv-mt")
        self.assertEqual(helper_llm.generate.call_count, 0)

        # Turn 2: follow-up without repeating topic triggers rewrite + passes history
        agent.respond("त्यसको जरिवाना कति हो?", session_id="conv-mt")
        self.assertEqual(helper_llm.generate.call_count, 1)
        self.assertEqual(
            mock_retriever.retrieve.call_args_list[1].kwargs["retrieval_query"],
            "कम्पनी ऐन २०६३ दफा ४३ को जरिवाना",
        )
        self.assertTrue(len(mock_rag.assemble_context.call_args_list[1].kwargs.get("conversation_history", [])) >= 2)
        self.assertEqual(len(store.get("conv-mt").messages), 4)  # type: ignore[union-attr]

    def test_regenerate_replaces_last_assistant_message_without_duplicating_user(self):
        from src.agent.agent import LegalAgent
        from src.agent.iterative_retriever import RetrievalResult
        from src.agent.session_store import InMemorySessionStore

        primary_llm = MagicMock()
        primary_llm.generate.side_effect = ["First answer", "Regenerated answer"]
        helper_llm = MagicMock()
        helper_llm.generate.return_value = "दफा ३"

        mock_rag = MagicMock()
        mock_rag.graph_store = None
        mock_rag.llm_client = primary_llm
        mock_rag.verifier = None
        mock_rag.assemble_context.return_value = "context"

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = RetrievalResult(chunks=[], graph=[], loops_used=1)

        store = InMemorySessionStore()
        agent = LegalAgent(
            rag_pipeline=mock_rag,
            graph_store=None,
            llm_client=primary_llm,
            helper_llm_client=helper_llm,
            session_store=store,
        )
        agent._retriever = mock_retriever

        agent.respond("दफा ३ के भन्छ?", session_id="conv-rg")
        resp2 = agent.respond("दफा ३ के भन्छ?", session_id="conv-rg", regenerate=True)

        self.assertEqual(resp2.answer, "Regenerated answer")
        msgs = store.get("conv-rg").messages  # type: ignore[union-attr]
        self.assertEqual(len(msgs), 2)
        self.assertEqual([m.role for m in msgs], ["user", "assistant"])
        self.assertEqual(msgs[1].content, "Regenerated answer")

    def test_respond_stream_emits_step_token_and_final_events_and_persists(self):
        from src.agent.agent import LegalAgent
        from src.agent.iterative_retriever import RetrievalResult
        from src.agent.session_store import InMemorySessionStore

        primary_llm = MagicMock()
        primary_llm.stream.return_value = iter(["Hello ", "Nepal."])

        mock_rag = MagicMock()
        mock_rag.graph_store = None
        mock_rag.llm_client = primary_llm
        mock_rag.verifier = None
        mock_rag.assemble_context.return_value = "context"
        mock_rag.format_citations.return_value = [
            {"label": "कम्पनी ऐन, २०६३, दफा 3", "excerpt": "पाठ", "status": "active"}
        ]

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = RetrievalResult(
            chunks=[{"chunk_id": "c-1", "act_source": "company_act_2063", "hierarchy": {"sec": 3}, "text": "पाठ"}],
            graph=[],
            loops_used=1,
        )

        store = InMemorySessionStore()
        agent = LegalAgent(
            rag_pipeline=mock_rag,
            graph_store=None,
            llm_client=primary_llm,
            session_store=store,
        )
        agent._retriever = mock_retriever

        events = list(agent.respond_stream("What is section 3?", session_id="conv-stream"))
        names = [e[0] for e in events]
        self.assertIn("step", names)
        self.assertEqual([e[1]["text"] for e in events if e[0] == "token"], ["Hello ", "Nepal."])
        finals = [e[1] for e in events if e[0] == "final"]
        self.assertEqual(len(finals), 1)
        self.assertEqual(finals[0]["answer"], "Hello Nepal.")
        self.assertEqual(len(finals[0]["citations"]), 1)
        self.assertEqual(len(store.get("conv-stream").messages), 2)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# 4. QueryTranslator
# ---------------------------------------------------------------------------

class TestQueryTranslator(unittest.TestCase):

    def test_detect_language_devanagari(self):
        self.assertEqual(detect_language("कम्पनी दर्ता कसरी गर्ने?"), "ne")

    def test_detect_language_non_devanagari_fast_path(self):
        self.assertEqual(detect_language("How to register a private company?"), "en")

    def test_devanagari_skips_llm_call(self):
        llm = MagicMock()
        tr = QueryTranslator(llm)
        res = tr.translate("दफा ४५ को व्यवस्था के छ?")
        self.assertEqual(res.language, "ne")
        self.assertFalse(res.was_translated)
        self.assertEqual(res.retrieval_query, "दफा ४५ को व्यवस्था के छ?")
        llm.generate.assert_not_called()

    def test_romanized_json_detects_language_and_translates_in_one_call(self):
        llm = MagicMock()
        llm.generate.return_value = (
            '```json\n{"language": "roman", "translated_query": "कम्पनी दर्ता प्रक्रिया के हो?"}\n```'
        )
        tr = QueryTranslator(llm)
        res = tr.translate("company darta prakriya k ho?")
        self.assertEqual(res.language, "roman")
        self.assertTrue(res.was_translated)
        self.assertEqual(res.original_query, "company darta prakriya k ho?")
        self.assertEqual(res.retrieval_query, "कम्पनी दर्ता प्रक्रिया के हो?")
        llm.generate.assert_called_once()

    def test_plain_text_fallback_when_llm_skips_json(self):
        llm = MagicMock()
        llm.generate.return_value = '"कम्पनी दर्ता शुल्क कति छ?"'
        tr = QueryTranslator(llm)
        res = tr.translate("What is the company registration fee?")
        self.assertEqual(res.language, "en")
        self.assertTrue(res.was_translated)
        self.assertEqual(res.retrieval_query, "कम्पनी दर्ता शुल्क कति छ?")

    def test_llm_error_falls_back_gracefully(self):
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("Ollama timeout")
        tr = QueryTranslator(llm)
        res = tr.translate("What is the registration fee?")
        self.assertEqual(res.language, "en")
        self.assertFalse(res.was_translated)
        self.assertEqual(res.retrieval_query, "What is the registration fee?")

    def test_llm_error_string_falls_back_gracefully(self):
        llm = MagicMock()
        llm.generate.return_value = "[Error: LLM generation failed — timed out]"
        tr = QueryTranslator(llm)
        res = tr.translate("What is the registration fee?")
        self.assertEqual(res.language, "en")
        self.assertFalse(res.was_translated)
        self.assertEqual(res.retrieval_query, "What is the registration fee?")

    def test_is_romanized_nepali_detection(self):
        self.assertTrue(is_romanized_nepali("company darta kasari garne?"))
        self.assertTrue(is_romanized_nepali("public company ksri suru garne"))
        self.assertFalse(is_romanized_nepali("what is public company in nepal"))
        self.assertFalse(is_romanized_nepali("कम्पनी दर्ता कसरी गर्ने?"))


if __name__ == "__main__":
    unittest.main()
