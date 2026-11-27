import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from src.agent.agent import LegalAgent
from src.agent.intent_classifier import IntentType
from src.agent.iterative_retriever import IterativeRetriever
from src.config import get_settings
from src.rag.rag_pipeline import RAG_SYSTEM_PROMPT, RAGPipeline


class _FakeVec(list):
    def tolist(self) -> list[float]:
        return list(self)


class _FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> dict[str, Any]:
        return {
            "dense_vecs": [_FakeVec([0.1] * 4) for _ in texts],
            "lexical_weights": [{1: 0.5} for _ in texts],
        }


class _FakeQdrantClient:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self._payloads = payloads
        self.last_scroll_filter = None

    def scroll(
        self,
        collection_name: str,
        scroll_filter: Any = None,
        limit: int = 20,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> tuple[list[SimpleNamespace], None]:
        self.last_scroll_filter = scroll_filter
        matched = []
        must = getattr(scroll_filter, "must", []) or []
        for payload in self._payloads:
            ok = True
            for cond in must:
                key = cond.key
                expected = cond.match.value
                # Resolve dotted key like "hierarchy.sec"
                cur: Any = payload
                for part in key.split("."):
                    cur = cur.get(part) if isinstance(cur, dict) else None
                if cur != expected:
                    ok = False
                    break
            if ok:
                matched.append(SimpleNamespace(payload=payload))
        return matched[:limit], None


class _FakeVectorStore:
    def __init__(self, payloads: list[dict[str, Any]], search_results: list[dict[str, Any]] | None = None) -> None:
        self.collection_name = "nepali_legal_chunks"
        self.client = _FakeQdrantClient(payloads)
        self._search_results = search_results or []

    def search(self, **kwargs: Any) -> list[dict[str, Any]]:
        return list(self._search_results)


class _FakeLLMClient:
    model = "fake-model"

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or [])
        self.calls: list[tuple[str, str]] = []

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float | None = None,
    ) -> str:
        self.calls.append((system_prompt, user_message))
        if self._responses:
            return self._responses.pop(0)
        return '{"sufficient": true, "missing": "", "next_actions": []}'

    def stream(self, system_prompt: str, user_message: str, temperature: float | None = None):
        yield self.generate(system_prompt, user_message, temperature)


class TestRAGPipelineIterativeTools(unittest.TestCase):
    def test_config_defaults_and_fetch_section_without_chapter(self) -> None:
        settings = get_settings()
        self.assertEqual(settings.AGENT_MAX_LOOPS, 2)
        self.assertEqual(settings.AGENT_MAX_PROVISIONS, 8)

        payloads = [
            {
                "chunk_id": "c-sec166-1",
                "act_source": "company_act_2063",
                "type": "subsection",
                "hierarchy": {"sec": 166, "sub": 1},
                "text": "मुनाफा वितरण नगर्ने कम्पनी स्थापना गर्न सकिनेछ।",
            },
            {
                "chunk_id": "c-sec166-other-act",
                "act_source": "other_act_2070",
                "type": "subsection",
                "hierarchy": {"sec": 166, "sub": 1},
                "text": "अन्य ऐनको दफा १६६ को व्यवस्था।",
            },
        ]
        rag = RAGPipeline(
            embedder=_FakeEmbedder(),
            vector_store=_FakeVectorStore(payloads),
            llm_client=_FakeLLMClient(),
            use_graph=False,
        )

        all_sec = rag.fetch_section(166)
        self.assertEqual(len(all_sec), 2)

        filtered = rag.fetch_section(166, act_source="company_act_2063")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["chunk_id"], "c-sec166-1")

        self.assertIn("partial", RAG_SYSTEM_PROMPT.lower())


class _FakeGraphStore:
    def __init__(self, processes: list[dict[str, Any]] | None = None) -> None:
        self._processes = processes or []

    def find_relevant_processes(self, query: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return list(self._processes)


class _ErrorLLMClient(_FakeLLMClient):
    def generate(self, system_prompt: str, user_message: str, temperature: float | None = None) -> str:
        raise RuntimeError("LLM rate limit exceeded")


def _make_chunk(cid: str, sec: int = 3, text: str = "कम्पनी संस्थापना सम्बन्धी व्यवस्था") -> dict[str, Any]:
    return {
        "chunk_id": cid,
        "act_source": "company_act_2063",
        "type": "subsection",
        "hierarchy": {"sec": sec, "sub": 1},
        "text": text,
    }


class TestIterativeRetriever(unittest.TestCase):
    def test_loop1_early_exit_when_sufficient(self) -> None:
        initial = [_make_chunk("c-1", sec=3)]
        rag = RAGPipeline(
            embedder=_FakeEmbedder(),
            vector_store=_FakeVectorStore([], search_results=initial),
            graph_store=_FakeGraphStore([{"process_name": "Company Registration", "steps": []}]),
            llm_client=_FakeLLMClient(['{"sufficient": true, "missing": "", "next_actions": []}']),
            use_graph=True,
        )
        events: list[tuple[str, str]] = []
        retriever = IterativeRetriever(rag_pipeline=rag, max_loops=3, max_provisions=12)
        result = retriever.retrieve("कम्पनी दर्ता कसरी गर्ने?", on_step=lambda s, m: events.append((s, m)))

        self.assertEqual(result.loops_used, 1)
        self.assertTrue(result.sufficient)
        self.assertEqual([c["chunk_id"] for c in result.chunks], ["c-1"])
        self.assertEqual(len(result.graph), 1)
        stages = [s for s, _ in events]
        self.assertIn("retrieve_loop", stages)
        self.assertIn("retrieve_graph", stages)

    def test_follows_cross_reference_with_fetch_section_and_deduplicates(self) -> None:
        initial = [_make_chunk("c-1", sec=3, text="दफा १६६ को अधीनमा रही कम्पनी स्थापना हुनेछ।")]
        sec166_payloads = [
            _make_chunk("c-1", sec=3),  # duplicate id to verify deduplication
            _make_chunk("c-166-1", sec=166, text="मुनाफा वितरण नगर्ने कम्पनी सम्बन्धी व्यवस्था।"),
        ]
        llm = _FakeLLMClient([
            '```json\n{"sufficient": false, "missing": "Section 166 details", "next_actions": [{"tool": "fetch_section", "sec_num": 166, "act_source": "company_act_2063"}, {"tool": "search_graph", "query": "non-profit company"}, {"tool": "search_clauses", "query": "should be skipped because max 2 actions per loop"}]}\n```',
            '{"sufficient": true, "missing": "", "next_actions": []}',
        ])
        rag = RAGPipeline(
            embedder=_FakeEmbedder(),
            vector_store=_FakeVectorStore(sec166_payloads, search_results=initial),
            graph_store=_FakeGraphStore([{"process_name": "Non-Profit Registration", "steps": []}]),
            llm_client=llm,
            use_graph=True,
        )
        events: list[tuple[str, str]] = []
        retriever = IterativeRetriever(rag_pipeline=rag, max_loops=3, max_provisions=12)
        result = retriever.retrieve("What types of companies can be registered?", on_step=lambda s, m: events.append((s, m)))

        self.assertEqual(result.loops_used, 2)
        self.assertTrue(result.sufficient)
        self.assertEqual([c["chunk_id"] for c in result.chunks], ["c-1", "c-166-1"])
        stages = [s for s, _ in events]
        self.assertIn("fetch_section", stages)

    def test_respects_max_loops_and_max_provisions_cap(self) -> None:
        # 6 chunks per search_clauses call -> Loop 1 (6), Loop 2 (6 more = 12), stops at cap
        chunks = [_make_chunk(f"c-{i}", sec=i) for i in range(1, 15)]

        class _RotatingVectorStore(_FakeVectorStore):
            def __init__(self) -> None:
                super().__init__([])
                self.calls = 0

            def search(self, **kwargs: Any) -> list[dict[str, Any]]:
                self.calls += 1
                start = (self.calls - 1) * 5
                return chunks[start : start + 5]

        llm = _FakeLLMClient([
            '{"sufficient": false, "missing": "need more", "next_actions": [{"tool": "search_clauses", "query": "q2"}]}',
            '{"sufficient": false, "missing": "still need more", "next_actions": [{"tool": "search_clauses", "query": "q3"}]}',
            '{"sufficient": false, "missing": "never reached", "next_actions": [{"tool": "search_clauses", "query": "q4"}]}',
        ])
        rag = RAGPipeline(
            embedder=_FakeEmbedder(),
            vector_store=_RotatingVectorStore(),
            llm_client=llm,
            use_graph=False,
        )
        retriever = IterativeRetriever(rag_pipeline=rag, max_loops=3, max_provisions=12)
        result = retriever.retrieve("broad legal question")

        self.assertEqual(result.loops_used, 3)
        self.assertLessEqual(len(result.chunks), 12)
        self.assertEqual(len(result.chunks), 12)

    def test_graceful_fallback_on_malformed_json_and_llm_error(self) -> None:
        initial = [_make_chunk("c-1", sec=3)]
        rag_bad_json = RAGPipeline(
            embedder=_FakeEmbedder(),
            vector_store=_FakeVectorStore([], search_results=initial),
            llm_client=_FakeLLMClient(["not valid json at all"]),
            use_graph=False,
        )
        res1 = IterativeRetriever(rag_pipeline=rag_bad_json).retrieve("query")
        self.assertEqual(res1.loops_used, 1)
        self.assertEqual([c["chunk_id"] for c in res1.chunks], ["c-1"])

        rag_err = RAGPipeline(
            embedder=_FakeEmbedder(),
            vector_store=_FakeVectorStore([], search_results=initial),
            llm_client=_ErrorLLMClient(),
            use_graph=False,
        )
        res2 = IterativeRetriever(rag_pipeline=rag_err).retrieve("query")
        self.assertEqual(res2.loops_used, 1)
        self.assertEqual([c["chunk_id"] for c in res2.chunks], ["c-1"])

    def test_execute_actions_clamps_and_sanitizes_top_k(self) -> None:
        recorded_top_ks: list[int] = []

        class _RecordingVectorStore(_FakeVectorStore):
            def search(self, **kwargs: Any) -> list[dict[str, Any]]:
                recorded_top_ks.append(kwargs.get("top_k", -1))
                return list(self._search_results)

        rag = RAGPipeline(
            embedder=_FakeEmbedder(),
            vector_store=_RecordingVectorStore([], search_results=[_make_chunk("c-1", sec=3)]),
            llm_client=_FakeLLMClient(),
            use_graph=False,
        )
        retriever = IterativeRetriever(rag_pipeline=rag, max_loops=3, max_provisions=12)

        # CONTEXT_OVERFETCH is 3, so vector_store.search receives top_k * 3
        actions = [
            {"tool": "search_clauses", "query": "q1", "top_k": 99},
            {"tool": "search_clauses", "query": "q2", "top_k": -2},
            {"tool": "search_clauses", "query": "q3", "top_k": "bad"},
            {"tool": "search_clauses", "query": "q4", "top_k": None},
        ]
        retriever._execute_actions(
            actions,
            "default",
            [],
            lambda _: None,
            lambda _: None,
            lambda _s, _m: None,
        )
        # Clamped to 8, clamped to 1, fallback 6, fallback 6 (multiplied by CONTEXT_OVERFETCH=3)
        self.assertEqual(recorded_top_ks, [24, 3, 18, 18])


class TestLegalAgentIterativeWiring(unittest.TestCase):
    def test_legal_fact_and_procedural_without_graph_use_iterative_retriever(self) -> None:
        initial = [_make_chunk("c-3", sec=3, text="दफा १६६ अनुसार कम्पनी स्थापना गर्न सकिन्छ।")]
        sec166 = [_make_chunk("c-166", sec=166, text="मुनाफा वितरण नगर्ने कम्पनी सम्बन्धी व्यवस्था।")]
        llm = _FakeLLMClient([
            '{"sufficient": false, "missing": "sec 166", "next_actions": [{"tool": "fetch_section", "sec_num": 166}]}',
            "Final grounded answer citing दफा ३ and दफा १६६.",
        ])
        rag = RAGPipeline(
            embedder=_FakeEmbedder(),
            vector_store=_FakeVectorStore(sec166, search_results=initial),
            llm_client=llm,
            use_graph=False,
        )
        agent = LegalAgent(rag_pipeline=rag, graph_store=None, llm_client=llm, translate_backoff_sec=0.0)

        events: list[tuple[str, str]] = []
        resp = agent.respond("What does section 3 say about company incorporation?", on_step=lambda s, m: events.append((s, m)))
        self.assertEqual(resp.intent, IntentType.LEGAL_FACT)
        self.assertEqual([c["chunk_id"] for c in resp.chunks], ["c-3", "c-166"])
        self.assertIn("Final grounded answer", resp.answer)
        self.assertIn("fetch_section", [s for s, _ in events])

        # Procedural query with graph_store=None also uses IterativeRetriever
        llm_proc = _FakeLLMClient([
            '{"sufficient": true, "missing": "", "next_actions": []}',
            "Procedural answer from iterative RAG.",
        ])
        rag_proc = RAGPipeline(
            embedder=_FakeEmbedder(),
            vector_store=_FakeVectorStore([], search_results=initial),
            llm_client=llm_proc,
            use_graph=False,
        )
        agent_proc = LegalAgent(rag_pipeline=rag_proc, graph_store=None, llm_client=llm_proc, translate_backoff_sec=0.0)
        resp_proc = agent_proc.respond("How to register a company?")
        self.assertEqual([c["chunk_id"] for c in resp_proc.chunks], ["c-3"])
        self.assertIn("Procedural answer", resp_proc.answer)

    def test_procedural_with_graph_combines_graph_and_vector_via_iterative_rag(self) -> None:
        initial = [_make_chunk("c-3", sec=3, text="कम्पनी दर्ता सम्बन्धी दफा ३ को व्यवस्था।")]
        graph_procs = [{"process_name": "Company Registration", "steps": [{"step_number": 1, "action": "Apply online"}]}]
        llm = _FakeLLMClient([
            '{"sufficient": true, "missing": "", "next_actions": []}',
            "Combined graph and provision answer.",
        ])
        rag = RAGPipeline(
            embedder=_FakeEmbedder(),
            vector_store=_FakeVectorStore([], search_results=initial),
            graph_store=_FakeGraphStore(graph_procs),
            llm_client=llm,
            use_graph=True,
        )
        agent = LegalAgent(
            rag_pipeline=rag,
            graph_store=_FakeGraphStore(graph_procs),
            llm_client=llm,
            translate_backoff_sec=0.0,
        )
        resp = agent.respond("How to register a company?")
        self.assertEqual(resp.intent, IntentType.PROCEDURAL)
        self.assertEqual([c["chunk_id"] for c in resp.chunks], ["c-3"])
        self.assertEqual(len(resp.graph), 1)
        self.assertIn("Combined graph and provision answer.", resp.answer)

    def test_insufficient_retrieval_appends_gap_note_to_context_and_response(self) -> None:
        initial = [_make_chunk("c-3", sec=3)]
        llm = _FakeLLMClient([
            '{"sufficient": false, "missing": "fee schedule in Rule 5", "next_actions": [{"tool": "search_clauses", "query": "fee schedule"}]}',
            "Partial answer based on Section 3.",
        ])
        rag = RAGPipeline(
            embedder=_FakeEmbedder(),
            vector_store=_FakeVectorStore([], search_results=initial),
            llm_client=llm,
            use_graph=False,
        )
        agent = LegalAgent(
            rag_pipeline=rag,
            graph_store=None,
            llm_client=llm,
            translate_backoff_sec=0.0,
        )
        resp = agent.respond("What is the registration fee?")
        self.assertIn("fee schedule in Rule 5", resp.answer)
        # Generation prompt (last LLM call) also received the gap note in context
        self.assertIn("fee schedule in Rule 5", llm.calls[-1][1])

    def test_multi_query_rrf_fuses_and_deduplicates_results(self) -> None:
        from src.agent.iterative_retriever import IterativeRetriever

        mock_rag = MagicMock()
        mock_rag.use_graph = False
        mock_rag.graph_store = None
        mock_rag.search_clauses.side_effect = [
            [_make_chunk("c-1", sec=1), _make_chunk("c-2", sec=2)],
            [_make_chunk("c-2", sec=2), _make_chunk("c-3", sec=3)],
        ]
        retriever = IterativeRetriever(
            mock_rag,
            llm_client=MagicMock(),
            max_loops=1,
            max_provisions=5,
        )
        res = retriever.retrieve(
            "original",
            search_queries=["query 1", "query 2"],
        )
        ids = [c["chunk_id"] for c in res.chunks]
        self.assertEqual(ids[0], "c-2")  # Appears in both lists -> highest RRF score
        self.assertEqual(len(ids), 3)
        self.assertEqual(mock_rag.search_clauses.call_count, 2)

    def test_graph_process_matching_handles_nepali_inflections_and_multi_query(self) -> None:
        from src.agent.iterative_retriever import IterativeRetriever
        from src.knowledge_base.graph_store import Neo4jGraphStore

        store = object.__new__(Neo4jGraphStore)
        store.get_all_processes = MagicMock(return_value=["कम्पनी दर्ता", "कर दर्ता"])  # type: ignore[method-assign]
        store.get_steps_for_process = MagicMock(return_value=[{"step_number": 1, "action": "आवेदन"}])  # type: ignore[method-assign]
        store.get_documents_for_process = MagicMock(return_value=[])  # type: ignore[method-assign]
        store.get_offices_for_process = MagicMock(return_value=[])  # type: ignore[method-assign]

        matched = store.find_relevant_processes("कम्पनीको दर्तामा लाग्ने कागजात", [])
        self.assertEqual(len(matched), 2)
        self.assertEqual(matched[0]["process_name"], "कम्पनी दर्ता")

        mock_rag = MagicMock()
        mock_rag.use_graph = True
        mock_rag.graph_store = MagicMock()
        mock_rag.search_clauses.return_value = [_make_chunk("c-1", sec=4)]
        mock_rag.search_graph.return_value = []
        retriever = IterativeRetriever(mock_rag, llm_client=MagicMock(), max_loops=1)
        retriever.retrieve("original", search_queries=["कम्पनी दर्ता कागजात", "कम्पनी दर्ता समयसीमा"])
        graph_query_arg = mock_rag.search_graph.call_args[0][0]
        self.assertIn("कागजात", graph_query_arg)
        self.assertIn("समयसीमा", graph_query_arg)


if __name__ == "__main__":
    unittest.main()
