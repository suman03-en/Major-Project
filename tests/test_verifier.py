import unittest
from types import SimpleNamespace
from typing import Any

from src.agent.agent import LegalAgent
from src.rag.rag_pipeline import RAGPipeline
from src.rag.verifier import VerificationLayer


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
    def scroll(self, **kwargs: Any) -> tuple[list[SimpleNamespace], None]:
        return [], None


class _FakeVectorStore:
    def __init__(self, search_results: list[dict[str, Any]]) -> None:
        self.collection_name = "nepali_legal_chunks"
        self.client = _FakeQdrantClient()
        self._search_results = search_results

    def search(self, **kwargs: Any) -> list[dict[str, Any]]:
        return list(self._search_results)


class _FakeLLMClient:
    model = "fake-model"

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float | None = None,
    ) -> str:
        self.calls.append((system_prompt, user_message))
        return self._responses.pop(0)

    def stream(self, system_prompt: str, user_message: str, temperature: float | None = None):
        yield self.generate(system_prompt, user_message, temperature)


class TestVerificationLayerTemporal(unittest.TestCase):
    def test_flags_inactive_provisions_and_leaves_active_untouched(self) -> None:
        verifier = VerificationLayer()

        active_chunks = [
            {
                "chunk_id": "c-1",
                "act_source": "company_act_2063",
                "hierarchy": {"sec": 3},
                "status": "active",
                "text": "कम्पनीको संस्थापना।",
            },
            {
                "chunk_id": "c-2",
                "act_source": "company_act_2063",
                "hierarchy": {"sec": 4},
                "text": "निजी कम्पनी।",
            },
        ]
        res_ok = verifier.verify("दफा ३ अनुसार कम्पनी स्थापना हुन्छ।", chunks=active_chunks)
        self.assertTrue(res_ok.is_valid)
        self.assertEqual(res_ok.inactive_provisions, [])
        self.assertEqual(res_ok.answer, "दफा ३ अनुसार कम्पनी स्थापना हुन्छ।")

        mixed_chunks = active_chunks + [
            {
                "chunk_id": "c-old",
                "act_source": "company_act_2053",
                "hierarchy": {"sec": 10},
                "status": "repealed",
                "text": "पुरानो व्यवस्था।",
            },
            {
                "chunk_id": "c-amended",
                "act_source": "company_act_2063",
                "hierarchy": {"sec": 12},
                "status": "amended",
                "text": "संशोधित व्यवस्था।",
            },
        ]
        res_warn = verifier.verify("दफा ३ अनुसार कम्पनी स्थापना हुन्छ।", chunks=mixed_chunks)
        self.assertFalse(res_warn.is_valid)
        self.assertEqual(len(res_warn.inactive_provisions), 2)
        self.assertEqual(
            [p["status"] for p in res_warn.inactive_provisions],
            ["repealed", "amended"],
        )
        self.assertIn("repealed", res_warn.answer.lower())
        self.assertIn("amended", res_warn.answer.lower())


class TestVerificationLayerCitations(unittest.TestCase):
    def test_grounded_and_ungrounded_section_citations(self) -> None:
        verifier = VerificationLayer()
        chunks = [
            {
                "chunk_id": "c-3",
                "act_source": "company_act_2063",
                "hierarchy": {"sec": 3},
                "status": "active",
                "text": "कम्पनीको संस्थापना।",
            },
            {
                "chunk_id": "c-166",
                "act_source": "company_act_2063",
                "hierarchy": {"sec": 166},
                "status": "active",
                "text": "मुनाफा वितरण नगर्ने कम्पनी।",
            },
        ]
        graph = [
            {
                "process_name": "Company Registration",
                "steps": [{"step_number": 1, "action": "Apply under Section 4", "section": 4}],
            }
        ]

        valid_answer = "दफा ३, Section 4, र दफा १६६ अनुसार कम्पनी दर्ता गर्न सकिन्छ।"
        res_valid = verifier.verify(valid_answer, chunks=chunks, graph=graph)
        self.assertTrue(res_valid.is_valid)
        self.assertEqual(res_valid.cited_sections, [3, 4, 166])
        self.assertEqual(res_valid.ungrounded_citations, [])
        self.assertEqual(res_valid.answer, valid_answer)

        hallucinated_answer = "दफा ३ र दफा ९९ तथा Section 420 अनुसार आवेदन दिनुपर्छ।"
        res_bad = verifier.verify(hallucinated_answer, chunks=chunks, graph=graph)
        self.assertFalse(res_bad.is_valid)
        self.assertEqual(res_bad.cited_sections, [3, 99, 420])
        self.assertEqual(len(res_bad.ungrounded_citations), 2)
        self.assertIn("९९", " ".join(res_bad.ungrounded_citations))
        self.assertIn("420", " ".join(res_bad.ungrounded_citations))
        self.assertIn("Verification Notice", res_bad.answer)
        self.assertTrue(res_bad.answer.startswith(hallucinated_answer))

    def test_grounded_and_ungrounded_act_citations(self) -> None:
        verifier = VerificationLayer()
        chunks = [
            {
                "chunk_id": "c-3",
                "act_source": "company_act_2063",
                "hierarchy": {"sec": 3},
                "status": "active",
                "text": "कम्पनीको संस्थापना।",
            },
        ]

        # Grounded Nepali and English Act citations matching company_act_2063
        valid_ne = "कम्पनी ऐन, २०६३ को दफा ३ अनुसार कम्पनी स्थापना हुन्छ।"
        res_ne = verifier.verify(valid_ne, chunks=chunks)
        self.assertTrue(res_ne.is_valid)
        self.assertEqual(res_ne.ungrounded_citations, [])
        self.assertEqual(res_ne.answer, valid_ne)

        valid_en = "Under Section 3 of the Companies Act, 2063, a company may be incorporated."
        res_en = verifier.verify(valid_en, chunks=chunks)
        self.assertTrue(res_en.is_valid)
        self.assertEqual(res_en.ungrounded_citations, [])
        self.assertEqual(res_en.answer, valid_en)

        # Ungrounded Act citation (Labor Act, 2074 / श्रम ऐन, २०७४ not in context)
        bad_act = "श्रम ऐन, २०७४ र Labor Act, 2074 को दफा ३ अनुसार दर्ता हुन्छ।"
        res_bad = verifier.verify(bad_act, chunks=chunks)
        self.assertFalse(res_bad.is_valid)
        self.assertIn("श्रम ऐन, २०७४", res_bad.ungrounded_citations)
        self.assertIn("Labor Act, 2074", res_bad.ungrounded_citations)
        self.assertIn("Verification Notice", res_bad.answer)


class TestVerificationPipelineIntegration(unittest.TestCase):
    def test_rag_pipeline_verifies_with_zero_extra_llm_calls(self) -> None:
        chunks = [
            {
                "chunk_id": "c-3",
                "act_source": "company_act_2063",
                "hierarchy": {"sec": 3},
                "status": "repealed",
                "text": "कम्पनीको संस्थापना सम्बन्धी पुरानो व्यवस्था।",
            }
        ]
        llm = _FakeLLMClient(["दफा ३ र दफा ९९ अनुसार दर्ता हुन्छ।"])
        rag = RAGPipeline(
            embedder=_FakeEmbedder(),
            vector_store=_FakeVectorStore(chunks),
            llm_client=llm,
            use_graph=False,
        )

        result = rag.answer("कम्पनी दर्ता कसरी हुन्छ?")
        self.assertEqual(len(llm.calls), 1)
        self.assertIn("Temporal Warning", result["answer"])
        self.assertIn("Verification Notice", result["answer"])
        self.assertIn("९९", result["answer"])
        self.assertFalse(result["verification"].is_valid)

    def test_legal_agent_verifies_iterative_rag_with_zero_extra_llm_calls(self) -> None:
        chunks = [
            {
                "chunk_id": "c-3",
                "act_source": "company_act_2063",
                "hierarchy": {"sec": 3},
                "status": "active",
                "text": "कम्पनीको संस्थापना सम्बन्धी व्यवस्था।",
            }
        ]
        llm = _FakeLLMClient([
            '{"sufficient": true, "missing": "", "next_actions": []}',
            "दफा ३ र दफा ४०४ अनुसार कम्पनी दर्ता हुन्छ।",
        ])
        rag = RAGPipeline(
            embedder=_FakeEmbedder(),
            vector_store=_FakeVectorStore(chunks),
            llm_client=llm,
            use_graph=False,
        )
        agent = LegalAgent(
            rag_pipeline=rag,
            graph_store=None,
            llm_client=llm,
            translate_backoff_sec=0.0,
        )

        events: list[tuple[str, str]] = []
        resp = agent.respond(
            "What is section 3?",
            on_step=lambda s, m: events.append((s, m)),
        )
        # 1 completeness check + 1 answer generation = 2 LLM calls (0 for verification)
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("Verification Notice", resp.answer)
        self.assertIn("४०४", resp.answer)
        self.assertIn("verify", [s for s, _ in events])


if __name__ == "__main__":
    unittest.main()
