import unittest
from collections.abc import Iterator
from typing import Any

import numpy as np

from src.embedding.vector_store import QdrantVectorStore
from src.rag.rag_pipeline import RAGPipeline


class FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> dict[str, Any]:
        return {
            "dense_vecs": np.zeros((len(texts), 4), dtype=np.float32),
            "lexical_weights": [{"1": 0.5} for _ in texts],
        }


class FakeVectorStore:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self._results = results

    def search(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._results[: kwargs.get("top_k", 5)]


class FakeGraphStore:
    def find_relevant_processes(
        self, query: str, chunks: list[dict[str, Any]], max_processes: int = 3
    ) -> list[dict[str, Any]]:
        return [
            {
                "process_name": "कम्पनी दर्ता",
                "steps": [
                    {
                        "step_number": 1,
                        "action": "अनलाइन आवेदन दिनुहोस्",
                        "office": "कम्पनी रजिष्ट्रारको कार्यालय",
                        "duration": "७ दिन",
                        "documents": ["प्रबन्धपत्र"],
                        "fees": ["रु. १,०००"],
                    }
                ],
                "documents": ["प्रबन्धपत्र"],
                "offices": ["कम्पनी रजिष्ट्रारको कार्यालय"],
            }
        ]

    def close(self) -> None:
        pass


class FakeLLMClient:
    model = "fake-model"

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float | None = None,
    ) -> str:
        return f"MOCK_ANSWER: {len(user_message)} chars"

    def stream(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float | None = None,
    ) -> Iterator[str]:
        yield "MOCK_"
        yield "STREAM"


class TestRAGPipelineSeams(unittest.TestCase):
    def test_answer_uses_injected_adapters(self) -> None:
        fake_chunks = [
            {
                "score": 0.9,
                "chunk_id": "company_act-2063-sec3-sub1",
                "act_source": "company_act",
                "type": "subsection",
                "hierarchy": {"sec": 3, "sec_title": "संस्थापना", "sub": 1},
                "text": "कुनै व्यक्तिले कम्पनी संस्थापना गर्न सक्नेछ।",
            },
            {
                "score": 0.8,
                "chunk_id": "stub",
                "act_source": "company_act",
                "type": "section",
                "hierarchy": {},
                "text": "छोटो",  # Should be filtered out (< MIN_CHUNK_TEXT_LEN)
            },
        ]

        rag = RAGPipeline(
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
            vector_store=FakeVectorStore(fake_chunks),  # type: ignore[arg-type]
            graph_store=FakeGraphStore(),  # type: ignore[arg-type]
            llm_client=FakeLLMClient(),
            use_graph=True,
            use_reranker=False,
        )

        result = rag.answer("कम्पनी दर्ता कसरी गर्ने?")
        self.assertTrue(result["answer"].startswith("MOCK_ANSWER:"))
        self.assertEqual(len(result["chunks"]), 1)
        self.assertEqual(len(result["graph"]), 1)
        self.assertEqual(result["graph"][0]["process_name"], "कम्पनी दर्ता")

    def test_vector_store_upsert_chunks_in_memory(self) -> None:
        store = QdrantVectorStore(url=":memory:", collection_name="test_col", vector_dim=4)
        store.ensure_collection()
        chunks = [
            {
                "id": "act-1-sec1",
                "type": "section",
                "hierarchy": {"sec": 1},
                "text": "परीक्षण पाठ यहाँ छ।",
            }
        ]
        count = store.upsert_chunks(
            chunks=chunks,
            dense_vecs=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
            sparse_vecs=[{"10": 0.75}],
            act_source="test_act",
        )
        self.assertEqual(count, 1)
        info = store.get_collection_info()
        self.assertEqual(info["points_count"], 1)

    def test_relevance_gate_skips_llm_when_reranker_score_low(self) -> None:
        fake_chunks = [
            {
                "score": 0.03,
                "chunk_id": "low-score-chunk",
                "act_source": "company_act",
                "type": "subsection",
                "hierarchy": {"sec": 1},
                "text": "यो असम्बन्धित पाठ हो जुन सन्दर्भमा मिल्दैन।",
            }
        ]

        class FakeReranker:
            def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
                return candidates[:top_k]

        rag = RAGPipeline(
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
            vector_store=FakeVectorStore(fake_chunks),  # type: ignore[arg-type]
            reranker=FakeReranker(),  # type: ignore[arg-type]
            llm_client=FakeLLMClient(),
            use_graph=False,
            use_reranker=True,
        )

        result = rag.answer("what is public company in nepal", response_language="en")
        self.assertEqual(result["chunks"], [])
        self.assertNotIn("MOCK_ANSWER:", result["answer"])
        self.assertIn("could not find sufficient", result["answer"])

    def test_graph_match_bypasses_low_reranker_score_gate(self) -> None:
        fake_chunks = [
            {
                "score": 0.03,
                "chunk_id": "indirect-chunk",
                "act_source": "company_act",
                "type": "subsection",
                "hierarchy": {"sec": 4, "sub": 1},
                "text": "पब्लिक कम्पनी संस्थापनाको लागि निवेदन दिनुपर्नेछ।",
            }
        ]

        class FakeReranker:
            def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
                return candidates[:top_k]

        rag = RAGPipeline(
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
            vector_store=FakeVectorStore(fake_chunks),  # type: ignore[arg-type]
            graph_store=FakeGraphStore(),  # type: ignore[arg-type]
            reranker=FakeReranker(),  # type: ignore[arg-type]
            llm_client=FakeLLMClient(),
            use_graph=True,
            use_reranker=True,
        )

        result = rag.answer("nepal ma public company register garne tarika kk hun")
        self.assertTrue(result["answer"].startswith("MOCK_ANSWER:"))
        self.assertEqual(len(result["graph"]), 1)

    def test_estimate_tokens_weights_devanagari_higher_than_ascii(self) -> None:
        from src.rag.rag_pipeline import estimate_tokens

        nepali = "कम्पनी ऐन २०६३ को दफा ३ बमोजिम संस्थापना गर्न सकिनेछ।"
        english = "a" * len(nepali)
        self.assertGreater(estimate_tokens(nepali), estimate_tokens(english) * 2)

    def test_assemble_context_caps_section_siblings_and_prioritizes_primary(self) -> None:
        from types import SimpleNamespace

        primary_chunks = [
            {
                "score": 0.95,
                "chunk_id": "act-sec43-sub1",
                "act_source": "company_act_2063",
                "type": "subsection",
                "hierarchy": {"ch": 3, "sec": 43, "sub": 1},
                "text": "प्राथमिक दफा ४३ उपदफा १ को मुख्य व्यवस्था यहाँ छ।",
            },
            {
                "score": 0.90,
                "chunk_id": "act-sec99-sub1",
                "act_source": "company_act_2063",
                "type": "subsection",
                "hierarchy": {"ch": 5, "sec": 99, "sub": 1},
                "text": "दोस्रो प्राथमिक दफा ९९ को मुख्य व्यवस्था यहाँ छ।",
            },
        ]
        sibling_payloads = [
            SimpleNamespace(
                payload={
                    "chunk_id": f"act-sec43-sib{i}",
                    "act_source": "company_act_2063",
                    "type": "subsection",
                    "hierarchy": {"ch": 3, "sec": 43, "sub": i + 2},
                    "text": f"दफा ४३ को अतिरिक्त उपदफा {i + 2} को पाठ।",
                }
            )
            for i in range(6)
        ]

        class FakeScrollVectorStore(FakeVectorStore):
            def __init__(self, results):
                super().__init__(results)
                self.collection_name = "nepali_legal_chunks"
                self.client = self

            def scroll(self, scroll_filter=None, **kwargs):
                must = getattr(scroll_filter, "must", []) or []
                sec_vals = [c.match.value for c in must if c.key == "hierarchy.sec"]
                if sec_vals and sec_vals[0] != 43:
                    return [], None
                return sibling_payloads, None

        rag = RAGPipeline(
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
            vector_store=FakeScrollVectorStore(primary_chunks),  # type: ignore[arg-type]
            llm_client=FakeLLMClient(),
            use_graph=False,
            max_context_tokens=3500,
            max_siblings_per_section=2,
        )

        ctx = rag.assemble_context("दफा ४३ र ९९ के भन्छ?", primary_chunks, [])
        self.assertIn("प्राथमिक दफा ४३ उपदफा १", ctx)
        self.assertIn("दोस्रो प्राथमिक दफा ९९", ctx)
        self.assertIn("📌 उद्धृत गर्नुहोस्:", ctx)
        included_siblings = [i for i in range(6) if f"अतिरिक्त उपदफा {i + 2}" in ctx]
        self.assertLessEqual(len(included_siblings), 2)

    def test_assemble_context_enforces_token_budget_on_dense_devanagari(self) -> None:
        from src.rag.rag_pipeline import estimate_tokens

        big_chunks = [
            {
                "score": 0.9 - i * 0.1,
                "chunk_id": f"c-{i}",
                "act_source": "company_act_2063",
                "type": "section",
                "hierarchy": {"sec": i + 1},
                "text": ("कम्पनी दर्ता र सञ्चालन सम्बन्धी विस्तृत कानूनी व्यवस्था " * 25),
                "explanations": ["लामो स्पष्टीकरण विवरण " * 15],
            }
            for i in range(6)
        ]
        rag = RAGPipeline(
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
            vector_store=FakeVectorStore(big_chunks),  # type: ignore[arg-type]
            llm_client=FakeLLMClient(),
            use_graph=False,
            max_context_tokens=900,
            max_siblings_per_section=2,
        )

        ctx = rag.assemble_context("कम्पनी दर्ता", big_chunks, [])
        self.assertLessEqual(estimate_tokens(ctx), 950)
        self.assertIn("दफा 1", ctx)
        self.assertNotIn("दफा 6", ctx)

    def test_assemble_context_omits_duplicate_procedure_rollups(self) -> None:
        rag = RAGPipeline(
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
            vector_store=FakeVectorStore([]),  # type: ignore[arg-type]
            llm_client=FakeLLMClient(),
            use_graph=False,
        )
        graph = [
            {
                "process_name": "कम्पनी दर्ता",
                "steps": [
                    {
                        "step_number": 1,
                        "action": "अनलाइन आवेदन दिनुहोस्",
                        "office": "कम्पनी रजिष्ट्रारको कार्यालय",
                        "documents": ["प्रबन्धपत्र"],
                        "fees": ["रु. १,०००"],
                    }
                ],
                "documents": ["प्रबन्धपत्र"],
                "offices": ["कम्पनी रजिष्ट्रारको कार्यालय"],
            }
        ]
        ctx = rag.assemble_context("कम्पनी दर्ता", [], graph)
        self.assertIn("कम्पनी रजिष्ट्रारको कार्यालय", ctx)
        self.assertIn("प्रबन्धपत्र", ctx)
        self.assertNotIn("सम्पूर्ण आवश्यक कागजातहरू", ctx)
        self.assertNotIn("सम्बन्धित कार्यालयहरू", ctx)

    def test_answer_and_stream_return_clean_message_on_rate_limit(self) -> None:
        class RateLimitedLLMClient(FakeLLMClient):
            def generate(self, system_prompt: str, user_message: str, temperature=None) -> str:
                return '[Error: LLM generation failed — HTTP 429: {"error":{"message":"TPM limit"}}]'

            def stream(self, system_prompt: str, user_message: str, temperature=None):
                yield '\n[Error: streaming failed — HTTP 429: {"error":{"message":"TPM limit"}}]'

        fake_chunks = [
            {
                "score": 0.9,
                "chunk_id": "c-1",
                "act_source": "company_act",
                "type": "subsection",
                "hierarchy": {"sec": 3, "sub": 1},
                "text": "कुनै व्यक्तिले कम्पनी संस्थापना गर्न सक्नेछ।",
            }
        ]
        rag = RAGPipeline(
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
            vector_store=FakeVectorStore(fake_chunks),  # type: ignore[arg-type]
            llm_client=RateLimitedLLMClient(),
            use_graph=False,
        )
        res = rag.answer("What is section 3?", response_language="en")
        self.assertNotIn("[Error:", res["answer"])
        self.assertIn("rate", res["answer"].lower())

        streamed = "".join(rag.answer_stream("दफा ३ के हो?", response_language="ne"))
        self.assertNotIn("[Error:", streamed)
        self.assertIn("मिनेट", streamed)

        res_inferred_en = rag.answer("What is section 3?")
        self.assertIn("rate", res_inferred_en["answer"].lower())

    def test_assemble_context_truncates_oversized_first_procedure_and_stops_sibling_scroll(self) -> None:
        from src.rag.rag_pipeline import estimate_tokens

        scroll_calls = 0

        class CountingVectorStore(FakeVectorStore):
            def __init__(self, results):
                super().__init__(results)
                self.collection_name = "nepali_legal_chunks"
                self.client = self

            def scroll(self, **kwargs):
                nonlocal scroll_calls
                scroll_calls += 1
                return [], None

        big_proc = [
            {
                "process_name": "लामो प्रक्रिया",
                "steps": [
                    {
                        "step_number": i,
                        "action": "कम्पनी दर्ताको लागि अनलाइन फारम भर्नुहोस् र आवश्यक कागजात पेश गर्नुहोस्।",
                        "office": "कम्पनी रजिष्ट्रारको कार्यालय",
                    }
                    for i in range(1, 15)
                ],
            }
        ]
        rag = RAGPipeline(
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
            vector_store=CountingVectorStore([]),  # type: ignore[arg-type]
            llm_client=FakeLLMClient(),
            use_graph=False,
            max_context_tokens=200,
            max_siblings_per_section=2,
        )
        ctx = rag.assemble_context("कम्पनी दर्ता?", [], big_proc)
        self.assertLessEqual(estimate_tokens(ctx), 220)

        tight_chunks = [
            {
                "score": 0.9,
                "chunk_id": f"c-{i}",
                "act_source": "company_act_2063",
                "type": "subsection",
                "hierarchy": {"ch": 1, "sec": i + 1, "sub": 1},
                "text": "कम्पनी संस्थापना सम्बन्धी व्यवस्था " * 12,
            }
            for i in range(3)
        ]
        rag2 = RAGPipeline(
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
            vector_store=CountingVectorStore(tight_chunks),  # type: ignore[arg-type]
            llm_client=FakeLLMClient(),
            use_graph=False,
            max_context_tokens=300,
            max_siblings_per_section=2,
        )
        rag2.assemble_context("कम्पनी", tight_chunks, [])
        self.assertEqual(scroll_calls, 0)

    def test_assemble_context_includes_bounded_conversation_history(self) -> None:
        from src.rag.rag_pipeline import estimate_tokens

        rag = RAGPipeline(
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
            vector_store=FakeVectorStore([]),  # type: ignore[arg-type]
            llm_client=FakeLLMClient(),
            use_graph=False,
            max_context_tokens=400,
        )
        history = [
            {"role": "user", "content": "पहिलो पुरानो प्रश्न " * 20},
            {"role": "assistant", "content": "पहिलो पुरानो जवाफ " * 20},
            {"role": "user", "content": "दफा ४३ के भन्छ?"},
            {"role": "assistant", "content": "दफा ४३ मा कम्पनी दर्ता व्यवस्था छ।"},
        ]
        ctx = rag.assemble_context(
            "त्यसको दस्तुर कति हो?",
            [],
            [],
            conversation_history=history,
        )
        self.assertIn("दफा ४३ मा कम्पनी दर्ता व्यवस्था छ।", ctx)
        self.assertLessEqual(estimate_tokens(ctx), 420)

    def test_format_citations_deduplicates_and_formats_readable_labels(self) -> None:
        from src.rag.rag_pipeline import RAG_SYSTEM_PROMPT

        self.assertIn("inline", RAG_SYSTEM_PROMPT.lower())
        self.assertNotIn('end with a "स्रोत"', RAG_SYSTEM_PROMPT)

        rag = RAGPipeline(
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
            vector_store=FakeVectorStore([]),  # type: ignore[arg-type]
            llm_client=FakeLLMClient(),
            use_graph=False,
        )
        chunks = [
            {
                "chunk_id": "c1",
                "act_source": "company_act_2063",
                "hierarchy": {"sec": 3, "sub": 1},
                "text": "कम्पनी संस्थापना सम्बन्धी व्यवस्था।",
                "status": "active",
            },
            {
                "chunk_id": "c1-dup",
                "act_source": "company_act_2063",
                "hierarchy": {"sec": 3, "sub": 1},
                "text": "कम्पनी संस्थापना सम्बन्धी व्यवस्था।",
                "status": "active",
            },
        ]
        citations = rag.format_citations(chunks)
        self.assertEqual(len(citations), 1)
        self.assertIn("दफा 3", citations[0]["label"])
        self.assertEqual(citations[0]["status"], "active")
        self.assertIn("कम्पनी संस्थापना", citations[0]["excerpt"])

    def test_format_citations_filters_to_only_used_sources_in_answer(self) -> None:
        rag = RAGPipeline(
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
            vector_store=FakeVectorStore([]),  # type: ignore[arg-type]
            llm_client=FakeLLMClient(),
            use_graph=False,
        )
        chunks = [
            {
                "chunk_id": "c1",
                "act_source": "company_act_2063",
                "hierarchy": {"sec": 3, "sub": 1},
                "text": "कम्पनी संस्थापना सम्बन्धी व्यवस्था।",
                "status": "active",
            },
            {
                "chunk_id": "c2",
                "act_source": "company_act_2063",
                "hierarchy": {"sec": 43},
                "text": "अर्को दफाको व्यवस्था।",
                "status": "active",
            },
            {
                "chunk_id": "c3",
                "act_source": "company_act_2063",
                "hierarchy": {"sec": 99},
                "text": "असम्बन्धित दफा।",
                "status": "active",
            },
        ]
        answer = "कम्पनी ऐन, २०६३, दफा ३(१) अनुसार कम्पनी संस्थापना गर्न सकिन्छ।"
        citations = rag.format_citations(chunks, answer=answer)
        self.assertEqual(len(citations), 1)
        self.assertIn("दफा 3", citations[0]["label"])

        no_cite_answer = "उपलब्ध सन्दर्भमा यस प्रश्नको पर्याप्त जानकारी छैन।"
        self.assertEqual(rag.format_citations(chunks, answer=no_cite_answer), [])


if __name__ == "__main__":
    unittest.main()
