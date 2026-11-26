import json
import unittest
from unittest.mock import MagicMock

from src.agent.query_transformer import QueryTransformer, QueryType


class TestQueryTransformer(unittest.TestCase):
    def test_broad_query_expands_to_multiple_nepali_queries(self) -> None:
        llm = MagicMock()
        llm.generate.return_value = json.dumps(
            {
                "query_type": "BROAD",
                "sub_queries": [
                    "नेपालमा कम्पनी दर्ता सम्बन्धी कानूनी व्यवस्था",
                    "प्राइभेट र पब्लिक कम्पनी बीचको भिन्नता",
                    "कम्पनी ऐन २०६३ को मुख्य नियम",
                ],
            },
            ensure_ascii=False,
        )
        qt = QueryTransformer(llm)
        res = qt.transform("Tell me about company rules in Nepal")
        self.assertEqual(res.query_type, QueryType.BROAD)
        self.assertEqual(len(res.sub_queries), 3)
        self.assertTrue(res.was_transformed)

    def test_multipart_query_decomposes_into_subqueries(self) -> None:
        llm = MagicMock()
        llm.generate.return_value = (
            '```json\n{"query_type": "MULTI_PART", "sub_queries": '
            '["प्राइभेट कम्पनी दर्ता शुल्क कति छ?", "पब्लिक कम्पनीको न्यूनतम पूँजी कति चाहिन्छ?"]}\n```'
        )
        qt = QueryTransformer(llm)
        res = qt.transform("What is private company fee and public company minimum capital?")
        self.assertEqual(res.query_type, QueryType.MULTI_PART)
        self.assertEqual(len(res.sub_queries), 2)
        self.assertTrue(res.was_transformed)

    def test_pluggable_classifier_overrides_llm_type(self) -> None:
        llm = MagicMock()
        llm.generate.return_value = json.dumps(
            {
                "query_type": "BROAD",
                "sub_queries": ["दफा ४३", "दफा ४४"],
            },
            ensure_ascii=False,
        )
        ext_clf = MagicMock()
        ext_clf.classify.return_value = QueryType.SIMPLE
        qt = QueryTransformer(llm, classifier=ext_clf)
        res = qt.transform("What is section 43?")
        self.assertEqual(res.query_type, QueryType.SIMPLE)
        self.assertEqual(res.sub_queries, ["दफा ४३"])
        ext_clf.classify.assert_called_once_with("What is section 43?")

    def test_llm_error_or_malformed_json_fails_open_to_simple(self) -> None:
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("429")
        qt = QueryTransformer(llm)
        res = qt.transform("company darta")
        self.assertEqual(res.query_type, QueryType.SIMPLE)
        self.assertEqual(res.sub_queries, ["company darta"])
        self.assertFalse(res.was_transformed)

        llm2 = MagicMock()
        llm2.generate.return_value = "not valid json"
        res2 = QueryTransformer(llm2).transform("company darta")
        self.assertEqual(res2.query_type, QueryType.SIMPLE)
        self.assertEqual(res2.sub_queries, ["company darta"])


if __name__ == "__main__":
    unittest.main()
