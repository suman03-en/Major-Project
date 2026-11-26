"""
Query Transformation Layer
==========================
Classifies query shape (``SIMPLE``, ``BROAD``, ``MULTI_PART``) via a pluggable
:class:`QueryTypeClassifier` seam (ready for a future trained mBERT model) and
transforms the query into 1–3 focused Nepali Devanagari search queries.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from src.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```", re.IGNORECASE)

_TRANSFORM_SYSTEM_PROMPT = """You are a Nepali legal search query planner.
Given a user question (in Nepali, Romanized Nepali, or English), classify its shape and produce Nepali Devanagari search queries for hybrid vector+keyword retrieval.

Respond with ONLY a compact JSON object (no prose):
{
  "query_type": "SIMPLE" | "BROAD" | "MULTI_PART",
  "sub_queries": ["नेपाली खोज प्रश्न १", "..."]
}

Rules:
- Always write every entry in "sub_queries" in clear Nepali Devanagari legal terminology.
- SIMPLE: single focused legal fact or section lookup -> 1 concise Nepali query.
- BROAD: vague, broad, or semantic question -> 2 to 3 complementary Nepali queries covering different statutory angles.
- MULTI_PART: compound or comparison question with multiple sub-questions -> 2 to 3 independent Nepali sub-queries (one per sub-question or compared entity).
- Never emit more than 3 sub_queries."""


class QueryType(str, Enum):
    """Query shape driving retrieval transformation."""

    SIMPLE = "SIMPLE"
    BROAD = "BROAD"
    MULTI_PART = "MULTI_PART"


class QueryTypeClassifier(Protocol):
    """Pluggable classifier seam (e.g., future trained mBERT model)."""

    def classify(self, query: str) -> QueryType:
        """Return the predicted :class:`QueryType` for *query*."""
        ...


@dataclass(frozen=True)
class TransformedQuery:
    """Result of :meth:`QueryTransformer.transform`."""

    original_query: str
    query_type: QueryType
    sub_queries: list[str]
    was_transformed: bool


def _parse_transform_json(raw: str) -> tuple[QueryType | None, list[str]]:
    """Parse ``{query_type, sub_queries}`` from LLM output."""
    cleaned = _CODE_FENCE_RE.sub("", raw.strip()).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None, []
    if not isinstance(data, dict):
        return None, []

    raw_type = str(data.get("query_type", "")).strip().upper()
    qtype: QueryType | None = None
    for member in QueryType:
        if member.value == raw_type:
            qtype = member
            break

    raw_queries = data.get("sub_queries")
    queries: list[str] = []
    if isinstance(raw_queries, list):
        seen: set[str] = set()
        for item in raw_queries:
            q = str(item).strip()
            if q and q not in seen:
                seen.add(q)
                queries.append(q)

    return qtype, queries


class QueryTransformer:
    """
    Single-call query classifier, Nepali translator, and multi-query/decomposition planner.

    Parameters
    ----------
    llm_client:
        Helper LLM client used to generate Nepali sub-queries.
    classifier:
        Optional external :class:`QueryTypeClassifier` (such as a future
        mBERT model). When provided, its predicted :class:`QueryType` is
        passed as a hint and used as the authoritative ``query_type``.
    max_sub_queries:
        Upper bound on emitted sub-queries (default ``3``).
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        classifier: QueryTypeClassifier | None = None,
        max_sub_queries: int = 3,
    ) -> None:
        self._llm = llm_client
        self._classifier = classifier
        self.max_sub_queries = max(1, max_sub_queries)

    def transform(
        self,
        query: str,
        query_type_override: QueryType | None = None,
    ) -> TransformedQuery:
        """Classify *query* and return 1–3 Nepali retrieval queries (fail-open to ``SIMPLE``)."""
        text = query.strip()
        if not text:
            return TransformedQuery(
                original_query=query,
                query_type=QueryType.SIMPLE,
                sub_queries=[query],
                was_transformed=False,
            )

        forced_type = query_type_override
        if forced_type is None and self._classifier is not None:
            try:
                forced_type = self._classifier.classify(text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("External QueryTypeClassifier failed (%s); falling back to LLM.", exc)

        user_msg = (
            f"Hint query_type: {forced_type.value}\nQuestion: {text}"
            if forced_type is not None
            else f"Question: {text}"
        )

        try:
            raw = self._llm.generate(_TRANSFORM_SYSTEM_PROMPT, user_msg)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query transformation failed (%s); failing open to SIMPLE.", exc)
            return TransformedQuery(
                original_query=query,
                query_type=forced_type or QueryType.SIMPLE,
                sub_queries=[text],
                was_transformed=False,
            )

        if not isinstance(raw, str) or not raw.strip() or raw.lstrip().startswith("[Error:"):
            return TransformedQuery(
                original_query=query,
                query_type=forced_type or QueryType.SIMPLE,
                sub_queries=[text],
                was_transformed=False,
            )

        parsed_type, parsed_queries = _parse_transform_json(raw)
        final_type = forced_type or parsed_type or QueryType.SIMPLE
        if not parsed_queries:
            return TransformedQuery(
                original_query=query,
                query_type=final_type,
                sub_queries=[text],
                was_transformed=False,
            )

        capped = parsed_queries[: 1 if final_type == QueryType.SIMPLE else self.max_sub_queries]
        was_transformed = len(capped) > 1 or capped != [text]
        return TransformedQuery(
            original_query=query,
            query_type=final_type,
            sub_queries=capped,
            was_transformed=was_transformed,
        )
