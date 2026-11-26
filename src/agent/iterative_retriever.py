"""
Bounded Iterative Retriever
===========================
Multi-loop retrieval orchestrator (``max_loops = 2``, ``max_provisions = 8``)
that combines vector clause search, section cross-reference lookup, and
Neo4j knowledge-graph procedural search.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.config import get_settings

if TYPE_CHECKING:
    from src.llm.base import BaseLLMClient
    from src.rag.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

_MAX_ACTIONS_PER_LOOP: int = 2
_MAX_PROVISION_PREVIEW_CHARS: int = 280

_COMPLETENESS_SYSTEM_PROMPT = """You are a legal retrieval planner for Nepali law and business registration.
Inspect the user's question and the retrieved Provisions and Procedures.
Respond with ONLY a compact JSON object (no prose):
{
  "sufficient": true | false,
  "missing": "short description of missing section or detail, or empty string",
  "next_actions": [
    {"tool": "fetch_section", "sec_num": 166, "act_source": "company_act_2063"},
    {"tool": "search_clauses", "query": "focused search query"},
    {"tool": "search_graph", "query": "procedure name or topic"}
  ]
}
Rules:
- Set "sufficient": true and "next_actions": [] if the context already answers the question or no clear follow-up exists.
- If a retrieved provision explicitly cross-references another section (e.g. "दफा १६६" or "Section 166"), emit a "fetch_section" action with integer "sec_num".
- Emit at most 2 actions in "next_actions"."""


@dataclass
class RetrievalResult:
    """Output of :meth:`IterativeRetriever.retrieve`."""

    chunks: list[dict[str, Any]] = field(default_factory=list)
    graph: list[dict[str, Any]] = field(default_factory=list)
    loops_used: int = 1
    sufficient: bool = False
    missing: str = ""


def _parse_completeness_json(raw: str) -> dict[str, Any] | None:
    """Parse compact JSON from LLM output, stripping optional code fences."""
    cleaned = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _summarize_context(
    query: str,
    chunks: list[dict[str, Any]],
    graph: list[dict[str, Any]],
) -> str:
    """Build a compact summary of gathered Provisions and Procedures for the planner."""
    lines = [f"Question: {query}", "", f"Provisions ({len(chunks)}):"]
    for idx, ch in enumerate(chunks, 1):
        h = ch.get("hierarchy", {})
        sec = h.get("sec", "?")
        act = ch.get("act_source", "")
        text = ch.get("text", "").replace("\n", " ")[:_MAX_PROVISION_PREVIEW_CHARS]
        lines.append(f"{idx}. [{act} दफा {sec}] {text}")

    if graph:
        lines.append("")
        lines.append(f"Procedures ({len(graph)}):")
        for proc in graph:
            lines.append(f"- {proc.get('process_name', 'Unknown')}")

    return "\n".join(lines)


def _fuse_rrf(
    ranked_lists: list[list[dict[str, Any]]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse multiple ranked chunk lists using Reciprocal Rank Fusion (RRF)."""
    scores: dict[str, float] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for chunks in ranked_lists:
        for rank, ch in enumerate(chunks, start=1):
            cid = ch.get("chunk_id") or f"{ch.get('act_source', '')}:{ch.get('text', '')[:40]}"
            by_id.setdefault(cid, ch)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    ordered_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [by_id[cid] for cid in ordered_ids]


class IterativeRetriever:
    """Bounded iterative retrieval loop over ``RAGPipeline`` tools."""

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        llm_client: BaseLLMClient | None = None,
        max_loops: int | None = None,
        max_provisions: int | None = None,
    ) -> None:
        settings = get_settings()
        self._rag = rag_pipeline
        self._llm = llm_client or rag_pipeline.llm_client
        self.max_loops = max_loops if max_loops is not None else settings.AGENT_MAX_LOOPS
        self.max_provisions = (
            max_provisions if max_provisions is not None else settings.AGENT_MAX_PROVISIONS
        )

    def retrieve(
        self,
        query: str,
        retrieval_query: str | None = None,
        on_step: Callable[[str, str], None] | None = None,
        search_queries: list[str] | None = None,
    ) -> RetrievalResult:
        """Run up to ``max_loops`` retrieval rounds, capping at ``max_provisions``."""
        def _emit(stage: str, msg: str) -> None:
            if on_step:
                on_step(stage, msg)

        search_query = retrieval_query or query
        queries = [q.strip() for q in (search_queries or []) if q and q.strip()]
        if not queries:
            queries = [search_query]
        collected_chunks: list[dict[str, Any]] = []
        collected_graph: list[dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()
        seen_processes: set[str] = set()

        def _add_chunks(new_chunks: list[dict[str, Any]]) -> None:
            for ch in new_chunks:
                if len(collected_chunks) >= self.max_provisions:
                    break
                cid = ch.get("chunk_id") or f"{ch.get('act_source', '')}:{ch.get('text', '')[:40]}"
                if cid in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(cid)
                collected_chunks.append(ch)

        def _add_graph(new_graph: list[dict[str, Any]]) -> None:
            for proc in new_graph:
                pname = proc.get("process_name", "")
                if pname and pname in seen_processes:
                    continue
                if pname:
                    seen_processes.add(pname)
                collected_graph.append(proc)

        # Loop 1: initial vector clause search (single or RRF-fused multi-query) + graph search
        _emit("retrieve_loop", f"Retrieval loop 1/{self.max_loops}: searching legal clauses…")
        if len(queries) == 1:
            _add_chunks(self._rag.search_clauses(queries[0], top_k=8))
        else:
            ranked_lists = [self._rag.search_clauses(q, top_k=5) for q in queries]
            _add_chunks(_fuse_rrf(ranked_lists))

        if self._rag.use_graph and self._rag.graph_store:
            _emit("retrieve_graph", "Checking Neo4j knowledge graph for procedures…")
            _add_graph(self._rag.search_graph(search_query, collected_chunks))

        loops_used = 1
        sufficient = False
        missing = ""

        while loops_used < self.max_loops and len(collected_chunks) < self.max_provisions:
            plan = self._check_completeness(query, collected_chunks, collected_graph)
            if plan is None:
                break

            sufficient = bool(plan.get("sufficient", False))
            missing = str(plan.get("missing", ""))
            actions = plan.get("next_actions") or []
            if sufficient or not isinstance(actions, list) or not actions:
                break

            loops_used += 1
            _emit(
                "retrieve_loop",
                f"Retrieval loop {loops_used}/{self.max_loops}: following {min(len(actions), _MAX_ACTIONS_PER_LOOP)} gap(s)…",
            )
            self._execute_actions(
                actions[:_MAX_ACTIONS_PER_LOOP],
                search_query,
                collected_chunks,
                _add_chunks,
                _add_graph,
                _emit,
            )

        return RetrievalResult(
            chunks=collected_chunks,
            graph=collected_graph,
            loops_used=loops_used,
            sufficient=sufficient,
            missing=missing,
        )

    def _check_completeness(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        graph: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Ask the LLM whether gathered context is sufficient; return None on error."""
        prompt = _summarize_context(query, chunks, graph)
        try:
            raw = self._llm.generate(_COMPLETENESS_SYSTEM_PROMPT, prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Iterative completeness check failed: %s", exc)
            return None

        parsed = _parse_completeness_json(raw)
        if parsed is None:
            logger.warning("Malformed completeness JSON from LLM; stopping retrieval loop.")
        return parsed

    def _execute_actions(
        self,
        actions: list[Any],
        default_query: str,
        collected_chunks: list[dict[str, Any]],
        add_chunks: Callable[[list[dict[str, Any]]], None],
        add_graph: Callable[[list[dict[str, Any]]], None],
        emit: Callable[[str, str], None],
    ) -> None:
        """Run up to ``_MAX_ACTIONS_PER_LOOP`` retrieval tool calls."""
        for action in actions:
            if not isinstance(action, dict):
                continue
            tool = action.get("tool")
            if tool == "fetch_section":
                sec_raw = action.get("sec_num")
                try:
                    sec_num = int(sec_raw)
                except (TypeError, ValueError):
                    continue
                act_source = action.get("act_source") or None
                emit("fetch_section", f"Fetching section दफा {sec_num}…")
                add_chunks(self._rag.fetch_section(sec_num, act_source=act_source))
            elif tool == "search_clauses":
                if len(collected_chunks) >= self.max_provisions:
                    continue
                q = str(action.get("query") or default_query)
                try:
                    top_k = max(1, min(int(action.get("top_k", 6)), 8))
                except (TypeError, ValueError):
                    top_k = 6
                add_chunks(self._rag.search_clauses(q, top_k=top_k))
            elif tool == "search_graph":
                q = str(action.get("query") or default_query)
                emit("retrieve_graph", f"Searching knowledge graph for: {q}")
                add_graph(self._rag.search_graph(q, collected_chunks))
