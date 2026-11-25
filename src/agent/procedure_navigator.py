"""
Procedure Navigator
===================
Wraps the Neo4j graph store to provide interactive, session-aware
step-by-step navigation of Nepali legal procedures (e.g. company registration).

The navigator is *stateless* itself — all mutable state lives in
:class:`~src.agent.session_store.SessionState`.  Call :meth:`navigate` with
the current session and it updates the session in-place.

Usage::

    from src.agent.procedure_navigator import ProcedureNavigator, NavigationResult
    from src.knowledge_base.graph_store import Neo4jGraphStore

    nav = ProcedureNavigator(graph_store)
    result = nav.navigate("कम्पनी दर्ता कसरी गर्ने?", session)
    print(result.step["action"])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.agent.session_store import SessionState
    from src.knowledge_base.graph_store import Neo4jGraphStore

logger = logging.getLogger(__name__)


@dataclass
class NavigationResult:
    """
    Structured result returned by :meth:`ProcedureNavigator.navigate`.

    Attributes
    ----------
    process_name:
        Human-readable Neo4j ``Process.name``.
    total_steps:
        Total number of steps in the process.
    current_step:
        1-based index of the step the user is *now on*.
    step:
        Dict with keys ``action``, ``office``, ``documents``, ``fees``,
        ``duration``, ``prerequisite`` for the current step.  Empty dict
        when ``is_complete`` is ``True``.
    is_complete:
        ``True`` when the user has walked past the last step.
    progress_pct:
        Percentage of steps completed (0–100).
    """

    process_name: str
    total_steps: int
    current_step: int
    step: dict[str, Any]
    is_complete: bool
    progress_pct: float


class ProcedureNavigator:
    """
    Interactive procedure walker backed by :class:`~src.knowledge_base.graph_store.Neo4jGraphStore`.

    Parameters
    ----------
    graph_store:
        Connected Neo4j store instance.
    """

    def __init__(self, graph_store: Neo4jGraphStore) -> None:
        self._graph = graph_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def navigate(
        self,
        query: str,
        session: SessionState,
        advance: bool = False,
    ) -> NavigationResult | None:
        """
        Handle a user query and return the current navigation state.

        Parameters
        ----------
        query:
            Raw user text (used for process discovery when session has no
            active process).
        session:
            Mutable session state; updated in-place.
        advance:
            When ``True`` (follow-up "next" intent), increment
            ``session.current_step`` before returning.

        Returns
        -------
        NavigationResult | None
            ``None`` if no matching process could be found in the graph.
        """
        # --- Step 1: Resolve the active process -------------------------
        if not session.has_active_process:
            process_name = self._find_process(query)
            if not process_name:
                logger.info("No matching Neo4j process for query: %s", query[:60])
                return None
            session.active_process = process_name
            session.current_step = 1

        process_name = session.active_process  # guaranteed non-None here

        # --- Step 2: Load all steps (once per process) ------------------
        steps = self._graph.get_steps_for_process(process_name)
        total = len(steps)
        session.total_steps = total

        # --- Step 3: Advance step if requested --------------------------
        if advance and session.current_step <= total:
            session.current_step += 1

        # --- Step 4: Check completion -----------------------------------
        if session.current_step > total:
            return NavigationResult(
                process_name=process_name,
                total_steps=total,
                current_step=session.current_step,
                step={},
                is_complete=True,
                progress_pct=100.0,
            )

        # --- Step 5: Return current step --------------------------------
        # steps list is 0-indexed, step numbers are 1-based
        idx = session.current_step - 1
        step_data = steps[idx] if idx < len(steps) else {}

        progress = (session.current_step - 1) / total * 100 if total else 0.0

        logger.info(
            "Navigator: process=%s  step=%d/%d",
            process_name,
            session.current_step,
            total,
        )
        return NavigationResult(
            process_name=process_name,
            total_steps=total,
            current_step=session.current_step,
            step=step_data,
            is_complete=False,
            progress_pct=round(progress, 1),
        )

    def reset(self, session: SessionState) -> None:
        """Clear the active procedure from *session* (user said "done" or cancelled)."""
        session.active_process = None
        session.current_step = 0
        session.total_steps = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_process(self, query: str) -> str | None:
        """
        Return the best-matching Neo4j process name for *query*, or ``None``.

        Uses the same keyword-overlap heuristic as
        :meth:`~src.knowledge_base.graph_store.Neo4jGraphStore.find_relevant_processes`
        but returns only the top-1 name so the navigator can focus on a
        single procedure.
        """
        try:
            all_processes = self._graph.get_all_processes()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch process list from Neo4j: %s", exc)
            return None

        if not all_processes:
            return None

        text = query.lower()

        # Exact substring match first (fastest)
        for name in all_processes:
            if name.lower() in text:
                return name

        # Word-overlap fallback
        query_words = {w for w in query.split() if len(w) > 2}
        best_name: str | None = None
        best_overlap = 0
        for name in all_processes:
            overlap = len(query_words & set(name.split()))
            if overlap > best_overlap:
                best_overlap = overlap
                best_name = name

        return best_name if best_overlap > 0 else None

    @staticmethod
    def format_step_text(result: NavigationResult, language: str = "ne") -> str:
        """
        Render *result* as a human-readable string for the LLM or direct reply.

        Parameters
        ----------
        result:
            Navigation result to render.
        language:
            ``"ne"`` for Nepali labels, ``"en"`` for English.
        """
        if result.is_complete:
            if language == "ne":
                return (
                    f"✅ **{result.process_name}** सम्पन्न भयो!\n"
                    "सबै चरणहरू पूरा गर्नुभयो। कुनै प्रश्न छ भने सोध्नुहोस्।"
                )
            return (
                f"✅ **{result.process_name}** — All steps complete!\n"
                "Feel free to ask any follow-up questions."
            )

        s = result.step
        lines: list[str] = []

        if language == "ne":
            lines.append(
                f"📋 **{result.process_name}**  "
                f"[चरण {result.current_step}/{result.total_steps}  "
                f"— {result.progress_pct:.0f}% पूरा]"
            )
            lines.append(f"\n**{result.current_step}. {s.get('action', '')}**")
            if s.get("office"):
                lines.append(f"🏢 **कार्यालय:** {s['office']}")
            docs = [d for d in (s.get("documents") or []) if d]
            if docs:
                lines.append("📄 **आवश्यक कागजातहरू:**")
                lines.extend(f"   • {d}" for d in docs)
            fees = [f for f in (s.get("fees") or []) if f]
            if fees:
                lines.append(f"💰 **दस्तुर:** {', '.join(str(f) for f in fees)}")
            if s.get("duration"):
                lines.append(f"⏱ **समय:** {s['duration']}")
            if s.get("prerequisite"):
                lines.append(f"⚠️ **पूर्वशर्त:** {s['prerequisite']}")
            if result.current_step < result.total_steps:
                lines.append("\n> अर्को चरणको लागि **\"next\"** वा **\"अर्को\"** टाइप गर्नुहोस्।")
        else:
            lines.append(
                f"📋 **{result.process_name}**  "
                f"[Step {result.current_step}/{result.total_steps}  "
                f"— {result.progress_pct:.0f}% done]"
            )
            lines.append(f"\n**{result.current_step}. {s.get('action', '')}**")
            if s.get("office"):
                lines.append(f"🏢 **Office:** {s['office']}")
            docs = [d for d in (s.get("documents") or []) if d]
            if docs:
                lines.append("📄 **Required documents:**")
                lines.extend(f"   • {d}" for d in docs)
            fees = [f for f in (s.get("fees") or []) if f]
            if fees:
                lines.append(f"💰 **Fees:** {', '.join(str(f) for f in fees)}")
            if s.get("duration"):
                lines.append(f"⏱ **Time:** {s['duration']}")
            if s.get("prerequisite"):
                lines.append(f"⚠️ **Prerequisite:** {s['prerequisite']}")
            if result.current_step < result.total_steps:
                lines.append('\n> Type **"next"** to continue to the next step.')

        return "\n".join(lines)
