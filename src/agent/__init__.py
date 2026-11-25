"""
Legal AI Agent package.

Exports the top-level orchestrator used by web/main.py and the CLI.

Usage::

    from src.agent import LegalAgent
    from src.agent.session_store import InMemorySessionStore

    store = InMemorySessionStore()
    agent = LegalAgent(rag_pipeline=rag, graph_store=graph, llm_client=llm, session_store=store)
    response = agent.respond("कम्पनी दर्ता कसरी गर्ने?", session_id="abc-123")
"""

from src.agent.agent import AgentResponse, LegalAgent

__all__ = ["AgentResponse", "LegalAgent"]
