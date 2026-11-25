"""
Base LLM Client Protocol
========================
Defines the seam for LLM providers so the RAG pipeline can swap between
Mistral AI, local Ollama, or test fakes without changing caller code.
"""

from collections.abc import Iterator
from typing import Protocol


class BaseLLMClient(Protocol):
    """Interface for chat-completion LLM providers."""

    model: str

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float | None = None,
    ) -> str:
        """Generate a complete response for the given prompt."""
        ...

    def stream(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """Stream response tokens one-by-one."""
        ...
