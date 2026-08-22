"""
Mistral LLM Client
===================
Reusable wrapper around the Mistral AI API for generating natural language
responses in the RAG pipeline. Uses the same ``mistralai`` SDK already
used by the NER extractor.

Usage:
    from src.llm.llm_client import MistralLLMClient

    client = MistralLLMClient()
    answer = client.generate(system_prompt, user_message)

    # Streaming:
    for token in client.stream(system_prompt, user_message):
        print(token, end="", flush=True)
"""

import time
import logging
from typing import Optional, Iterator, List, Dict

from mistralai.client import Mistral

from src.config import get_settings

logger = logging.getLogger(__name__)


class MistralLLMClient:
    """Thin wrapper around the Mistral chat completions API with retry logic."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        temperature: float = 0.3,
    ) -> None:
        """
        Parameters
        ----------
        model : str | None
            Mistral model identifier.  Defaults to ``Settings.MISTRAL_MODEL``.
        api_key : str | None
            API key.  Defaults to ``Settings.MISTRAL_API_KEY``.
        max_retries : int
            Number of retry attempts on transient HTTP errors (429, 5xx).
        temperature : float
            Sampling temperature for generation.
        """
        settings = get_settings()
        self.model = model or settings.MISTRAL_MODEL
        self.temperature = temperature
        self.max_retries = max_retries

        resolved_key = api_key or settings.MISTRAL_API_KEY
        if not resolved_key:
            raise ValueError(
                "MISTRAL_API_KEY is required. Set it in .env or pass api_key= explicitly."
            )

        self.client = Mistral(api_key=resolved_key)
        logger.info("MistralLLMClient initialised  ▸ model=%s", self.model)

    # ------------------------------------------------------------------
    # Non-streaming generation
    # ------------------------------------------------------------------

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a chat completion request and return the full response text.

        Parameters
        ----------
        system_prompt : str
            System-level instructions (RAG grounding prompt).
        user_message : str
            The assembled context + user query.
        temperature : float | None
            Override instance-level temperature for this call.

        Returns
        -------
        str
            The model's generated text, or an error message on failure.
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.complete(
                    model=self.model,
                    messages=messages,
                    temperature=temperature or self.temperature,
                )
                return response.choices[0].message.content

            except Exception as e:
                error_str = str(e)
                is_retriable = any(
                    code in error_str for code in ["429", "500", "502", "503"]
                )

                if is_retriable and attempt < self.max_retries:
                    wait_time = 3 ** attempt
                    logger.warning(
                        "Mistral API error (attempt %d/%d): %s. Retrying in %ds...",
                        attempt, self.max_retries, e, wait_time,
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        "Mistral API failed after %d attempts: %s", attempt, e
                    )
                    return f"[Error: LLM generation failed — {e}]"

        return "[Error: LLM generation failed after all retries]"

    # ------------------------------------------------------------------
    # Streaming generation
    # ------------------------------------------------------------------

    def stream(
        self,
        system_prompt: str,
        user_message: str,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        """
        Stream chat completion tokens one-by-one.

        Yields
        ------
        str
            Individual text tokens as they arrive from the API.
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            stream_response = self.client.chat.stream(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
            )

            for event in stream_response:
                token = event.data.choices[0].delta.content
                if token:
                    yield token

        except Exception as e:
            logger.error("Mistral streaming failed: %s", e)
            yield f"\n[Error: streaming failed — {e}]"
