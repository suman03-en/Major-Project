"""
Standard OpenAI-Compatible LLM Client
=====================================
Provider-agnostic LLM module speaking the standard OpenAI Chat Completions
protocol (``POST <base_url>/chat/completions``). Works with any provider that
supports this protocol (Groq, Gemini, Mistral, OpenRouter, Together, Cerebras,
DeepSeek, Ollama ``/v1``, vLLM, LM Studio, etc.) without hardcoding provider
names or SDKs.

Usage::

    from src.llm import create_llm_client

    client = create_llm_client()
    answer = client.generate(system_prompt, user_message)
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from typing import Any

from src.config import get_settings
from src.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

_DURATION_PART_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h)", re.IGNORECASE)
_TPM_LIMIT_RE = re.compile(
    r"Limit\s+(\d+)\s*,\s*Used\s+\d+\s*,\s*Requested\s+(\d+)",
    re.IGNORECASE,
)


def _parse_duration_seconds(raw: str | None) -> float | None:
    """Parse numeric seconds or Go-style duration strings (e.g. ``1m5.5s``, ``500ms``)."""
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        val = float(text)
        return val if val >= 0 else None
    except ValueError:
        pass

    parts = _DURATION_PART_RE.findall(text)
    if not parts:
        return None
    total = 0.0
    for amount, unit in parts:
        num = float(amount)
        u = unit.lower()
        if u == "ms":
            total += num / 1000.0
        elif u == "s":
            total += num
        elif u == "m":
            total += num * 60.0
        elif u == "h":
            total += num * 3600.0
    return total


def _extract_reset_seconds(headers: Any) -> float | None:
    """Extract wait duration in seconds from standard rate-limit response headers."""
    if headers is None:
        return None
    for key in ("Retry-After", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        raw = headers.get(key) if hasattr(headers, "get") else None
        parsed = _parse_duration_seconds(raw)
        if parsed is not None:
            return parsed
    return None


def _is_single_request_over_limit(body: str) -> bool:
    """Return True when a 429 body shows one request exceeds the model's total limit."""
    match = _TPM_LIMIT_RE.search(body)
    if not match:
        return False
    limit = int(match.group(1))
    requested = int(match.group(2))
    return requested > limit


def _normalize_completions_url(base_url: str) -> str:
    """Normalize a base URL into a ``.../chat/completions`` endpoint."""
    cleaned = base_url.strip().rstrip("/")
    if not cleaned:
        raise ValueError("LLM_BASE_URL is required (e.g. https://api.groq.com/openai/v1).")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    return f"{cleaned}/chat/completions"


def _is_error_output(text: str) -> bool:
    """Return True when a provider returned an in-band ``[Error: ...]`` string."""
    return text.lstrip().startswith("[Error:")


def is_rate_limit_error(text: str) -> bool:
    """Return True when *text* is an in-band error output caused by HTTP 429 rate limits."""
    if not _is_error_output(text):
        return False
    lowered = text.lower()
    return "429" in lowered or "rate limit" in lowered or "rate_limit" in lowered


def format_rate_limit_message(lang: str | None = None) -> str:
    """Return a clean user-facing retry message for exhausted rate limits."""
    if lang == "en":
        return "The LLM service is temporarily rate-limited. Please retry in about 1 minute."
    return "अहिले सेवा धेरै व्यस्त छ (rate limit)। कृपया करिब १ मिनेटपछि फेरि प्रयास गर्नुहोस्।"


def handle_rate_limit_response(
    text: str,
    lang: str | None,
    stage: str,
) -> str | None:
    """Log and return a localized message if *text* is a rate-limit error, else ``None``."""
    if not is_rate_limit_error(text):
        return None
    logger.error("Rate limit reached during %s: %s", stage, text.strip())
    return format_rate_limit_message(lang)


class StandardLLMClient:
    """
    Universal chat-completions client using the OpenAI-compatible REST + SSE protocol.

    Parameters
    ----------
    base_url:
        Provider base URL (e.g. ``https://api.groq.com/openai/v1``) or full
        ``/chat/completions`` URL.
    api_key:
        Bearer API key (optional for local servers; defaults to ``"local"``).
    model:
        Model identifier expected by the target endpoint.
    max_retries:
        Retry attempts on transient HTTP errors (429, 5xx).
    temperature:
        Default sampling temperature.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int | None = None,
        temperature: float = 0.3,
        timeout: int = 60,
        retry_base_seconds: float | None = None,
        retry_max_wait_seconds: float | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        jitter_fn: Callable[[], float] | None = None,
    ) -> None:
        settings = get_settings()
        resolved_base = base_url if base_url is not None else settings.LLM_BASE_URL
        resolved_model = model if model is not None else settings.LLM_MODEL
        resolved_key = api_key if api_key is not None else settings.LLM_API_KEY

        if not resolved_model or not resolved_model.strip():
            raise ValueError("LLM_MODEL is required.")

        self.endpoint = _normalize_completions_url(resolved_base)
        self.model = resolved_model.strip()
        self.temperature = temperature
        self.max_retries = max(
            1,
            max_retries
            if max_retries is not None
            else int(getattr(settings, "LLM_MAX_RETRIES", 4)),
        )
        self.retry_base_seconds = (
            retry_base_seconds
            if retry_base_seconds is not None
            else float(getattr(settings, "LLM_RETRY_BASE_SECONDS", 2.0))
        )
        self.retry_max_wait_seconds = (
            retry_max_wait_seconds
            if retry_max_wait_seconds is not None
            else float(getattr(settings, "LLM_RETRY_MAX_WAIT_SECONDS", 60.0))
        )
        self._sleep = sleep_fn
        self._jitter = jitter_fn or (lambda: random.uniform(0.0, 1.0))
        self.timeout = timeout
        self._api_key = resolved_key.strip() if resolved_key else "local"
        logger.info(
            "StandardLLMClient initialised  ▸ endpoint=%s  ▸ model=%s",
            self.endpoint,
            self.model,
        )

    def _compute_wait_seconds(self, attempt: int, headers: Any = None) -> float:
        """Compute wait time from reset headers or jittered exponential backoff."""
        header_wait = _extract_reset_seconds(headers)
        if header_wait is not None:
            base_wait = header_wait
        else:
            base_wait = self.retry_base_seconds * (2 ** (attempt - 1))
        capped = min(base_wait + self._jitter(), self.retry_max_wait_seconds)
        return max(0.0, capped)

    def _should_retry_http(self, code: int, body: str, attempt: int) -> bool:
        """Return True if an HTTP error is transient and retries remain."""
        if attempt >= self.max_retries:
            return False
        if code == 429 and _is_single_request_over_limit(body):
            return False
        return code in (429, 500, 502, 503)

    def _build_request(
        self,
        system_prompt: str,
        user_message: str,
        stream: bool,
        temperature: float | None = None,
    ) -> urllib.request.Request:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature if temperature is not None else self.temperature,
            "stream": stream,
        }
        return urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "NepaliLegalRAG/1.0",
            },
            method="POST",
        )

    def _handle_http_error(
        self,
        err: urllib.error.HTTPError,
        attempt: int,
        is_stream: bool = False,
    ) -> str | None:
        """Sleep and return ``None`` if retriable; otherwise log and return an error string."""
        body = err.read().decode("utf-8", errors="replace")
        snippet = body[:400] if err.code == 429 else body[:200]
        if self._should_retry_http(err.code, body, attempt):
            wait_time = self._compute_wait_seconds(attempt, err.headers)
            kind = "stream HTTP" if is_stream else "HTTP"
            logger.warning(
                "LLM %s %d (%s, attempt %d/%d): %s. Retrying in %.2fs...",
                kind, err.code, self.model, attempt, self.max_retries, snippet, wait_time,
            )
            (self._sleep or time.sleep)(wait_time)
            return None
        logger.error(
            "LLM %sfailed after %d attempts (%s, HTTP %d): %s",
            "streaming " if is_stream else "", attempt, self.model, err.code, snippet,
        )
        prefix = "\n[Error: streaming failed" if is_stream else "[Error: LLM generation failed"
        return f"{prefix} — HTTP {err.code}: {snippet}]"

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float | None = None,
    ) -> str:
        """Send a chat completion request and return the full response text."""
        for attempt in range(1, self.max_retries + 1):
            req = self._build_request(
                system_prompt, user_message, stream=False, temperature=temperature
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                err_msg = self._handle_http_error(e, attempt, is_stream=False)
                if err_msg is not None:
                    return err_msg
            except (urllib.error.URLError, json.JSONDecodeError, KeyError, OSError) as e:
                logger.error("LLM request failed (%s): %s", self.model, e)
                return f"[Error: LLM generation failed — {e}]"

        return "[Error: LLM generation failed after all retries]"

    def stream(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """Stream chat completion tokens one-by-one from an SSE endpoint."""
        for attempt in range(1, self.max_retries + 1):
            req = self._build_request(
                system_prompt, user_message, stream=True, temperature=temperature
            )
            yielded_any = False
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    for raw_line in resp:
                        line = raw_line.decode("utf-8").strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        chunk = json.loads(payload)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        token = delta.get("content")
                        if token:
                            yielded_any = True
                            yield token
                return
            except urllib.error.HTTPError as e:
                if yielded_any:
                    body = e.read().decode("utf-8", errors="replace")
                    logger.error("LLM streaming failed mid-stream (%s, HTTP %d): %s", self.model, e.code, body[:200])
                    yield f"\n[Error: streaming failed — HTTP {e.code}: {body[:200]}]"
                    return
                err_msg = self._handle_http_error(e, attempt, is_stream=True)
                if err_msg is not None:
                    yield err_msg
                    return
            except Exception as e:  # noqa: BLE001
                logger.error("LLM streaming failed (%s): %s", self.model, e)
                yield f"\n[Error: streaming failed — {e}]"
                return

        yield "\n[Error: streaming failed after all retries]"


class FallbackLLMClient:
    """
    Failover router behind :class:`~src.llm.base.BaseLLMClient`.

    Tries configured endpoints in order; on rate limits (429), auth/tier
    restrictions (401/403), or network errors, puts the failing endpoint on
    cooldown and transparently fails over to the next one. When all endpoints
    are rate-limited or cooling down, optional exponential backoff retries the
    chain after waiting.
    """

    def __init__(
        self,
        clients: list[BaseLLMClient],
        cooldown_seconds: float = 60.0,
        auth_cooldown_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
        max_exhausted_retries: int = 0,
        retry_base_seconds: float = 2.0,
        retry_max_wait_seconds: float = 60.0,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        if not clients:
            raise ValueError("FallbackLLMClient requires at least one client.")
        self._clients = list(clients)
        self._cooldown_seconds = cooldown_seconds
        self._auth_cooldown_seconds = auth_cooldown_seconds
        self._clock = clock
        self._max_exhausted_retries = max(0, max_exhausted_retries)
        self._retry_base_seconds = retry_base_seconds
        self._retry_max_wait_seconds = retry_max_wait_seconds
        self._sleep = sleep_fn
        self._cooldown_until: dict[int, float] = {}
        self._active_idx = 0

    @property
    def model(self) -> str:
        """Model identifier of the currently active client."""
        return self._clients[self._active_idx].model

    def _ordered_indices(self) -> list[int]:
        now = self._clock()
        healthy = [i for i in range(len(self._clients)) if self._cooldown_until.get(i, 0.0) <= now]
        cooled = sorted(
            (i for i in range(len(self._clients)) if self._cooldown_until.get(i, 0.0) > now),
            key=lambda i: self._cooldown_until[i],
        )
        return healthy + cooled

    def _mark_failed(self, idx: int, reason: str) -> None:
        is_auth = any(tok in reason for tok in ("401", "403", "Invalid API Key", "tier_not_allowed"))
        duration = self._auth_cooldown_seconds if is_auth else self._cooldown_seconds
        self._cooldown_until[idx] = self._clock() + duration
        logger.warning(
            "Endpoint (%s) failed — cooling down for %.0fs: %s",
            self._clients[idx].model,
            duration,
            reason[:160],
        )

    def _should_retry_exhausted(self, round_idx: int, last_error: str) -> bool:
        """Wait and return True if another pass over all endpoints should run."""
        if round_idx >= self._max_exhausted_retries or _is_single_request_over_limit(last_error):
            return False
        wait_time = min(
            self._retry_base_seconds * (2 ** round_idx),
            self._retry_max_wait_seconds,
        )
        logger.warning(
            "All LLM fallback endpoints failed (round %d/%d); retrying in %.2fs...",
            round_idx + 1,
            self._max_exhausted_retries,
            wait_time,
        )
        (self._sleep or time.sleep)(wait_time)
        return True

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float | None = None,
    ) -> str:
        """Try endpoints in order until one succeeds without an error."""
        last_error = "[Error: No LLM endpoints succeeded]"
        for round_idx in range(self._max_exhausted_retries + 1):
            for idx in self._ordered_indices():
                client = self._clients[idx]
                try:
                    result = client.generate(system_prompt, user_message, temperature=temperature)
                except Exception as e:  # noqa: BLE001
                    last_error = f"[Error: LLM generation failed — {e}]"
                    self._mark_failed(idx, str(e))
                    continue

                if _is_error_output(result):
                    last_error = result
                    self._mark_failed(idx, result)
                    continue

                self._active_idx = idx
                return result

            if not self._should_retry_exhausted(round_idx, last_error):
                break

        return last_error

    def stream(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """Stream from the first healthy endpoint; fail over if first token is an error."""
        last_error = "\n[Error: streaming failed — all endpoints unavailable]"
        for round_idx in range(self._max_exhausted_retries + 1):
            for idx in self._ordered_indices():
                client = self._clients[idx]
                try:
                    it = iter(client.stream(system_prompt, user_message, temperature=temperature))
                    first = next(it, None)
                except Exception as e:  # noqa: BLE001
                    last_error = f"\n[Error: streaming failed — {e}]"
                    self._mark_failed(idx, str(e))
                    continue

                if first is None:
                    continue
                if _is_error_output(first):
                    last_error = first
                    self._mark_failed(idx, first)
                    continue

                self._active_idx = idx
                yield first
                yield from it
                return

            if not self._should_retry_exhausted(round_idx, last_error):
                break

        yield last_error


def _parse_fallbacks(raw_json: str) -> list[dict[str, str]]:
    """Parse ``LLM_FALLBACKS`` JSON array of ``{base_url, api_key, model}`` dicts."""
    if not raw_json or not raw_json.strip():
        return []
    parsed = json.loads(raw_json)
    if not isinstance(parsed, list):
        raise TypeError("LLM_FALLBACKS must be a JSON array of {base_url, api_key, model} objects.")
    return [entry for entry in parsed if isinstance(entry, dict)]


def create_llm_client(
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> BaseLLMClient:
    """
    Build a standard OpenAI-compatible client (with optional failover chain).

    Reads ``LLM_BASE_URL``, ``LLM_API_KEY``, ``LLM_MODEL``, and optional
    ``LLM_FALLBACKS`` from :func:`~src.config.get_settings`.
    """
    settings = get_settings()
    fallback_entries = [
        fb
        for fb in _parse_fallbacks(getattr(settings, "LLM_FALLBACKS", ""))
        if str(fb.get("base_url", "")).strip() and str(fb.get("model", "")).strip()
    ]
    has_fallbacks = len(fallback_entries) > 0
    per_client_retries = 1 if has_fallbacks else None

    primary = StandardLLMClient(
        base_url=base_url if base_url is not None else settings.LLM_BASE_URL,
        api_key=api_key if api_key is not None else settings.LLM_API_KEY,
        model=model if model is not None else settings.LLM_MODEL,
        max_retries=per_client_retries,
    )

    if not has_fallbacks:
        return primary

    clients: list[BaseLLMClient] = [primary]
    for fb in fallback_entries:
        clients.append(
            StandardLLMClient(
                base_url=str(fb["base_url"]).strip(),
                api_key=str(fb.get("api_key", "")),
                model=str(fb["model"]).strip(),
                max_retries=1,
            )
        )

    logger.info(
        "Initialised FallbackLLMClient chain (%d endpoints): %s",
        len(clients),
        " → ".join(c.model for c in clients),
    )
    max_retries = int(getattr(settings, "LLM_MAX_RETRIES", 4))
    return FallbackLLMClient(
        clients,
        max_exhausted_retries=max(0, max_retries - 1),
        retry_base_seconds=float(getattr(settings, "LLM_RETRY_BASE_SECONDS", 2.0)),
        retry_max_wait_seconds=float(getattr(settings, "LLM_RETRY_MAX_WAIT_SECONDS", 60.0)),
    )


def create_helper_llm_client() -> BaseLLMClient | None:
    """
    Build an optional lighter/separate-TPM LLM client from ``LLM_HELPER_MODEL``.

    Returns ``None`` when ``LLM_HELPER_MODEL`` is unset so callers fall back to
    the primary LLM client.
    """
    settings = get_settings()
    helper_model = str(getattr(settings, "LLM_HELPER_MODEL", "")).strip()
    if not helper_model:
        return None
    return StandardLLMClient(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        model=helper_model,
    )
