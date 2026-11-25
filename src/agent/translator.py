"""
Query Translator
================
Normalizes English and Romanized Nepali user queries into Devanagari Nepali
for internal retrieval (Qdrant hybrid search, BGE reranker, Neo4j graph
matching, and Nepali keyword intent classification) in a **single LLM call**
that also identifies the input language (``"en"`` or ``"roman"``).

Fast paths (0 LLM calls):
- Queries predominantly in Devanagari Nepali are returned as-is (``"ne"``).
- Empty/whitespace queries are returned as-is.

Usage::

    from src.agent.translator import QueryTranslator

    translator = QueryTranslator(llm_client)
    res = translator.translate("company darta kasari garne?")
    # res.language == "roman"
    # res.original_query == "company darta kasari garne?"
    # res.retrieval_query == "कम्पनी दर्ता कसरी गर्ने?"
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from src.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

LanguageCode = Literal["ne", "en", "roman"]

_TRANSLATE_SYSTEM_PROMPT = (
    "You are a Nepali legal query analyzer and translator.\n"
    "Given a user query in English or Romanized Nepali (Nepali written in Latin script):\n"
    '1. Identify the input language: "en" (English) or "roman" (Romanized Nepali).\n'
    "2. Translate/convert the query into natural, accurate Devanagari Nepali suitable "
    "for searching Nepali laws and government business registration procedures.\n"
    "   - Preserve act names, section numbers (दफा), and subsection numbers (उपदफा) accurately.\n"
    "Output ONLY valid JSON with no extra text:\n"
    '{"language": "en" | "roman", "translated_query": "<Devanagari Nepali query>"}'
)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

_ROMAN_NEPALI_MARKERS: frozenset[str] = frozenset(
    {
        "kasari", "ksri", "kaha", "garne", "darta", "prakriya", "praliya",
        "kagajat", "kagaj", "ho", "chha", "xa", "cha", "ko", "ma", "le",
        "lai", "bata", "huncha", "hunchha", "pardacha", "parcha", "parne",
        "kholne", "suru", "banne", "banao", "sajaya", "jarimana", "dafa",
        "dhara", "ain", "kanun", "kanoon", "niyam", "niyamawali", "sarkar",
        "karyalaya", "nibedan", "praman", "patra", "lagne", "kati", "kina",
        "kahile", "kun", "mero", "hamro", "tapai", "hajur", "ke", "k",
        "arko", "thap", "jari", "sakiyo", "bujhe", "agadi", "pachhadi",
    }
)
_ROMAN_NEPALI_RE = re.compile(
    r"\b(?:" + "|".join(sorted(_ROMAN_NEPALI_MARKERS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def is_romanized_nepali(text: str) -> bool:
    """Return ``True`` when non-Devanagari *text* contains Romanized Nepali markers."""
    if not text or is_devanagari_query(text):
        return False
    return bool(_ROMAN_NEPALI_RE.search(text))


def _is_devanagari(ch: str) -> bool:
    return "\u0900" <= ch <= "\u097F"


def is_devanagari_query(text: str) -> bool:
    """Return ``True`` when *text* is predominantly Devanagari script."""
    if not text or not text.strip():
        return True
    dev_count = sum(1 for ch in text if _is_devanagari(ch))
    latin_count = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    total_alpha = dev_count + latin_count
    return dev_count > 0 and (total_alpha == 0 or dev_count / total_alpha >= 0.4)


def detect_language(text: str) -> LanguageCode:
    """
    Lightweight script-only language check (0 LLM calls).

    Used on fast paths (greetings/follow-ups); full ``"en"`` vs ``"roman"``
    detection is performed by the LLM inside :meth:`QueryTranslator.translate`.
    """
    return "ne" if is_devanagari_query(text) else "en"


def _normalize_lang(raw: str) -> LanguageCode:
    val = raw.strip().lower()
    if "roman" in val:
        return "roman"
    if val in ("ne", "nepali", "devanagari"):
        return "ne"
    return "en"


def _parse_translation_response(raw_text: str) -> tuple[LanguageCode | None, str]:
    """Parse JSON ``{language, translated_query}`` with a plain-text fallback."""
    cleaned = _CODE_FENCE_RE.sub("", raw_text.strip()).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            lang_raw = str(data.get("language", ""))
            translated = str(data.get("translated_query", "")).strip()
            lang = _normalize_lang(lang_raw) if lang_raw else None
            return lang, translated
    except json.JSONDecodeError:
        pass

    # Fallback if model returned plain text instead of JSON
    return None, cleaned.strip('"').strip("'").strip()


@dataclass
class TranslationResult:
    """Result of normalizing an input query for retrieval."""

    original_query: str
    retrieval_query: str
    language: LanguageCode
    was_translated: bool = False


class QueryTranslator:
    """
    Detects input language (``"en"`` / ``"roman"``) and translates non-Devanagari
    queries into Devanagari Nepali in a single LLM call.

    Parameters
    ----------
    llm_client:
        LLM client used to detect language and translate non-Devanagari queries.
        If ``None``, translation is skipped and queries pass through unchanged.
    """

    def __init__(self, llm_client: BaseLLMClient | None = None) -> None:
        self._llm = llm_client

    def translate(self, query: str) -> TranslationResult:
        """
        Return a :class:`TranslationResult` for *query*.

        Devanagari Nepali queries skip the LLM completely (0 ms).
        """
        cleaned = query.strip()

        if not cleaned or is_devanagari_query(cleaned):
            return TranslationResult(
                original_query=query,
                retrieval_query=cleaned or query,
                language="ne",
                was_translated=False,
            )

        if self._llm is None:
            return TranslationResult(
                original_query=query,
                retrieval_query=cleaned,
                language="en",
                was_translated=False,
            )

        try:
            raw_resp = self._llm.generate(_TRANSLATE_SYSTEM_PROMPT, f"Query: {cleaned}")
            if raw_resp.strip().startswith("[Error:"):
                logger.warning("LLM returned error during translation: %s", raw_resp.strip())
                return TranslationResult(
                    original_query=query,
                    retrieval_query=cleaned,
                    language="en",
                    was_translated=False,
                )

            detected_lang, translated = _parse_translation_response(raw_resp)
            lang: LanguageCode = detected_lang or "en"

            if not translated or translated.startswith("[Error:"):
                logger.warning("Empty translation returned; falling back to original query.")
                return TranslationResult(
                    original_query=query,
                    retrieval_query=cleaned,
                    language=lang,
                    was_translated=False,
                )

            logger.info(
                "Query translated (%s -> ne): %r -> %r",
                lang,
                cleaned[:60],
                translated[:60],
            )
            return TranslationResult(
                original_query=query,
                retrieval_query=translated,
                language=lang,
                was_translated=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query translation failed (%s); using original query.", exc)
            return TranslationResult(
                original_query=query,
                retrieval_query=cleaned,
                language="en",
                was_translated=False,
            )
