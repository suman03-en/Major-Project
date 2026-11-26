"""
Intent Classifier
=================
Classifies a user query into one of :class:`IntentType` without an LLM call
(keyword-based, O(n) in keyword table size, <1 ms per query).

The classifier is intentionally lightweight — it only escalates to an LLM
when the keyword pass returns ``UNKNOWN``, which is rare for the legal domain.

Usage::

    from src.agent.intent_classifier import IntentClassifier, IntentType

    clf = IntentClassifier()
    intent = clf.classify("कम्पनी दर्ता कसरी गर्ने?")
    # → IntentType.PROCEDURAL
"""

from __future__ import annotations

import re
from enum import Enum


class IntentType(str, Enum):
    """Possible high-level query intents for the legal agent."""

    GREETING = "GREETING"
    PROCEDURAL = "PROCEDURAL"
    LEGAL_FACT = "LEGAL_FACT"
    DOCUMENT_PREP = "DOCUMENT_PREP"
    FOLLOW_UP = "FOLLOW_UP"
    HYBRID = "HYBRID"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Keyword tables (all lower-case; Nepali and English)
# ---------------------------------------------------------------------------

_GREETING_KW: frozenset[str] = frozenset(
    {
        "hi", "hello", "hey", "namaste", "namaskar", "dhanyabad",
        "k chha", "k xa", "ke chha", "के छ",
        "नमस्ते", "नमस्कार", "धन्यवाद", "thank you", "thanks",
        "bye", "goodbye", "good morning", "good evening", "शुभ प्रभात",
    }
)

_PROCEDURAL_KW: frozenset[str] = frozenset(
    {
        # Nepali
        "दर्ता", "प्रक्रिया", "चरण", "कसरी", "कहाँ", "कहाँ जाने",
        "के गर्नुपर्छ", "के गर्ने", "कार्यविधि", "निर्देशन",
        "विधि", "तरिका", "नियम", "पालना",
        # Romanized Nepali
        "darta", "prakriya", "praliya", "charan", "kasari", "kaha",
        "garne", "tarika", "karyabidhi", "karyabidi", "karyalaya",
        # English
        "register", "registration", "process", "procedure", "steps",
        "how to", "where to", "what to do", "workflow", "guideline",
        "requirements", "apply", "application",
    }
)

_LEGAL_FACT_KW: frozenset[str] = frozenset(
    {
        # Nepali
        "दफा", "ऐन", "धारा", "उपदफा", "खण्ड", "अनुच्छेद",
        "कानून", "नियमावली", "व्यवस्था", "परिभाषा", "अपराध",
        "सजाय", "जरिमाना", "जरिवाना", "दण्ड", "हक", "अधिकार",
        # Romanized Nepali
        "dafa", "ain", "dhara", "upadafa", "khanda", "kanun", "kanoon",
        "niyamawali", "sajaya", "jarimana", "adhikar",
        # English
        "section", "act", "clause", "subsection", "article",
        "law", "legal", "regulation", "provision", "penalty",
        "fine", "right", "definition", "what is", "what does",
        "define", "explain", "meaning of",
    }
)

_DOCUMENT_KW: frozenset[str] = frozenset(
    {
        # Nepali
        "फारम", "कागजात", "कागज", "प्रमाण", "कागजात तयार",
        "दस्तावेज", "निवेदन", "पत्र", "आवेदन",
        # Romanized Nepali
        "pharam", "kagajat", "kagaj", "praman", "dastabej",
        "nibedan", "patra", "abedan",
        # English
        "form", "document", "template", "draft", "prepare",
        "fill", "checklist", "certificate", "letter",
    }
)

_FOLLOW_UP_KW: frozenset[str] = frozenset(
    {
        # Nepali
        "अर्को", "थप", "जारी", "सकियो", "गरिसकें",
        "बुझें", "अगाडि", "पछाडि",
        # Romanized Nepali
        "arko", "thap", "jari", "sakiyo", "bujhe", "agadi", "pachhadi",
        # English
        "next", "continue", "done", "more", "back",
        "previous", "step", "proceed", "go ahead", "got it",
    }
)

# Compile a single fast regex for each group (word boundaries for ASCII terms)
def _build_pattern(keywords: frozenset[str]) -> re.Pattern[str]:
    # Sort longest-first so overlapping phrases match correctly
    parts: list[str] = []
    for kw in sorted(keywords, key=len, reverse=True):
        esc = re.escape(kw)
        parts.append(rf"\b{esc}\b" if kw.isascii() else esc)
    return re.compile("|".join(parts), re.IGNORECASE | re.UNICODE)


_GREETING_RE = _build_pattern(_GREETING_KW)
_PROCEDURAL_RE = _build_pattern(_PROCEDURAL_KW)
_LEGAL_RE = _build_pattern(_LEGAL_FACT_KW)
_DOCUMENT_RE = _build_pattern(_DOCUMENT_KW)
_FOLLOW_UP_RE = _build_pattern(_FOLLOW_UP_KW)


class IntentClassifier:
    """
    Keyword-based intent classifier for Nepali/English legal queries.

    The decision tree runs in fixed order and is deterministic:

    1. Greeting → ``GREETING``
    2. Navigation/follow-up → ``FOLLOW_UP``
    3. Document prep → ``DOCUMENT_PREP``
    4. Procedural **and** legal keywords both present → ``HYBRID``
    5. Procedural only → ``PROCEDURAL``
    6. Legal only → ``LEGAL_FACT``
    7. No match → ``UNKNOWN``

    Parameters
    ----------
    has_active_session : bool
        When ``True``, ambiguous short queries (e.g. "थप") lean toward
        ``FOLLOW_UP`` instead of ``UNKNOWN``.
    """

    def classify(
        self,
        query: str,
        has_active_session: bool = False,
    ) -> IntentType:
        """
        Classify *query* and return the most likely :class:`IntentType`.

        Parameters
        ----------
        query:
            Raw user text (any language).
        has_active_session:
            Pass ``True`` when the user has an ongoing procedure session so
            short follow-up phrases are routed correctly.
        """
        text = query.strip().lower()

        # Very short input + active session → treat as follow-up
        if has_active_session and len(text.split()) <= 3 and _FOLLOW_UP_RE.search(text):
            return IntentType.FOLLOW_UP

        is_greeting = bool(_GREETING_RE.search(text))
        is_followup = bool(_FOLLOW_UP_RE.search(text))
        if is_followup and has_active_session:
            return IntentType.FOLLOW_UP

        is_doc = bool(_DOCUMENT_RE.search(text))
        is_proc = bool(_PROCEDURAL_RE.search(text))
        is_legal = bool(_LEGAL_RE.search(text))

        if is_greeting and not (is_doc or is_proc or is_legal) and len(text.split()) <= 4:
            return IntentType.GREETING

        if is_doc:
            return IntentType.DOCUMENT_PREP

        is_proc = bool(_PROCEDURAL_RE.search(text))
        is_legal = bool(_LEGAL_RE.search(text))

        if is_proc and is_legal:
            return IntentType.HYBRID
        if is_proc:
            return IntentType.PROCEDURAL
        if is_legal:
            return IntentType.LEGAL_FACT

        return IntentType.UNKNOWN
