"""
Clause pre-filter for the NER extraction pipeline.

Filters chunks from the structured dataset JSON to identify only those
clauses that are likely to contain extractable process-related entities
(offices, documents, steps, fees, durations, conditions, incentives).

Uses a two-layer strategy:
  1. Keyword group scoring (entity categories)
  2. Procedural verb detection (must describe an action, not just define terms)

This reduces unnecessary LLM calls by ~60-70%.
"""

import re
from typing import List, Dict, Tuple


# Nepali keywords that indicate a clause is likely process-related.
# Grouped by entity category for scoring.
KEYWORD_GROUPS = {
    "office": [
        "कार्यालय", "विभाग", "मन्त्रालय", "बोर्ड", "निकाय",
        "रजिष्ट्रार", "अधिकारी", "कमिसन", "प्राधिकरण",
        "एकल बिन्दु सेवा केन्द्र",
    ],
    "document": [
        "निवेदन", "कागजात", "प्रमाणपत्र", "प्रतिलिपि",
        "दरखास्त", "फारम", "प्रतिवेदन", "विवरण",
        "नागरिकता", "अनुमतिपत्र", "इजाजतपत्र",
    ],
    "process": [
        "दर्ता", "नवीकरण", "खारेज", "नामसारी",
        "अनुमति", "अनुमती", "स्वीकृति", "सिफारिस",
        "निर्णय", "आवेदन", "जाँचबुझ",
    ],
    "fee": [
        "दस्तुर", "शुल्क", "रकम", "रुपैयाँ",
        "जरिवाना", "विगो", "मूल्य",
    ],
    "duration": [
        "दिनभित्र", "दिन भित्र", "म्याद", "अवधि",
        "समयावधि", "महिना", "वर्ष",
    ],
}

# Procedural verbs/suffixes that indicate the clause describes an action,
# not just a definition or declaration. A chunk must contain at least one
# of these to be considered actionable.
PROCEDURAL_VERBS = [
    "पर्नेछ",          # must do (पर्नेछ / दिनु पर्नेछ)
    "गर्नेछ",          # will do (गर्नेछ)
    "दिनु",            # to give / submit (दिनु पर्ने)
    "पेश गर्ने",       # to present/submit
    "बुझाउनु",         # to submit/hand over
    "दर्ता गर्ने",     # to register
    "खुलाउनु",         # to disclose/show
    "सक्नेछ",          # may/can (permissive action)
    "लिनु",            # to take (अनुमति लिनु)
    "गराउनु",          # causative — to get done
    "गर्न",            # to do (infinitive — गर्न सक्ने)
    "लिन",             # to take (infinitive)
    "पठाउनु",          # to send
    "पठाउने",          # will send
    "प्रदान गर्ने",    # to provide/grant
    "जारी गर्ने",      # to issue
    "हुनेछ",           # it shall be (prescriptive)
]

# Chunk types that are never relevant for NER extraction
SKIP_TYPES = {"preamble"}

# Section titles that indicate non-extractable content (definitions, interpretation)
SKIP_SECTION_TITLES = [
    "परिभाषा",       # Definitions
    "संक्षिप्त नाम",  # Short title
    "प्रारम्भ",       # Commencement
]

# Minimum relevance score to pass the filter
MIN_RELEVANCE_SCORE = 2


class ClauseFilter:
    """
    Filters chunks from the structured dataset JSON, selecting only
    those likely to contain extractable process-related entities.
    
    Two-layer filtering:
    1. Keyword scoring: Each keyword group match adds +1. Score >= MIN_RELEVANCE_SCORE.
    2. Procedural verb check: Chunk must contain at least one actionable verb.
       This catches declarative-only clauses that have keywords but no procedure.
    
    Definition sections (परिभाषा) are always skipped.
    """

    def __init__(self, min_score: int = MIN_RELEVANCE_SCORE):
        self.min_score = min_score
        # Pre-compile keyword patterns per group for faster matching
        self._patterns = {}
        for group_name, keywords in KEYWORD_GROUPS.items():
            # Join keywords into a single alternation pattern
            pattern = "|".join(re.escape(kw) for kw in keywords)
            self._patterns[group_name] = re.compile(pattern)
        
        # Pre-compile procedural verb pattern
        verb_pattern = "|".join(re.escape(v) for v in PROCEDURAL_VERBS)
        self._verb_pattern = re.compile(verb_pattern)

    def _has_procedural_verb(self, text: str) -> bool:
        """Check if the text contains at least one procedural/actionable verb."""
        return bool(self._verb_pattern.search(text))

    def score_chunk(self, chunk: dict) -> Tuple[int, Dict[str, List[str]]]:
        """
        Score a single chunk for NER relevance.
        
        Returns:
            (score, matched_groups) where matched_groups maps 
            group_name -> list of matched keywords.
        """
        text = chunk.get("text", "")
        hierarchy = chunk.get("hierarchy", {})
        chunk_type = chunk.get("type", "")
        
        # Skip definition sections
        sec_title = hierarchy.get("sec_title", "")
        if any(skip in sec_title for skip in SKIP_SECTION_TITLES):
            return 0, {}
        
        # Skip explicitly excluded chunk types
        if chunk_type in SKIP_TYPES:
            return 0, {}
        
        # Skip very short chunks (likely headers or labels)
        if len(text) < 20:
            return 0, {}

        # Procedural verb gate: if the text has no actionable verb,
        # it's likely a declarative/definitional statement — skip it.
        if not self._has_procedural_verb(text):
            return 0, {}

        score = 0
        matched_groups = {}
        
        for group_name, pattern in self._patterns.items():
            matches = pattern.findall(text)
            if matches:
                score += 1
                matched_groups[group_name] = list(set(matches))
        
        return score, matched_groups

    def filter_chunks(self, chunks: List[dict]) -> List[dict]:
        """
        Filter a list of chunks, returning only those that pass
        the relevance threshold.
        
        Each returned chunk is annotated with:
          - '_ner_score': int relevance score
          - '_ner_matched_groups': dict of matched keyword groups
        """
        filtered = []
        for chunk in chunks:
            score, matched = self.score_chunk(chunk)
            if score >= self.min_score:
                # Annotate the chunk (non-destructive copy)
                annotated = dict(chunk)
                annotated["_ner_score"] = score
                annotated["_ner_matched_groups"] = matched
                filtered.append(annotated)
        return filtered

    def get_stats(self, chunks: List[dict]) -> dict:
        """Return filtering statistics for logging/debugging."""
        total = len(chunks)
        filtered = self.filter_chunks(chunks)
        passed = len(filtered)
        
        # Score distribution
        scores = {}
        verb_rejected = 0
        for chunk in chunks:
            text = chunk.get("text", "")
            score, _ = self.score_chunk(chunk)
            scores[score] = scores.get(score, 0) + 1
            
            # Count chunks rejected specifically by verb filter
            if not self._has_procedural_verb(text) and len(text) >= 20:
                verb_rejected += 1
        
        return {
            "total_chunks": total,
            "passed_filter": passed,
            "rejected": total - passed,
            "filter_rate_pct": round((total - passed) / total * 100, 1) if total > 0 else 0,
            "verb_rejected": verb_rejected,
            "score_distribution": dict(sorted(scores.items())),
        }
