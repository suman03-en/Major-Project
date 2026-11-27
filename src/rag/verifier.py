"""
Deterministic Citation and Temporal Verification Layer
======================================================
Zero-LLM post-generation verifier that checks:
1. Temporal status of retrieved Provisions (flags any ``status != "active"``).
2. Citation grounding of Section (``दफा <num>`` / ``Section <num>``) and Act
   references against retrieved Provisions and Procedures.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.utils import nepali_to_int

_SECTION_PATTERN = re.compile(
    r"(?:दफा\s*([०-९\d]+)|Section\s+(\d+))",
    re.IGNORECASE,
)

_NEPALI_ACT_PATTERN = re.compile(
    r"([\u0900-\u097F]+(?:\s+[\u0900-\u097F]+){0,3}\s+(?:ऐन|नियमावली)(?:,?\s*[०-९\d]{4})?)"
)

_ENGLISH_ACT_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z&'-]*(?:\s+[A-Z][A-Za-z&'-]*){0,4}\s+(?:(?:Act|Regulations?)(?:,?\s*\d{4})?|Rules,?\s*\d{4}))\b"
)

_NEPALI_GENERIC_PREFIXES = frozenset({
    "यस", "उक्त", "सो", "यस्तो", "कुनै", "अन्य", "सम्बन्धित",
    "प्रचलित", "नयाँ", "पुरानो", "तत्सम्बन्धी", "नेपाली",
})

_NEPALI_STOPWORDS = frozenset({"र", "वा", "तथा", "एवं", "अनुसार", "बमोजिम", "मा", "को", "का", "की"})

_ACT_TOKEN_MAP: dict[str, str] = {
    "कम्पनी": "company",
    "companies": "company",
    "औद्योगिक": "industrial",
    "व्यवसाय": "enterprise",
    "enterprises": "enterprise",
    "श्रम": "labor",
    "labour": "labor",
    "आयकर": "income_tax",
    "कर": "tax",
    "विदेशी": "foreign",
    "लगानी": "investment",
    "साझेदारी": "partnership",
    "प्राइभेट": "private",
    "फर्म": "firm",
    "ऐन": "act",
    "नियमावली": "regulation",
    "regulations": "regulation",
    "rules": "regulation",
}

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def _normalize_act_tokens(raw: str) -> tuple[set[str], str | None]:
    """Return ``(canonical_tokens, year)`` for an Act title or ``act_source`` slug."""
    ascii_text = raw.translate(_DEVANAGARI_DIGITS).lower()
    year_match = re.search(r"\b(\d{4})\b", ascii_text)
    year = year_match.group(1) if year_match else None

    tokens: set[str] = set()
    for tok in re.split(r"[\s,._/-]+", ascii_text):
        if not tok or tok == year:
            continue
        mapped = _ACT_TOKEN_MAP.get(tok, tok)
        if mapped not in {"act", "regulation", "of", "the", "and"}:
            tokens.add(mapped)
    return tokens, year


def _to_int_section(val: Any) -> int | None:
    """Convert a section number (int, ASCII str, or Devanagari str) to int."""
    if isinstance(val, int) and not isinstance(val, bool):
        return val
    if isinstance(val, str):
        return nepali_to_int(val.strip())
    return None


@dataclass
class VerificationResult:
    """Output of :meth:`VerificationLayer.verify`."""

    answer: str
    is_valid: bool = True
    cited_sections: list[int] = field(default_factory=list)
    cited_acts: list[str] = field(default_factory=list)
    ungrounded_citations: list[str] = field(default_factory=list)
    inactive_provisions: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class VerificationLayer:
    """Deterministic post-generation verifier (zero LLM calls)."""

    def check_temporal(
        self, chunks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Return metadata for any retrieved Provision whose ``status`` is not ``"active"``."""
        inactive: list[dict[str, Any]] = []
        for chunk in chunks:
            raw_status = chunk.get("status")
            if raw_status is None or str(raw_status).strip() == "":
                continue
            status = str(raw_status).strip().lower()
            if status != "active":
                hierarchy = chunk.get("hierarchy") or {}
                inactive.append(
                    {
                        "chunk_id": chunk.get("chunk_id", ""),
                        "act_source": chunk.get("act_source", ""),
                        "sec": hierarchy.get("sec"),
                        "status": status,
                    }
                )
        return inactive

    @staticmethod
    def filter_active_provisions(
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return only Provisions that are active or have no explicit ``status``."""
        active: list[dict[str, Any]] = []
        for chunk in chunks:
            raw_status = chunk.get("status")
            if raw_status is None or str(raw_status).strip().lower() in ("", "active"):
                active.append(chunk)
        return active

    @staticmethod
    def _collect_grounded_sections(
        chunks: list[dict[str, Any]],
        graph: list[dict[str, Any]],
    ) -> set[int]:
        """Collect all section numbers present in retrieved Provisions and Procedures."""
        sections: set[int] = set()
        for chunk in chunks:
            sec = _to_int_section((chunk.get("hierarchy") or {}).get("sec"))
            if sec is not None:
                sections.add(sec)
            body = chunk.get("text")
            if isinstance(body, str) and body:
                for match in _SECTION_PATTERN.finditer(body):
                    raw_num = match.group(1) or match.group(2)
                    ref_sec = nepali_to_int(raw_num)
                    if ref_sec is not None:
                        sections.add(ref_sec)

        for proc in graph:
            for key in ("section", "sec"):
                sec = _to_int_section(proc.get(key))
                if sec is not None:
                    sections.add(sec)
            for raw_sec in proc.get("sections") or []:
                sec = _to_int_section(raw_sec)
                if sec is not None:
                    sections.add(sec)
            for step in proc.get("steps") or []:
                if not isinstance(step, dict):
                    continue
                for key in ("section", "sec"):
                    sec = _to_int_section(step.get(key))
                    if sec is not None:
                        sections.add(sec)
        return sections

    @staticmethod
    def _extract_cited_sections(answer: str) -> list[tuple[int, str]]:
        """Extract ``(section_int, raw_citation)`` pairs from *answer* in order."""
        seen: set[int] = set()
        cited: list[tuple[int, str]] = []
        for match in _SECTION_PATTERN.finditer(answer):
            raw_num = match.group(1) or match.group(2)
            sec_num = nepali_to_int(raw_num)
            if sec_num is not None and sec_num not in seen:
                seen.add(sec_num)
                cited.append((sec_num, match.group(0).strip()))
        return cited

    @staticmethod
    def _collect_grounded_acts(
        chunks: list[dict[str, Any]],
        graph: list[dict[str, Any]],
    ) -> list[tuple[set[str], str | None]]:
        """Collect normalized ``(tokens, year)`` pairs for all retrieved Acts."""
        acts: list[tuple[set[str], str | None]] = []
        candidates: list[str] = []
        for chunk in chunks:
            for key in ("act_source", "act_name"):
                val = chunk.get(key)
                if isinstance(val, str) and val.strip():
                    candidates.append(val.strip())
            h_act = (chunk.get("hierarchy") or {}).get("act_name")
            if isinstance(h_act, str) and h_act.strip():
                candidates.append(h_act.strip())
            body = chunk.get("text")
            if isinstance(body, str) and body:
                for pattern in (_NEPALI_ACT_PATTERN, _ENGLISH_ACT_PATTERN):
                    for match in pattern.finditer(body):
                        candidates.append(match.group(1).strip())

        for proc in graph:
            for key in ("act_source", "act_name", "process_name"):
                val = proc.get(key)
                if isinstance(val, str) and val.strip():
                    candidates.append(val.strip())

        for cand in candidates:
            tokens, year = _normalize_act_tokens(cand)
            if tokens:
                acts.append((tokens, year))
        return acts

    @staticmethod
    def _extract_cited_acts(answer: str) -> list[str]:
        """Extract named Act / Regulation citations from *answer* in order."""
        seen: set[str] = set()
        cited: list[str] = []

        for match in _NEPALI_ACT_PATTERN.finditer(answer):
            phrase = match.group(1).strip()
            words = phrase.split()
            while len(words) > 2 and words[0] in _NEPALI_STOPWORDS:
                words.pop(0)
            if len(words) < 2 or words[0] in _NEPALI_GENERIC_PREFIXES:
                continue
            cleaned = " ".join(words)
            if cleaned not in seen:
                seen.add(cleaned)
                cited.append(cleaned)

        for match in _ENGLISH_ACT_PATTERN.finditer(answer):
            cleaned = match.group(1).strip()
            if cleaned not in seen:
                seen.add(cleaned)
                cited.append(cleaned)

        return cited

    @staticmethod
    def _is_act_grounded(
        cited_act: str,
        grounded_acts: list[tuple[set[str], str | None]],
    ) -> bool:
        """Return True if *cited_act* matches any retrieved Act."""
        cited_tokens, cited_year = _normalize_act_tokens(cited_act)
        if not cited_tokens:
            return False
        for g_tokens, g_year in grounded_acts:
            if cited_year and g_year and cited_year != g_year:
                continue
            overlap = cited_tokens & g_tokens
            if cited_tokens <= g_tokens or (len(cited_tokens) >= 2 and len(overlap) >= 2):
                return True
        return False

    def verify(
        self,
        answer: str,
        chunks: list[dict[str, Any]] | None = None,
        graph: list[dict[str, Any]] | None = None,
    ) -> VerificationResult:
        """Verify temporal status and citation grounding with zero LLM calls."""
        provisions = chunks or []
        procedures = graph or []

        inactive_provisions = self.check_temporal(provisions)
        grounded_sections = self._collect_grounded_sections(provisions, procedures)
        grounded_acts = self._collect_grounded_acts(provisions, procedures)

        cited_pairs = self._extract_cited_sections(answer)
        cited_sections = [sec for sec, _ in cited_pairs]
        ungrounded_citations = [
            raw for sec, raw in cited_pairs if sec not in grounded_sections
        ]

        cited_acts = self._extract_cited_acts(answer)
        for act in cited_acts:
            if not self._is_act_grounded(act, grounded_acts):
                ungrounded_citations.append(act)

        warnings: list[str] = []

        if inactive_provisions:
            details = ", ".join(
                f"{p['act_source'] or 'unknown'} (दफा {p['sec']}: {p['status']})"
                if p.get("sec") is not None
                else f"{p['act_source'] or 'unknown'} ({p['status']})"
                for p in inactive_provisions
            )
            warnings.append(
                f"⚠️ Temporal Warning: Retrieved context includes inactive provisions: {details}."
            )

        if ungrounded_citations:
            warnings.append(
                "⚠️ Verification Notice: The following citations were not found in the retrieved legal context: "
                + ", ".join(ungrounded_citations)
                + "."
            )

        verified_answer = answer
        if warnings:
            verified_answer = f"{answer.rstrip()}\n\n" + "\n".join(warnings)

        return VerificationResult(
            answer=verified_answer,
            is_valid=not inactive_provisions and not ungrounded_citations,
            cited_sections=cited_sections,
            cited_acts=cited_acts,
            ungrounded_citations=ungrounded_citations,
            inactive_provisions=inactive_provisions,
            warnings=warnings,
        )
