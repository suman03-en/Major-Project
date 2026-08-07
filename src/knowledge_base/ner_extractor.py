"""
Enhanced NER Extractor using Mistral AI API.

Extracts structured business registration entities from Nepali legal
text chunks using Mistral's structured JSON output mode.

Features:
- Pydantic schema-enforced structured output
- Exponential backoff retry on rate limits and transient errors
- Batch processing with configurable concurrency
- Context windowing: merges contiguous sub-clauses for richer extraction
- Robust JSON parsing with fallback sanitization
"""

import os
import re
import json
import time
import logging
from typing import List, Optional, Tuple

from pydantic import ValidationError
from mistralai.client import Mistral

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config import get_settings
from src.knowledge_base.schemas import ClauseNEROutput, ExtractedEntity, PriceFee, OfficeEntity

logger = logging.getLogger(__name__)

# --- Nepali digit utilities ---

NEPALI_DIGITS = '०१२३४५६७८९'
ENGLISH_DIGITS = '0123456789'
_TRANS_TABLE = str.maketrans(NEPALI_DIGITS, ENGLISH_DIGITS)


def nepali_to_int(text: str) -> Optional[int]:
    """Convert a Nepali numeral string to integer."""
    clean = re.sub(r'[^\\d०-९,]', '', text)
    clean = clean.replace(',', '')
    if not clean:
        return None
    try:
        return int(clean.translate(_TRANS_TABLE))
    except ValueError:
        return None


def parse_price_fee(raw_text: Optional[str]) -> List[PriceFee]:
    """
    Parse a price/fee string into structured PriceFee objects.
    Handles formats like: 'रु. १०,०००', '५०० रुपैयाँ', 'दस्तुर रु.५,०००'
    """
    if not raw_text:
        return []
    
    fees = []
    # Pattern: optional label + रु./रुपैयाँ + amount
    patterns = [
        r'(?:रु\.?\s*)([\d०-९,\.]+)',
        r'([\d०-९,\.]+)\s*रुपैयाँ',
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, raw_text):
            amount_raw = match.group(0)
            amount_int = nepali_to_int(match.group(1))
            fees.append(PriceFee(
                type=None,
                amount_raw=amount_raw,
                amount_npr=amount_int,
            ))
    
    # If no pattern matched but there's content, return raw
    if not fees and raw_text.strip():
        fees.append(PriceFee(type=None, amount_raw=raw_text.strip(), amount_npr=None))
    
    return fees


def parse_office(raw_text: Optional[str]) -> Optional[OfficeEntity]:
    """Parse an office string into a structured OfficeEntity."""
    if not raw_text or not raw_text.strip():
        return None
    
    # Detect government level from office name
    level = None
    if any(kw in raw_text for kw in ["प्रदेश", "प्रदेशको"]):
        level = "प्रदेश"
    elif any(kw in raw_text for kw in ["स्थानीय", "नगरपालिका", "गाउँपालिका"]):
        level = "स्थानीय"
    else:
        level = "केन्द्र"
    
    return OfficeEntity(name=raw_text.strip(), level=level)


# --- System prompt for Mistral NER extraction ---

SYSTEM_PROMPT = """You are a Named Entity Recognition assistant specializing in Nepali legal and administrative text.

Your task is to extract business registration entities from the provided Nepali text chunk.

Return a JSON object with EXACTLY these fields:
{
  "office": "string or null — the government office, department, or authority mentioned (e.g., कम्पनी रजिष्ट्रारको कार्यालय, उद्योग विभाग)",
  "documents_required": ["string1", "string2"] — any documents, certificates, applications, or forms required,
  "steps": ["string1", "string2"] — sequential actions, duties, or procedural steps mentioned,
  "price": "string or null — any fee, fine, cost, or monetary amount (e.g., रु. ५००, दस्तुर)",
  "duration_days": "string or null — any time duration, deadline, or validity period (e.g., ३५ दिनभित्र, एक वर्ष)",
  "prerequisites": "string or null — any eligibility criteria, conditions, or prerequisites"
}

Rules:
- Extract entities ONLY from the given text. Do not infer or hallucinate.
- If no entity of a type is found, use null for strings and [] for arrays.
- Keep extracted text in original Nepali language.
- For steps, preserve the logical order as they appear in the text.
- Output ONLY the JSON object. No markdown, no explanation."""


def build_clause_ref(chunk: dict) -> str:
    """
    Build a compact, human-readable breadcrumb string from a chunk's
    hierarchy metadata and type.

    Format: 'ch{n} (title) › sec{n} (title) › sub{n} [type]'
    Examples:
      'ch2 (उद्योग दर्ता तथा नियमन) › sec3 (उद्योग दर्ता गराउनु पर्ने) › sub6 [subsection]'
      'ch1 (प्रारम्भिक) › sec2 (परिभाषा) › clक [clause]'
      'anusuchi1 (अनुसूची-१) › sub2 [anusuchi_subsection]'

    The chunk_id in the same ExtractedEntity is the programmatic key
    for looking up the full clause text in extracted_jsons/.
    """
    hierarchy = chunk.get("hierarchy", {})
    chunk_type = chunk.get("type", "")
    parts = []

    # Anusuchi (annex) or regular chapter
    if hierarchy.get("ch") is not None:
        ch_n = hierarchy["ch"]
        ch_title = hierarchy.get("ch_title", "")
        # Shorten title to 30 chars to keep ref compact
        if ch_title:
            short_title = ch_title[:30].rstrip() + ("..." if len(ch_title) > 30 else "")
            # Detect if this is an anusuchi chapter
            if "anusuchi" in chunk_type:
                parts.append(f"anusuchi{ch_n} ({short_title})")
            else:
                parts.append(f"ch{ch_n} ({short_title})")
        else:
            if "anusuchi" in chunk_type:
                parts.append(f"anusuchi{ch_n}")
            else:
                parts.append(f"ch{ch_n}")

    if hierarchy.get("sec") is not None:
        sec_n = hierarchy["sec"]
        sec_title = hierarchy.get("sec_title", "")
        if sec_title:
            short_title = sec_title[:30].rstrip() + ("..." if len(sec_title) > 30 else "")
            parts.append(f"sec{sec_n} ({short_title})")
        else:
            parts.append(f"sec{sec_n}")

    if hierarchy.get("sub") is not None:
        parts.append(f"sub{hierarchy['sub']}")

    if hierarchy.get("clause") is not None:
        parts.append(f"cl{hierarchy['clause']}")

    breadcrumb = " › ".join(parts) if parts else "(root)"
    return f"{breadcrumb} [{chunk_type}]"


class NERExtractor:
    """
    Extracts structured business registration entities from Nepali
    legal text chunks using the Mistral AI API.
    """

    def __init__(self, model: str = "mistral-large-latest", max_retries: int = 5):
        settings = get_settings()
        self.client = Mistral(api_key=settings.MISTRAL_API_KEY)
        self.model = model
        self.max_retries = max_retries

    def _call_mistral(self, text: str) -> Optional[str]:
        """
        Call Mistral API with exponential backoff retry.
        Returns raw JSON string content or None on failure.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.complete(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": text},
                    ],
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                return content

            except Exception as e:
                error_str = str(e)
                # Detect rate limit (429) or server errors (5xx)
                is_retriable = any(code in error_str for code in ["429", "500", "502", "503"])

                if is_retriable and attempt < self.max_retries:
                    wait_time = 3 ** attempt  # 3s, 9s, 27s, 81s — aggressive backoff for free tier
                    logger.warning(
                        f"Mistral API error (attempt {attempt}/{self.max_retries}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Mistral API failed after {attempt} attempts: {e}")
                    return None

    def _parse_response(self, content: str) -> Optional[ClauseNEROutput]:
        """
        Parse and validate the Mistral JSON response into a ClauseNEROutput.
        Handles common formatting issues (markdown backticks, trailing commas).
        """
        if not content:
            return None

        # Strip markdown code fences if present
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        # Remove trailing commas before } or ] (common LLM JSON issue)
        content = re.sub(r',\s*([}\]])', r'\1', content)

        try:
            data = json.loads(content)
            return ClauseNEROutput(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"Failed to parse Mistral response: {e}\nContent: {content[:200]}")
            return None

    def extract_from_chunk(self, chunk: dict) -> Optional[ExtractedEntity]:
        """
        Extract NER entities from a single dataset chunk.
        
        Args:
            chunk: A chunk dict from the dataset JSON with 'id', 'type', 
                   'hierarchy', 'text' fields.
        
        Returns:
            ExtractedEntity with source traceability, or None if extraction fails.
        """
        text = chunk.get("text", "")
        if not text.strip():
            return None

        raw_response = self._call_mistral(text)
        ner_output = self._parse_response(raw_response)

        if ner_output is None:
            return None

        # Check if any meaningful entities were extracted
        has_entities = (
            ner_output.office
            or ner_output.documents_required
            or ner_output.steps
            or ner_output.price
            or ner_output.duration_days
            or ner_output.prerequisites
        )
        if not has_entities:
            return None

        # Build compact entity with clause_ref for traceability (no full text duplication)
        return ExtractedEntity(
            chunk_id=chunk.get("id", "unknown"),
            clause_ref=build_clause_ref(chunk),
            office=parse_office(ner_output.office),
            documents_required=ner_output.documents_required,
            steps=ner_output.steps,
            price_fees=parse_price_fee(ner_output.price),
            duration=ner_output.duration_days,
            prerequisites=ner_output.prerequisites,
        )

    def extract_batch(
        self,
        chunks: List[dict],
        delay_between_calls: float = 2.0,
        progress_callback=None,
    ) -> List[ExtractedEntity]:
        """
        Extract NER entities from a batch of chunks.
        
        Args:
            chunks: List of chunk dicts to process.
            delay_between_calls: Seconds to wait between API calls to respect rate limits.
            progress_callback: Optional callable(current, total) for progress reporting.
        
        Returns:
            List of successfully extracted entities.
        """
        entities = []
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            entity = self.extract_from_chunk(chunk)
            if entity:
                entities.append(entity)

            if progress_callback:
                progress_callback(i + 1, total)

            # Rate limiting delay between calls
            if i < total - 1:
                time.sleep(delay_between_calls)

        return entities

    @staticmethod
    def build_context_window(chunks: List[dict], window_size: int = 3) -> List[dict]:
        """
        Merge contiguous sub-clauses that share the same parent section
        into combined context windows for richer extraction.
        
        This helps when a registration process is described across multiple
        consecutive sub-sections (e.g., sec3-sub1 through sec3-sub5).
        
        Args:
            chunks: Filtered chunks to potentially merge.
            window_size: Maximum number of contiguous chunks to merge.
        
        Returns:
            List of chunks, where some may have merged text from neighbors.
        """
        if not chunks or window_size <= 1:
            return chunks

        windowed = []
        i = 0
        while i < len(chunks):
            current = chunks[i]
            current_hierarchy = current.get("hierarchy", {})
            current_sec = current_hierarchy.get("sec")

            # Collect contiguous chunks with the same parent section
            group = [current]
            j = i + 1
            while j < len(chunks) and len(group) < window_size:
                next_chunk = chunks[j]
                next_hierarchy = next_chunk.get("hierarchy", {})
                next_sec = next_hierarchy.get("sec")

                if current_sec is not None and next_sec == current_sec:
                    group.append(next_chunk)
                    j += 1
                else:
                    break

            if len(group) > 1:
                # Merge texts, keep first chunk's metadata
                merged_text = "\n\n".join(c["text"] for c in group)
                merged_chunk = dict(current)
                merged_chunk["text"] = merged_text
                merged_chunk["_merged_ids"] = [c["id"] for c in group]
                windowed.append(merged_chunk)
                i = j  # Skip merged chunks
            else:
                windowed.append(current)
                i += 1

        return windowed
