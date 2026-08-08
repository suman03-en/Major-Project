"""
Enhanced NER Extractor supporting both Local Ollama (e.g. Qwen 2.5) and Mistral AI API.

Extracts structured business registration entities from Nepali legal
text chunks using structured JSON output mode.

Features:
- Dual provider support: Local Ollama (Qwen 2.5 / Llama 3.2) & Mistral AI API
- Auto-provider selection: Uses local Ollama if running, falls back to Mistral API
- Pydantic schema-enforced structured output
- Exponential backoff retry on API rate limits and transient errors
- Batch processing with zero-delay option for local execution
- Context windowing: merges contiguous sub-clauses for richer extraction
"""

import os
import re
import json
import time
import logging
import urllib.request
import urllib.error
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
    clean = re.sub(r'[^\d०-९,]', '', text)
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


# --- System prompt for NER extraction ---

SYSTEM_PROMPT = """You are a Named Entity Recognition assistant specializing in Nepali legal and administrative text.

Your task is to extract business registration entities from the provided Nepali text chunk.

Return a JSON object with EXACTLY these fields:
{
  "office": "string or null",
  "documents_required": ["string1", "string2"],
  "steps": ["string1", "string2"],
  "price": "string or null",
  "duration_days": "string or null",
  "prerequisites": "string or null"
}

═══════════════════════════════════════════
FIELD DEFINITIONS AND RULES
═══════════════════════════════════════════

1. "office" — The government office, department, ministry, or authority that performs the action.
   ✅ VALID examples: "कम्पनी रजिष्ट्रारको कार्यालय", "उद्योग विभाग", "मन्त्रालय", "उद्योग दर्ता गर्ने निकाय"
   ❌ INVALID — Do NOT extract these as office:
     - Generic words like "बमोजिम", "बमोजिम कार्यालय", "निकायको सिफारिसमा"
     - Sentence fragments that are not proper office names
   → Must be a specific institutional name. If unsure, use null.
   → Return as a plain string, NOT as an object/dictionary.

2. "documents_required" — Specific named documents, certificates, forms, or applications that must be submitted.
   ✅ VALID examples: "उद्योग दर्ता प्रमाणपत्र", "वातावरणीय प्रभाव मूल्याङ्कन प्रतिवेदन", "निवेदन", "अनुमति पत्र", "प्रतिलिपि"
   ❌ INVALID — Do NOT extract these as documents:
     - Legal cross-references: "उपदफा (१)", "उपदफा (४)", "दफा १३", "बमोजिम"
     - Types of services/activities: "उद्योग दर्ता", "नवीकरण", "नामसारी", "नाम परिवर्तन", "स्थानान्तरण", "क्षमता वृद्धि", "पुँजी वृद्धि"
     - Single characters or Nepali list markers: "क", "ख", "ग", "ढ", "ण", "त", "थ"
     - Generic words: "बमोजिम", "मिति", "एक पटक", "सुझाव", "सिफारिस", "प्रतिशत"
   → Each document name must be at least 3 words or a specific named form/certificate.

3. "steps" — Complete, meaningful procedural actions or duties described in the text.
   ✅ VALID examples: "उद्योग दर्ता गर्ने निकाय समक्ष निवेदन दिनुपर्छ", "उपदफा बमोजिम प्राप्त निवेदन जाँचबुझ गर्ने", "निवेदकलाई सोको जानकारी दिनुपर्नेछ"
   ❌ INVALID — Do NOT extract these as steps:
     - Individual words: "पेश", "सम्बन्धमा", "हुने", "निर्णय"
     - Two-word fragments: "अनुगमन गरिए", "सिफारिस गरे"
     - Incomplete phrases that do not describe a complete action
   → Each step MUST be a complete sentence or phrase describing a specific action (minimum 5 words).
   → If the text does not describe a sequential procedure, return [].

4. "price" — ONLY official government fees with specific amounts.
   ✅ VALID: "रु. १०,०००", "दस्तुर रु.५,०००", "पाँच हजार रुपैयाँ दस्तुर"
   ❌ INVALID: Capital amounts, penalties/जरिवाना, percentages like "पैंतीस प्रतिशत"
   → Must contain a specific monetary amount with रु./रुपैयाँ/शुल्क/दस्तुर.

5. "duration_days" — Time periods, deadlines, or validity periods.
   ✅ VALID: "तीस दिनभित्र", "सात कार्य दिनभित्र", "एक वर्ष", "पाँच दिनभित्र", "३ महिनाभित्र"
   ❌ INVALID: Words that are not time periods: "अद्यावधि", "तीन"
   → Must contain a time unit word: दिन, दिनभित्र, महिना, वर्ष, कार्य दिन, etc.

6. "prerequisites" — Eligibility criteria or conditions that must be met BEFORE the process.
   → Must describe a specific condition, not a general statement.

═══════════════════════════════════════════
WHEN TO RETURN ALL NULLS / EMPTY
═══════════════════════════════════════════
If the text is:
  - A definition section (परिभाषा)
  - A list of industry categories or types
  - About board governance structure or meeting rules with no registration procedure
  - A penalty/punishment section (सजाय/जरिवाना) with no registration process
Then return: {"office": null, "documents_required": [], "steps": [], "price": null, "duration_days": null, "prerequisites": null}

═══════════════════════════════════════════
EXAMPLES
═══════════════════════════════════════════

Example Input: "(१) अनुसूची-१ मा उल्लिखित उद्योग दर्ता गराउन चाहने व्यक्तिले तोकिएको विवरण सहित उद्योग दर्ता गर्ने निकाय मार्फत्‌ बोर्ड समक्ष निवेदन दिनुपर्नेछ।"
Example Output:
{
  "office": "उद्योग दर्ता गर्ने निकाय",
  "documents_required": ["तोकिएको विवरण सहित निवेदन"],
  "steps": ["उद्योग दर्ता गराउन चाहने व्यक्तिले तोकिएको विवरण सहित उद्योग दर्ता गर्ने निकाय मार्फत्‌ बोर्ड समक्ष निवेदन दिनुपर्नेछ"],
  "price": null,
  "duration_days": null,
  "prerequisites": null
}

Example Input: "बोर्डको बैठकमा पेश हुने कार्यसूचीको सम्बन्धमा बोर्डको कुनै सदस्यको निजी सरोकार वा स्वार्थ रहेको भएमा त्यस्तो सदस्यले त्यस्तो कार्यसूचीका सम्बन्धमा हुने निर्णय प्रक्रियामा भाग लिन पाउने छैन।"
Example Output:
{
  "office": null,
  "documents_required": [],
  "steps": [],
  "price": null,
  "duration_days": null,
  "prerequisites": null
}

═══════════════════════════════════════════
FINAL RULES
═══════════════════════════════════════════
- Extract entities ONLY from the given text. Do not infer or hallucinate.
- If no entity of a type is found, use null for strings and [] for arrays.
- Keep extracted text in original Nepali language.
- For steps, preserve the logical order as they appear in the text.
- Output ONLY the JSON object. No markdown, no explanation.
- QUALITY CHECK: Before returning, verify each extracted item meets the validity rules above."""


def build_clause_ref(chunk: dict) -> str:
    """
    Build a compact, human-readable breadcrumb string from a chunk's
    hierarchy metadata and type.

    Format: 'ch{n} (title) › sec{n} (title) › sub{n} [type]'
    """
    hierarchy = chunk.get("hierarchy", {})
    chunk_type = chunk.get("type", "")
    parts = []

    # Anusuchi (annex) or regular chapter
    if hierarchy.get("ch") is not None:
        ch_n = hierarchy["ch"]
        ch_title = hierarchy.get("ch_title", "")
        if ch_title:
            short_title = ch_title[:30].rstrip() + ("..." if len(ch_title) > 30 else "")
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


def get_ollama_installed_models(host: str = "http://localhost:11434") -> List[str]:
    """Retrieve list of installed model names from local Ollama server."""
    try:
        req = urllib.request.Request(f"{host}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        pass
    return []


def is_ollama_available(host: str = "http://localhost:11434") -> bool:
    """Check if local Ollama server is running and accessible."""
    return len(get_ollama_installed_models(host)) > 0 or False


class NERExtractor:
    """
    Extracts structured business registration entities from Nepali legal
    text chunks using Local Ollama (Qwen 2.5 / Llama 3.2) or Mistral AI API.
    """

    def __init__(
        self,
        provider: str = "auto",
        ollama_host: Optional[str] = None,
        ollama_model: Optional[str] = None,
        mistral_model: str = "mistral-large-latest",
        max_retries: int = 5,
    ):
        settings = get_settings()
        self.ollama_host = ollama_host or settings.OLLAMA_HOST or "http://localhost:11434"
        requested_ollama_model = ollama_model or settings.OLLAMA_MODEL or "qwen2.5:3b"
        self.mistral_model = mistral_model
        self.max_retries = max_retries

        # Resolve installed Ollama models for auto tag matching
        installed_ollama = get_ollama_installed_models(self.ollama_host)
        if installed_ollama:
            # Match exact, prefix (e.g. qwen2.5 -> qwen2.5:3b), or fallback to first installed model
            if requested_ollama_model in installed_ollama:
                self.ollama_model = requested_ollama_model
            else:
                matches = [m for m in installed_ollama if m.startswith(requested_ollama_model)]
                if matches:
                    self.ollama_model = matches[0]
                else:
                    self.ollama_model = installed_ollama[0]
        else:
            self.ollama_model = requested_ollama_model

        # Provider selection logic
        if provider == "auto":
            if installed_ollama:
                self.active_provider = "ollama"
                logger.info(
                    f"Local Ollama detected at {self.ollama_host} (model: {self.ollama_model}). "
                    "Using Local LLM for NER extraction (No rate limits!)."
                )
            else:
                self.active_provider = "mistral"
                logger.info(
                    f"Ollama not detected at {self.ollama_host}. Falling back to Mistral API ({self.mistral_model})."
                )
        elif provider == "ollama":
            self.active_provider = "ollama"
            if not is_ollama_available(self.ollama_host):
                logger.warning(
                    f"Provider explicitly set to 'ollama' but {self.ollama_host} is not responding."
                )
        elif provider == "mistral":
            self.active_provider = "mistral"
        else:
            raise ValueError(f"Unknown provider '{provider}'. Must be 'auto', 'ollama', or 'mistral'.")

        # Initialize Mistral client if needed
        self.mistral_client = None
        if self.active_provider == "mistral":
            if not settings.MISTRAL_API_KEY:
                raise ValueError("MISTRAL_API_KEY is required when using Mistral provider.")
            self.mistral_client = Mistral(api_key=settings.MISTRAL_API_KEY)

    def _call_ollama(self, text: str) -> Optional[str]:
        """Call Local Ollama API (e.g. Qwen 2.5) with JSON format enforcement."""
        url = f"{self.ollama_host}/api/chat"
        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "format": "json",
            "stream": False,
            "think": False,          # Qwen3: disable chain-of-thought reasoning
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=180.0) as resp:
                    if resp.status == 200:
                        body = resp.read().decode("utf-8")
                        resp_data = json.loads(body)
                        content = resp_data.get("message", {}).get("content", "")
                        return content
            except Exception as e:
                if attempt < self.max_retries:
                    wait_time = 2 * attempt
                    logger.warning(f"Ollama call failed (attempt {attempt}/{self.max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Ollama call failed after {attempt} attempts: {e}")
                    return None

    def _call_mistral(self, text: str) -> Optional[str]:
        """Call Mistral API with exponential backoff retry."""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.mistral_client.chat.complete(
                    model=self.mistral_model,
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
                is_retriable = any(code in error_str for code in ["429", "500", "502", "503"])

                if is_retriable and attempt < self.max_retries:
                    wait_time = 3 ** attempt
                    logger.warning(
                        f"Mistral API error (attempt {attempt}/{self.max_retries}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Mistral API failed after {attempt} attempts: {e}")
                    return None

    def _parse_response(self, content: Optional[str]) -> Optional[ClauseNEROutput]:
        """
        Parse and validate JSON response into a ClauseNEROutput.
        Handles markdown backticks, minor JSON syntax errors, and
        normalizes office field from dict to string.
        """
        if not content:
            return None

        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        # Remove trailing commas before } or ]
        content = re.sub(r',\s*([}\]])', r'\1', content)

        try:
            data = json.loads(content)

            # Normalize office field: dict → string
            if isinstance(data.get("office"), dict):
                office_dict = data["office"]
                data["office"] = office_dict.get("name") or office_dict.get("office") or None

            return ClauseNEROutput(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}\nContent: {content[:200]}")
            return None

    def _validate_and_clean(self, ner_output: ClauseNEROutput) -> ClauseNEROutput:
        """
        Post-processing validation layer that cleans LLM output to fix common
        extraction errors: word fragmentation, category confusion, garbage text.
        """
        # --- Clean office ---
        clean_office = ner_output.office
        if clean_office:
            clean_office = clean_office.strip()
            # Reject generic non-office strings
            invalid_office_patterns = [
                r'^बमोजिम$',
                r'^बमोजिम\s',
                r'सिफारिसमा$',
                r'^निकायको\s',
            ]
            for pattern in invalid_office_patterns:
                if re.search(pattern, clean_office):
                    clean_office = None
                    break
            # Reject very short "office" names (< 3 chars)
            if clean_office and len(clean_office) < 3:
                clean_office = None

        # --- Clean documents_required ---
        # Patterns that indicate legal cross-references, not actual documents
        doc_reject_patterns = [
            r'^उपदफा\s*\(.*\)$',          # "उपदफा (१)" etc.
            r'^दफा\s*[०-९\d]+',             # "दफा १३" etc.
            r'^बमोजिम$',                    # standalone "बमोजिम"
            r'^मिति$',                       # standalone "मिति"
            r'^एक पटक$',                    # "एक पटक"
            r'^प्रतिशत$',                   # "प्रतिशत"
            r'^सुझाव$',                      # "सुझाव"
            r'^सिफारिस$',                   # "सिफारिस"
            r'^दस्तुर$',                     # bare "दस्तुर" without amount
            r'^जानकारी$',                    # bare "जानकारी"
            r'^अभिलेख$',                    # bare "अभिलेख"
            r'^कागजात$',                    # bare "कागजात"
        ]
        # Service type names that are NOT documents
        service_type_keywords = {
            'उद्योग दर्ता', 'नवीकरण', 'नामसारी', 'नाम परिवर्तन',
            'स्थानान्तरण', 'क्षमता वृद्धि', 'पुँजी वृद्धि',
        }
        clean_docs = []
        for doc in ner_output.documents_required:
            doc = doc.strip()
            # Skip empty or single-character entries (Nepali list markers)
            if len(doc) <= 2:
                continue
            # Skip legal cross-reference patterns
            if any(re.match(p, doc) for p in doc_reject_patterns):
                continue
            # Skip service type names
            if doc in service_type_keywords:
                continue
            clean_docs.append(doc)

        # --- Clean steps ---
        clean_steps = []
        for step in ner_output.steps:
            step = step.strip()
            # Count words (Nepali words separated by spaces)
            word_count = len(step.split())
            # Reject steps with fewer than 4 words (fragments/individual words)
            if word_count < 4:
                continue
            clean_steps.append(step)

        # --- Clean duration ---
        clean_duration = ner_output.duration_days
        if clean_duration:
            clean_duration = clean_duration.strip()
            # Must contain a time-related word
            time_keywords = ['दिन', 'महिना', 'वर्ष', 'कार्य दिन', 'भित्र', 'सम्म']
            has_time_word = any(kw in clean_duration for kw in time_keywords)
            if not has_time_word:
                clean_duration = None

        # --- Clean price ---
        clean_price = ner_output.price
        if clean_price:
            clean_price = clean_price.strip()
            # Must contain fee-related indicators
            fee_indicators = ['रु', 'रुपैयाँ', 'शुल्क', 'दस्तुर', 'हजार']
            has_fee_word = any(kw in clean_price for kw in fee_indicators)
            if not has_fee_word:
                clean_price = None
            # Reject penalty amounts
            penalty_indicators = ['जरिवाना', 'जरिबाना', 'सजाय']
            if clean_price and any(kw in clean_price for kw in penalty_indicators):
                clean_price = None

        # --- Clean prerequisites ---
        clean_prereqs = ner_output.prerequisites
        if clean_prereqs:
            clean_prereqs = clean_prereqs.strip()
            if len(clean_prereqs) < 10:
                clean_prereqs = None

        return ClauseNEROutput(
            office=clean_office,
            documents_required=clean_docs,
            steps=clean_steps,
            price=clean_price,
            duration_days=clean_duration,
            prerequisites=clean_prereqs,
        )

    def extract_from_chunk(self, chunk: dict) -> Optional[ExtractedEntity]:
        """Extract NER entities from a single dataset chunk."""
        text = chunk.get("text", "")
        if not text.strip():
            return None

        if self.active_provider == "ollama":
            raw_response = self._call_ollama(text)
        else:
            raw_response = self._call_mistral(text)

        ner_output = self._parse_response(raw_response)

        if ner_output is None:
            return None

        # Apply post-processing validation and cleaning
        ner_output = self._validate_and_clean(ner_output)

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
        delay_between_calls: Optional[float] = None,
        progress_callback=None,
    ) -> List[ExtractedEntity]:
        """
        Extract NER entities from a batch of chunks.
        Automatically uses 0s delay for Ollama and 2s delay for Mistral API.
        """
        if delay_between_calls is None:
            delay_between_calls = 0.0 if self.active_provider == "ollama" else 2.0

        entities = []
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            entity = self.extract_from_chunk(chunk)
            if entity:
                entities.append(entity)

            if progress_callback:
                progress_callback(i + 1, total)

            if delay_between_calls > 0 and i < total - 1:
                time.sleep(delay_between_calls)

        return entities

    @staticmethod
    def build_context_window(chunks: List[dict], window_size: int = 3) -> List[dict]:
        """Merge contiguous sub-clauses sharing the same parent section."""
        if not chunks or window_size <= 1:
            return chunks

        windowed = []
        i = 0
        while i < len(chunks):
            current = chunks[i]
            current_hierarchy = current.get("hierarchy", {})
            current_sec = current_hierarchy.get("sec")

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
                merged_text = "\n\n".join(c["text"] for c in group)
                merged_chunk = dict(current)
                merged_chunk["text"] = merged_text
                merged_chunk["_merged_ids"] = [c["id"] for c in group]
                windowed.append(merged_chunk)
                i = j
            else:
                windowed.append(current)
                i += 1

        return windowed
