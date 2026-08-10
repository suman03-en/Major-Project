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


from src.config import get_settings
from src.knowledge_base.schemas import ClauseNEROutput, RawStepOutput, ExtractedEntity, ProcessStep, PriceFee, OfficeEntity

logger = logging.getLogger(__name__)

from src.utils import nepali_to_int, get_ollama_installed_models, is_ollama_available


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


# --- System prompt for NER extraction (step-centric for Neo4j graph DB) ---

SYSTEM_PROMPT = """You are a Named Entity Recognition assistant specializing in Nepali legal and administrative text.

Your task is to extract STEP-BY-STEP procedural information from Nepali legal text about business registration.
Each step must carry its OWN associated office, documents, fee, and duration.

This data will be stored in a Neo4j graph database where users ask questions like:
"How do I register a company?" and expect to receive:
- Ordered steps
- Which office to visit at each step
- What documents to carry at each step
- What fees to pay at each step
- How long each step takes

Return a JSON object with EXACTLY these fields:
{
  "process_name": "string or null",
  "steps": [
    {
      "step_number": 1,
      "action": "complete action description in Nepali",
      "office": "government office name or null",
      "documents_required": ["doc1", "doc2"],
      "fee": "fee amount text or null",
      "duration": "time period or null",
      "prerequisite": "condition or null"
    }
  ]
}

═══════════════════════════════════════════
FIELD DEFINITIONS AND RULES
═══════════════════════════════════════════

1. "process_name" — The high-level name of the process or procedure described.
   ✅ VALID: "उद्योग दर्ता", "उद्योग नवीकरण", "नामसारी", "उद्योग खारेज", "अनुमतिपत्र"
   → If the text describes part of a known process, use the process name.
   → If unclear, use null.

2. "steps" — Array of procedural steps. Each step is a COMPLETE action with its metadata.

   2a. "step_number" — Sequential integer starting from 1.

   2b. "action" — A complete, meaningful procedural action described in the text.
       ✅ VALID: "उद्योग दर्ता गराउन चाहने व्यक्तिले तोकिएको विवरण सहित निकाय मार्फत्‌ बोर्ड समक्ष निवेदन दिनुपर्नेछ"
       ❌ INVALID: Individual words like "पेश", fragments like "अनुगमन गरिए"
       → Must be a complete sentence or phrase describing a specific action (minimum 5 words).

   2c. "office" — The government office where THIS SPECIFIC step is performed.
       ✅ VALID: "कम्पनी रजिष्ट्रारको कार्यालय", "उद्योग विभाग", "उद्योग दर्ता गर्ने निकाय", "मन्त्रालय", "बोर्ड", "विभाग"
       ❌ INVALID:
         - Generic words: "बमोजिम", sentence fragments
         - Descriptions of people/roles: "अनुमति प्राप्त गर्ने आवेदक", "निवेदक"
       → Must be a specific institutional name (office/department/ministry/board/authority).
       → If no office is mentioned for this step, use null.
       → Return as a plain string, NOT as an object/dictionary.

   2d. "documents_required" — ONLY specific named documents, certificates, forms, or applications needed for this step.
       ✅ VALID: "उद्योग दर्ता प्रमाणपत्र", "वातावरणीय प्रभाव मूल्याङ्कन प्रतिवेदन", "तोकिएको ढाँचामा निवेदन", "नागरिकताको प्रतिलिपि"
       ❌ INVALID — Do NOT extract ANY of these as documents:
         - Single words: "अनुमति", "आवश्यकता", "कारण", "कारोबार", "जग्गा", "निवेदन" (alone)
         - Legal cross-references: "उपदफा (१)", "दफा १३", "प्रचलित कानून", "खण्ड (ख)"
         - Sentence fragments: "उपदफा (१) मा जुनसुकै कुरा लेखिएको", "अन्य कुराका अतिरिकत देहायका विवरण"
         - Service/activity types: "उद्योग दर्ता", "नवीकरण", "नामसारी", "उद्योग सञ्चालन", "व्यावसायिक उत्पादन", "कारोबार"
         - Generic nouns: "आवश्यक जाँचबुझ", "आवश्यकता", "सिफारिस", "प्रचलित कानून"
       → A valid document must be something you can PHYSICALLY carry or submit (certificate, form, application, report).
       → If no specific document name is mentioned, return [].

   2e. "fee" — Government fee for THIS SPECIFIC step.
       ✅ VALID: "रु. १०,०००", "दस्तुर रु.५,०००", "पाँच हजार रुपैयाँ दस्तुर"
       ❌ INVALID: Capital amounts, penalties/जरिवाना, percentages
       → Must contain a specific monetary amount with रु./रुपैयाँ/शुल्क/दस्तुर.

   2f. "duration" — An ACTUAL time period or deadline for THIS SPECIFIC step.
       ✅ VALID: "तीस दिनभित्र", "सात कार्य दिनभित्र", "एक वर्ष", "पाँच दिनभित्र", "नब्बे दिन"
       ❌ INVALID:
         - Legal references to deadlines: "उपदफा (३) बमोजिमको म्यादभित्र" — this is a REFERENCE, not a time period
         - Words without time units: "अद्यावधि", "तीन"
       → Must be a CONCRETE time value with a unit: दिन, दिनभित्र, महिना, वर्ष, कार्य दिन
       → Do NOT extract legal cross-references to deadlines (containing बमोजिम/उपदफा/दफा)

   2g. "prerequisite" — Condition that must be met BEFORE this step can be taken.
       → Must describe a specific condition, not a general statement.
       → Must be at least 10 characters.
       → Do NOT use legal references like "उपदफा (१) बमोजिम" as prerequisites.

═══════════════════════════════════════════
WHEN TO RETURN EMPTY STEPS
═══════════════════════════════════════════
If the text is:
  - A definition section (परिभाषा)
  - A list of industry categories or types
  - About board governance or meeting rules with no registration procedure
  - A penalty/punishment section (सजाय/जरिवाना) with no registration process
Then return: {"process_name": null, "steps": []}

═══════════════════════════════════════════
EXAMPLES
═══════════════════════════════════════════

Example Input: "(१) अनुसूची-१ मा उल्लिखित उद्योग दर्ता गराउन चाहने व्यक्तिले तोकिएको विवरण सहित उद्योग दर्ता गर्ने निकाय मार्फत्‌ बोर्ड समक्ष निवेदन दिनुपर्नेछ।"
Example Output:
{
  "process_name": "उद्योग दर्ता",
  "steps": [
    {
      "step_number": 1,
      "action": "अनुसूची-१ मा उल्लिखित उद्योग दर्ता गराउन चाहने व्यक्तिले तोकिएको विवरण सहित उद्योग दर्ता गर्ने निकाय मार्फत्‌ बोर्ड समक्ष निवेदन दिनुपर्नेछ",
      "office": "उद्योग दर्ता गर्ने निकाय",
      "documents_required": ["तोकिएको विवरण सहित निवेदन"],
      "fee": null,
      "duration": null,
      "prerequisite": null
    }
  ]
}

Example Input: "बोर्डको बैठकमा पेश हुने कार्यसूचीको सम्बन्धमा बोर्डको कुनै सदस्यको निजी सरोकार वा स्वार्थ रहेको भएमा त्यस्तो सदस्यले त्यस्तो कार्यसूचीका सम्बन्धमा हुने निर्णय प्रक्रियामा भाग लिन पाउने छैन।"
Example Output:
{
  "process_name": null,
  "steps": []
}

Example Input: "(३) उपदफा (१) बमोजिम प्राप्त निवेदन जाँचबुझ गर्दा आवश्यक विवरण तथा कागजात पूरा भएको देखिएमा त्यस्तो विवरण वा कागजात प्राप्त भएको पाँच दिनभित्र उद्योग दर्ता गर्ने निकायले उद्योग दर्ता गरी तोकिएको ढाँचामा उद्योग दर्ताको प्रमाणपत्र दिनु पर्नेछ।"
Example Output:
{
  "process_name": "उद्योग दर्ता",
  "steps": [
    {
      "step_number": 1,
      "action": "उपदफा (१) बमोजिम प्राप्त निवेदन जाँचबुझ गर्दा आवश्यक विवरण तथा कागजात पूरा भएको देखिएमा उद्योग दर्ता गर्ने निकायले उद्योग दर्ता गरी तोकिएको ढाँचामा उद्योग दर्ताको प्रमाणपत्र दिनु पर्नेछ",
      "office": "उद्योग दर्ता गर्ने निकाय",
      "documents_required": ["आवश्यक विवरण तथा कागजात"],
      "fee": null,
      "duration": "पाँच दिनभित्र",
      "prerequisite": "आवश्यक विवरण तथा कागजात पूरा भएको"
    }
  ]
}

Example Input: "(२) दफा ३ बमोजिम उद्योग दर्ताको लागि तोकिएको दस्तुर रु. ५,००० बुझाउनु पर्नेछ।"
Example Output:
{
  "process_name": "उद्योग दर्ता",
  "steps": [
    {
      "step_number": 1,
      "action": "दफा ३ बमोजिम उद्योग दर्ताको लागि तोकिएको दस्तुर बुझाउनु पर्नेछ",
      "office": null,
      "documents_required": [],
      "fee": "दस्तुर रु. ५,०००",
      "duration": null,
      "prerequisite": null
    }
  ]
}

═══════════════════════════════════════════
FINAL RULES
═══════════════════════════════════════════
- Extract entities ONLY from the given text. Do not infer or hallucinate.
- If the text describes multiple sequential actions, create multiple steps.
- If the text describes a single action, create one step.
- Each step's office, documents, fee, and duration must be SPECIFIC to that step.
- Do NOT duplicate the same information across all steps — only assign metadata to the step it belongs to.
- Keep extracted text in original Nepali language.
- Output ONLY the JSON object. No markdown, no explanation.
- QUALITY CHECK: Before returning, verify each step has a meaningful action (5+ words)."""


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
        normalizes step-level office fields from dict to string.
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

            # Ensure steps is a list
            if not isinstance(data.get("steps"), list):
                data["steps"] = []

            # Normalize each step's office field: dict → string
            for step in data.get("steps", []):
                if isinstance(step, dict) and isinstance(step.get("office"), dict):
                    office_dict = step["office"]
                    step["office"] = office_dict.get("name") or office_dict.get("office") or None

            return ClauseNEROutput(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}\nContent: {content[:200]}")
            return None

    def _validate_and_clean(self, ner_output: ClauseNEROutput) -> ClauseNEROutput:
        """
        Post-processing validation layer that cleans step-centric LLM output.
        Iterates over each step and cleans its fields individually.
        """
        # --- Clean process_name ---
        clean_process = ner_output.process_name
        if clean_process:
            clean_process = clean_process.strip()
            if len(clean_process) < 3:
                clean_process = None

        # --- Clean each step ---
        clean_steps = []
        for step in ner_output.steps:
            cleaned = self._clean_step(step)
            if cleaned is not None:
                clean_steps.append(cleaned)

        # Re-number steps sequentially after cleaning
        for i, step in enumerate(clean_steps):
            clean_steps[i] = RawStepOutput(
                step_number=i + 1,
                action=step.action,
                office=step.office,
                documents_required=step.documents_required,
                fee=step.fee,
                duration=step.duration,
                prerequisite=step.prerequisite,
            )

        return ClauseNEROutput(
            process_name=clean_process,
            steps=clean_steps,
        )

    def _clean_step(self, step: RawStepOutput) -> Optional[RawStepOutput]:
        """Clean a single step's fields. Returns None if step should be discarded."""
        action = step.action.strip() if step.action else ""

        # Reject steps with too-short actions (fragments)
        word_count = len(action.split())
        if word_count < 4:
            return None

        # --- Clean office ---
        clean_office = step.office
        if clean_office:
            clean_office = clean_office.strip()
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
            if clean_office and len(clean_office) < 3:
                clean_office = None

        # --- Clean documents ---
        doc_reject_patterns = [
            r'^उपदफा\s*\(.*\)',          # legal refs: "उपदफा (१)", "उपदफा (१) मा..."
            r'^दफा\s*[०-९\d]+',           # "दफा १३"
            r'^खण्ड\s*\(',                 # "खण्ड (ख) मा..."
            r'^बमोजिम',                    # "बमोजिम" or "बमोजिमको"
            r'^मिति$',
            r'^एक पटक$',
            r'^प्रतिशत$',
            r'^सुझाव$',
            r'^सिफारिस$',
            r'^दस्तुर$',
            r'^जानकारी$',
            r'^अभिलेख$',
            r'^कागजात$',
            r'^अन्य कुरा',                 # "अन्य कुराका अतिरिकत..."
            r'लेखिएको$',                   # sentence fragments ending in "लेखिएको"
        ]
        # Single-word generic nouns that are NOT documents
        single_word_rejects = {
            'अनुमति', 'आवश्यकता', 'कारण', 'कारोबार', 'जग्गा',
            'निवेदन', 'प्रचलित कानून', 'अनुमतिपत्र', 'आवश्यक जाँचबुझ',
            'उद्योग सञ्चालन', 'व्यावसायिक उत्पादन', 'आवश्यकता अनुसार',
        }
        # Service/activity type names that are NOT documents
        service_type_keywords = {
            'उद्योग दर्ता', 'नवीकरण', 'नामसारी', 'नाम परिवर्तन',
            'स्थानान्तरण', 'क्षमता वृद्धि', 'पुँजी वृद्धि',
        }
        clean_docs = []
        for doc in step.documents_required:
            doc = doc.strip()
            if len(doc) <= 2:
                continue
            # Reject single-word docs (too generic to be a real document name)
            if len(doc.split()) == 1:
                continue
            if any(re.match(p, doc) for p in doc_reject_patterns):
                continue
            if doc in service_type_keywords or doc in single_word_rejects:
                continue
            clean_docs.append(doc)

        # --- Clean fee ---
        clean_fee = step.fee
        if clean_fee:
            clean_fee = clean_fee.strip()
            fee_indicators = ['रु', 'रुपैयाँ', 'शुल्क', 'दस्तुर', 'हजार']
            has_fee_word = any(kw in clean_fee for kw in fee_indicators)
            if not has_fee_word:
                clean_fee = None
            penalty_indicators = ['जरिवाना', 'जरिबाना', 'सजाय']
            if clean_fee and any(kw in clean_fee for kw in penalty_indicators):
                clean_fee = None

        # --- Clean duration ---
        clean_duration = step.duration
        if clean_duration:
            clean_duration = clean_duration.strip()
            # Must have a concrete time unit
            time_keywords = ['दिन', 'महिना', 'वर्ष', 'कार्य दिन', 'भित्र', 'सम्म']
            has_time_word = any(kw in clean_duration for kw in time_keywords)
            if not has_time_word:
                clean_duration = None
            # Reject legal cross-references to deadlines (not actual time values)
            if clean_duration:
                legal_ref_patterns = ['बमोजिम', 'उपदफा', 'दफा', 'खण्ड']
                if any(ref in clean_duration for ref in legal_ref_patterns):
                    clean_duration = None

        # --- Clean prerequisite ---
        clean_prereq = step.prerequisite
        if clean_prereq:
            clean_prereq = clean_prereq.strip()
            if len(clean_prereq) < 10:
                clean_prereq = None

        return RawStepOutput(
            step_number=step.step_number,
            action=action,
            office=clean_office,
            documents_required=clean_docs,
            fee=clean_fee,
            duration=clean_duration,
            prerequisite=clean_prereq,
        )

    def extract_from_chunk(self, chunk: dict) -> Optional[ExtractedEntity]:
        """Extract step-centric NER entities from a single dataset chunk."""
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

        # Discard if no steps survived cleaning
        if not ner_output.steps:
            return None

        # Convert RawStepOutput → ProcessStep (with structured office & fees)
        process_steps = []
        for raw_step in ner_output.steps:
            process_steps.append(ProcessStep(
                step_number=raw_step.step_number,
                action=raw_step.action,
                office=parse_office(raw_step.office),
                documents_required=raw_step.documents_required,
                price_fees=parse_price_fee(raw_step.fee),
                duration=raw_step.duration,
                prerequisite=raw_step.prerequisite,
            ))

        return ExtractedEntity(
            chunk_id=chunk.get("id", "unknown"),
            clause_ref=build_clause_ref(chunk),
            process_name=ner_output.process_name,
            steps=process_steps,
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
