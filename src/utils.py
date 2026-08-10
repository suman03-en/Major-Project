"""
Shared utilities for the Nepali Legal RAG pipeline.

Contains common helpers used across extraction, embedding, and knowledge_base modules.
"""

import re
from typing import Optional

# Nepali ↔ English digit translation table
NEPALI_DIGITS = '०१२३४५६७८९'
ENGLISH_DIGITS = '0123456789'
_TRANS_TABLE = str.maketrans(NEPALI_DIGITS, ENGLISH_DIGITS)


def nepali_to_int(text: str) -> Optional[int]:
    """
    Convert a Nepali numeral string to an integer.

    Handles mixed Nepali/English digits, commas, and surrounding whitespace.
    Returns None if no digits are found or conversion fails.
    """
    clean = re.sub(r'[^\d०-९,]', '', text)
    clean = clean.replace(',', '')
    if not clean:
        return None
    try:
        return int(clean.translate(_TRANS_TABLE))
    except ValueError:
        return None

def get_ollama_installed_models(host: str = "http://localhost:11434") -> list[str]:
    """Retrieve list of installed model names from local Ollama server."""
    import urllib.request
    import json
    
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
    return len(get_ollama_installed_models(host)) > 0

