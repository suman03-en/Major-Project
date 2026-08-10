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
