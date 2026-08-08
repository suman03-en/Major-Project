import http
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    BASE_DIR: str
    QDRANT_URL: str
    MISTRAL_API_KEY: str
    # Derived directory paths — always relative to BASE_DIR
    EXTRACTED_JSONS_DIR: str  # Output of PDF extraction pipeline; input to embedding & NER
    NER_OUTPUTS_DIR: str      # Output of NER extraction pipeline

def get_settings() -> Settings:
    """Generate settings from environment"""
    qdrant_url = os.getenv("QDRANT_URL")
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    if not qdrant_url:
        raise ValueError("QDRANT_URL is not defined in environment variables")

    if not mistral_api_key:
        raise ValueError("MISTRAL_API_KEY is not defined in environment variables")

    return Settings(
        BASE_DIR=base_dir,
        QDRANT_URL=qdrant_url,
        MISTRAL_API_KEY=mistral_api_key,
        EXTRACTED_JSONS_DIR=os.path.join(base_dir, "extracted_jsons"),
        NER_OUTPUTS_DIR=os.path.join(base_dir, "ner_outputs"),
    )