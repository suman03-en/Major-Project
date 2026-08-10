import os
from dataclasses import dataclass
from functools import lru_cache
import logging
from dotenv import load_dotenv

load_dotenv()

def setup_logging(level: int = logging.INFO):
    """Simple centralized logging configuration."""
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")


@dataclass(frozen=True)
class Settings:
    BASE_DIR: str
    QDRANT_URL: str
    MISTRAL_API_KEY: str
    OLLAMA_HOST: str 
    OLLAMA_MODEL: str
    # Derived directory paths — always relative to BASE_DIR
    EXTRACTED_JSONS_DIR: str  # Output of PDF extraction pipeline; input to embedding & NER
    NER_OUTPUTS_DIR: str      # Output of NER extraction pipeline

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Generate settings from environment"""
    qdrant_url = os.getenv("QDRANT_URL")
    mistral_api_key = os.getenv("MISTRAL_API_KEY", "")
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    if not qdrant_url:
        raise ValueError("QDRANT_URL is not defined in environment variables")

    return Settings(
        BASE_DIR=base_dir,
        QDRANT_URL=qdrant_url,
        MISTRAL_API_KEY=mistral_api_key,
        OLLAMA_HOST=ollama_host,
        OLLAMA_MODEL=ollama_model,
        EXTRACTED_JSONS_DIR=os.path.join(base_dir, "extracted_jsons"),
        NER_OUTPUTS_DIR=os.path.join(base_dir, "ner_outputs"),
    )