import os
import logging
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

def setup_logging(level: int = logging.INFO):
    """Simple centralized logging configuration."""
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")


class Settings(BaseSettings):
    BASE_DIR: str = Field(default_factory=lambda: os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    QDRANT_URL: str
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "neo4j_pass"
    MISTRAL_API_KEY: str = ""
    MISTRAL_MODEL: str = "mistral-large-latest"
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:3b"
    
    # Derived directory paths
    EXTRACTED_JSONS_DIR: str = ""
    NER_OUTPUTS_DIR: str = ""

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    def model_post_init(self, __context):
        if not self.EXTRACTED_JSONS_DIR:
            self.EXTRACTED_JSONS_DIR = os.path.join(self.BASE_DIR, "extracted_jsons")
        if not self.NER_OUTPUTS_DIR:
            self.NER_OUTPUTS_DIR = os.path.join(self.BASE_DIR, "ner_outputs")

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Generate settings from environment"""
    return Settings()