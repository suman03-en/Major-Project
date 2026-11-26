import logging
import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def setup_logging(level: int = logging.INFO):
    """Simple centralized logging configuration."""
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")


class Settings(BaseSettings):
    BASE_DIR: str = Field(default_factory=lambda: os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    QDRANT_URL: str = "http://localhost:6333"
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "neo4j_pass"
    # Standard OpenAI-compatible LLM configuration (provider-agnostic)
    LLM_BASE_URL: str = ""
    LLM_API_KEY: str = ""
    LLM_MODEL: str = ""
    LLM_HELPER_MODEL: str = ""
    LLM_FALLBACKS: str = ""
    LLM_MAX_RETRIES: int = 4
    LLM_RETRY_BASE_SECONDS: float = 2.0
    LLM_RETRY_MAX_WAIT_SECONDS: float = 60.0
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:3b"
    ENABLE_RERANK: bool = False
    USE_GRAPH: bool = False
    AGENT_MAX_LOOPS: int = 2
    AGENT_MAX_PROVISIONS: int = 8
    MAX_CONTEXT_TOKENS: int = 3500
    MAX_SIBLINGS_PER_SECTION: int = 2
    MIN_RELEVANCE_SCORE: float = 0.28
    
    # Derived directory paths
    EXTRACTED_JSONS_DIR: str = ""
    NER_OUTPUTS_DIR: str = ""
    CHAT_DB_PATH: str = ""

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    def model_post_init(self, __context, /) -> None:
        if not self.EXTRACTED_JSONS_DIR:
            self.EXTRACTED_JSONS_DIR = os.path.join(self.BASE_DIR, "extracted_jsons")
        if not self.NER_OUTPUTS_DIR:
            self.NER_OUTPUTS_DIR = os.path.join(self.BASE_DIR, "ner_outputs")
        if not self.CHAT_DB_PATH:
            self.CHAT_DB_PATH = os.path.join(self.BASE_DIR, "data", "chat_history.db")

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Generate settings from environment"""
    return Settings()