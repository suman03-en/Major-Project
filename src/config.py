import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    BASE_DIR: str
    QDRANT_URL: str
    
def get_settings() -> Settings:
    """Generate settings from environment"""
    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        raise ValueError("QDRANT_URL is not defined in environment variables")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    return Settings(BASE_DIR=base_dir, QDRANT_URL=qdrant_url)

    