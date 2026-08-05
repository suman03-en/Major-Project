import os
import json
from pydantic import BaseModel, Field
from typing import List, Optional
from mistralai.client import Mistral

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config import get_settings

settings = get_settings()

class RegistrationProcess(BaseModel):
    office: Optional[str] = Field(default=None, description="The government office involved, e.g., कम्पनी रजिष्ट्रारको कार्यालय")
    documents_required: List[str] = Field(default_factory=list, description="List of documents needed, e.g., नागरिकता, निवेदन")
    steps: List[str] = Field(default_factory=list, description="Sequential steps to follow")
    price: Optional[str] = Field(default=None, description="Cost or fee, e.g., १००० रुपैयाँ")
    duration_days: Optional[str] = Field(default=None, description="Time taken, e.g., ३ दिन")

class NERExtractor:
    def __init__(self):
        api_key = settings.MISTRAL_API_KEY
        self.client = Mistral(api_key=api_key)
        self.model = "mistral-large-latest" # mistral-large supports structured output well

    def extract_entities(self, text: str) -> Optional[RegistrationProcess]:
        system_prompt = """You are a Named Entity Recognition assistant for Nepali text.
Your task is to extract business registration entities from the provided text according to this JSON schema:
{
  "office": "string or null",
  "documents_required": ["string1", "string2"],
  "steps": ["string1", "string2"],
  "price": "string or null",
  "duration_days": "string or null"
}

- office: The government office or authority (e.g., कम्पनी रजिष्ट्रारको कार्यालय).
- documents_required: Any documents, certificates, or applications required.
- steps: Sequential actions, duties, or procedures mentioned.
- price: Any fee, fine, or monetary cost (e.g., ५० रुपैयाँ, दस्तुर).
- duration_days: Any time duration or deadline (e.g., ३५ दिनभित्र, ३ महिना).

Output ONLY the JSON object. Do not include markdown formatting or additional explanation."""

        try:
            chat_response = self.client.chat.complete(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"}
            )
            
            content = chat_response.choices[0].message.content
            
            # Remove markdown backticks if Mistral included them despite the prompt
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
                
            data = json.loads(content)
            return RegistrationProcess(**data)
            
        except Exception as e:
            print(f"Error during extraction: {e}")
            return None
