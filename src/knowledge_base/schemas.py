"""
Pydantic data models for the NER extraction pipeline.

Defines the structured output schemas for:
- Clause-level Mistral API extraction results
- Normalized entity representations (compact, no source text duplication)
- Task-level aggregated registration workflows
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class PriceFee(BaseModel):
    """A single fee or monetary cost extracted from a legal clause."""
    type: Optional[str] = Field(default=None, description="Fee category, e.g., दर्ता दस्तुर, जरिवाना")
    amount_raw: Optional[str] = Field(default=None, description="Original Nepali text, e.g., रु. १०,०००")
    amount_npr: Optional[int] = Field(default=None, description="Normalized integer amount in NPR")


class OfficeEntity(BaseModel):
    """A government office or authority referenced in the clause."""
    name: str = Field(description="Office name in Nepali, e.g., कम्पनी रजिष्ट्रारको कार्यालय")
    level: Optional[str] = Field(default=None, description="Government level: केन्द्र, प्रदेश, स्थानीय")


class ClauseNEROutput(BaseModel):
    """
    Structured NER output from a single chunk/clause sent to Mistral.
    This is the JSON schema the Mistral API is prompted to return.
    """
    office: Optional[str] = Field(default=None, description="Government office or authority name")
    documents_required: List[str] = Field(default_factory=list, description="List of required documents, certificates, or applications")
    steps: List[str] = Field(default_factory=list, description="Sequential procedural steps or duties")
    price: Optional[str] = Field(default=None, description="Fee, fine, or monetary cost as text")
    duration_days: Optional[str] = Field(default=None, description="Time duration or deadline as text")
    prerequisites: Optional[str] = Field(default=None, description="Any eligibility criteria or prerequisites")


class ExtractedEntity(BaseModel):
    """
    A compact NER extraction result for a single clause.

    Traceability is maintained via:
      - chunk_id: programmatic key to look up the full clause in extracted_jsons/*.json
      - clause_ref: human-readable breadcrumb string showing the hierarchy path

    Redundant fields removed vs. earlier version:
      - hierarchy (dict) → encoded as clause_ref string
      - chunk_type → encoded inside clause_ref
      - source_text → available in extracted_jsons via chunk_id lookup
    """
    chunk_id: str = Field(
        description="Source chunk ID for programmatic lookup in extracted_jsons/, e.g. -2076-ch2-sec3-sub1"
    )
    clause_ref: str = Field(
        description=(
            "Human-readable breadcrumb of the clause hierarchy. "
            "Format: 'ch{n} (title) › sec{n} (title) › sub{n} [type]'. "
            "Example: 'ch2 (उद्योग दर्ता) › sec3 (दर्ता गराउनु पर्ने) › sub6 [subsection]'"
        )
    )
    office: Optional[OfficeEntity] = None
    documents_required: List[str] = Field(default_factory=list)
    steps: List[str] = Field(default_factory=list)
    price_fees: List[PriceFee] = Field(default_factory=list)
    duration: Optional[str] = None
    prerequisites: Optional[str] = None


class RegistrationTaskWorkflow(BaseModel):
    """
    Aggregated registration workflow grouping multiple clause-level
    extractions into a single coherent business registration task.
    """
    task_name: str = Field(description="High-level task name, e.g., उद्योग दर्ता")
    description: Optional[str] = None
    offices: List[OfficeEntity] = Field(default_factory=list, description="All offices involved")
    all_documents: List[str] = Field(default_factory=list, description="Deduplicated list of all required documents")
    all_steps: List[str] = Field(default_factory=list, description="Ordered list of all procedural steps")
    all_fees: List[PriceFee] = Field(default_factory=list, description="All fees and costs")
    durations: List[str] = Field(default_factory=list, description="All time durations and deadlines")
    prerequisites: List[str] = Field(default_factory=list, description="All eligibility criteria")
    source_clauses: List[str] = Field(default_factory=list, description="List of chunk_ids that contributed to this workflow")


class NERPipelineResult(BaseModel):
    """Top-level output of the entire NER extraction pipeline for one act/document."""
    act_title: str
    act_slug: str
    total_chunks_processed: int
    total_chunks_filtered: int
    entities: List[ExtractedEntity] = Field(default_factory=list)
    workflows: List[RegistrationTaskWorkflow] = Field(default_factory=list)
