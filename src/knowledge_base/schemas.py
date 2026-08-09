"""
Pydantic data models for the NER extraction pipeline.

Defines step-centric structured output schemas optimized for Neo4j graph database ingestion.
Each procedural step carries its own associated office, documents, fees, and duration,
enabling rich graph relationships like:
  (:Step)-[:PERFORMED_AT]->(:Office)
  (:Step)-[:REQUIRES_DOCUMENT]->(:Document)
  (:Step)-[:HAS_FEE]->(:Fee)
  (:Step)-[:NEXT_STEP]->(:Step)
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class PriceFee(BaseModel):
    """A single fee or monetary cost extracted from a legal clause."""
    type: Optional[str] = Field(default=None, description="Fee category, e.g., दर्ता दस्तुर, नवीकरण दस्तुर")
    amount_raw: Optional[str] = Field(default=None, description="Original Nepali text, e.g., रु. १०,०००")
    amount_npr: Optional[int] = Field(default=None, description="Normalized integer amount in NPR")


class OfficeEntity(BaseModel):
    """A government office or authority referenced in the clause."""
    name: str = Field(description="Office name in Nepali, e.g., कम्पनी रजिष्ट्रारको कार्यालय")
    level: Optional[str] = Field(default=None, description="Government level: केन्द्र, प्रदेश, स्थानीय")


class RawStepOutput(BaseModel):
    """
    Raw step output from LLM before post-processing.
    This is the JSON schema the LLM is prompted to return per step.
    """
    step_number: int = Field(description="Sequential position of this step in the process")
    action: str = Field(default="", description="Complete description of the action/step in Nepali")
    office: Optional[str] = Field(default=None, description="Government office where this step is performed")
    documents_required: List[str] = Field(default_factory=list, description="Documents needed for this specific step")
    fee: Optional[str] = Field(default=None, description="Government fee for this step as Nepali text")
    duration: Optional[str] = Field(default=None, description="Time limit or deadline for this step")
    prerequisite: Optional[str] = Field(default=None, description="Condition that must be met before this step")

    @field_validator('documents_required', mode='before')
    @classmethod
    def filter_invalid_documents(cls, v):
        """Remove single-character entries and obvious non-documents."""
        if not isinstance(v, list):
            return []
        return [item for item in v if isinstance(item, str) and len(item.strip()) > 2]

    @field_validator('action', mode='before')
    @classmethod
    def ensure_action_string(cls, v):
        """Ensure action is always a non-None string."""
        if v is None:
            return ""
        return str(v)


class ClauseNEROutput(BaseModel):
    """
    Step-centric structured NER output from a single chunk/clause.
    This is the JSON schema the LLM is prompted to return.

    Designed for Neo4j graph DB: each step carries its own office,
    documents, fee, and duration — enabling per-step graph relationships.
    """
    process_name: Optional[str] = Field(default=None, description="Name of the registration/administrative process described")
    steps: List[RawStepOutput] = Field(default_factory=list, description="Sequential procedural steps with per-step metadata")

    @field_validator('process_name', mode='before')
    @classmethod
    def normalize_process_name(cls, v):
        """Handle LLM returning dict instead of string for process_name."""
        if isinstance(v, dict):
            return v.get('name') or v.get('process_name') or None
        return v


class ProcessStep(BaseModel):
    """
    A cleaned, structured procedural step ready for Neo4j graph DB ingestion.

    Maps to Neo4j relationships:
      (:Step)-[:PERFORMED_AT]->(:Office)
      (:Step)-[:REQUIRES_DOCUMENT]->(:Document)
      (:Step)-[:HAS_FEE]->(:Fee)
      (:Step)-[:NEXT_STEP]->(:Step)
    """
    step_number: int = Field(description="Sequential position (1-based) within the process")
    action: str = Field(description="Complete description of the action/step in Nepali")
    office: Optional[OfficeEntity] = Field(default=None, description="Office where this step is performed")
    documents_required: List[str] = Field(default_factory=list, description="Documents needed for this specific step")
    price_fees: List[PriceFee] = Field(default_factory=list, description="Fees for this specific step")
    duration: Optional[str] = Field(default=None, description="Time limit or deadline for this step")
    prerequisite: Optional[str] = Field(default=None, description="Condition required before this step")


class ExtractedEntity(BaseModel):
    """
    Step-centric NER extraction result for a single clause, optimized for Neo4j.

    Each entity represents a process (or fragment of a process) extracted from
    a legal clause, with ordered steps carrying per-step metadata.

    Traceability is maintained via:
      - chunk_id: programmatic key to look up the full clause in extracted_jsons/*.json
      - clause_ref: human-readable breadcrumb string showing the hierarchy path
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
    process_name: Optional[str] = Field(default=None, description="Name of the process described in this clause")
    steps: List[ProcessStep] = Field(default_factory=list, description="Ordered procedural steps with per-step metadata")


class RegistrationTaskWorkflow(BaseModel):
    """
    Aggregated registration workflow grouping multiple clause-level
    extractions into a single coherent business registration task.

    Steps are ordered and carry per-step office, documents, fees —
    enabling graph queries like "What documents do I need at step 3?"
    """
    task_name: str = Field(description="High-level task name, e.g., उद्योग दर्ता")
    description: Optional[str] = None
    steps: List[ProcessStep] = Field(default_factory=list, description="All procedural steps with per-step metadata, ordered")
    source_clauses: List[str] = Field(default_factory=list, description="List of chunk_ids that contributed to this workflow")


class NERPipelineResult(BaseModel):
    """Top-level output of the entire NER extraction pipeline for one act/document."""
    act_title: str
    act_slug: str
    total_chunks_processed: int
    total_chunks_filtered: int
    entities: List[ExtractedEntity] = Field(default_factory=list)
    workflows: List[RegistrationTaskWorkflow] = Field(default_factory=list)
