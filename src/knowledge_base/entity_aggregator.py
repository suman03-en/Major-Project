"""
Entity Aggregator for the NER extraction pipeline.

Groups clause-level NER extractions into coherent business
registration task workflows. Each workflow contains ordered steps
with per-step office, documents, fees — ready for Neo4j ingestion.

Handles:
- Step deduplication by action text
- Sequential step re-numbering
- Task-level workflow construction from clause-level extractions
"""

import re
from typing import List, Dict
from collections import defaultdict

from src.knowledge_base.schemas import (
    ExtractedEntity,
    RegistrationTaskWorkflow,
    ProcessStep,
)

# Mapping patterns to high-level task names
TASK_NAME_PATTERNS = [
    (r"दर्ता", "उद्योग दर्ता"),
    (r"नवीकरण", "उद्योग नवीकरण"),
    (r"खारेज|विघटन", "उद्योग खारेज"),
    (r"नामसारी|हस्तान्तरण", "उद्योग नामसारी"),
    (r"अनुमति|इजाजत", "उद्योग अनुमतिपत्र"),
    (r"विस्तार|क्षमता", "उद्योग विस्तार"),
    (r"वातावरण", "वातावरणीय अनुमति"),
]


def _infer_task_name(entity: ExtractedEntity) -> str:
    """
    Infer a high-level task name from the entity's process_name,
    clause_ref, and step actions.
    """
    # First, check process_name (most reliable — set by LLM)
    if entity.process_name:
        for pattern, task_name in TASK_NAME_PATTERNS:
            if re.search(pattern, entity.process_name):
                return task_name

    # Then check clause_ref and step actions
    text_content = [entity.clause_ref]
    text_content.extend(step.action for step in entity.steps)
    combined = " ".join(text_content)

    for pattern, task_name in TASK_NAME_PATTERNS:
        if re.search(pattern, combined):
            return task_name

    return "सामान्य प्रक्रिया"


def _deduplicate_steps(steps: List[ProcessStep]) -> List[ProcessStep]:
    """Deduplicate steps by action text, preserving order and re-numbering."""
    seen_actions = set()
    unique_steps = []
    for step in steps:
        normalized = step.action.strip()
        if normalized and normalized not in seen_actions:
            seen_actions.add(normalized)
            unique_steps.append(step)

    # Re-number sequentially
    renumbered = []
    for i, step in enumerate(unique_steps):
        renumbered.append(step.model_copy(update={"step_number": i + 1}))

    return renumbered


class EntityAggregator:
    """
    Aggregates clause-level NER extractions into task-level
    registration workflows with ordered, per-step metadata.
    """

    def aggregate(self, entities: List[ExtractedEntity]) -> List[RegistrationTaskWorkflow]:
        """
        Group entities by inferred task name and build aggregated workflows.

        Args:
            entities: List of clause-level extracted entities.

        Returns:
            List of RegistrationTaskWorkflow objects.
        """
        if not entities:
            return []

        # Group entities by inferred task name
        task_groups: Dict[str, List[ExtractedEntity]] = defaultdict(list)
        for entity in entities:
            task_name = _infer_task_name(entity)
            task_groups[task_name].append(entity)

        # Build workflows from each group
        workflows = []
        for task_name, group_entities in task_groups.items():
            workflow = self._build_workflow(task_name, group_entities)
            workflows.append(workflow)

        return workflows

    def _build_workflow(
        self, task_name: str, entities: List[ExtractedEntity]
    ) -> RegistrationTaskWorkflow:
        """Build a single aggregated workflow from a group of related entities."""

        all_steps = []
        source_clauses = []

        for entity in entities:
            source_clauses.append(entity.chunk_id)
            all_steps.extend(entity.steps)

        # Deduplicate steps and re-number sequentially
        unique_steps = _deduplicate_steps(all_steps)

        # Build description from first entity with a process_name or section title
        description = None
        for entity in entities:
            if entity.process_name:
                description = entity.process_name
                break
            sec_match = re.search(r'sec\d+\s*\((.*?)\)', entity.clause_ref)
            if sec_match:
                description = sec_match.group(1)
                break

        return RegistrationTaskWorkflow(
            task_name=task_name,
            description=description,
            steps=unique_steps,
            source_clauses=source_clauses,
        )
