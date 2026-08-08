"""
Entity Aggregator for the NER extraction pipeline.

Groups clause-level NER extractions into coherent business
registration task workflows. Handles:
- Deduplication of documents and steps
- Office grouping and normalization
- Fee aggregation
- Task-level workflow construction
"""

import re
from typing import List, Dict, Optional
from collections import defaultdict

from src.knowledge_base.schemas import (
    ExtractedEntity,
    RegistrationTaskWorkflow,
    OfficeEntity,
    PriceFee,
)

# Mapping section titles (or keywords in them) to high-level task names
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
    Infer a high-level task name from the entity's clause_ref breadcrumb
    and any extracted step/document content.
    """
    # Combine the clause reference and extracted fields for keyword searching
    text_content = [entity.clause_ref]
    text_content.extend(entity.steps)
    text_content.extend(entity.documents_required)
    combined = " ".join(text_content)

    for pattern, task_name in TASK_NAME_PATTERNS:
        if re.search(pattern, combined):
            return task_name

    return "सामान्य प्रक्रिया"


def _deduplicate_strings(items: List[str]) -> List[str]:
    """Deduplicate a list of strings preserving order."""
    seen = set()
    result = []
    for item in items:
        normalized = item.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _deduplicate_fees(fees: List[PriceFee]) -> List[PriceFee]:
    """Deduplicate fees by raw amount text."""
    seen = set()
    result = []
    for fee in fees:
        key = fee.amount_raw or ""
        if key and key not in seen:
            seen.add(key)
            result.append(fee)
        elif not key:
            result.append(fee)
    return result


def _deduplicate_offices(offices: List[OfficeEntity]) -> List[OfficeEntity]:
    """Deduplicate offices by name."""
    seen = set()
    result = []
    for office in offices:
        if office.name not in seen:
            seen.add(office.name)
            result.append(office)
    return result


class EntityAggregator:
    """
    Aggregates clause-level NER extractions into task-level
    registration workflows.
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

        all_offices = []
        all_documents = []
        all_steps = []
        all_fees = []
        all_durations = []
        all_prerequisites = []
        source_clauses = []

        for entity in entities:
            source_clauses.append(entity.chunk_id)

            if entity.office:
                all_offices.append(entity.office)

            all_documents.extend(entity.documents_required)
            all_steps.extend(entity.steps)
            all_fees.extend(entity.price_fees)

            if entity.duration:
                all_durations.append(entity.duration)

            if entity.prerequisites:
                all_prerequisites.append(entity.prerequisites)

        # Build a description from the first entity's section title in clause_ref
        description = None
        for entity in entities:
            sec_match = re.search(r'sec\d+\s*\((.*?)\)', entity.clause_ref)
            if sec_match:
                description = sec_match.group(1)
                break

        return RegistrationTaskWorkflow(
            task_name=task_name,
            description=description,
            offices=_deduplicate_offices(all_offices),
            all_documents=_deduplicate_strings(all_documents),
            all_steps=_deduplicate_strings(all_steps),
            all_fees=_deduplicate_fees(all_fees),
            durations=_deduplicate_strings(all_durations),
            prerequisites=_deduplicate_strings(all_prerequisites),
            source_clauses=source_clauses,
        )
