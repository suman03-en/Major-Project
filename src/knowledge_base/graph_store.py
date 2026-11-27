"""
Neo4j Graph Store
=================
Manages a Neo4j database for storing and searching Nepali legal-text
NER extracted entities.

Usage:
    from src.knowledge_base.graph_store import Neo4jGraphStore
    from src.config import get_settings

    settings = get_settings()
    with Neo4jGraphStore(settings.NEO4J_URI, settings.NEO4J_USER, settings.NEO4J_PASSWORD) as store:
        store.ensure_constraints()
        store.ingest_ner_result(ner_data)
"""

import logging
from typing import Any

from neo4j import GraphDatabase, Transaction

logger = logging.getLogger(__name__)

_NEPALI_CASE_SUFFIXES = ("सम्बन्धी", "हरू", "हरु", "बाट", "लाई", "को", "का", "की", "मा", "ले")


def _nepali_stem(word: str) -> str:
    """Strip common Nepali postpositions/case suffixes from tokens longer than 4 chars."""
    cleaned = word.strip(".,?!:;()[]\"'।")
    for suffix in _NEPALI_CASE_SUFFIXES:
        if len(cleaned) > len(suffix) + 2 and cleaned.endswith(suffix):
            return cleaned[: -len(suffix)]
    return cleaned


class Neo4jGraphStore:
    """Wrapper around Neo4j driver for legal NER graph."""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
    ) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s", uri)

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def run_cypher(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Run arbitrary Cypher query and return results as list of dicts."""
        parameters = parameters or {}
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]

    def ensure_constraints(self) -> None:
        """Create uniqueness constraints for node properties."""
        constraints = [
            "CREATE CONSTRAINT act_slug IF NOT EXISTS FOR (a:Act) REQUIRE a.slug IS UNIQUE",
            "CREATE CONSTRAINT process_uid IF NOT EXISTS FOR (p:Process) REQUIRE p.uid IS UNIQUE",
            "CREATE CONSTRAINT step_uid IF NOT EXISTS FOR (s:Step) REQUIRE s.uid IS UNIQUE",
            "CREATE CONSTRAINT office_name IF NOT EXISTS FOR (o:Office) REQUIRE o.name IS UNIQUE",
            "CREATE CONSTRAINT document_name IF NOT EXISTS FOR (d:Document) REQUIRE d.name IS UNIQUE",
        ]
        with self.driver.session() as session:
            for query in constraints:
                session.run(query)
        logger.info("Neo4j constraints verified.")

    def clear_graph(self) -> None:
        """Delete all nodes and relationships in the database efficiently."""
        query = "CALL { MATCH (n) DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS"
        with self.driver.session() as session:
            session.run(query)
        logger.info("Neo4j graph cleared.")

    def ingest_ner_result(self, ner_data: dict[str, Any]) -> None:
        """Ingest a full NER result JSON (NERPipelineResult schema) into Neo4j."""
        act_title = ner_data.get("act_title", "Unknown Act")
        act_slug = ner_data.get("act_slug", "unknown_act")
        entities = ner_data.get("entities", [])

        if not entities:
            logger.warning("No entities to ingest for act '%s'", act_title)
            return

        def _ingest_tx(tx: Transaction):
            # 1. Merge Act
            tx.run(
                """
                MERGE (a:Act {slug: $slug})
                SET a.title = $title
                """,
                slug=act_slug, title=act_title
            )

            # Flatten steps for batching
            all_steps = []
            all_documents = []
            all_fees = []

            for entity in entities:
                process_name = entity.get("process_name")
                if not process_name:
                    continue  # Skip entities with no process

                process_name = process_name.strip()
                process_uid = f"{act_slug}_{process_name}"
                chunk_id = entity.get("chunk_id", "")
                clause_ref = entity.get("clause_ref", "")
                steps = entity.get("steps", [])

                # 2. Merge Process and link to Act
                tx.run(
                    """
                    MATCH (a:Act {slug: $act_slug})
                    MERGE (p:Process {uid: $process_uid})
                    ON CREATE SET p.name = $process_name
                    MERGE (a)-[:HAS_PROCESS]->(p)
                    """,
                    act_slug=act_slug, process_uid=process_uid, process_name=process_name
                )

                prev_step_uid = None

                for step in steps:
                    step_number = step.get("step_number")
                    step_uid = f"{act_slug}_{chunk_id}_{step_number}"
                    
                    office_name = None
                    office_level = None
                    office = step.get("office")
                    if office and office.get("name"):
                        office_name = office.get("name").strip()
                        office_level = office.get("level")

                    all_steps.append({
                        "process_uid": process_uid,
                        "uid": step_uid,
                        "step_number": step_number,
                        "action": step.get("action", ""),
                        "duration": step.get("duration"),
                        "prerequisite": step.get("prerequisite"),
                        "chunk_id": chunk_id,
                        "clause_ref": clause_ref,
                        "office_name": office_name,
                        "office_level": office_level,
                        "prev_uid": prev_step_uid
                    })

                    prev_step_uid = step_uid

                    for doc_name in step.get("documents_required", []):
                        if doc_name:
                            all_documents.append({"step_uid": step_uid, "name": doc_name.strip()})

                    for i, fee in enumerate(step.get("price_fees", [])):
                        all_fees.append({
                            "step_uid": step_uid,
                            "uid": f"{step_uid}_fee_{i}",
                            "amount_raw": fee.get("amount_raw"),
                            "amount_npr": fee.get("amount_npr"),
                            "type": fee.get("type")
                        })

            # 3. Batch Create Steps
            if all_steps:
                tx.run(
                    """
                    UNWIND $steps AS step
                    MATCH (p:Process {uid: step.process_uid})
                    MERGE (s:Step {uid: step.uid})
                    ON CREATE SET 
                        s.step_number = step.step_number,
                        s.action = step.action,
                        s.duration = step.duration,
                        s.prerequisite = step.prerequisite,
                        s.chunk_id = step.chunk_id,
                        s.clause_ref = step.clause_ref
                    MERGE (p)-[:HAS_STEP]->(s)
                    
                    WITH s, step
                    WHERE step.prev_uid IS NOT NULL
                    MATCH (prev:Step {uid: step.prev_uid})
                    MERGE (prev)-[:NEXT_STEP]->(s)
                    
                    WITH s, step
                    WHERE step.office_name IS NOT NULL
                    MERGE (o:Office {name: step.office_name})
                    ON CREATE SET o.level = step.office_level
                    MERGE (s)-[:PERFORMED_AT]->(o)
                    """,
                    steps=all_steps
                )

            # 4. Batch Create Documents
            if all_documents:
                tx.run(
                    """
                    UNWIND $docs AS doc
                    MATCH (s:Step {uid: doc.step_uid})
                    MERGE (d:Document {name: doc.name})
                    MERGE (s)-[:REQUIRES_DOCUMENT]->(d)
                    """,
                    docs=all_documents
                )

            # 5. Batch Create Fees
            if all_fees:
                tx.run(
                    """
                    UNWIND $fees AS fee
                    MATCH (s:Step {uid: fee.step_uid})
                    MERGE (f:Fee {uid: fee.uid})
                    ON CREATE SET 
                        f.amount_raw = fee.amount_raw,
                        f.amount_npr = fee.amount_npr,
                        f.type = fee.type
                    MERGE (s)-[:HAS_FEE]->(f)
                    """,
                    fees=all_fees
                )

        with self.driver.session() as session:
            session.execute_write(_ingest_tx)
        
        logger.info("Ingested NER results for act '%s'", act_title)

    def get_graph_stats(self) -> dict[str, Any]:
        """Return counts of different node and relationship types."""
        node_query = "MATCH (n) RETURN labels(n)[0] AS name, count(*) AS count"
        nodes = self.run_cypher(node_query)
        
        rel_query = "MATCH ()-[r]->() RETURN type(r) AS name, count(r) AS count"
        rels = self.run_cypher(rel_query)
        
        return {
            "nodes": [n for n in nodes if n["name"] is not None],
            "relationships": [r for r in rels if r["name"] is not None]
        }

    def get_all_processes(self) -> list[str]:
        """Return list of all process names."""
        query = "MATCH (p:Process) RETURN p.name AS name ORDER BY name"
        results = self.run_cypher(query)
        return [r["name"] for r in results if r["name"]]

    def get_steps_for_process(self, process_name: str) -> list[dict[str, Any]]:
        """Return ordered steps for a given process."""
        query = """
        MATCH (p:Process {name: $process_name})-[:HAS_STEP]->(s:Step)
        OPTIONAL MATCH (s)-[:PERFORMED_AT]->(o:Office)
        OPTIONAL MATCH (s)-[:REQUIRES_DOCUMENT]->(d:Document)
        OPTIONAL MATCH (s)-[:HAS_FEE]->(f:Fee)
        RETURN s.step_number AS step_number, 
               s.action AS action, 
               s.duration AS duration, 
               s.prerequisite AS prerequisite,
               s.clause_ref AS clause_ref,
               o.name AS office,
               collect(DISTINCT d.name) AS documents,
               collect(DISTINCT f.amount_raw) AS fees
        ORDER BY s.step_number
        """
        return self.run_cypher(query, {"process_name": process_name})

    def get_documents_for_process(self, process_name: str) -> list[str]:
        """Return all unique documents required across a whole process."""
        query = """
        MATCH (:Process {name: $process_name})-[:HAS_STEP]->(:Step)-[:REQUIRES_DOCUMENT]->(d:Document)
        RETURN DISTINCT d.name AS document
        ORDER BY document
        """
        results = self.run_cypher(query, {"process_name": process_name})
        return [r["document"] for r in results if r["document"]]

    def get_offices_for_process(self, process_name: str) -> list[str]:
        """Return all unique offices involved in a whole process."""
        query = """
        MATCH (:Process {name: $process_name})-[:HAS_STEP]->(:Step)-[:PERFORMED_AT]->(o:Office)
        RETURN DISTINCT o.name AS office
        ORDER BY office
        """
        results = self.run_cypher(query, {"process_name": process_name})
        return [r["office"] for r in results if r["office"]]

    def find_relevant_processes(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        max_processes: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Match Neo4j processes against the user query and retrieved chunks,
        returning full step, document, and office details.
        """
        all_processes = self.get_all_processes()
        if not all_processes:
            return []

        search_text = query.lower()
        for chunk in chunks:
            search_text += " " + chunk.get("text", "").lower()

        matching_processes: list[str] = [
            name for name in all_processes if name.lower() in search_text
        ]

        if not matching_processes:
            query_stems = {_nepali_stem(w) for w in query.lower().split() if len(w) > 2}
            scored: list[tuple[int, str]] = []
            for name in all_processes:
                proc_stems = {_nepali_stem(w) for w in name.lower().split() if len(w) > 1}
                overlap = sum(
                    1
                    for ps in proc_stems
                    if any(
                        ps == qs or (len(ps) >= 3 and len(qs) >= 3 and (qs.startswith(ps) or ps.startswith(qs)))
                        for qs in query_stems
                    )
                )
                if overlap > 0:
                    scored.append((overlap, name))
            scored.sort(key=lambda pair: pair[0], reverse=True)
            matching_processes = [name for _, name in scored]

        graph_results: list[dict[str, Any]] = []
        for process_name in matching_processes[:max_processes]:
            graph_results.append({
                "process_name": process_name,
                "steps": self.get_steps_for_process(process_name),
                "documents": self.get_documents_for_process(process_name),
                "offices": self.get_offices_for_process(process_name),
            })

        return graph_results
