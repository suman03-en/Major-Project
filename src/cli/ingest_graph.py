"""
Graph Ingestion Script
======================
Reads all NER JSON datasets from `ner_outputs/`, and ingests them into the Neo4j graph database.

Usage:
    python -m src.cli.ingest_graph                      # Ingest all datasets
    python -m src.cli.ingest_graph --clear              # Drop all existing graph nodes/edges first
    python -m src.cli.ingest_graph --input path/to.json # Ingest specific file
"""

import sys
import os
import glob
import json
import logging
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.knowledge_base.graph_store import Neo4jGraphStore
from src.config import get_settings, setup_logging

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
setup_logging()
logger = logging.getLogger("ingest_graph")

# Fix Windows console encoding for Nepali text output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_settings = get_settings()
NER_OUTPUTS_DIR = _settings.NER_OUTPUTS_DIR


def load_datasets(directory: str, specific_input: str = None) -> list[tuple[str, dict]]:
    """Load JSON files to ingest."""
    if specific_input:
        files = [specific_input]
    else:
        pattern = os.path.join(directory, "*.json")
        files = sorted(glob.glob(pattern))

    if not files:
        logger.warning("No JSON files found in %s", directory)
        return []

    datasets = []
    for filepath in files:
        act_source = Path(filepath).stem
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            # Basic validation
            if "entities" in data:
                entity_count = len(data["entities"])
                logger.info("Loaded %-35s (%d entities)", act_source, entity_count)
                datasets.append((act_source, data))
            else:
                logger.warning("Skipping %s: no 'entities' key found", act_source)
        except Exception as e:
            logger.error("Failed to load %s: %s", filepath, e)

    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest NER outputs into Neo4j graph database."
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the graph database before ingesting.",
    )
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="Path to a specific NER output JSON file. Defaults to all in ner_outputs/.",
    )
    args = parser.parse_args()

    datasets = load_datasets(NER_OUTPUTS_DIR, args.input)
    if not datasets:
        logger.error("Nothing to ingest.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  NEO4J GRAPH INGESTION")
    print("=" * 60)

    try:
        with Neo4jGraphStore(_settings.NEO4J_URI, _settings.NEO4J_USER, _settings.NEO4J_PASSWORD) as store:
            if args.clear:
                store.clear_graph()
            
            store.ensure_constraints()
            
            for act_source, data in datasets:
                print(f"Ingesting: {data.get('act_title', act_source)} ...")
                store.ingest_ner_result(data)
            
            # Print final stats
            stats = store.get_graph_stats()
            print("\n" + "=" * 60)
            print("  GRAPH STATISTICS")
            print("=" * 60)
            print("Nodes:")
            for node in stats.get("nodes", []):
                print(f"  {node['name']:<15}: {node['count']}")
            print("\nRelationships:")
            for rel in stats.get("relationships", []):
                print(f"  {rel['name']:<15}: {rel['count']}")
            print("=" * 60 + "\n")

    except Exception as e:
        logger.error("Graph operations failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
