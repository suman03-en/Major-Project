"""
Ingestion Script
================
Reads all JSON datasets from ``output_jsons/``, embeds each chunk using
BAAI/bge-m3, and upserts the vectors into Qdrant.

Usage:
    python src/ingest.py                      # Ingest all datasets
    python src/ingest.py --recreate           # Drop & recreate collection first
"""

import sys
import os

# Add project root to python path to allow running directly from src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
import json
import logging
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.embedding.embedder import LegalChunkEmbedder
from src.embedding.vector_store import QdrantVectorStore

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_JSONS_DIR = os.path.join(PROJECT_ROOT, "output_jsons")
QDRANT_URL = "http://localhost:6333"


def load_datasets(directory: str) -> list[tuple[str, dict]]:
    """
    Load all ``*_dataset.json`` files from *directory*.

    Returns a list of ``(act_source_name, dataset_dict)`` tuples.
    """
    pattern = os.path.join(directory, "*_dataset.json")
    files = sorted(glob.glob(pattern))

    if not files:
        logger.warning("No *_dataset.json files found in %s", directory)
        return []

    datasets: list[tuple[str, dict]] = []
    for filepath in files:
        act_source = Path(filepath).stem  # e.g. "comapy_act_dataset"
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        chunk_count = len(data.get("chunks", []))
        logger.info("Loaded %-35s  (%d chunks)", act_source, chunk_count)
        datasets.append((act_source, data))

    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed output_jsons and upsert into Qdrant."
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the Qdrant collection before ingesting.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=QDRANT_URL,
        help="Qdrant server URL (default: %(default)s).",
    )
    args = parser.parse_args()

    # ── 1. Load datasets ────────────────────────────────────────────────
    datasets = load_datasets(OUTPUT_JSONS_DIR)
    if not datasets:
        logger.error("Nothing to ingest. Place *_dataset.json files in %s/", OUTPUT_JSONS_DIR)
        sys.exit(1)

    # ── 2. Initialize embedder & vector store ───────────────────────────
    embedder = LegalChunkEmbedder()
    store = QdrantVectorStore(url=args.qdrant_url)

    if args.recreate:
        try:
            store.delete_collection()
            logger.info("Existing collection dropped.")
        except Exception:
            pass  # Collection may not exist yet.

    store.ensure_collection()

    # ── 3. Embed & upsert each dataset ──────────────────────────────────
    total_points = 0
    for act_source, dataset in datasets:
        chunks = dataset.get("chunks", [])
        if not chunks:
            logger.warning("Skipping '%s' – no chunks.", act_source)
            continue

        logger.info("Embedding %d chunks from '%s'...", len(chunks), act_source)
        points = embedder.embed_chunks(chunks, act_source=act_source)

        logger.info("Upserting %d points into Qdrant...", len(points))
        store.upsert_points(points)
        total_points += len(points)

    # ── 4. Summary ──────────────────────────────────────────────────────
    info = store.get_collection_info()
    print("\n" + "=" * 60)
    print("  INGESTION COMPLETE")
    print("=" * 60)
    print(f"  Acts ingested   : {len(datasets)}")
    print(f"  Points upserted : {total_points}")
    print(f"  Collection      : {info['name']}")
    print(f"  Total points    : {info['points_count']}")
    print(f"  Status          : {info['status']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
