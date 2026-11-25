"""
Ingestion Script
================
Reads all JSON datasets from ``extracted_jsons/``, embeds each chunk using
BAAI/bge-m3, and upserts the vectors into Qdrant.

Usage:
    python src/ingest.py                      # Ingest all datasets
    python src/ingest.py --recreate           # Drop & recreate collection first
"""

import argparse
import glob
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

from src.config import get_settings, setup_logging

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.embedding.embedder import LegalChunkEmbedder
from src.embedding.vector_store import QdrantVectorStore

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
setup_logging()
logger = logging.getLogger("ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_settings = get_settings()
EXTRACTED_JSONS_DIR = _settings.EXTRACTED_JSONS_DIR
QDRANT_URL = _settings.QDRANT_URL


def _file_sha256(filepath: str) -> str:
    """Compute SHA-256 hex digest of a file for ingestion idempotency."""
    h = hashlib.sha256()
    with open(filepath, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _load_manifest(manifest_path: Path) -> dict[str, str]:
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _save_manifest(manifest_path: Path, manifest: dict[str, str]) -> None:
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)


def load_datasets(directory: str) -> list[tuple[str, str, str, dict]]:
    """
    Load all ``*_dataset.json`` files from *directory*.

    Returns a list of ``(act_source_name, filepath, file_hash, dataset_dict)`` tuples.
    """
    pattern = os.path.join(directory, "*_dataset.json")
    files = sorted(glob.glob(pattern))

    if not files:
        logger.warning("No *_dataset.json files found in %s", directory)
        return []

    datasets: list[tuple[str, str, str, dict]] = []
    for filepath in files:
        act_source = Path(filepath).stem  # e.g. "comapy_act_dataset"
        file_hash = _file_sha256(filepath)
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        chunk_count = len(data.get("chunks", []))
        logger.info("Loaded %-35s  (%d chunks)", act_source, chunk_count)
        datasets.append((act_source, filepath, file_hash, data))

    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed extracted_jsons and upsert into Qdrant."
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the Qdrant collection before ingesting.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed and upsert all datasets even if unchanged.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=QDRANT_URL,
        help="Qdrant server URL (default: %(default)s).",
    )
    args = parser.parse_args()

    # ── 1. Load datasets ────────────────────────────────────────────────
    datasets = load_datasets(EXTRACTED_JSONS_DIR)
    if not datasets:
        logger.error("Nothing to ingest. Place *_dataset.json files in %s/", EXTRACTED_JSONS_DIR)
        sys.exit(1)

    manifest_path = Path(EXTRACTED_JSONS_DIR) / ".ingest_manifest.json"
    manifest = {} if args.recreate else _load_manifest(manifest_path)

    # Filter out unchanged datasets unless --recreate or --force is set
    pending = []
    for act_source, filepath, file_hash, dataset in datasets:
        if not args.force and not args.recreate and manifest.get(act_source) == file_hash:
            logger.info("Skipping unchanged '%s' (hash match). Use --force to re-ingest.", act_source)
            continue
        pending.append((act_source, filepath, file_hash, dataset))

    # ── 2. Initialize embedder & vector store ───────────────────────────
    store = QdrantVectorStore(url=args.qdrant_url)

    if args.recreate:
        try:
            store.delete_collection()
            logger.info("Existing collection dropped.")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Collection delete skipped: %s", exc)

    store.ensure_collection()

    if not pending:
        logger.info("All datasets are up to date. Nothing to ingest.")
        return

    embedder = LegalChunkEmbedder()

    # ── 3. Embed & upsert each dataset ──────────────────────────────────
    total_points = 0
    for act_source, filepath, file_hash, dataset in pending:
        chunks = dataset.get("chunks", [])
        if not chunks:
            logger.warning("Skipping '%s' – no chunks.", act_source)
            continue

        logger.info("Embedding %d chunks from '%s'...", len(chunks), act_source)
        texts = embedder.build_embedding_texts(chunks)
        embeddings = embedder.embed_texts(texts)

        logger.info("Upserting %d points into Qdrant...", len(chunks))
        upserted = store.upsert_chunks(
            chunks=chunks,
            dense_vecs=embeddings["dense_vecs"],
            sparse_vecs=embeddings["lexical_weights"],
            act_source=act_source,
        )
        total_points += upserted
        manifest[act_source] = file_hash

    _save_manifest(manifest_path, manifest)

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
