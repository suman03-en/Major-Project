"""
Search Script
=============
Interactive CLI that encodes a user query with BAAI/bge-m3 and retrieves
the most relevant Nepali legal-text chunks from Qdrant.

Usage:
    python search.py                      # Interactive REPL
    python search.py -q "कम्पनी दर्ता"   # One-shot query
    python search.py --top-k 10           # Return 10 results
"""
'''testing'''
import os
import sys
import logging
import argparse
from typing import Optional

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.embedding.embedder import LegalChunkEmbedder
from src.embedding.vector_store import QdrantVectorStore

# ---------------------------------------------------------------------------
# Logging (quiet by default for clean CLI output)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("search")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
QDRANT_URL = "http://localhost:6333"


def format_hierarchy(hierarchy: dict) -> str:
    """Build a human-readable breadcrumb from a hierarchy dict."""
    parts: list[str] = []
    if "ch_title" in hierarchy:
        parts.append(f"परिच्छेद {hierarchy.get('ch', '?')}: {hierarchy['ch_title']}")
    if "sec_title" in hierarchy:
        parts.append(f"दफा {hierarchy.get('sec', '?')}: {hierarchy['sec_title']}")
    if "sub" in hierarchy:
        parts.append(f"उपदफा ({hierarchy['sub']})")
    if "clause" in hierarchy:
        parts.append(f"खण्ड ({hierarchy['clause']})")
    return " → ".join(parts) if parts else "—"


def display_results(results: list[dict], query: str) -> None:
    """Pretty-print search results to stdout."""
    print(f"\n{'─' * 70}")
    print(f"  Query: {query}")
    print(f"  Results: {len(results)}")
    print(f"{'─' * 70}")

    if not results:
        print("  No matching chunks found.\n")
        return

    for i, hit in enumerate(results, 1):
        score = hit["score"]
        chunk_id = hit["chunk_id"]
        act_source = hit["act_source"]
        hierarchy = hit.get("hierarchy", {})
        text = hit["text"]

        # Truncate long text for display
        display_text = text if len(text) <= 300 else text[:300] + "..."

        print(f"\n  [{i}]  Score: {score:.4f}")
        print(f"       Act   : {act_source}")
        print(f"       ID    : {chunk_id}")
        print(f"       Path  : {format_hierarchy(hierarchy)}")
        print(f"       Text  : {display_text}")

    print(f"\n{'─' * 70}\n")


def run_search(
    embedder: LegalChunkEmbedder,
    store: QdrantVectorStore,
    query: str,
    search_type: str = "hybrid",
    top_k: int = 5,
    act_filter: Optional[str] = None,
) -> list[dict]:
    """Encode query (dense & sparse) and search Qdrant."""
    embeddings = embedder.embed_texts([query])
    query_dense = embeddings["dense_vecs"][0].tolist()
    query_sparse = embeddings["lexical_weights"][0]
    
    results = store.search(
        query_dense=query_dense,
        query_sparse=query_sparse,
        search_type=search_type,
        top_k=top_k,
        act_filter=act_filter,
    )
    return results


def interactive_mode(
    embedder: LegalChunkEmbedder,
    store: QdrantVectorStore,
    search_type: str,
    top_k: int,
    act_filter: Optional[str],
) -> None:
    """Run the interactive REPL search loop."""
    # Show collection stats on startup
    try:
        info = store.get_collection_info()
        print(f"\n  Qdrant collection : {info['name']}")
        print(f"  Total points      : {info['points_count']}")
        print(f"  Status            : {info['status']}")
    except Exception as exc:
        print(f"\n  ⚠  Could not reach Qdrant: {exc}")
        sys.exit(1)

    print(f"\n  Type your query in Nepali or English.")
    print(f"  Commands: 'exit'/'quit' to leave, 'top <N>' to change count, 'type <dense|sparse|hybrid>' to change search type.\n")

    current_type = search_type
    
    while True:
        try:
            query = input(f"  🔍 Query ({current_type}) > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!\n")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("  Goodbye!\n")
            break

        # Allow changing top_k on the fly: "top 10"
        if query.lower().startswith("top "):
            try:
                top_k = int(query.split()[1])
                print(f"  ✓ Showing top {top_k} results.\n")
            except (IndexError, ValueError):
                print("  Usage: top <number>\n")
            continue
            
        # Allow changing search_type on the fly: "type dense"
        if query.lower().startswith("type "):
            new_type = query.split()[1].lower()
            if new_type in ("dense", "sparse", "hybrid"):
                current_type = new_type
                print(f"  ✓ Search type set to '{current_type}'.\n")
            else:
                print("  Usage: type <dense|sparse|hybrid>\n")
            continue

        results = run_search(embedder, store, query, current_type, top_k, act_filter)
        display_results(results, query)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search Nepali legal chunks in Qdrant using BAAI/bge-m3."
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        default=None,
        help="One-shot query (skip interactive mode).",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["dense", "sparse", "hybrid"],
        default="hybrid",
        help="Search type (default: %(default)s).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5).",
    )
    parser.add_argument(
        "--act",
        type=str,
        default=None,
        help="Filter results to a specific act source name.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=QDRANT_URL,
        help="Qdrant server URL (default: %(default)s).",
    )
    args = parser.parse_args()

    # ── Initialize ──────────────────────────────────────────────────────
    print("\n  Loading BAAI/bge-m3 model... (first run downloads ~2 GB)")
    embedder = LegalChunkEmbedder(show_progress=False)
    store = QdrantVectorStore(url=args.qdrant_url)
    print("  Model ready.\n")

    # ── One-shot or interactive ─────────────────────────────────────────
    if args.query:
        results = run_search(embedder, store, args.query, args.type, args.top_k, args.act)
        display_results(results, args.query)
    else:
        interactive_mode(embedder, store, args.type, args.top_k, args.act)


if __name__ == "__main__":
    main()
