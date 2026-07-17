"""
Search Script
=============
Interactive CLI that encodes a user query with BAAI/bge-m3, retrieves
the most relevant Nepali legal-text chunks from Qdrant, and optionally
re-ranks candidates using a BAAI/bge-reranker-v2-m3 cross-encoder for
higher accuracy.

Usage:
    python src/search.py                      # Interactive REPL (with re-ranking)
    python src/search.py -q "कम्पनी दर्ता"   # One-shot query
    python src/search.py --top-k 10           # Return 10 results
    python src/search.py --no-rerank          # Disable cross-encoder re-ranking
"""

import sys
import os

# Add project root to python path to allow running directly from src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import argparse
from typing import Optional

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.embedding.embedder import LegalChunkEmbedder
from src.embedding.vector_store import QdrantVectorStore
from src.embedding.reranker import CrossEncoderReranker
from src.config import get_settings

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
QDRANT_URL = get_settings().QDRANT_URL


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


def display_results(results: list[dict], query: str, reranked: bool = False) -> None:
    """Pretty-print search results to stdout."""
    mode_label = "re-ranked" if reranked else "retrieval"
    print(f"\n{'─' * 70}")
    print(f"  Query   : {query}")
    print(f"  Results : {len(results)}  ({mode_label})")
    print(f"{'─' * 70}")

    if not results:
        print("  No matching chunks found.\n")
        return

    for i, hit in enumerate(results, 1):
        chunk_id = hit["chunk_id"]
        act_source = hit["act_source"]
        hierarchy = hit.get("hierarchy", {})
        text = hit["text"]

        # Truncate long text for display
        display_text = text if len(text) <= 300 else text[:300] + "..."

        # Show cross-encoder score when available, with original score.
        if "rerank_score" in hit:
            score_line = (
                f"Score: {hit['rerank_score']:.4f} (rerank)  "
                f"│ {hit['original_score']:.4f} (retrieval)"
            )
        else:
            score_line = f"Score: {hit['score']:.4f}"

        print(f"\n  [{i}]  {score_line}")
        print(f"       Act   : {act_source}")
        print(f"       ID    : {chunk_id}")
        print(f"       Path  : {format_hierarchy(hierarchy)}")
        print(f"       Text  : {display_text}")

    print(f"\n{'─' * 70}\n")


# ---------------------------------------------------------------------------
# Over-fetch multiplier: when re-ranking is active, we retrieve this many
# times more candidates from Qdrant so the cross-encoder has a richer pool.
# ---------------------------------------------------------------------------
RERANK_FETCH_MULTIPLIER: int = 4


def run_search(
    embedder: LegalChunkEmbedder,
    store: QdrantVectorStore,
    query: str,
    search_type: str = "hybrid",
    top_k: int = 5,
    act_filter: Optional[str] = None,
    reranker: Optional[CrossEncoderReranker] = None,
) -> list[dict]:
    """Encode query (dense & sparse), search Qdrant, and optionally re-rank."""
    embeddings = embedder.embed_texts([query])
    query_dense = embeddings["dense_vecs"][0].tolist()
    query_sparse = embeddings["lexical_weights"][0]

    # When re-ranking, over-fetch candidates for a richer candidate pool.
    fetch_k = top_k * RERANK_FETCH_MULTIPLIER if reranker else top_k

    results = store.search(
        query_dense=query_dense,
        query_sparse=query_sparse,
        search_type=search_type,
        top_k=fetch_k,
        act_filter=act_filter,
    )

    # Stage 2: Cross-encoder re-ranking.
    if reranker and results:
        results = reranker.rerank(query=query, candidates=results, top_k=top_k)

    return results


def interactive_mode(
    embedder: LegalChunkEmbedder,
    store: QdrantVectorStore,
    search_type: str,
    top_k: int,
    act_filter: Optional[str],
    reranker: Optional[CrossEncoderReranker] = None,
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

    rerank_active = reranker is not None

    print(f"\n  Type your query in Nepali or English.")
    print(f"  Commands: 'exit'/'quit' to leave, 'top <N>' to change count,")
    print(f"            'type <dense|sparse|hybrid>' to change search type,")
    print(f"            'rerank on/off' to toggle cross-encoder re-ranking.\n")

    current_type = search_type
    
    while True:
        rerank_tag = "+rerank" if rerank_active else ""
        try:
            query = input(f"  🔍 Query ({current_type}{rerank_tag}) > ").strip()
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

        # Allow toggling cross-encoder re-ranking: "rerank on" / "rerank off"
        if query.lower().startswith("rerank "):
            toggle = query.split()[1].lower()
            if toggle == "on" and reranker is not None:
                rerank_active = True
                print("  ✓ Cross-encoder re-ranking enabled.\n")
            elif toggle == "on" and reranker is None:
                print("  ⚠  Reranker was not loaded (started with --no-rerank).\n")
            elif toggle == "off":
                rerank_active = False
                print("  ✓ Cross-encoder re-ranking disabled.\n")
            else:
                print("  Usage: rerank <on|off>\n")
            continue

        active_reranker = reranker if rerank_active else None
        results = run_search(
            embedder, store, query, current_type, top_k, act_filter,
            reranker=active_reranker,
        )
        display_results(results, query, reranked=active_reranker is not None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search Nepali legal chunks in Qdrant using BAAI/bge-m3 "
                    "with optional cross-encoder re-ranking.",
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
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable cross-encoder re-ranking (faster, less accurate).",
    )
    args = parser.parse_args()

    # ── Initialize embedder ─────────────────────────────────────────────
    print("\n  Loading BAAI/bge-m3 embedder... (first run downloads ~2 GB)")
    embedder = LegalChunkEmbedder(show_progress=False)
    store = QdrantVectorStore(url=args.qdrant_url)
    print("  Embedder ready.")

    # ── Initialize cross-encoder reranker ───────────────────────────────
    reranker: Optional[CrossEncoderReranker] = None
    if not args.no_rerank:
        print("  Loading cross-encoder reranker (BAAI/bge-reranker-v2-m3)...")
        reranker = CrossEncoderReranker()
        print("  Reranker ready.")
    else:
        print("  Cross-encoder re-ranking disabled (--no-rerank).")
    print()

    # ── One-shot or interactive ─────────────────────────────────────────
    if args.query:
        results = run_search(
            embedder, store, args.query, args.type, args.top_k, args.act,
            reranker=reranker,
        )
        display_results(results, args.query, reranked=reranker is not None)
    else:
        interactive_mode(
            embedder, store, args.type, args.top_k, args.act,
            reranker=reranker,
        )


if __name__ == "__main__":
    main()
