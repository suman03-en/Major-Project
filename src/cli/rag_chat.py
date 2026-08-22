"""
RAG Chat CLI
=============
Interactive CLI for the full Retrieval-Augmented Generation pipeline.
Retrieves context from Qdrant (chunks) and Neo4j (procedural steps),
then generates grounded answers via Mistral LLM with token streaming.

Usage:
    python -m src.cli.rag_chat                         # Interactive REPL
    python -m src.cli.rag_chat -q "कम्पनी दर्ता कसरी गर्ने?"  # One-shot query
    python -m src.cli.rag_chat --no-graph              # Skip Neo4j graph lookup
    python -m src.cli.rag_chat --top-k 10              # Retrieve 10 chunks
    python -m src.cli.rag_chat --rerank                # Enable cross-encoder reranking
"""

import sys
import os
import logging
import argparse
from typing import Optional

from src.config import get_settings, setup_logging
from src.rag.rag_pipeline import RAGPipeline
from src.rag.rag_pipeline import RAG_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Logging (quiet by default for clean CLI output)
# ---------------------------------------------------------------------------
setup_logging(level=logging.WARNING)
logger = logging.getLogger("rag_chat")

# Fix Windows console encoding for Nepali text output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def print_banner(rag: RAGPipeline) -> None:
    """Print startup banner with connection status."""
    print(f"\n{'═' * 70}")
    print(f"  नेपाली कानूनी RAG प्रणाली  (Nepali Legal RAG System)")
    print(f"{'═' * 70}")

    # Qdrant status
    try:
        info = rag.vector_store.get_collection_info()
        print(f"  Qdrant        : {info['name']} ({info['points_count']} chunks)")
    except Exception as exc:
        print(f"  Qdrant        : ⚠ Not reachable ({exc})")

    # Neo4j status
    if rag.graph_store:
        try:
            stats = rag.graph_store.get_graph_stats()
            node_count = sum(n["count"] for n in stats.get("nodes", []))
            print(f"  Neo4j Graph   : Connected ({node_count} nodes)")
        except Exception:
            print(f"  Neo4j Graph   : ⚠ Connected but query failed")
    else:
        print(f"  Neo4j Graph   : Disabled")

    # LLM status
    print(f"  LLM Model     : {rag.llm_client.model}")

    # Reranker status
    rerank_status = "Enabled" if rag.reranker else "Disabled"
    print(f"  Reranker      : {rerank_status}")

    print(f"{'═' * 70}")
    print()


def print_help() -> None:
    """Print available interactive commands."""
    print("\n  Commands:")
    print("    context         — Show retrieval context from last query")
    print("    top <N>         — Change number of retrieved chunks")
    print("    type <mode>     — Change search type (dense|sparse|hybrid)")
    print("    rerank on/off   — Toggle cross-encoder reranking")
    print("    graph on/off    — Toggle Neo4j graph context")
    print("    help            — Show this message")
    print("    exit / quit     — Exit\n")


def display_context(rag: RAGPipeline) -> None:
    """Display the retrieval context from the last query."""
    ctx = getattr(rag, "last_retrieval_context", None)
    if not ctx:
        print("\n  No previous query context available.\n")
        return

    chunks = ctx.get("chunks", [])
    graph = ctx.get("graph", [])

    print(f"\n{'─' * 70}")
    print(f"  Retrieved Chunks: {len(chunks)}")
    print(f"{'─' * 70}")

    for i, chunk in enumerate(chunks, 1):
        act = chunk.get("act_source", "?")
        text = chunk.get("text", "")
        display_text = text if len(text) <= 200 else text[:200] + "..."

        if "rerank_score" in chunk:
            score = f"{chunk['rerank_score']:.4f} (rerank)"
        else:
            score = f"{chunk.get('score', 0):.4f}"

        print(f"\n  [{i}] Score: {score}  |  Act: {act}")
        print(f"      {display_text}")

    if graph:
        print(f"\n{'─' * 70}")
        print(f"  Graph Processes: {len(graph)}")
        print(f"{'─' * 70}")
        for proc in graph:
            name = proc["process_name"]
            step_count = len(proc.get("steps", []))
            doc_count = len(proc.get("documents", []))
            print(f"\n  Process: {name}")
            print(f"    Steps: {step_count} | Documents: {doc_count}")

    print(f"\n{'─' * 70}\n")


def run_one_shot(rag: RAGPipeline, query: str, show_context: bool = False) -> None:
    """Run a single query and print the answer."""
    print(f"\n  Query: {query}\n")
    print(f"  Retrieving context...", end=" ", flush=True)

    # Use streaming for the one-shot output
    print("Generating answer...\n")
    print(f"{'─' * 70}")

    full_response = []
    for token in rag.answer_stream(query):
        print(token, end="", flush=True)
        full_response.append(token)

    print(f"\n{'─' * 70}\n")

    if show_context:
        display_context(rag)


def interactive_mode(rag: RAGPipeline) -> None:
    """Run the interactive REPL chat loop."""
    print_banner(rag)
    print("  Ask questions about Nepali business registration laws.")
    print("  Type 'help' for commands.\n")

    while True:
        try:
            query = input("  💬 rag> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!\n")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("  Goodbye!\n")
            break
        if query.lower() in ("help", "h"):
            print_help()
            continue
        if query.lower() == "context":
            display_context(rag)
            continue

        # ── top <N> command ──────────────────────────────────────────
        if query.lower().startswith("top "):
            try:
                rag.top_k = int(query.split()[1])
                print(f"  ✓ Retrieving top {rag.top_k} chunks.\n")
            except (IndexError, ValueError):
                print("  Usage: top <number>\n")
            continue

        # ── type <mode> command ──────────────────────────────────────
        if query.lower().startswith("type "):
            new_type = query.split()[1].lower()
            if new_type in ("dense", "sparse", "hybrid"):
                rag.search_type = new_type
                print(f"  ✓ Search type: {rag.search_type}\n")
            else:
                print("  Usage: type <dense|sparse|hybrid>\n")
            continue

        # ── rerank on/off command ────────────────────────────────────
        if query.lower().startswith("rerank "):
            toggle = query.split()[1].lower()
            if toggle == "on":
                if rag.reranker:
                    rag.use_reranker = True
                    print("  ✓ Reranking enabled.\n")
                else:
                    print("  ⚠ Reranker not loaded (start with --rerank).\n")
            elif toggle == "off":
                rag.use_reranker = False
                print("  ✓ Reranking disabled.\n")
            else:
                print("  Usage: rerank <on|off>\n")
            continue

        # ── graph on/off command ─────────────────────────────────────
        if query.lower().startswith("graph "):
            toggle = query.split()[1].lower()
            if toggle == "on":
                if rag.graph_store:
                    rag.use_graph = True
                    print("  ✓ Graph context enabled.\n")
                else:
                    print("  ⚠ Neo4j not connected.\n")
            elif toggle == "off":
                rag.use_graph = False
                print("  ✓ Graph context disabled.\n")
            else:
                print("  Usage: graph <on|off>\n")
            continue

        # ── Run RAG query ────────────────────────────────────────────
        print()
        print(f"  Retrieving context... ", end="", flush=True)

        # To show reranking progress, we manually step through the pipeline
        chunks = rag.retrieve_chunks(query)
        if rag.reranker:
            # Notice the user that reranker is running
            print("Reranking... ", end="", flush=True)
            
        graph_context = []
        if rag.use_graph and rag.graph_store:
            graph_context = rag.retrieve_graph_context(chunks, query)

        context = rag.assemble_context(query, chunks, graph_context)
        rag.last_retrieval_context = {
            "chunks": chunks,
            "graph": graph_context,
            "context": context,
        }

        # Stream the response
        first_token = True
        for token in rag.llm_client.stream(RAG_SYSTEM_PROMPT, context):
            if first_token:
                print(f"Done.\n")
                print(f"{'─' * 70}")
                first_token = False
            print(token, end="", flush=True)

        if first_token:
            # No tokens received
            print("No response generated.\n")
        else:
            print(f"\n{'─' * 70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nepali Legal RAG Chat — Ask questions about business registration laws.",
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
        help="Qdrant search type (default: hybrid).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5).",
    )
    parser.add_argument(
        "--act",
        type=str,
        default=None,
        help="Filter results to a specific act source.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help="Qdrant server URL (default: from .env).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Mistral model name (default: from .env or mistral-large-latest).",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        default=False,
        help="Enable cross-encoder reranking (disabled by default).",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        default=False,
        help="Disable Neo4j graph context lookup.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        default=False,
        help="Show retrieval context after one-shot query.",
    )
    args = parser.parse_args()

    settings = get_settings()

    # ── Initialize components ────────────────────────────────────────
    print("\n  Initialising RAG pipeline...")

    from src.embedding.embedder import LegalChunkEmbedder
    from src.embedding.vector_store import QdrantVectorStore
    from src.llm.llm_client import MistralLLMClient

    print("  Loading BGE-M3 embedder... (first run downloads ~2 GB)")
    embedder = LegalChunkEmbedder(show_progress=False)
    print("  ✓ Embedder ready.")

    qdrant_url = args.qdrant_url or settings.QDRANT_URL
    vector_store = QdrantVectorStore(url=qdrant_url)

    reranker = None
    if args.rerank:
        from src.embedding.reranker import CrossEncoderReranker
        print("  Loading cross-encoder reranker...")
        reranker = CrossEncoderReranker()
        print("  ✓ Reranker ready.")

    llm_client = MistralLLMClient(model=args.model)
    print(f"  ✓ LLM client ready ({llm_client.model}).")

    rag = RAGPipeline(
        embedder=embedder,
        vector_store=vector_store,
        llm_client=llm_client,
        reranker=reranker,
        use_graph=not args.no_graph,
        use_reranker=args.rerank,
        top_k=args.top_k,
        search_type=args.type,
    )

    # ── Run ──────────────────────────────────────────────────────────
    try:
        if args.query:
            run_one_shot(rag, args.query, show_context=args.show_context)
        else:
            interactive_mode(rag)
    finally:
        rag.close()


if __name__ == "__main__":
    main()
