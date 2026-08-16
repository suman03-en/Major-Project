"""
Graph Query CLI
===============
Interactive CLI for querying the Neo4j knowledge graph of Nepali legal procedures.

Usage:
    python -m src.cli.query_graph                      # Interactive REPL
    python -m src.cli.query_graph --query "processes"  # One-shot command
"""

import sys
import logging
import argparse
import json
from typing import Optional

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.knowledge_base.graph_store import Neo4jGraphStore
from src.config import get_settings, setup_logging

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
setup_logging(level=logging.WARNING)
logger = logging.getLogger("query_graph")

# Fix Windows console encoding for Nepali text output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def display_processes(store: Neo4jGraphStore) -> None:
    processes = store.get_all_processes()
    print(f"\nFound {len(processes)} Processes:")
    for p in processes:
        print(f"  - {p}")
    print()


def display_steps(store: Neo4jGraphStore, process_name: str) -> None:
    steps = store.get_steps_for_process(process_name)
    if not steps:
        print(f"\nNo steps found for process: '{process_name}'\n")
        return

    print(f"\nSteps for '{process_name}' ({len(steps)} total):")
    for s in steps:
        print(f"\n  Step {s['step_number']}:")
        print(f"    Action : {s['action']}")
        if s['office']:
            print(f"    Office : {s['office']}")
        if s['documents']:
            print(f"    Docs   : {', '.join(str(d) for d in s['documents'] if d)}")
        if s['fees']:
            print(f"    Fees   : {', '.join(str(f) for f in s['fees'] if f)}")
        if s['duration']:
            print(f"    Time   : {s['duration']}")
        if s['prerequisite']:
            print(f"    Pre-req: {s['prerequisite']}")
        if s['clause_ref']:
            print(f"    Source : {s['clause_ref']}")
    print()


def display_docs(store: Neo4jGraphStore, process_name: str) -> None:
    docs = store.get_documents_for_process(process_name)
    if not docs:
        print(f"\nNo documents found for process: '{process_name}'\n")
        return
    print(f"\nDocuments required for '{process_name}':")
    for d in docs:
        print(f"  - {d}")
    print()


def display_offices(store: Neo4jGraphStore, process_name: str) -> None:
    offices = store.get_offices_for_process(process_name)
    if not offices:
        print(f"\nNo offices found for process: '{process_name}'\n")
        return
    print(f"\nOffices involved in '{process_name}':")
    for o in offices:
        print(f"  - {o}")
    print()


def display_stats(store: Neo4jGraphStore) -> None:
    stats = store.get_graph_stats()
    print("\nGraph Statistics:")
    print("  Nodes:")
    for node in stats.get("nodes", []):
        print(f"    {node['name']:<15}: {node['count']}")
    print("  Relationships:")
    for rel in stats.get("relationships", []):
        print(f"    {rel['name']:<15}: {rel['count']}")
    print()


def run_command(store: Neo4jGraphStore, cmd_text: str) -> None:
    cmd_text = cmd_text.strip()
    if not cmd_text:
        return

    parts = cmd_text.split(" ", 1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "processes":
        display_processes(store)
    elif cmd == "steps" and arg:
        display_steps(store, arg)
    elif cmd == "docs" and arg:
        display_docs(store, arg)
    elif cmd == "offices" and arg:
        display_offices(store, arg)
    elif cmd == "stats":
        display_stats(store)
    elif cmd == "cypher" and arg:
        try:
            res = store.run_cypher(arg)
            print(f"\nCypher Result ({len(res)} records):")
            for record in res:
                print(json.dumps(record, ensure_ascii=False, indent=2))
            print()
        except Exception as e:
            print(f"\nCypher Error: {e}\n")
    else:
        print("\nInvalid command or missing arguments.")
        print_help()


def print_help() -> None:
    print("\nAvailable Commands:")
    print("  processes              - List all business processes in the graph")
    print("  steps <process>        - List step-by-step procedure for a process")
    print("  docs <process>         - List all documents required for a process")
    print("  offices <process>      - List all offices involved in a process")
    print("  stats                  - Show graph node and relationship counts")
    print("  cypher <query>         - Run an arbitrary Cypher query")
    print("  help                   - Show this message")
    print("  exit / quit            - Exit interactive mode\n")


def interactive_mode(store: Neo4jGraphStore) -> None:
    print("\n" + "=" * 60)
    print("  NEO4J GRAPH QUERY CLI")
    print("=" * 60)
    print_help()

    while True:
        try:
            cmd_text = input("graph> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!\n")
            break

        if cmd_text.lower() in ("exit", "quit", "q"):
            print("Goodbye!\n")
            break
        elif cmd_text.lower() in ("help", "h"):
            print_help()
            continue

        run_command(store, cmd_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query Neo4j graph of NER outputs.")
    parser.add_argument("--query", "-q", default=None, help="Run a specific command and exit (e.g., 'processes')")
    args = parser.parse_args()

    settings = get_settings()

    try:
        store = Neo4jGraphStore(settings.NEO4J_URI, settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    except Exception as e:
        print(f"\nFailed to connect to Neo4j at {settings.NEO4J_URI}: {e}\n")
        sys.exit(1)

    with store:
        if args.query:
            run_command(store, args.query)
        else:
            interactive_mode(store)


if __name__ == "__main__":
    main()
