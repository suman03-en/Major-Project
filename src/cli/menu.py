"""
Interactive Menu CLI
====================
Numbered interactive launcher for all Nepali Legal RAG operations.

Usage:
    python -m src.cli.menu
"""

import glob
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.config import get_settings


@dataclass(frozen=True)
class MenuItem:
    """Represents a single selectable action in the CLI menu."""

    label: str
    action: Callable[[], None]


def _run_module(module: str, *args: str) -> None:
    """Execute a Python module as a foreground subprocess."""
    cmd = [sys.executable, "-m", module, *args]
    print(f"\n▶ Running: {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print("\nInterrupted. Returning to menu...")


def _prompt_yes_no(prompt: str, default: bool = False) -> bool:
    """Prompt the user for a y/n confirmation."""
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input(f"{prompt} {suffix}: ").strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


def _action_extract_pdfs() -> None:
    _run_module("src.cli.pdf_extractor")


def _action_ingest_qdrant() -> None:
    args: list[str] = []
    if _prompt_yes_no("Drop and recreate Qdrant collection first?", default=False):
        args.append("--recreate")
    elif _prompt_yes_no("Force re-embed unchanged datasets?", default=False):
        args.append("--force")
    _run_module("src.cli.ingest", *args)


def _action_extract_ner() -> None:
    settings = get_settings()
    pattern = os.path.join(settings.EXTRACTED_JSONS_DIR, "*_dataset.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"\nNo *_dataset.json files found in {settings.EXTRACTED_JSONS_DIR}.")
        return

    print("\nSelect a dataset JSON for NER extraction:")
    for idx, filepath in enumerate(files, 1):
        print(f"  {idx}. {Path(filepath).name}")
    print("  0. Cancel")

    raw = input("\nEnter file number: ").strip()
    if not raw.isdigit():
        print("Invalid selection.")
        return
    choice = int(raw)
    if choice == 0:
        return
    if not (1 <= choice <= len(files)):
        print("Choice out of range.")
        return

    selected_file = files[choice - 1]
    print("\nSelect LLM provider:")
    print("  1. auto (Local Ollama if running, else Mistral)")
    print("  2. ollama")
    print("  3. mistral")
    prov_raw = input("Enter provider number [1]: ").strip() or "1"
    provider_map = {"1": "auto", "2": "ollama", "3": "mistral"}
    provider = provider_map.get(prov_raw, "auto")

    _run_module("src.cli.extract_ner", "--input", selected_file, "--provider", provider)


def _action_ingest_graph() -> None:
    args: list[str] = []
    if _prompt_yes_no("Clear existing Neo4j graph before ingesting?", default=False):
        args.append("--clear")
    _run_module("src.cli.ingest_graph", *args)


def _action_rag_chat() -> None:
    args: list[str] = []
    if _prompt_yes_no("Enable cross-encoder reranking?", default=False):
        args.append("--rerank")
    _run_module("src.cli.rag_chat", *args)


def _action_search_cli() -> None:
    _run_module("src.cli.search")


def _action_query_graph() -> None:
    _run_module("src.cli.query_graph")


def _action_web_server() -> None:
    _run_module("uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "8000")


def build_menu_items() -> list[MenuItem]:
    """Return the ordered list of interactive menu actions."""
    return [
        MenuItem("Extract text from PDFs (Surya OCR)", _action_extract_pdfs),
        MenuItem("Ingest extracted JSONs into Qdrant", _action_ingest_qdrant),
        MenuItem("Run NER extraction on a dataset", _action_extract_ner),
        MenuItem("Ingest NER outputs into Neo4j graph", _action_ingest_graph),
        MenuItem("Start interactive RAG Chat CLI", _action_rag_chat),
        MenuItem("Start Vector Search CLI", _action_search_cli),
        MenuItem("Start Knowledge Graph Query CLI", _action_query_graph),
        MenuItem("Start FastAPI Web Server (port 8000)", _action_web_server),
    ]


def run_menu(items: list[MenuItem] | None = None) -> None:
    """Run the interactive numbered menu loop."""
    menu_items = items if items is not None else build_menu_items()

    while True:
        print("\n" + "=" * 54)
        print("  NEPALI LEGAL RAG — INTERACTIVE MENU")
        print("=" * 54)
        for idx, item in enumerate(menu_items, 1):
            print(f"  {idx}. {item.label}")
        print("  0. Exit")
        print("=" * 54)

        try:
            raw = input("Select an option (number): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not raw.isdigit():
            print("Please enter a valid number.")
            continue

        choice = int(raw)
        if choice == 0:
            print("Goodbye!")
            break
        if 1 <= choice <= len(menu_items):
            menu_items[choice - 1].action()
        else:
            print(f"Invalid option '{choice}'. Choose 0–{len(menu_items)}.")


def main() -> None:
    run_menu()


if __name__ == "__main__":
    main()
