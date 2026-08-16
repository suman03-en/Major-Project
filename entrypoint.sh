#!/bin/sh
set -e

# Configurable environment flags with default fallbacks
RUN_EXTRACT="${RUN_EXTRACT:-false}"
RUN_INGEST="${RUN_INGEST:-false}"
RECREATE_COLLECTION="${RECREATE_COLLECTION:-false}"
RUN_GRAPH_INGEST="${RUN_GRAPH_INGEST:-false}"

# Step 1: Extract text from PDFs into structured JSONs
if [ "$RUN_EXTRACT" = "true" ]; then
    echo "========================================"
    echo "Step 1: Extracting text from PDFs..."
    echo "========================================"
    python src/cli/pdf_extractor.py
fi

# Step 2: Embed JSON datasets and ingest into Qdrant
if [ "$RUN_INGEST" = "true" ]; then
    echo "========================================"
    echo "Step 2: Ingesting datasets into Qdrant..."
    echo "========================================"
    INGEST_ARGS=""
    if [ "$RECREATE_COLLECTION" = "true" ]; then
        INGEST_ARGS="--recreate"
    fi
    python src/cli/ingest.py $INGEST_ARGS
fi

# Step 3: Ingest graph data into Neo4j
if [ "$RUN_GRAPH_INGEST" = "true" ]; then
    echo "========================================"
    echo "Step 3: Ingesting NER outputs into Neo4j graph..."
    echo "========================================"
    GRAPH_INGEST_ARGS=""
    if [ "$RECREATE_COLLECTION" = "true" ]; then
        GRAPH_INGEST_ARGS="--clear"
    fi
    python -m src.cli.ingest_graph $GRAPH_INGEST_ARGS
fi

# Handle custom command arguments or fallback to keep-alive mode
if [ $# -gt 0 ]; then
    exec "$@"
else
    echo ""
    echo "Container ready."
    echo "Commands you can run inside container:"
    echo "  Extract PDFs  : docker compose exec rag python src/cli/pdf_extractor.py"
    echo "  Ingest JSONs  : docker compose exec rag python src/cli/ingest.py"
    echo "  Run NER       : docker compose exec rag python -m src.cli.extract_ner --input extracted_jsons/<file>.json"
    echo "  Ingest Graph  : docker compose exec rag python -m src.cli.ingest_graph"
    echo "  Query Graph   : docker compose exec -it rag python -m src.cli.query_graph"
    echo "  Search CLI    : docker compose exec -it rag python src/cli/search.py"
    echo ""
    tail -f /dev/null
fi