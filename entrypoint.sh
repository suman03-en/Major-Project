#!/bin/sh
set -e

# Configurable environment flags with default fallbacks
RUN_EXTRACT="${RUN_EXTRACT:-false}"
RUN_INGEST="${RUN_INGEST:-false}"
RECREATE_COLLECTION="${RECREATE_COLLECTION:-false}"

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

# Handle custom command arguments or fallback to keep-alive mode
if [ $# -gt 0 ]; then
    exec "$@"
else
    echo ""
    echo "Container ready."
    echo "Commands you can run inside container:"
    echo "  Extract PDFs  : docker compose exec rag python src/cli/pdf_extractor.py"
    echo "  Ingest JSONs  : docker compose exec rag python src/cli/ingest.py"
    echo "  Search CLI    : docker compose exec -it rag python src/cli/search.py"
    echo ""
    tail -f /dev/null
fi