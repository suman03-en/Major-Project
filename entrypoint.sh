#!/bin/sh
set -e

if [ "$RUN_INGEST" = "true" ]; then
    python src/cli/ingest.py
fi

# Pass any extra arguments (e.g. docker compose run rag python src/cli/search.py)
# Default: just keep the container alive so you can exec into it
if [ $# -gt 0 ]; then
    exec "$@"
else
    echo "Container ready. Use 'docker compose exec rag python src/cli/search.py' to search."
    echo "Or 'docker compose run rag python src/cli/pdf_extractor.py' to extract PDFs."
    tail -f /dev/null
fi